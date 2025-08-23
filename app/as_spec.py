# filepath: app/as_spec.py
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# ---- small util ----------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

class Dot:
    """
    Simple dict -> attribute access for safe eval in router rules:
      db.secs_since_last_msg, features.energy, text.fan_last_n, ...
    """
    def __init__(self, d: Dict[str, Any]): self.__dict__.update(d)
    def to_dict(self) -> Dict[str, Any]:  return dict(self.__dict__)

# ---- feature extraction ---------------------------------------------------------

_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")
_EXCLAIM_RE = re.compile(r"[!?]")
_WHITESPACE_RE = re.compile(r"\s+")

# keyword sets (tuned to be broad but safe)
_PRICE_PAT = re.compile(
    r"\b(price|how\s*much|cost|menu|ppv|unlock|pay(ment)?|tip|charge|buy|"
    r"send\s*(me)?\s*price|discount)\b",
    re.I,
)

# direct requests for media
_EXPLICIT_REQ_PAT = re.compile(
    r"(send|show|see|share).{0,18}\b(pic|photo|vid(eo)?|menu|nude|nudes|boobs|tits|ass|pussy|explicit)\b",
    re.I,
)

# sexual/NSFW cues (not a blocker; used to pivot to tease)
_SEX_PAT = re.compile(
    r"\b(horny|spicy|nude|naked|nudes|boobs|tits|pussy|dick|cock|cum|bj|blowjob|"
    r"sext|hard|fuck|anal|fetish|kinky)\b",
    re.I,
)

# low-energy / no-time / cool-down signals
_NEG_PAT = re.compile(
    r"\b(not now|later|busy|tired|sleep|working|no\s*time|cant|can't|sorry|"
    r"not\s*interested|stop|leave|maybe\s*later)\b|^(k|idk|ok)$",
    re.I,
)

def _normalize(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", (s or "")).strip().lower()

def _avg_len(texts: List[str]) -> float:
    if not texts: return 0.0
    return sum(len(_normalize(t)) for t in texts) / len(texts)

def _emoji_score(texts: List[str]) -> float:
    if not texts: return 0.0
    cnt = sum(len(_EMOJI_RE.findall(t)) for t in texts)
    return _clamp(cnt / (len(texts) * 4.0), 0.0, 1.0)  # ~4 emojis per msg = 1.0

def _exclaim_score(texts: List[str]) -> float:
    if not texts: return 0.0
    cnt = sum(len(_EXCLAIM_RE.findall(t)) for t in texts)
    return _clamp(cnt / (len(texts) * 6.0), 0.0, 1.0)  # ~6 !/? across chunk = 1.0

def _gap_score(seconds: int) -> float:
    # 0s=>1.0, 5min=>~0.6, 30min=>0, 2h=>0
    return _clamp(1.0 - (seconds / 1800.0), 0.0, 1.0)

def _regex_hits(texts: List[str], pat: re.Pattern) -> int:
    return sum(1 for t in texts if pat.search(t or ""))

# ---- DB pulls (kept here to keep main.py small) --------------------------------

def _recent_messages(conn, thread_id: str, limit: int = 16) -> List[Dict[str, Any]]:
    cast = "::uuid" if "-" in thread_id else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select role, text, created_at
            from messages
            where thread_id=%s{cast}
            order by id desc
            limit %s
            """,
            (thread_id, limit),
        )
        rows = cur.fetchall()
    # newest first ➜ reverse to oldest..newest
    return rows[::-1]

def _message_stats(conn, thread_id: str) -> Dict[str, Any]:
    cast = "::uuid" if "-" in thread_id else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select
              count(*) filter (where role='fan')     as fan_count,
              count(*) filter (where role='creator') as creator_count,
              max(created_at)                        as last_any_at,
              max(created_at) filter (where role='fan')     as last_fan_at,
              max(created_at) filter (where role='creator') as last_creator_at
            from messages where thread_id=%s{cast}
            """,
            (thread_id,),
        )
        r = cur.fetchone() or {}
    now = datetime.now(timezone.utc)
    def _gap(ts):
        if not ts: return 10**9
        return int((now - ts).total_seconds())
    secs_since_last_any = _gap(r.get("last_any_at"))
    last_fan_gap_secs = _gap(r.get("last_fan_at"))
    last_creator_gap_secs = _gap(r.get("last_creator_at"))
    return {
        "fan_msg_count": int(r.get("fan_count") or 0),
        "creator_msg_count": int(r.get("creator_count") or 0),
        "secs_since_last_msg": secs_since_last_any,
        "last_fan_gap_secs": last_fan_gap_secs,
        "last_creator_gap_secs": last_creator_gap_secs,
    }

def _ltv_cents(conn, thread_id: str) -> int:
    cast = "::uuid" if "-" in thread_id else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select coalesce(sum(price_cents) filter (where status='paid'),0)::int as ltv
            from ppv_offers where thread_id=%s{cast}
            """,
            (thread_id,),
        )
        r = cur.fetchone() or {"ltv": 0}
    return int(r["ltv"])

# ---- helpers object -------------------------------------------------------------

@dataclass
class Helpers:
    fan_texts: List[str]
    features: Dot
    db: Dot

    def fan_text_any(self, kws: List[str]) -> bool:
        texts = " ".join(self.fan_texts).lower()
        return any(k.lower() in texts for k in kws)

    def fan_text_re(self, pattern: str) -> bool:
        try:
            pat = re.compile(pattern, re.I)
        except re.error:
            return False
        return any(bool(pat.search(t or "")) for t in self.fan_texts)

    def price_intent(self) -> bool:
        return self.features.price_intent >= 0.45

    def explicit_request(self) -> bool:
        return any(bool(_EXPLICIT_REQ_PAT.search(t or "")) for t in self.fan_texts)

    def low_energy(self) -> bool:
        return (self.features.energy < 0.35) or (self.db.last_fan_gap_secs > 3600)

    def energy_rising(self) -> bool:
        lens = [len(_normalize(t)) for t in self.fan_texts[-6:]]
        if len(lens) < 4: return False
        prev = sum(lens[:-2]) / max(1, len(lens) - 2)
        recent = (lens[-1] + lens[-2]) / 2.0
        return (recent > prev * 1.2) or (recent >= 40 and self.db.last_fan_gap_secs < 600)

    def new_conversation(self) -> bool:
        # "fresh" if little history OR thread just restarted
        few_msgs = (self.db.fan_msg_count + self.db.creator_msg_count) <= 3
        recent = self.db.secs_since_last_msg < 24 * 3600
        return few_msgs and recent

# ---- main entry: build as_spec_v0_2 --------------------------------------------

def build_as_spec_v0_2(conn, thread_id: str, n: int = 8) -> Dict[str, Any]:
    """
    Returns:
      {
        'db': Dot(...),
        'text': Dot({'fan_last_n':[...], 'creator_last_n':[...]}),
        'features': Dot({'energy':float,'sexual_cue':float,'price_intent':float,'negativity':float}),
        'helpers': Helpers(...),    # callables used by the router
        'recent_messages': [...],   # (optional) for logging/diagnostics
      }
    """
    recent = _recent_messages(conn, thread_id, limit=max(8, n * 2))
    fan_texts = [r["text"] for r in recent if r["role"] == "fan"][-n:]
    creator_texts = [r["text"] for r in recent if r["role"] == "creator"][-n:]

    # DB stats
    stats = _message_stats(conn, thread_id)
    stats["ltv_cents"] = _ltv_cents(conn, thread_id)

    # --- features ---
    avg_len = _avg_len(fan_texts[-4:])
    len_score = _clamp(avg_len / 120.0)                # ~120 chars avg => 1.0
    gap_sc = _gap_score(stats["last_fan_gap_secs"])
    emo_sc = _clamp(0.5 * _emoji_score(fan_texts[-4:]) + 0.5 * _exclaim_score(fan_texts[-4:]))
    energy = _clamp(0.4 * len_score + 0.4 * gap_sc + 0.2 * emo_sc)

    total = max(1, len(fan_texts))
    price_hits = _regex_hits(fan_texts, _PRICE_PAT)
    explicit_hits = _regex_hits(fan_texts, _EXPLICIT_REQ_PAT)
    sex_hits = _regex_hits(fan_texts, _SEX_PAT)
    neg_hits = _regex_hits(fan_texts, _NEG_PAT)

    price_intent = _clamp((price_hits / total) + 0.25 * (1 if explicit_hits > 0 else 0))
    sexual_cue = _clamp((sex_hits / total) + 0.30 * (1 if explicit_hits > 0 else 0))
    negativity = _clamp(neg_hits / total)

    db = Dot(stats)
    text = Dot({"fan_last_n": fan_texts, "creator_last_n": creator_texts})
    features = Dot({"energy": energy, "sexual_cue": sexual_cue, "price_intent": price_intent, "negativity": negativity})
    helpers = Helpers(fan_texts=fan_texts, features=features, db=db)

    return {
        "db": db,
        "text": text,
        "features": features,
        "helpers": helpers,
        "recent_messages": [{"role": r["role"], "text": r["text"]} for r in recent],
    }
