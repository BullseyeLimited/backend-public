# app/cards_router.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

_BASE = Path(__file__).resolve().parent
_DET_DIR = _BASE / "coach_cards" / "stage_detectors"

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

class DotDict(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v

def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def _list_detectors() -> List[dict]:
    out: List[dict] = []
    if _DET_DIR.exists():
        for p in sorted(_DET_DIR.glob("*.json")):
            try:
                d = _load_json(p)
                if isinstance(d, dict) and d.get("type") == "stage_detector":
                    d["_file"] = p.name
                    out.append(d)
            except Exception:
                pass
    return out

_RE_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")
_RE_SEX   = re.compile(r"(nude|tits?|boobs?|ass|pussy|dick|bj|blowjob|nsfw|spicy|naked|cum|horny)", re.I)
_RE_PRICE = re.compile(r"(\$|\bprice|how\s*much|cost|menu|ppv|tip|unlock|buy)\b", re.I)
_RE_NEG   = re.compile(r"\b(later|busy|tired|no time|not now|broke|expensive|idk|nah|maybe)\b", re.I)
_RE_EXPL  = re.compile(r"(send|show|drop).{0,22}\b(pic|photo|video|vid|preview)\b", re.I)

def _energy_score(fan_texts: List[str]) -> float:
    if not fan_texts: return 0.0
    tail = fan_texts[-5:]
    avg_len = sum(len(t) for t in tail) / max(1, len(tail))
    exclam  = sum(t.count("!") for t in tail)
    emojis  = sum(len(_RE_EMOJI.findall(t)) for t in tail)
    base = _clamp(avg_len / 100.0, 0.0, 0.85)
    bonus = _clamp(0.05 * exclam + 0.06 * emojis, 0.0, 0.30)
    return _clamp(base + bonus)

def _features_from_recent(recent: List[Dict[str, Any]]) -> Tuple[dict, dict]:
    fan = [m.get("text","") for m in recent if (m.get("role") == "fan")]
    cr  = [m.get("text","") for m in recent if (m.get("role") == "creator")]

    energy = _energy_score(fan)
    sexual = _clamp(sum(1 for t in fan[-8:] if _RE_SEX.search(t)) / 4.0)
    price  = _clamp(sum(1 for t in fan[-8:] if _RE_PRICE.search(t)) / 3.0)
    neg    = _clamp(sum(1 for t in fan[-8:] if _RE_NEG.search(t)) / 4.0)

    return ({"fan_last_n": fan[-8:], "creator_last_n": cr[-8:]},
            {"energy": energy, "sexual_cue": sexual, "price_intent": price, "negativity": neg})

_HEARTS_RE   = re.compile(r"[❤♥️💖💘💗💓💞💝😍🤗🥰]", re.U)
_GRAT_RE     = re.compile(r"\b(thank(s| you)|appreciate|grateful)\b", re.I)
_PET_RE      = re.compile(r"\b(babe|baby|hun|honey|handsome|love|luv|sweet(ie)?)\b", re.I)
_SOFTENER_RE = re.compile(r"\b(ok if not|no worries|i understand|hope that’s ok|hope thats ok)\b", re.I)
_APOL_RE     = re.compile(r"\b(sorry|apologize|my bad)\b", re.I)
_FORG_RE     = re.compile(r"\b(forgive|we’re ok|we are ok|we’re good|we are good|we ok|all good)\b", re.I)
_ACCEPT_RE   = re.compile(r"\b(no problem|i understand|i get it|that’s ok|thats ok)\b", re.I)
_BOUND_REQ_RE= re.compile(r"\b(meet|number|whats(app)?|free\b.*(pic|photo|video|preview)|call|video\s*chat)\b", re.I)
_HOSTILE_RE  = re.compile(r"\b(deserve|other creators|why not|you should|deal with it)\b", re.I)
_JEALOUS_RE  = re.compile(r"\b(saw you (reply|online)|ignored me|others|why not me)\b", re.I)
_CONTENT_REF = re.compile(r"\b(video|clip|set|photo|pic|post|story|bikini|cosplay)\b", re.I)
_LAUGH_RE    = re.compile(r"\b(lol|lmao|rofl|haha|hehe)\b", re.I)

def _mk_helpers(db: dict, text: dict, features: dict):
    fan_tail = [x.lower() for x in text.get("fan_last_n", [])]
    cr_tail  = [x.lower() for x in text.get("creator_last_n", [])]

    def fan_text_any(words: List[str]) -> bool:
        if not words: return False
        ws = [w.lower() for w in words]
        return any(any(w in t for w in ws) for t in fan_tail)

    def fan_text_re(pattern: str) -> bool:
        try: r = re.compile(pattern, re.I)
        except Exception: return False
        return any(r.search(t) for t in fan_tail)

    def price_intent() -> bool:
        return features.get("price_intent", 0.0) > 0.45 or fan_text_re(r"(\$|\bprice|how\s*much|cost|menu|ppv|tip|unlock|buy)\b")

    def explicit_request() -> bool:
        return any(_RE_EXPL.search(t) for t in fan_tail)

    def low_energy() -> bool:
        if features.get("energy", 0.0) < 0.35: return True
        last = [len(t) for t in fan_tail[-4:]]
        return last and sum(1 for L in last if L <= 7) >= max(2, len(last)//2)

    def energy_rising() -> bool:
        last = [len(t) for t in fan_tail[-5:]]
        return len(last) >= 3 and (last[-1] - last[0]) > 20

    def new_conversation() -> bool:
        if (db.get("fan_msg_count",0) + db.get("creator_msg_count",0)) <= 3: return True
        return fan_text_any(["hey","hi","hello"]) and len(fan_tail) <= 2

    def tier_is(name: str) -> bool:
        tg = (db.get("tier_guess") or "").lower()
        return tg == (name or "").lower()

    # gold-related lexical helpers (kept)
    def gratitude_present() -> bool:
        return any(_GRAT_RE.search(t) for t in fan_tail)

    def affection_present() -> bool:
        hearts = any(_HEARTS_RE.search(t) for t in fan_tail)
        pets   = any(_PET_RE.search(t) for t in fan_tail)
        return hearts or pets

    def petname_present() -> bool:
        return any(_PET_RE.search(t) for t in fan_tail)

    def content_ref() -> bool:
        return any(_CONTENT_REF.search(t) for t in fan_tail)

    def apology_present() -> bool:
        return any(_APOL_RE.search(t) for t in fan_tail + cr_tail)

    def forgiveness_present() -> bool:
        return any(_FORG_RE.search(t) for t in fan_tail + cr_tail)

    def acceptance_present() -> bool:
        return any(_ACCEPT_RE.search(t) for t in fan_tail + cr_tail)

    def boundary_request_present() -> bool:
        return any(_BOUND_REQ_RE.search(t) for t in fan_tail)

    def softener_present() -> bool:
        return any(_SOFTENER_RE.search(t) for t in fan_tail + cr_tail)

    def hostile_push_present() -> bool:
        return any(_HOSTILE_RE.search(t) for t in fan_tail)

    def jealousy_trigger_present() -> bool:
        return any(_JEALOUS_RE.search(t) for t in fan_tail)

    def warm_reentry_present() -> bool:
        return fan_text_any(["missed you","welcome back","there you are"]) or affection_present()

    def pre_signal_present() -> bool:
        return fan_text_any(["no rush","ttyl","talk later","busy today","back later"])

    def gap_hours() -> float:
        return float(db.get("secs_since_last_msg", 999999)) / 3600.0

    def recent_paid(hours: int = 24) -> bool:
        st  = (db.get("ppv_last_status") or "").lower()
        sec = int(db.get("ppv_last_secs") or 10**9)
        if st == "paid" and sec <= hours*3600: return True
        return fan_text_any(["just subscribed","i subscribed","unlocked","i tipped","first time here"])

    def laughter_present() -> bool:
        return any(_LAUGH_RE.search(t) for t in fan_tail + cr_tail)

    # memory-based helpers
    def inside_joke_memory() -> bool:
        return int(db.get("mem_inside_jokes_count", 0)) > 0

    def petname_memory(by: str = "any") -> bool:
        fc = int(db.get("mem_petnames_fan_count", 0))
        cc = int(db.get("mem_petnames_creator_count", 0))
        if by == "fan": return fc > 0
        if by == "creator": return cc > 0
        return (fc + cc) > 0

    # scheduling proximity (optional use)
    def next_schedule_within(hours: float) -> bool:
        ns = db.get("next_schedule_secs")
        return ns is not None and ns <= hours*3600

    return {
        # base
        "fan_text_any": fan_text_any,
        "fan_text_re":  fan_text_re,
        "price_intent": price_intent,
        "explicit_request": explicit_request,
        "low_energy": low_energy,
        "energy_rising": energy_rising,
        "new_conversation": new_conversation,
        "tier_is": tier_is,
        # gold helpers
        "gratitude_present": gratitude_present,
        "affection_present": affection_present,
        "petname_present": petname_present,
        "content_ref": content_ref,
        "apology_present": apology_present,
        "forgiveness_present": forgiveness_present,
        "acceptance_present": acceptance_present,
        "boundary_request_present": boundary_request_present,
        "softener_present": softener_present,
        "hostile_push_present": hostile_push_present,
        "jealousy_trigger_present": jealousy_trigger_present,
        "warm_reentry_present": warm_reentry_present,
        "pre_signal_present": pre_signal_present,
        "gap_hours": gap_hours,
        "recent_paid": recent_paid,
        "laughter_present": laughter_present,
        # memory-based
        "inside_joke_memory": inside_joke_memory,
        "petname_memory": petname_memory,
        # schedule proximity
        "next_schedule_within": next_schedule_within,
        # misc
        "min": min, "max": max
    }

def _eval_expr(expr: str, ctx: dict) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": {}}, ctx))
    except Exception:
        return False

def _score_detector(det: dict, ctx: dict) -> Tuple[bool, int, List[str]]:
    m = det.get("match") or {}
    must    = m.get("must") or []
    should  = m.get("should") or []
    must_not= m.get("must_not") or []

    hits: List[str] = []

    for cond in must:
        cond_s = cond if isinstance(cond,str) else (cond.get("if") if isinstance(cond,dict) else "")
        if not cond_s or not _eval_expr(cond_s, ctx):
            return (False, 0, hits)
        hits.append(f"+must {cond_s}")

    for cond in must_not:
        cond_s = cond if isinstance(cond,str) else (cond.get("if") if isinstance(cond,dict) else "")
        if cond_s and _eval_expr(cond_s, ctx):
            hits.append(f"-must_not {cond_s}")
            return (False, 0, hits)

    score = 0
    for item in should:
        if isinstance(item, str):
            ok = _eval_expr(item, ctx)
            if ok: score += 1; hits.append(f"+should {item} (+1)")
        elif isinstance(item, dict):
            cond_s = item.get("if","")
            add    = int(item.get("add",1))
            if cond_s and _eval_expr(cond_s, ctx):
                score += add; hits.append(f"+should {cond_s} (+{add})")

    return (True, score, hits)

_DETECTORS_CACHE: List[dict] | None = None

def _tier_guess_from_ltv(ltv_cents: int) -> str:
    if ltv_cents >= 250000: return "emerald"
    if ltv_cents >= 50000:  return "diamond"
    if ltv_cents > 0:       return "gold"
    return "silver"

def _prepare_context(router_input: Dict[str, Any]) -> Tuple[dict, dict, dict, dict]:
    db_raw = dict(router_input.get("db") or {})
    recent = list(router_input.get("recent_messages") or [])
    text, features = _features_from_recent(recent)

    ltv = int(db_raw.get("ltv_cents") or 0)
    db = {
        "fan_msg_count": int(db_raw.get("fan_msg_count") or 0),
        "creator_msg_count": int(db_raw.get("creator_msg_count") or 0),
        "ltv_cents": ltv,
        "secs_since_last_msg": int(db_raw.get("secs_since_last_msg") or 999999),
        "last_fan_gap_secs": int(db_raw.get("last_fan_gap_secs") or 999999),
        "last_creator_gap_secs": int(db_raw.get("last_creator_gap_secs") or 999999),
        "ppv_last_status": db_raw.get("ppv_last_status"),
        "ppv_last_secs":  int(db_raw.get("ppv_last_secs") or 10**9),
        "tier_guess": db_raw.get("tier_guess") or _tier_guess_from_ltv(ltv),
        # memory counts (0-safe)
        "mem_petnames_fan_count": int(db_raw.get("mem_petnames_fan_count") or 0),
        "mem_petnames_creator_count": int(db_raw.get("mem_petnames_creator_count") or 0),
        "mem_inside_jokes_count": int(db_raw.get("mem_inside_jokes_count") or 0),
        "mem_topics_count": int(db_raw.get("mem_topics_count") or 0),
        "mem_boundaries_count": int(db_raw.get("mem_boundaries_count") or 0),
        # next schedule proximity (seconds; may be None)
        "next_schedule_secs": db_raw.get("next_schedule_secs"),
    }

    helpers = _mk_helpers(db, text, features)
    ctx = {"db": DotDict(db), "text": DotDict(text), "features": DotDict(features), **helpers}
    return db, text, features, ctx

def pick_stage(router_input: Dict[str, Any]) -> str:
    return pick_stage_with_reason(router_input).get("stage") or "stage_opener"

def pick_stage_with_reason(router_input: Dict[str, Any]) -> Dict[str, Any]:
    global _DETECTORS_CACHE
    if _DETECTORS_CACHE is None:
        _DETECTORS_CACHE = _list_detectors()

    db, text, feats, ctx = _prepare_context(router_input)

    best_name, best_score = "stage_opener", -10
    best_pri, best_hits, chosen_file = 10**9, [], None

    for det in _DETECTORS_CACHE:
        ok, score, hits = _score_detector(det, ctx)
        if not ok: continue
        thr = int(((det.get("thresholds") or {}).get("choose_if_score_gte")) or 0)
        if score < thr: continue
        pri = int((det.get("thresholds") or {}).get("priority") or det.get("priority") or 100)
        if (score > best_score) or (score == best_score and pri < best_pri):
            best_name, best_score, best_pri = det.get("name") or best_name, score, pri
            best_hits, chosen_file = hits, det.get("_file")

    reason = f"{best_name} (score={best_score}, pri={best_pri}) via: " + "; ".join(best_hits)
    return {
        "stage": best_name,
        "reason": reason,
        "debug": {
            "file": chosen_file,
            "db": db,
            "features": feats,
            "last_fan_texts": text.get("fan_last_n", [])[-3:],
        }
    }
