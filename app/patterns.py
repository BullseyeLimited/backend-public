# app/patterns.py
# Boundary-safe pattern mining for both creator and fan messages.
# - Records into thread_pattern_stats(actor, pattern, hits, reply_hits, paid_hits, ...)
# - Updates thread_profiles.jsonb with richer fan-centric memory (kinks, petnames, style)
# - Uses fan replies + PPV paid proximity as success signals for creator patterns

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Iterable, Optional
from datetime import timedelta, datetime, timezone

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

# ---------- boundary-safe regex helpers ----------
def phrase_regex(phrase: str) -> re.Pattern:
    """
    Compile a whole-phrase regex with word boundaries.
    Example: "good boy" -> r'\bgood\s+boy\b'  (case-insensitive)
    """
    tokens = [re.escape(t) for t in phrase.strip().split()]
    body = r"\s+".join(tokens)
    return re.compile(rf"\b{body}\b", re.IGNORECASE)

def word_regex(word: str) -> re.Pattern:
    """Whole word only (no matches inside 'smirking')."""
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)

# ---------- pattern catalog ----------
PETNAMES_CREATOR = {
    "petname:dear": word_regex("dear"),
    "petname:good boy": phrase_regex("good boy"),
    "petname:baby": word_regex("baby"),
    "petname:babe": word_regex("babe"),
    "petname:daddy": word_regex("daddy"),
    "petname:handsome": word_regex("handsome"),
    "petname:sir": word_regex("sir"),
}

# CTAs (creator)
CTA_CREATOR = {
    "cta:soft": re.compile(r"\b(want me to|should i|can send|if you want|do you want)\b", re.IGNORECASE),
    "cta:hard": re.compile(r"\b(unlock|buy|it'?s?\s*\$|pay|tip)\b", re.IGNORECASE),
}

# Emoji/tone markers (creator)
EMOJI_CREATOR = {
    "emoji:smirk": re.compile("[😏😉]", re.UNICODE),
}
TONE_CREATOR = {
    "tone:dominant": re.compile(r"\b(be a good boy|do it now|i want you to|say please|come here)\b", re.IGNORECASE),
    "tone:submissive": re.compile(r"\b(for you|i'?m yours|please\b.*(baby|daddy)|good girl)\b", re.IGNORECASE),
}

# Fan-side kinks & signals (mined into memory)
FAN_KINKS = {
    "kink:praise": re.compile(r"\b(praise|good girl|good boy|call me|pet me)\b", re.IGNORECASE),
    "kink:daddy": word_regex("daddy"),
    "kink:lingerie": word_regex("lingerie"),
    "kink:feet": word_regex("feet"),
    "kink:roleplay": re.compile(r"\b(role[-\s]?play|rp|cosplay)\b", re.IGNORECASE),
    "kink:submissive": re.compile(r"\b(i'?m (a )?sub|i like when you lead)\b", re.IGNORECASE),
    "kink:dominant": re.compile(r"\b(dom(me)?|i like to lead|be my good girl)\b", re.IGNORECASE),
}

FAN_PETNAMES = {
    "petname:princess": word_regex("princess"),
    "petname:queen": word_regex("queen"),
    "petname:angel": word_regex("angel"),
    "petname:baby": word_regex("baby"),
    "petname:babe": word_regex("babe"),
}

POSITIVE_FAN = re.compile(
    r"(😍|😘|🥵|🔥|😮‍💨|🤤|😉|😏|🥰|❤️|❤️‍🔥|\b(yes|yeah|more|please|want|how much|price|send|show|do it|omg|i need)\b)",
    re.IGNORECASE,
)
NEGATIVE_FAN = re.compile(
    r"\b(not now|busy|no|stop|maybe later|can'?t|broke|no money)\b", re.IGNORECASE
)

# ---------- config ----------
LOOKAHEAD_MSGS = 2          # check up to next 2 fan messages for "reply_hits"
LOOKAHEAD_MINS = 240        # and up to 4 hours for a paid hit after a creator pattern

@dataclass
class Msg:
    id: int
    role: str
    text: str
    created_at: datetime

# ---------- core ingest ----------
def _fetch_thread_messages(conn, thread_id: str) -> List[Msg]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select id, role, text, created_at
            from messages
            where thread_id = %s::uuid
            order by id asc
            """,
            (thread_id,),
        )
        rows = cur.fetchall()
    out: List[Msg] = []
    for r in rows:
        out.append(Msg(id=r["id"], role=r["role"], text=r["text"] or "", created_at=r["created_at"]))
    return out

def _fetch_paid_times(conn, thread_id: str) -> List[datetime]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select created_at
            from ppv_offers
            where thread_id = %s::uuid and status = 'paid'
            order by created_at asc
            """,
            (thread_id,),
        )
        rows = cur.fetchall()
    return [r["created_at"] for r in rows]

def _update_stat(conn, thread_id: str, actor: str, pattern: str, hits: int, reply_hits: int, paid_hits: int, last_msg_id: int, last_seen_at: datetime):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into thread_pattern_stats (thread_id, actor, pattern, hits, reply_hits, paid_hits, last_msg_id, last_seen_at)
            values (%s::uuid, %s, %s, %s, %s, %s, %s, %s)
            on conflict (thread_id, actor, pattern)
            do update set
                hits = thread_pattern_stats.hits + excluded.hits,
                reply_hits = thread_pattern_stats.reply_hits + excluded.reply_hits,
                paid_hits = thread_pattern_stats.paid_hits + excluded.paid_hits,
                last_msg_id = greatest(thread_pattern_stats.last_msg_id, excluded.last_msg_id),
                last_seen_at = greatest(thread_pattern_stats.last_seen_at, excluded.last_seen_at)
            """,
            (thread_id, actor, pattern, hits, reply_hits, paid_hits, last_msg_id, last_seen_at),
        )

def _append_profile_set(obj: Dict[str, Any], key: str, value: str):
    arr = list(obj.get(key) or [])
    if value not in arr:
        arr.append(value)
    obj[key] = arr

def _bump_profile_counter(obj: Dict[str, Any], key: str, value: str, by: int = 1):
    d = dict(obj.get(key) or {})
    d[value] = int(d.get(value) or 0) + by
    obj[key] = d

def _persist_profile(conn, thread_id: str, patch: Dict[str, Any]):
    # Merge jsonb: profile = profile || patch
    with conn.cursor() as cur:
        cur.execute(
            """
            with t as (
              select fan_id from threads where id=%s::uuid
            ), merged as (
              update fans
                 set profile = coalesce(profile,'{}'::jsonb) || %s::jsonb
              from t
              where fans.id = t.fan_id
              returning fans.profile
            )
            select 1
            """,
            (thread_id, Json(patch)),
        )

def is_positive_reply(text: str) -> bool:
    # richer heuristic later; this is enough to start
    if POSITIVE_FAN.search(text or ""):
        return True
    t = (text or "").strip()
    return len(t) >= 20 and not NEGATIVE_FAN.search(t)

def _collect_creator_patterns(text: str) -> List[str]:
    hits: List[str] = []
    for label, rx in PETNAMES_CREATOR.items():
        if rx.search(text):
            hits.append(label)
    for label, rx in CTA_CREATOR.items():
        if rx.search(text):
            hits.append(label)
    for label, rx in EMOJI_CREATOR.items():
        if rx.search(text):
            hits.append(label)
    for label, rx in TONE_CREATOR.items():
        if rx.search(text):
            hits.append(label)
    return hits

def _collect_fan_patterns(text: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Returns (pattern_labels, profile_patch_from_this_line)
    """
    labels: List[str] = []
    patch: Dict[str, Any] = {"memory": {"fan": {}}}
    mm = patch["memory"]["fan"]

    # kinks
    kinks_hit: List[str] = []
    for k, rx in FAN_KINKS.items():
        if rx.search(text):
            labels.append(k)
            kinks_hit.append(k)
    if kinks_hit:
        for k in kinks_hit:
            _append_profile_set(mm, "kinks", k.split(":", 1)[1])

    # petnames he uses for the model persona
    for p, rx in FAN_PETNAMES.items():
        if rx.search(text):
            labels.append(p)
            _bump_profile_counter(mm, "petnames_used", p.split(":",1)[1], 1)

    # style markers (very short / emoji-heavy)
    t = (text or "").strip()
    if len(t) <= 6:
        _bump_profile_counter(mm, "style_fragments", "very_short", 1)
    if re.search("[!?]{2,}", t):
        _bump_profile_counter(mm, "style_fragments", "exclaimy", 1)
    if re.search(r"[😂😍😘🥵🔥😮‍💨🤤😉😏🥰❤️]", t):
        _bump_profile_counter(mm, "style_fragments", "emoji_heavy", 1)

    return labels, patch

def ingest_thread_patterns(conn, thread_id: str) -> Dict[str, Any]:
    """
    Walk the thread once, update:
      - thread_pattern_stats for actor='creator' (what we tried) and actor='fan' (what he says)
      - fans.profile (jsonb) with richer memory
    This function is idempotent-ish because we upsert with incremental counters.
    """
    msgs = _fetch_thread_messages(conn, thread_id)
    if not msgs:
        return {"ok": True, "updated": 0, "patterns": 0}

    paid_times = _fetch_paid_times(conn, thread_id)
    paid_idx = 0  # pointer as we sweep time

    # For profile aggregation in this pass
    profile_patch: Dict[str, Any] = {"memory": {"fan": {}}}
    mm = profile_patch["memory"]["fan"]

    updated_stats = 0
    now = datetime.now(timezone.utc)

    for i, m in enumerate(msgs):
        if m.role == "creator":
            labels = _collect_creator_patterns(m.text)
            if not labels:
                continue

            # Look ahead window end (next creator msg or time window)
            next_creator_time: Optional[datetime] = None
            for j in range(i + 1, len(msgs)):
                if msgs[j].role == "creator":
                    next_creator_time = msgs[j].created_at
                    break
            time_cap = m.created_at + timedelta(minutes=LOOKAHEAD_MINS)
            end_time = min(next_creator_time or time_cap, time_cap)

            # Fan reply signals
            reply_hits = 0
            fan_seen = 0
            for j in range(i + 1, len(msgs)):
                if msgs[j].created_at > end_time:
                    break
                if msgs[j].role == "fan":
                    fan_seen += 1
                    if is_positive_reply(msgs[j].text):
                        reply_hits = 1  # count once per occurrence
                        break
                if fan_seen >= LOOKAHEAD_MSGS:
                    break

            # Paid proximity count
            paid_hits = 0
            while paid_idx < len(paid_times) and paid_times[paid_idx] < m.created_at:
                paid_idx += 1
            k = paid_idx
            while k < len(paid_times) and paid_times[k] <= end_time:
                paid_hits += 1
                k += 1

            for label in labels:
                _update_stat(
                    conn,
                    thread_id=thread_id,
                    actor="creator",
                    pattern=label,
                    hits=1,
                    reply_hits=reply_hits,
                    paid_hits=paid_hits,
                    last_msg_id=m.id,
                    last_seen_at=m.created_at or now,
                )
                updated_stats += 1

                # Track which petnames we used and if elicited reply
                if label.startswith("petname:"):
                    pname = label.split(":", 1)[1]
                    _bump_profile_counter(mm, "petnames_we_used", pname, 1)
                    if reply_hits:
                        _bump_profile_counter(mm, "petnames_replied", pname, 1)

        else:  # fan
            labels, patch_line = _collect_fan_patterns(m.text)
            # Merge per-line patch into staging patch
            # (simple deep-merge for two levels is enough)
            for k1, v1 in (patch_line.get("memory") or {}).items():
                tgt = mm if k1 == "fan" else None
                if tgt is None and k1 == "fan":
                    profile_patch["memory"]["fan"] = v1
                elif k1 == "fan":
                    # merge dict-of-arr / dict-of-dict
                    for k2, v2 in (v1 or {}).items():
                        if isinstance(v2, list):
                            cur = list(mm.get(k2) or [])
                            for x in v2:
                                if x not in cur:
                                    cur.append(x)
                            mm[k2] = cur
                        elif isinstance(v2, dict):
                            curd = dict(mm.get(k2) or {})
                            for kk, vv in v2.items():
                                curd[kk] = int(curd.get(kk) or 0) + int(vv or 0)
                            mm[k2] = curd

            for label in labels:
                _update_stat(
                    conn,
                    thread_id=thread_id,
                    actor="fan",
                    pattern=label,
                    hits=1,
                    reply_hits=0,
                    paid_hits=0,
                    last_msg_id=m.id,
                    last_seen_at=m.created_at or now,
                )
                updated_stats += 1

    # Persist aggregated profile patch
    if (profile_patch.get("memory") or {}).get("fan"):
        _persist_profile(conn, thread_id, profile_patch)

    return {"ok": True, "updated": updated_stats}

def get_thread_patterns(conn, thread_id: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select actor, pattern, hits, reply_hits, paid_hits, last_msg_id, last_seen_at
            from thread_pattern_stats
            where thread_id = %s::uuid
            order by actor asc, paid_hits desc, reply_hits desc, hits desc, last_seen_at desc
            """,
            (thread_id,),
        )
        rows = cur.fetchall()
    return {"patterns": rows}