# app/memory.py
from __future__ import annotations
import os, re, json, requests
from typing import List, Dict, Any, Iterable, Tuple, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ---------------- env for optional LLM summary ----------------
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY") or ""
RUNPOD_COACH_ENDPOINT_ID = os.getenv("RUNPOD_COACH_ENDPOINT_ID", "6pdohjnr3boind")
COACH_MODEL = os.getenv("COACH_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
RUNPOD_BASE = f"https://api.runpod.ai/v2/{RUNPOD_COACH_ENDPOINT_ID}/openai/v1"
SUMMARY_EVERY_TURNS = int(os.getenv("SUMMARY_EVERY_TURNS", "10"))
CREATOR_TZ = os.getenv("CREATOR_TZ", "UTC")

def _rp_headers():
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

def _safe_tz() -> ZoneInfo:
    try: return ZoneInfo(CREATOR_TZ)
    except Exception: return ZoneInfo("UTC")

# ---------------- core regexes (base memory) ----------------
PETNAME_RE = re.compile(r"\b(babe|baby|hun|honey|sweet(ie)?|handsome|king|sir|stud|cutie|love|luv|darling|angel|good\s*(?:boy|girl))\b", re.I)
REMEMBER_RE = re.compile(r"\bremember(?:\s+(?:that|the))?\s+([^?.!]{3,80})", re.I)  # "remember <phrase>"
OUR_X_RE = re.compile(r"\bour\s+([a-z][a-z\-]{2,18})\b", re.I)  # "our <thing>"
LIKE_RE = re.compile(r"\bi\s+(like|love|prefer|enjoy|am (?:super )?into)\s+([^,.!?;]{2,120})", re.I)
FAV_RE = re.compile(r"\bfavorite\s+([a-z ]{3,20})\s*(?:is|:)?\s+([^,.!?;]{2,120})", re.I)
BOUNDARY_RE = re.compile(r"\b(i\s+(?:don'?t|won'?t|can'?t)\s+(?:do|share|send|show|meet)|no\s+(?:free|pics?|videos?)|not\s+(?:into|comfortable with))\s+([^,.!?;]{2,120})", re.I)

# ---------------- topics (broad) ----------------
TOPICS = [
  # daily life
  "work","boss","manager","shift","overtime","night shift","schedule","deadline","project","promotion","raise",
  "school","exam","finals","class","college","university","homework","study",
  "gym","workout","leg day","push day","pull day","cardio","bulk","cut",
  "doctor","dentist","clinic","hospital","appointment","therapy","therapist",
  "family","mom","dad","sister","brother","kids","son","daughter","wife","ex","girlfriend","friends","roommate",
  "sleep","insomnia","coffee","tea","beer","whisky","wine","restaurant","bar","club","party","hangover",
  "car","truck","motorcycle","bike","bus","train","plane","flight","airport","uber",
  # hobbies & media
  "gaming","steam","playstation","xbox","nintendo","switch","fifa","2k","cod","fortnite","valorant","league","dota","csgo","apex","minecraft","gta",
  "anime","manga","cosplay","vtuber","naruto","one piece","attack on titan","demon slayer","jujutsu kaisen","bleach",
  "music","rap","rock","metal","edm","country","pop","jazz","blues","classical","guitar","piano","drums","dj",
  "movies","series","netflix","hbo","disney","marvel","dc","star wars","lotr","harry potter","horror","comedy","romcom",
  "photography","cooking","baking","chef","barista","coding","hiking","camping","fishing","hunting","sports cards","lego",
  # sports
  "nba","nfl","mlb","nhl","ufc","f1","premier league","la liga","bundesliga","serie a","champions league","euros","world cup","lakers","warriors","celtics","knicks","cowboys","patriots","chiefs","arsenal","man city","man united","liverpool","real madrid","barcelona","milan","juventus","psg",
  # travel/events
  "trip","vacation","holiday","beach","mountain","hotel","airbnb","concert","festival","birthday","anniversary","wedding","valentine","christmas","new year","halloween","thanksgiving",
]
TOPIC_RE = re.compile(r"\b(" + "|".join([re.escape(t) for t in TOPICS]) + r")\b", re.I)

# ---------------- kinks / NSFW interests (labels) ----------------
# Keep it compact but useful. Purely detection, no content generation here.
KINK_PATTERNS: List[Tuple[re.Pattern, str]] = [
  (re.compile(r"\bpraise\b|\b(call me|say)\s+(?:a\s*)?good\s*(?:boy|girl)\b", re.I), "praise"),
  (re.compile(r"\b(submissive|sub|dom(me)?|dominant|control me|be my (?:master|mistress)|i(?:'m| am)\s*your\s*(?:good\s*)?(?:boy|girl))\b", re.I), "dom_sub"),
  (re.compile(r"\b(edg(e|ing)|denial)\b", re.I), "edging"),
  (re.compile(r"\b(jo[iy]|jerk(?:\s*off)?\s*instruction)\b", re.I), "JOI"),
  (re.compile(r"\b(feet|toes|foot\s*(?:job)?)\b", re.I), "feet"),
  (re.compile(r"\b(stockings?|nylons?|pantyhose|lingerie|corset|garter)\b", re.I), "lingerie"),
  (re.compile(r"\b(cosplay|role\s*play|nurse|teacher|maid|secretary|step(?:mom|sis|bro))\b", re.I), "roleplay"),
  (re.compile(r"\b(boot(y|ies?)|ass|thick|peach)\b", re.I), "ass_focus"),
  (re.compile(r"\b(boobs?|tits?)\b", re.I), "boobs_focus"),
  (re.compile(r"\b(oral|bj|blow\s*job|suck)\b", re.I), "oral"),
  (re.compile(r"\b(anal)\b", re.I), "anal"),
  (re.compile(r"\b(cum|finish|load)\b", re.I), "cum_focus"),
  (re.compile(r"\b(choke|spank|rough|slap)\b", re.I), "rough_talk"),
  (re.compile(r"\b(cuck|cuckold)\b", re.I), "cuckold"),
  (re.compile(r"\b(teen|milf|mommy|daddy)\b", re.I), "archetype"),
]

# ---------------- creator pattern learning ----------------
# What we record as “patterns” when CREATOR speaks.
CREATOR_PETNAMES = [ "good boy","good girl","baby","babe","handsome","king","sir","stud","sweetie","love","luv","darling" ]
CTA_SOFT_RE   = re.compile(r"\b(?:want me to|should i|if you like|if you want|i can|let me)\b", re.I)
CTA_HARD_RE   = re.compile(r"\b(?:buy|unlock|tip|pay|now|tap|hit|send)\b", re.I)
TONE_SUB_RE   = re.compile(r"\b(for you|please|i(?:’|')ll behave|i’ll be good|just for you|want me to be yours)\b", re.I)
TONE_DOM_RE   = re.compile(r"\b(be a good boy|i want you to|you(?:'|’)ll|you will|i'm making you|kneel|obey)\b", re.I)
EMOJI_HEART_RE= re.compile(r"[❤♥️💖💘💗💓💞💝😍🥰]", re.U)
EMOJI_SMIRK_RE= re.compile(r"[😏😼😉]", re.U)
EMOJI_DEVIL_RE= re.compile(r"[😈]", re.U)
ROLEPLAY_RE   = re.compile(r"\b(role\s*play|rp|let(?:'|’)s pretend|i’ll be your (?:girlfriend|wife|teacher|nurse|maid))\b", re.I)

# ---------------- base extractors ----------------
def _normalize_phrase(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"[\"'`]+", "", s)
    return s[:160]

def _yield_petnames(text: str) -> Iterable[Tuple[str, str]]:
    for m in PETNAME_RE.finditer(text):
        token = m.group(1).lower()
        yield ("petname", token)

def _yield_inside_jokes(text: str) -> Iterable[Tuple[str, str]]:
    for m in REMEMBER_RE.finditer(text):
        phrase = _normalize_phrase(m.group(1))
        if len(phrase) >= 3:
            yield ("inside_joke", phrase)
    for m in OUR_X_RE.finditer(text):
        noun = _normalize_phrase(m.group(1))
        if len(noun) >= 3 and noun not in ("time", "day", "chat", "talk"):
            yield ("inside_joke", f"our {noun}")

def _yield_preferences(text: str) -> Iterable[Tuple[str, str]]:
    for m in LIKE_RE.finditer(text):
        obj = _normalize_phrase(m.group(2))
        yield ("preference", obj)
    for m in FAV_RE.finditer(text):
        cat = _normalize_phrase(m.group(1))
        val = _normalize_phrase(m.group(2))
        yield ("preference", f"{cat}: {val}")

def _yield_boundaries(text: str) -> Iterable[Tuple[str, str]]:
    for m in BOUNDARY_RE.finditer(text):
        obj = _normalize_phrase(m.group(2))
        yield ("boundary", obj)

def _yield_topics(text: str) -> Iterable[Tuple[str, str]]:
    for m in TOPIC_RE.finditer(text):
        yield ("topic", m.group(0).lower())

def _yield_kinks(text: str) -> Iterable[Tuple[str, str]]:
    for rx, label in KINK_PATTERNS:
        if rx.search(text):
            yield ("kink", label)

def extract_candidates(text: str) -> List[Tuple[str, str]]:
    text = text or ""
    out: List[Tuple[str, str]] = []
    out.extend(_yield_petnames(text))
    out.extend(_yield_inside_jokes(text))
    out.extend(_yield_preferences(text))
    out.extend(_yield_boundaries(text))
    out.extend(_yield_topics(text))
    out.extend(_yield_kinks(text))
    # de‑dup
    seen = set()
    uniq = []
    for k, v in out:
        key = (k, v.lower())
        if key not in seen:
            uniq.append((k, v))
            seen.add(key)
    return uniq

# ---------------- persistence: thread_memories ----------------
def upsert_memory(conn, thread_id: str, who: str, kind: str, key: str, value: str, confidence: float, last_msg_id=None):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into thread_memories (thread_id, who, kind, key, value, confidence, first_msg_id, last_msg_id)
            values (%s::uuid, %s, %s, %s, %s, %s, %s, %s)
            on conflict (thread_id, who, kind, key)
            do update set
              value = excluded.value,
              confidence = greatest(thread_memories.confidence, excluded.confidence),
              last_msg_id = excluded.last_msg_id,
              last_seen_at = now(),
              active = true
            """,
            (thread_id, who, kind, key, value, float(confidence), last_msg_id, last_msg_id),
        )

def extract_from_message(conn, thread_id: str, role: str, text: str, msg_id: int | None = None):
    who = "fan" if role == "fan" else "creator"
    for kind, key in extract_candidates(text):
        upsert_memory(conn, thread_id, who, kind, key, key, 0.80, last_msg_id=msg_id)

def summary(conn, thread_id: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select
              sum(case when kind='petname' and who='fan' then 1 else 0 end) as petnames_fan_count,
              sum(case when kind='petname' and who='creator' then 1 else 0 end) as petnames_creator_count,
              sum(case when kind='inside_joke' then 1 else 0 end) as inside_jokes_count,
              sum(case when kind='topic' then 1 else 0 end) as topics_count,
              sum(case when kind='boundary' then 1 else 0 end) as boundaries_count,
              sum(case when kind='kink' then 1 else 0 end) as kinks_count
            from thread_memories
            where thread_id=%s::uuid and active=true
            """,
            (thread_id,),
        )
        row = cur.fetchone() or {}
    with conn.cursor() as cur:
        cur.execute(
            """
            select kind, who, key, value
            from thread_memories
            where thread_id=%s::uuid and active=true
            order by last_seen_at desc
            limit 80
            """,
            (thread_id,),
        )
        items = cur.fetchall() or []

    return {"counts": {k: int(row.get(k) or 0) for k in [
        "petnames_fan_count","petnames_creator_count","inside_jokes_count","topics_count","boundaries_count","kinks_count"
    ]}, "items": items}

# ---------------- creator patterns (what we used) ----------------
def _extract_creator_patterns(text: str) -> List[str]:
    t = text or ""
    pats: List[str] = []
    low = t.lower()
    # petnames
    for p in CREATOR_PETNAMES:
        if p in low:
            pats.append(f"petname:{p}")
    # tone / CTA / roleplay / emoji
    if TONE_SUB_RE.search(t): pats.append("tone:submissive")
    if TONE_DOM_RE.search(t): pats.append("tone:dominant")
    if CTA_SOFT_RE.search(t): pats.append("cta:soft")
    if CTA_HARD_RE.search(t): pats.append("cta:hard")
    if ROLEPLAY_RE.search(t): pats.append("roleplay:girlfriend")
    if EMOJI_HEART_RE.search(t): pats.append("emoji:heart")
    if EMOJI_SMIRK_RE.search(t): pats.append("emoji:smirk")
    if EMOJI_DEVIL_RE.search(t): pats.append("emoji:devil")
    # keep unique
    u = []
    seen = set()
    for p in pats:
        if p not in seen:
            u.append(p); seen.add(p)
    return u

def _pattern_hit(conn, thread_id: str, pattern: str, *, inc_hit=False, inc_reply=False, inc_paid=False, msg_id: Optional[int]=None):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into thread_pattern_stats (thread_id, pattern, hits, reply_hits, paid_hits, last_msg_id)
            values (%s::uuid, %s, %s, %s, %s, %s)
            on conflict (thread_id, pattern)
            do update set
              hits = thread_pattern_stats.hits + %s,
              reply_hits = thread_pattern_stats.reply_hits + %s,
              paid_hits = thread_pattern_stats.paid_hits + %s,
              last_msg_id = greatest(coalesce(thread_pattern_stats.last_msg_id,0), coalesce(excluded.last_msg_id,0)),
              last_seen_at = now()
            """,
            (thread_id, pattern,
             1 if inc_hit else 0, 1 if inc_reply else 0, 1 if inc_paid else 0, msg_id,
             1 if inc_hit else 0, 1 if inc_reply else 0, 1 if inc_paid else 0)
        )

def track_patterns_on_creator(conn, thread_id: str, text: str, msg_id: int | None):
    for p in _extract_creator_patterns(text):
        _pattern_hit(conn, thread_id, p, inc_hit=True, msg_id=msg_id)

def track_patterns_on_fan_reply(conn, thread_id: str, fan_msg_id: int):
    """
    When the fan sends a message, give credit to recent creator patterns.
    Window: last 3 creator messages within 45 minutes.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            with last_creators as (
              select id, text
                from messages
               where thread_id=%s::uuid and role='creator'
                 and created_at >= now() - interval '45 minutes'
               order by id desc
               limit 3
            )
            select id, text from last_creators order by id asc
            """,
            (thread_id,)
        )
        rows = cur.fetchall() or []
    for r in rows:
        for p in _extract_creator_patterns(r["text"] or ""):
            _pattern_hit(conn, thread_id, p, inc_reply=True, msg_id=r["id"])

def track_patterns_on_paid(conn, thread_id: str):
    """
    When a PPV is marked paid, attribute success to recent creator patterns.
    Window: last 5 creator messages within 2 hours.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            with last_creators as (
              select id, text
                from messages
               where thread_id=%s::uuid and role='creator'
                 and created_at >= now() - interval '2 hours'
               order by id desc
               limit 5
            )
            select id, text from last_creators
            """,
            (thread_id,)
        )
        rows = cur.fetchall() or []
    seen = set()
    for r in rows:
        for p in _extract_creator_patterns(r["text"] or ""):
            if p in seen:  # count once per payment
                continue
            seen.add(p)
            _pattern_hit(conn, thread_id, p, inc_paid=True, msg_id=r["id"])

def patterns_snapshot(conn, thread_id: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select pattern, hits, reply_hits, paid_hits, last_seen_at
              from thread_pattern_stats
             where thread_id=%s::uuid
             order by last_seen_at desc
            """,
            (thread_id,)
        )
        rows = cur.fetchall() or []
    return {"patterns": rows}

# ---------------- texter style + best times ----------------
EMOJI_BLOCK = re.compile(r"[\U0001F300-\U0001FAFF]")
def _fan_last_n(conn, thread_id: str, n=80) -> List[Dict[str,Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select id, text, created_at
              from messages
             where thread_id=%s::uuid and role='fan'
             order by id desc
             limit %s
            """,
            (thread_id, n)
        )
        rows = cur.fetchall() or []
    rows.reverse()
    return rows

def _creator_last_n(conn, thread_id: str, n=80) -> List[Dict[str,Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select id, text, created_at
              from messages
             where thread_id=%s::uuid and role='creator'
             order by id desc
             limit %s
            """,
            (thread_id, n)
        )
        rows = cur.fetchall() or []
    rows.reverse()
    return rows

def _classify_texter_style(fan_msgs: List[Dict[str,Any]]) -> str:
    if not fan_msgs: return "unknown"
    lens = [len((m["text"] or "")) for m in fan_msgs]
    avg = sum(lens)/len(lens)
    qs = sum(1 for m in fan_msgs if "?" in (m["text"] or ""))
    em = sum(len(EMOJI_BLOCK.findall(m["text"] or "")) for m in fan_msgs)
    excl = sum((m["text"] or "").count("!") for m in fan_msgs)
    q_rate = qs/len(fan_msgs)
    emoji_rate = em / max(1, sum(lens))
    # simple thresholds
    if avg <= 35 and q_rate < 0.15 and emoji_rate < 0.03:
        return "short-bursts"
    if avg >= 120 and q_rate < 0.2:
        return "paragraph-dumper"
    if emoji_rate >= 0.08:
        return "emoji-heavy"
    if q_rate >= 0.28:
        return "question-seeker"
    if excl / max(1, len(fan_msgs)) >= 0.25:
        return "excitable"
    return "balanced"

def _best_times(fan_msgs: List[Dict[str,Any]]) -> List[str]:
    if not fan_msgs: return []
    tz = _safe_tz()
    buckets = {"morning":0,"afternoon":0,"evening":0,"late-night":0}
    for m in fan_msgs:
        dt = (m["created_at"]).astimezone(tz)
        h = dt.hour
        if 7 <= h < 12: buckets["morning"] += 1
        elif 12 <= h < 17: buckets["afternoon"] += 1
        elif 17 <= h < 23: buckets["evening"] += 1
        else: buckets["late-night"] += 1
    ranked = sorted(buckets.items(), key=lambda x: (-x[1], x[0]))
    return [f"{k} ({v})" for k,v in ranked if v > 0][:2]

def _topic_affinity(conn, thread_id: str) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select key, count(*) as c
              from thread_memories
             where thread_id=%s::uuid and kind='topic' and active=true
             group by key
             order by c desc, key asc
             limit 12
            """,
            (thread_id,)
        )
        rows = cur.fetchall() or []
    return [r["key"] for r in rows]

def _kink_list(conn, thread_id: str) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select key, count(*) as c
              from thread_memories
             where thread_id=%s::uuid and kind='kink' and active=true
             group by key
             order by c desc, key asc
             limit 12
            """,
            (thread_id,)
        )
        rows = cur.fetchall() or []
    return [r["key"] for r in rows]

def _boundaries(conn, thread_id: str) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select value
              from thread_memories
             where thread_id=%s::uuid and kind='boundary' and active=true
             order by last_seen_at desc
             limit 12
            """,
            (thread_id,)
        )
        rows = cur.fetchall() or []
    return [r["value"] for r in rows]

def _petnames_dict(conn, thread_id: str) -> Dict[str, List[str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select who, key
              from thread_memories
             where thread_id=%s::uuid and kind='petname' and active=true
            """,
            (thread_id,)
        )
        rows = cur.fetchall() or []
    fan = sorted({r["key"] for r in rows if r["who"]=="fan"})
    creator = sorted({r["key"] for r in rows if r["who"]=="creator"})
    return {"fan_used": fan, "creator_used": creator}

# ---------------- thread_profiles maintenance ----------------
def _ensure_profile(conn, thread_id: str):
    with conn.cursor() as cur:
        cur.execute(
            "insert into thread_profiles (thread_id) values (%s::uuid) on conflict (thread_id) do nothing",
            (thread_id,)
        )

def register_turn_and_maybe_refresh_profile(conn, thread_id: str):
    """
    Increment turns; refresh summary + traits every SUMMARY_EVERY_TURNS.
    """
    _ensure_profile(conn, thread_id)
    with conn.cursor() as cur:
        cur.execute(
            "update thread_profiles set turns=turns+1, updated_at=now() where thread_id=%s::uuid returning turns",
            (thread_id,)
        )
        turns = int((cur.fetchone() or {"turns":0})["turns"])
    if turns % max(1, SUMMARY_EVERY_TURNS) == 0:
        refresh_thread_profile(conn, thread_id)

def _patterns_learned(conn, thread_id: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select pattern, hits, reply_hits, paid_hits
              from thread_pattern_stats
             where thread_id=%s::uuid
            """,
            (thread_id,)
        )
        rows = cur.fetchall() or []
    # compute simple success rates
    best_reply, best_paid, avoid = [], [], []
    for r in rows:
        h = max(1, int(r["hits"] or 0))
        rr = (int(r["reply_hits"] or 0))/h
        pr = (int(r["paid_hits"] or 0))/h
        if h >= 2 and (rr >= 0.45 or pr >= 0.25):
            best_reply.append((r["pattern"], rr))
            best_paid.append((r["pattern"], pr))
        if h >= 3 and rr <= 0.10 and pr == 0:
            avoid.append(r["pattern"])
    best_reply.sort(key=lambda x: -x[1])
    best_paid.sort(key=lambda x: -x[1])
    return {
        "responds_best_reply": [p for p,_ in best_reply[:6]],
        "responds_best_paid": [p for p,_ in best_paid[:6]],
        "responds_avoid": sorted(list(set(avoid)))[:6],
    }

def refresh_thread_profile(conn, thread_id: str):
    """
    Recompute traits and summary. Uses LLM if RUNPOD_API_KEY set; otherwise rule-based writeup.
    """
    _ensure_profile(conn, thread_id)
    fan_msgs = _fan_last_n(conn, thread_id, n=80)
    creator_msgs = _creator_last_n(conn, thread_id, n=50)
    texter_style = _classify_texter_style(fan_msgs)
    best_times = _best_times(fan_msgs)
    topics = _topic_affinity(conn, thread_id)
    kinks = _kink_list(conn, thread_id)
    bounds = _boundaries(conn, thread_id)
    pets = _petnames_dict(conn, thread_id)
    learned = _patterns_learned(conn, thread_id)

    traits = {
        "texter_style": texter_style,
        "best_times": best_times,
        "topics": topics,
        "kinks": kinks,
        "boundaries": bounds,
        "petnames": pets,
        **learned
    }

    # Compose story summary
    summary_text = _compose_story_summary(conn, thread_id, fan_msgs, creator_msgs, traits)

    with conn.cursor() as cur:
        cur.execute(
            "update thread_profiles set traits=%s::jsonb, summary=%s, updated_at=now() where thread_id=%s::uuid",
            (json.dumps(traits, ensure_ascii=False), summary_text, thread_id)
        )

def _compose_story_summary(conn, thread_id: str, fan_msgs, creator_msgs, traits: Dict[str, Any]) -> str:
    # Use LLM if available
    if RUNPOD_API_KEY:
        sys = (
            "You are COACH Profile Summarizer. Output a compact 2–3 paragraph narrative (120–220 words) "
            "that captures who the FAN is (style, topics, kinks/interests, boundaries), how they respond (triggers that work), "
            "and any petnames/inside jokes (light touch). No prices, no explicit actions—just descriptive narrative. "
            "Write first-person from the creator’s perspective ('He tends to...', 'He lights up when...')."
        )
        # Sample last messages (shorten)
        fan_sample = [{"text": m["text"]} for m in fan_msgs[-12:]]
        cr_sample = [{"text": m["text"]} for m in creator_msgs[-8:]]
        user = {
            "traits": traits,
            "fan_recent": fan_sample,
            "creator_recent": cr_sample
        }
        try:
            r = requests.post(
                f"{RUNPOD_BASE}/chat/completions",
                headers=_rp_headers(),
                json={
                    "model": COACH_MODEL,
                    "messages": [
                        {"role":"system","content":sys},
                        {"role":"user","content":json.dumps(user, ensure_ascii=False)}
                    ],
                    "temperature": 0.35,
                    "max_tokens": 380
                },
                timeout=90
            )
            if r.status_code == 200:
                data = r.json()
                content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                return content.strip()[:2000]
        except Exception:
            pass

    # Fallback: deterministic compact writeup
    parts = []
    ts = traits.get("texter_style","unknown").replace("-", " ")
    bt = ", ".join(traits.get("best_times") or []) or "—"
    topics = ", ".join(traits.get("topics") or []) or "varied topics"
    kinks = ", ".join(traits.get("kinks") or []) or "keeps it soft so far"
    bounds = "; ".join(traits.get("boundaries") or []) or "no strong boundaries mentioned yet"
    pets_f = ", ".join(traits.get("petnames",{}).get("fan_used",[]) or [])
    pets_c = ", ".join(traits.get("petnames",{}).get("creator_used",[]) or [])
    best_r = ", ".join(traits.get("responds_best_reply",[]) or traits.get("responds_best_paid",[]) or [])
    avoid = ", ".join(traits.get("responds_avoid",[]) or [])

    parts.append(f"He writes in a {ts} style. Best times he pops up: {bt}.")
    parts.append(f"Topics he returns to: {topics}. Interests/kinks mentioned: {kinks}. Boundaries: {bounds}.")
    if pets_f or pets_c:
        pieces = []
        if pets_f: pieces.append(f"petnames he uses for me ({pets_f})")
        if pets_c: pieces.append(f"petnames I’ve used ({pets_c})")
        parts.append("We have " + " and ".join(pieces) + ".")
    if best_r:
        parts.append(f"He responds best to: {best_r}.")
    if avoid:
        parts.append(f"Less effective: {avoid}.")
    return " ".join(parts)[:2000]

# ---------------- SCHEDULE EXTRACTOR (unchanged from prior drop) ----------------
MONTHS = "(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DOW    = "(mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)"

RE_MONTH_DAY = re.compile(rf"\b{MONTHS}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(\d{{4}}))?", re.I)
RE_NUM_DATE  = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")
RE_DOW       = re.compile(rf"\b(?:this|next)?\s*{DOW}\b", re.I)
RE_TOD       = re.compile(r"\b(at\s*)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)
RE_REL_HM    = re.compile(r"\bin\s+(\d{1,3})\s*(minutes?|mins?|hours?|hrs?)\b", re.I)
RE_TOMORROW  = re.compile(r"\b(tomorrow|tmrw|tmr)\b", re.I)
RE_TONIGHT   = re.compile(r"\b(tonight|this\s+evening)\b", re.I)
RE_MORNING   = re.compile(r"\b(this\s+)?morning\b", re.I)
RE_AFTERNOON = re.compile(r"\b(this\s+)?afternoon\b", re.I)
RE_EVENING   = re.compile(r"\b(this\s+)?evening\b", re.I)

KIND_MAP: List[Tuple[str, str]] = [
    ("birthday|bday", "birthday"),
    ("shift|overtime|night\\s*shift", "work_shift"),
    ("meeting|meet|call|zoom", "appointment"),
    ("doctor|dentist|clinic|hospital|appointment", "appointment"),
    ("game|match|kickoff|tipoff", "game"),
    ("gym|workout|leg day|push day|pull day", "gym"),
    ("date\\s+night|date\\b", "date"),
    ("flight|airport|plane|travel|trip|vacation|holiday", "trip"),
    ("delivery|package|arrives|drops? off", "delivery"),
]

def _classify_kind_and_title(text: str) -> Tuple[str, str]:
    t = text.lower()
    for pat, kind in KIND_MAP:
        if re.search(pat, t):
            if kind == "birthday":  return (kind, "Birthday check‑in")
            if kind == "work_shift":return (kind, "Work shift check‑in")
            if kind == "appointment":return (kind, "Appointment check‑in")
            if kind == "game":      return (kind, "Game day check‑in")
            if kind == "gym":       return (kind, "Gym check‑in")
            if kind == "date":      return (kind, "Date night check‑in")
            if kind == "trip":      return (kind, "Trip/Travel check‑in")
            if kind == "delivery":  return (kind, "Delivery check‑in")
    return ("check-in", "Quick check‑in")

def _safe_tz_name(tz_name: Optional[str]) -> ZoneInfo:
    try: return ZoneInfo(tz_name or "UTC")
    except Exception: return ZoneInfo("UTC")

def _combine_date_time(base_date: datetime, h: Optional[int], m: Optional[int], tod: Optional[str]) -> datetime:
    hour = h if h is not None else 10
    minute = m if m is not None else 0
    if tod:
        tod = tod.lower()
        if tod == "pm" and hour < 12: hour += 12
        if tod == "am" and hour == 12: hour = 0
    return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

def _next_dow(from_dt: datetime, dow_name: str) -> datetime:
    targets = {"mon":0,"tue":1,"wed":2,"thu":3,"fri":4,"sat":5,"sun":6}
    target = targets[dow_name[:3].lower()]
    cur = from_dt.weekday()
    delta = (target - cur) % 7
    if delta == 0: delta = 7
    return (from_dt + timedelta(days=delta))

def _parse_time_of_day(text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    m = RE_TOD.search(text)
    if not m: return (None, None, None)
    h = int(m.group(2))
    mm = int(m.group(3)) if m.group(3) else None
    ampm = m.group(4).lower() if m.group(4) else None
    return (h, mm, ampm)

def _default_time_for_phrase(text: str) -> Tuple[int, int]:
    tl = text.lower()
    if RE_TONIGHT.search(tl) or RE_EVENING.search(tl): return (20, 0)
    if RE_AFTERNOON.search(tl): return (15, 0)
    if RE_MORNING.search(tl): return (10, 0)
    return (12, 0)

def extract_schedules(text: str, tz_name: Optional[str], now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    if not text or not text.strip(): return []
    tz = _safe_tz_name(tz_name)
    now_local = (now or datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))).astimezone(tz)
    items: List[Dict[str, Any]] = []
    text_l = text.lower()

    for m in RE_REL_HM.finditer(text_l):
        num = int(m.group(1)); unit = m.group(2)
        delta = timedelta(hours=num) if unit.startswith(("hour","hr")) else timedelta(minutes=num)
        dt_local = now_local + delta
        kind, title = _classify_kind_and_title(text)
        items.append({"title": title,"kind": kind,"scheduled_at": dt_local.astimezone(ZoneInfo("UTC")),"timezone": tz.key,"confidence": 0.65 if delta < timedelta(hours=48) else 0.55})

    if RE_TOMORROW.search(text_l) or RE_TONIGHT.search(text_l) or RE_MORNING.search(text_l) or RE_AFTERNOON.search(text_l) or RE_EVENING.search(text_l):
        base = now_local + timedelta(days=1) if RE_TOMORROW.search(text_l) else now_local
        h, mm, ampm = _parse_time_of_day(text_l)
        if h is None: h, mm = _default_time_for_phrase(text_l)
        dt_local = _combine_date_time(base, h, mm, ampm)
        kind, title = _classify_kind_and_title(text)
        conf = 0.75 if h is not None else 0.65
        items.append({"title": title,"kind": kind,"scheduled_at": dt_local.astimezone(ZoneInfo("UTC")),"timezone": tz.key,"confidence": conf})

    for m in RE_DOW.finditer(text_l):
        dow_token = m.group(0)
        base = now_local
        next_ = "next" in dow_token
        dow_name = re.findall(DOW, dow_token, re.I)[0]
        target = _next_dow(base, dow_name)
        if not next_ and target.date() == base.date():
            target = base
        h, mm, ampm = _parse_time_of_day(text_l)
        if h is None: h, mm = _default_time_for_phrase(text_l)
        dt_local = _combine_date_time(target, h, mm, ampm)
        kind, title = _classify_kind_and_title(text)
        conf = 0.75 if h is not None else 0.65
        items.append({"title": title,"kind": kind,"scheduled_at": dt_local.astimezone(ZoneInfo("UTC")),"timezone": tz.key,"confidence": conf})

    for m in RE_MONTH_DAY.finditer(text_l):
        mon_str = m.group(1); day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else now_local.year
        month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12}
        mkey = mon_str[:3].lower(); month = month_map.get(mkey, now_local.month)
        h, mm, ampm = _parse_time_of_day(text_l)
        if h is None: h, mm = _default_time_for_phrase(text_l)
        try:
            base = datetime(year, month, day, tzinfo=tz)
            dt_local = _combine_date_time(base, h, mm, ampm)
            if dt_local < now_local: dt_local = dt_local.replace(year=year+1)
            kind, title = _classify_kind_and_title(text)
            conf = 0.80 if h is not None else 0.70
            items.append({"title": title,"kind": kind,"scheduled_at": dt_local.astimezone(ZoneInfo("UTC")),"timezone": tz.key,"confidence": conf})
        except Exception:
            pass

    for m in RE_NUM_DATE.finditer(text_l):
        m1 = int(m.group(1)); m2 = int(m.group(2)); yy = m.group(3)
        candidates = []
        for (a,b) in [(m1,m2),(m2,m1)]:
            try:
                year = int(yy)+2000 if (yy and len(yy)==2) else (int(yy) if yy else now_local.year)
                base = datetime(year, a, b, tzinfo=tz)
                candidates.append(base); break
            except Exception:
                continue
        if not candidates: continue
        base = candidates[0]
        h, mm, ampm = _parse_time_of_day(text_l)
        if h is None: h, mm = _default_time_for_phrase(text_l)
        dt_local = _combine_date_time(base, h, mm, ampm)
        if dt_local < now_local: dt_local = dt_local.replace(year=dt_local.year+1)
        kind, title = _classify_kind_and_title(text)
        conf = 0.80 if h is not None else 0.70
        items.append({"title": title,"kind": kind,"scheduled_at": dt_local.astimezone(ZoneInfo("UTC")),"timezone": tz.key,"confidence": conf})

    uniq: List[Dict[str,Any]] = []
    seen = set()
    now_utc = now_local.astimezone(ZoneInfo("UTC"))
    for it in items:
        if it["scheduled_at"] <= now_utc + timedelta(seconds=30): continue
        if it["scheduled_at"] >= now_utc + timedelta(days=365): continue
        key = (int(it["scheduled_at"].timestamp()//60), it["title"])
        if key in seen: continue
        seen.add(key)
        uniq.append(it)
    return uniq
