# filepath: app/main.py
import os
import json
import re
import asyncio
import requests
import logging
from uuid import UUID
from typing import Optional, List, Literal, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from app.telemetry import router as telemetry_router  # <— you already have this

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from dotenv import load_dotenv

# ---------------- optional: cards loader & router ----------------
try:
    from app.cards_loader import load_card  # resolves includes + returns full system prompt text
except Exception as _e:
    load_card = None
    logging.warning("cards_loader not available: %s", _e)

try:
    import app.cards_router as _cards_router_mod  # type: ignore
    pick_stage_with_reason = getattr(_cards_router_mod, "pick_stage_with_reason", None)
    pick_stage = getattr(_cards_router_mod, "pick_stage", None)
except Exception as _e:
    pick_stage_with_reason = None
    pick_stage = None
    logging.warning("cards_router not available: %s", _e)

# ---- deep memory / profile / scheduler helpers
try:
    from app import memory as mem
except Exception as _e:
    mem = None
    logging.warning("memory module not available: %s", _e)

# ---- (optional) bulk pattern ingestion / stats from history
try:
    from app.patterns import ingest_thread_patterns, get_thread_patterns  # type: ignore
except Exception as _e:
    ingest_thread_patterns = None
    get_thread_patterns = None
    logging.warning("patterns module not available: %s", _e)

# ---------------- env ----------------
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
CREATOR_ID = os.getenv("CREATOR_ID")

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_COACH_ENDPOINT_ID = os.getenv("RUNPOD_COACH_ENDPOINT_ID", "6pdohjnr3boind")
COACH_MODEL = os.getenv("COACH_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
RUNPOD_BASE = f"https://api.runpod.ai/v2/{RUNPOD_COACH_ENDPOINT_ID}/openai/v1"

# Optional WRITER endpoint (persona/style layer)
RUNPOD_WRITER_ENDPOINT_ID = os.getenv("RUNPOD_WRITER_ENDPOINT_ID", RUNPOD_COACH_ENDPOINT_ID)
RUNPOD_WRITER_MODEL = os.getenv("RUNPOD_WRITER_MODEL", COACH_MODEL)
RUNPOD_WRITER_BASE = f"https://api.runpod.ai/v2/{RUNPOD_WRITER_ENDPOINT_ID}/openai/v1"

DEFAULT_STAGE_CARD = "stage_opener"

# Scheduler env
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "0") == "1"
CREATOR_TZ = os.getenv("CREATOR_TZ", "UTC")
SCHEDULER_POLL_SECS = int(os.getenv("SCHEDULER_POLL_SECS", "30"))
SCHEDULER_DUE_GRACE_SECS = int(os.getenv("SCHEDULER_DUE_GRACE_SECS", "45"))

# ---------------- app ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "tauri://localhost",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ADD THIS LINE to expose /telemetry/* endpoints
app.include_router(telemetry_router)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ---------------- helpers ----------------
def db():
    return psycopg.connect(DB_URL, row_factory=dict_row)

def is_uuid_like(x: Any) -> bool:
    try:
        UUID(str(x))
        return True
    except Exception:
        return False

def _thread_cast(thread_id: Any) -> str:
    return "::uuid" if is_uuid_like(thread_id) else ""

def log_event(conn, thread_id: Any, etype: str, data: Optional[dict] = None):
    cast = _thread_cast(thread_id)
    with conn.cursor() as cur:
        cur.execute(
            f"insert into events (thread_id, type, data) values (%s{cast}, %s, %s)",
            (thread_id, etype, Json(data or {})),
        )

def _extract_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\\s\\S]*\}", s)
        return json.loads(m.group(0)) if m else {}

def _choose_stage(router_input: Dict) -> Dict:
    try:
        if pick_stage_with_reason:
            result = pick_stage_with_reason(router_input)  # type: ignore
            if isinstance(result, dict) and result.get("stage"):
                return {
                    "stage": result.get("stage") or DEFAULT_STAGE_CARD,
                    "reason": result.get("reason") or "router: pick_stage_with_reason",
                    "debug": result.get("debug") or {},
                }
        if pick_stage:
            stage = pick_stage(router_input)  # type: ignore
            return {
                "stage": stage or DEFAULT_STAGE_CARD,
                "reason": "router: pick_stage (fallback)",
                "debug": {},
            }
    except Exception:
        logging.exception("router error")
    return {"stage": DEFAULT_STAGE_CARD, "reason": "router: default fallback", "debug": {}}

def ensure_test_thread(conn) -> str:
    """
    Keep a stable demo thread for legacy endpoints (/suggest, /messages, etc.).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into fans (creator_id, platform_user_id, profile)
            values (%s, %s, '{}'::jsonb)
            on conflict (creator_id, platform_user_id) do nothing
            returning id
            """,
            (CREATOR_ID, "test_fan"),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "select id from fans where creator_id=%s and platform_user_id=%s",
                (CREATOR_ID, "test_fan"),
            )
            row = cur.fetchone()
        fan_id = row["id"]

        cur.execute(
            """
            insert into threads (creator_id, fan_id)
            values (%s, %s)
            on conflict do nothing
            returning id
            """,
            (CREATOR_ID, fan_id),
        )
        t = cur.fetchone()
        if t is None:
            cur.execute(
                """
                select id from threads
                where creator_id=%s and fan_id=%s
                order by created_at asc
                limit 1
                """,
                (CREATOR_ID, fan_id),
            )
            t = cur.fetchone()
        return str(t["id"])

def ensure_thread_for_fan(conn, fan_id: Any) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "select id from threads where creator_id=%s and fan_id=%s order by created_at asc limit 1",
            (CREATOR_ID, fan_id),
        )
        t = cur.fetchone()
        if t:
            return str(t["id"])
    with conn.cursor() as cur:
        cur.execute(
            "insert into threads (creator_id, fan_id) values (%s, %s) returning id",
            (CREATOR_ID, fan_id),
        )
        t = cur.fetchone()
        return str(t["id"])

def _slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "fan"

def _compute_tier(explicit_tier: Optional[int], ltv_cents: int) -> int:
    if isinstance(explicit_tier, int) and explicit_tier in (-1, 0, 1, 2):
        return explicit_tier
    if ltv_cents >= 50000:
        return 2  # diamond
    if ltv_cents > 0:
        return 1  # gold
    return 0      # silver

def _safe_tz() -> ZoneInfo:
    try:
        return ZoneInfo(CREATOR_TZ)
    except Exception:
        return ZoneInfo("UTC")

# ---------------- DTOs ----------------
class FanIn(BaseModel):
    fan_message_text: str

class CreatorIn(BaseModel):
    text: str

class SuggestOut(BaseModel):
    drafts: List[str]

class UndoIn(BaseModel):
    count: Optional[int] = 1

class MsgOut(BaseModel):
    role: str
    text: str

class OfferStartIn(BaseModel):
    description: Optional[str] = None
    price_cents: Optional[int] = None
    asset_label: Optional[str] = None
    thread_id: Optional[str] = None

class OfferStartOut(BaseModel):
    offer_id: str

class OfferMarkIn(BaseModel):
    action: Literal["sent", "not_sent", "paid", "unpaid"]

class CustomerNewIn(BaseModel):
    display_name: str
    handle: Optional[str] = None

class CustomerOut(BaseModel):
    id: str
    display_name: str
    platform_user_id: str
    tier: int
    status: str
    ltv_cents: int
    last_msg_preview: str
    secs_since_last_msg: int
    thread_id: str
    last_stage: Optional[str] = None

# ---------------- health ----------------
@app.get("/health")
def health():
    return {"ok": True, "scheduler": SCHEDULER_ENABLED}

# ---------------- SCHEDULER helpers ----------------
def _ppv_last_status_secs(conn, thread_id: Any) -> Tuple[Optional[str], Optional[int]]:
    cast = _thread_cast(thread_id)
    with conn.cursor() as cur:
        cur.execute(
            f"select status, extract(epoch from now()-created_at)::int as age from ppv_offers where thread_id=%s{cast} order by created_at desc limit 1",
            (thread_id,),
        )
        row = cur.fetchone()
    if not row: return None, None
    return (row["status"], int(row["age"] or 10**9))

def _secs_since_last_msg(conn, thread_id: Any) -> int:
    cast = _thread_cast(thread_id)
    with conn.cursor() as cur:
        cur.execute(
            f"select extract(epoch from now()-max(created_at))::int as gap from messages where thread_id=%s{cast}",
            (thread_id,),
        )
        r = cur.fetchone()
    return int((r or {}).get("gap") or 999999)

def _memory_summary(conn, thread_id: Any) -> Dict[str, Any]:
    if mem is None:
        return {"counts": {"petnames_fan_count": 0, "petnames_creator_count": 0, "inside_jokes_count": 0, "topics_count": 0, "boundaries_count": 0, "kinks_count": 0}, "items": []}
    return mem.summary(conn, str(thread_id))

def _next_pending_schedule_secs(conn, thread_id: Any) -> Optional[int]:
    cast = _thread_cast(thread_id)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select extract(epoch from scheduled_at - now())::int as diff
              from thread_schedules
             where thread_id=%s{cast}
               and status in ('pending','ready')
             order by scheduled_at asc
             limit 1
            """,
            (thread_id,),
        )
        row = cur.fetchone()
    if not row: return None
    return int(row["diff"]) if row["diff"] is not None else None

def _build_router_input(conn, thread_id: str, recent: List[dict]) -> dict:
    cast = _thread_cast(thread_id)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select
              count(*) filter (where role='fan')     as fan_count,
              count(*) filter (where role='creator') as creator_count
            from messages where thread_id=%s{cast}
            """,
            (thread_id,),
        )
        counts = cur.fetchone() or {"fan_count": 0, "creator_count": 0}

    with conn.cursor() as cur:
        cur.execute(
            f"select coalesce(sum(price_cents) filter (where status='paid'),0)::int as ltv from ppv_offers where thread_id=%s{cast}",
            (thread_id,),
        )
        ltv_row = cur.fetchone() or {"ltv": 0}
    ltv_cents = int(ltv_row["ltv"] or 0)

    secs_gap  = _secs_since_last_msg(conn, thread_id)
    last_stat, last_age = _ppv_last_status_secs(conn, thread_id)
    memsum = _memory_summary(conn, thread_id)
    next_sched_secs = _next_pending_schedule_secs(conn, thread_id)

    router_input = {
        "db": {
            "fan_msg_count": int(counts["fan_count"]),
            "creator_msg_count": int(counts["creator_count"]),
            "ltv_cents": ltv_cents,
            "secs_since_last_msg": secs_gap,
            "ppv_last_status": last_stat,
            "ppv_last_secs": last_age,
            "tier_guess": "gold" if ltv_cents > 0 and ltv_cents < 50000 else ("diamond" if ltv_cents >= 50000 else "silver"),
            # memory counts for helpers
            "mem_petnames_fan_count": (memsum["counts"]["petnames_fan_count"] if memsum else 0),
            "mem_petnames_creator_count": (memsum["counts"]["petnames_creator_count"] if memsum else 0),
            "mem_inside_jokes_count": (memsum["counts"]["inside_jokes_count"] if memsum else 0),
            "mem_topics_count": (memsum["counts"]["topics_count"] if memsum else 0),
            "mem_boundaries_count": (memsum["counts"]["boundaries_count"] if memsum else 0),
            "mem_kinks_count": (memsum["counts"]["kinks_count"] if memsum else 0),
            # next schedule proximity
            "next_schedule_secs": next_sched_secs,
        },
        "recent_messages": recent,
    }
    return router_input

# ---------------- RunPod glue for reminders ----------------
def _rp_headers():
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

def _compose_reminder_text(conn, thread_id: str, title: str, kind: str) -> Dict[str, str]:
    cast = _thread_cast(thread_id)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select role, text
            from messages
            where thread_id=%s{cast}
            order by id desc
            limit 8
            """,
            (thread_id,),
        )
        recent = cur.fetchall()[::-1]
    memsum = _memory_summary(conn, thread_id)

    sys = (
        "You are COACH Reminder. Return one JSON object only.\\n"
        "Compose a short, warm, personal check-in the creator can paste.\\n"
        "- 1–2 sentences; no prices, no explicit.\\n"
        "- Acknowledge the specific event.\\n"
        '- JSON only: { \"text\": string, \"reason\": string }'
    )
    user = {
        "event": {"title": title, "kind": kind},
        "memory_counts": memsum.get("counts", {}),
        "recent_messages": recent,
    }

    if not RUNPOD_API_KEY:
        txt = f"Just checking in about {title.lower()} — thinking of you. How did it go? 💬"
        return {"text": txt, "reason": "fallback:no_api"}

    payload = {
        "model": COACH_MODEL,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ],
        "temperature": 0.2,
        "max_tokens": 120
    }
    try:
        r = requests.post(f"{RUNPOD_BASE}/chat/completions", headers=_rp_headers(), json=payload, timeout=90)
        if r.status_code != 200:
            raise RuntimeError(f"RunPod status {r.status_code}: {r.text[:180]}")
        data = r.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        j = _extract_json(content)
        if not isinstance(j, dict) or "text" not in j:
            raise RuntimeError("bad JSON from model")
        return {"text": (j.get("text") or "").strip(), "reason": j.get("reason") or "ok"}
    except Exception as e:
        logging.warning("reminder text compose failed: %s", e)
        txt = f"Just checking in about {title.lower()} — how did it go? 🙂"
        return {"text": txt, "reason": f"error:{e.__class__.__name__}"}

# ---------------- legacy single-thread (kept) ----------------
@app.post("/log_fan")
def log_fan(payload: FanIn):
    text = (payload.fan_message_text or "").strip()
    if not text:
        return {"ok": True}
    with db() as conn:
        thread_id = ensure_test_thread(conn)
        cast = _thread_cast(thread_id)
        with conn.cursor() as cur:
            cur.execute(
                f"insert into messages (thread_id, role, text) values (%s{cast}, 'fan', %s) returning id, created_at",
                (thread_id, text),
            )
            row = cur.fetchone()
            mid = row["id"]

        # Memory + schedules + pattern learning (reply credit) + profile turns
        if mem is not None:
            mem.extract_from_message(conn, thread_id, "fan", text, msg_id=mid)
            mem.track_patterns_on_fan_reply(conn, thread_id, fan_msg_id=mid)
            for ev in mem.extract_schedules(text, CREATOR_TZ):
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            insert into thread_schedules (thread_id, who, title, kind, scheduled_at, timezone, confidence, source_msg_id)
                            values (%s::uuid, 'fan', %s, %s, %s, %s, %s, %s)
                            on conflict (thread_id, title, scheduled_at) do nothing
                            """,
                            (thread_id, ev["title"], ev["kind"], ev["scheduled_at"], ev["timezone"], ev["confidence"], mid),
                        )
                except Exception:
                    logging.exception("failed to insert schedule")
            mem.register_turn_and_maybe_refresh_profile(conn, thread_id)

        log_event(conn, thread_id, "fan_paste", {"text": text})
        conn.commit()
    return {"ok": True}

@app.post("/send_creator")
def send_creator(payload: CreatorIn):
    text = (payload.text or "").strip()
    if not text:
        return {"ok": True}
    with db() as conn:
        thread_id = ensure_test_thread(conn)
        cast = _thread_cast(thread_id)
        with conn.cursor() as cur:
            cur.execute(
                f"insert into messages (thread_id, role, text) values (%s{cast}, 'creator', %s) returning id, created_at",
                (thread_id, text),
            )
            row = cur.fetchone()
            mid = row["id"]

        if mem is not None:
            mem.extract_from_message(conn, thread_id, "creator", text, msg_id=mid)
            mem.track_patterns_on_creator(conn, thread_id, text, msg_id=mid)
            for ev in mem.extract_schedules(text, CREATOR_TZ):
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            insert into thread_schedules (thread_id, who, title, kind, scheduled_at, timezone, confidence, source_msg_id)
                            values (%s::uuid, 'creator', %s, %s, %s, %s, %s, %s)
                            on conflict (thread_id, title, scheduled_at) do nothing
                            """,
                            (thread_id, ev["title"], ev["kind"], ev["scheduled_at"], ev["timezone"], ev["confidence"], mid),
                        )
                except Exception:
                    logging.exception("failed to insert schedule")
            mem.register_turn_and_maybe_refresh_profile(conn, thread_id)

        log_event(conn, thread_id, "creator_send", {"text": text})
        conn.commit()
    return {"ok": True}

@app.post("/suggest", response_model=SuggestOut)
def suggest_legacy():
    with db() as conn:
        thread_id = ensure_test_thread(conn)
        cast = _thread_cast(thread_id)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                select role, text
                from messages
                where thread_id=%s{cast}
                order by id desc
                limit 8
                """,
                (thread_id,),
            )
            recent = cur.fetchall()[::-1]
    fan_lines = [m["text"] for m in recent if m["role"] == "fan"]
    wants_burst = len(fan_lines) >= 2 or any("?" in x for x in fan_lines)
    reply_count = min(3, 2 if wants_burst else 1)
    drafts = (
        ["Hey babe 😏 tell me more about your day..."]
        if reply_count == 1
        else ["mm I was thinking about you…", "what part of your day made you smile? 😊"][:reply_count]
    )
    return {"drafts": drafts}

@app.post("/undo_last")
def undo_last(payload: UndoIn):
    n = max(1, min(5, payload.count or 1))
    with db() as conn:
        thread_id = ensure_test_thread(conn)
        cast = _thread_cast(thread_id)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                select id from messages
                where thread_id=%s{cast}
                order by id desc
                limit %s
                """,
                (thread_id, n),
            )
            ids = [r["id"] for r in cur.fetchall()]
        if not ids:
            return {"deleted": 0}
        with conn.cursor() as cur:
            cur.execute("delete from messages where id = any(%s)", (ids,))
        log_event(conn, thread_id, "undo", {"deleted_ids": ids})
        conn.commit()
    return {"deleted": len(ids)}

@app.get("/messages", response_model=List[MsgOut])
def get_messages_legacy():
    with db() as conn:
        thread_id = ensure_test_thread(conn)
        cast = _thread_cast(thread_id)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                select role, text
                from messages
                where thread_id=%s{cast}
                order by id asc
                limit 500
                """,
                (thread_id,),
            )
            rows = cur.fetchall()
    return [{"role": r["role"], "text": r["text"]} for r in rows]

# ---------------- per-thread messaging ----------------
@app.get("/thread/{thread_id}/messages", response_model=List[MsgOut])
def thread_messages(thread_id: str):
    cast = _thread_cast(thread_id)
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select role, text
            from messages
            where thread_id=%s{cast}
            order by id asc
            limit 500
            """,
            (thread_id,),
        )
        rows = cur.fetchall()
    return [{"role": r["role"], "text": r["text"]} for r in rows]

@app.post("/thread/{thread_id}/log_fan")
def thread_log_fan(thread_id: str, payload: FanIn):
    text = (payload.fan_message_text or "").strip()
    if not text:
        return {"ok": True}
    cast = _thread_cast(thread_id)
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"insert into messages (thread_id, role, text) values (%s{cast}, 'fan', %s) returning id, created_at",
                (thread_id, text),
            )
            row = cur.fetchone()
            mid = row["id"]
        if mem is not None:
            mem.extract_from_message(conn, thread_id, "fan", text, msg_id=mid)
            mem.track_patterns_on_fan_reply(conn, thread_id, fan_msg_id=mid)
            for ev in mem.extract_schedules(text, CREATOR_TZ):
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            insert into thread_schedules (thread_id, who, title, kind, scheduled_at, timezone, confidence, source_msg_id)
                            values (%s::uuid, 'fan', %s, %s, %s, %s, %s, %s)
                            on conflict (thread_id, title, scheduled_at) do nothing
                            """,
                            (thread_id, ev["title"], ev["kind"], ev["scheduled_at"], ev["timezone"], ev["confidence"], mid),
                        )
                except Exception:
                    logging.exception("failed to insert schedule")
            mem.register_turn_and_maybe_refresh_profile(conn, thread_id)
        log_event(conn, thread_id, "fan_paste", {"text": text})
        conn.commit()
    return {"ok": True}

@app.post("/thread/{thread_id}/send_creator")
def thread_send_creator(thread_id: str, payload: CreatorIn):
    text = (payload.text or "").strip()
    if not text:
        return {"ok": True}
    cast = _thread_cast(thread_id)
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"insert into messages (thread_id, role, text) values (%s{cast}, 'creator', %s) returning id, created_at",
                (thread_id, text),
            )
            row = cur.fetchone()
            mid = row["id"]
        if mem is not None:
            mem.extract_from_message(conn, thread_id, "creator", text, msg_id=mid)
            mem.track_patterns_on_creator(conn, thread_id, text, msg_id=mid)
            for ev in mem.extract_schedules(text, CREATOR_TZ):
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            insert into thread_schedules (thread_id, who, title, kind, scheduled_at, timezone, confidence, source_msg_id)
                            values (%s::uuid, 'creator', %s, %s, %s, %s, %s, %s)
                            on conflict (thread_id, title, scheduled_at) do nothing
                            """,
                            (thread_id, ev["title"], ev["kind"], ev["scheduled_at"], ev["timezone"], ev["confidence"], mid),
                        )
                except Exception:
                    logging.exception("failed to insert schedule")
            mem.register_turn_and_maybe_refresh_profile(conn, thread_id)
        log_event(conn, thread_id, "creator_send", {"text": text})
        conn.commit()
    return {"ok": True}

@app.post("/thread/{thread_id}/undo_last")
def thread_undo_last(thread_id: str, payload: UndoIn):
    n = max(1, min(5, payload.count or 1))
    cast = _thread_cast(thread_id)
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                select id from messages
                where thread_id=%s{cast}
                order by id desc
                limit %s
                """,
                (thread_id, n),
            )
            ids = [r["id"] for r in cur.fetchall()]
        if not ids:
            return {"deleted": 0}
        with conn.cursor() as cur:
            cur.execute("delete from messages where id = any(%s)", (ids,))
        log_event(conn, thread_id, "undo", {"deleted_ids": ids})
        conn.commit()
    return {"deleted": len(ids)}

# ---- aliases for backwards compat (/threads/...) ----
@app.get("/threads/{thread_id}/messages", response_model=List[MsgOut])
def threads_messages_alias(thread_id: str):
    return thread_messages(thread_id)

@app.post("/threads/{thread_id}/log_fan")
def threads_log_fan_alias(thread_id: str, payload: FanIn):
    return thread_log_fan(thread_id, payload)

@app.post("/threads/{thread_id}/send_creator")
def threads_send_creator_alias(thread_id: str, payload: CreatorIn):
    return thread_send_creator(thread_id, payload)

# ---------------- customers ----------------
@app.post("/customers/new", response_model=CustomerOut)
def customers_new(payload: CustomerNewIn):
    name = (payload.display_name or "").strip()
    if not name:
        raise HTTPException(400, "display_name is required")
    handle = (payload.handle or "").strip() or _slugify(name)

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into fans (creator_id, platform_user_id, profile)
                values (%s, %s, %s::jsonb)
                on conflict (creator_id, platform_user_id)
                do update set profile = fans.profile || excluded.profile
                returning id, profile
                """,
                (CREATOR_ID, handle, Json({"name": name})),
            )
            fan_id = cur.fetchone()["id"]

        thread_id = ensure_thread_for_fan(conn, fan_id)

        out = CustomerOut(
            id=str(fan_id),
            display_name=name,
            platform_user_id=handle,
            tier=0,
            status="active",
            ltv_cents=0,
            last_msg_preview="",
            secs_since_last_msg=999999,
            thread_id=str(thread_id),
            last_stage=None,
        )
        conn.commit()
    return out

@app.get("/customers", response_model=List[CustomerOut])
def customers_list():
    rows: List[dict]
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                with fan_base as (
                  select f.id as fan_id,
                         f.platform_user_id,
                         coalesce(f.profile->>'name', f.platform_user_id) as display_name,
                         (f.profile->>'tier')::int as explicit_tier,
                         coalesce(f.profile->>'status','active') as status
                  from fans f
                  where f.creator_id=%s
                    and f.platform_user_id <> 'test_fan'
                ),
                threads_one as (
                  select distinct on (t.fan_id) t.fan_id, t.id as thread_id
                  from threads t
                  join fan_base fb on fb.fan_id = t.fan_id
                  where t.creator_id=%s
                  order by t.fan_id, t.created_at asc, t.id asc
                ),
                ltv as (
                  select p.thread_id, coalesce(sum(p.price_cents) filter (where p.status='paid'),0)::int as ltv_cents
                  from ppv_offers p
                  group by p.thread_id
                ),
                last_msg as (
                  select m.thread_id,
                         max(m.id) as last_id,
                         max(m.created_at) as last_at
                  from messages m
                  group by m.thread_id
                )
                select
                  fb.fan_id,
                  fb.platform_user_id,
                  fb.display_name,
                  fb.explicit_tier,
                  fb.status,
                  t1.thread_id,
                  coalesce(l.ltv_cents,0) as ltv_cents,
                  coalesce(extract(epoch from now() - lm.last_at)::int, 999999) as secs_since_last_msg,
                  coalesce(m.text,'') as last_msg_preview
                from fan_base fb
                left join threads_one t1 on t1.fan_id = fb.fan_id
                left join ltv l on l.thread_id = t1.thread_id
                left join last_msg lm on lm.thread_id = t1.thread_id
                left join messages m on m.id = lm.last_id
                order by fb.display_name asc
                """,
                (CREATOR_ID, CREATOR_ID),
            )
            rows = cur.fetchall()

        # ensure every fan has a thread
        for r in rows:
            if not r["thread_id"]:
                r["thread_id"] = ensure_thread_for_fan(conn, r["fan_id"])
        conn.commit()

    out: List[CustomerOut] = []
    for r in rows:
        tier = _compute_tier(r.get("explicit_tier"), int(r["ltv_cents"]))
        out.append(
            CustomerOut(
                id=str(r["fan_id"]),
                display_name=r["display_name"],
                platform_user_id=r["platform_user_id"],
                tier=tier,
                status=r.get("status") or "active",
                ltv_cents=int(r["ltv_cents"] or 0),
                last_msg_preview=r.get("last_msg_preview") or "",
                secs_since_last_msg=int(r.get("secs_since_last_msg") or 999999),
                thread_id=str(r["thread_id"]),
                last_stage=None,
            )
        )
    return out

# ---------------- PPV helpers & endpoints ----------------
def _update_tier_on_payment(conn, thread_id: Any, action: str):
    """
    Policy:
      - On 'unpaid': unpaid_strikes += 1; if strikes >= 3 => tier = -1 (Bronze)
      - On 'paid': reset strikes to 0; tier = Gold if LTV>0; Diamond if LTV>=50000
    """
    cast = _thread_cast(thread_id)
    with conn.cursor() as cur:
        cur.execute(f"select fan_id from threads where id=%s{cast}", (thread_id,))
        row = cur.fetchone()
        if not row:
            return
        fan_id = row["fan_id"]

    if action == "unpaid":
        with conn.cursor() as cur:
            cur.execute(
                """
                update fans
                   set profile = coalesce(profile,'{}'::jsonb) ||
                                  jsonb_build_object('unpaid_strikes',
                                      coalesce((profile->>'unpaid_strikes')::int,0) + 1)
                 where id=%s
             returning coalesce((profile->>'unpaid_strikes')::int,0) as strikes
                """,
                (fan_id,),
            )
            strikes = int((cur.fetchone() or {"strikes": 0})["strikes"])
        if strikes >= 3:
            with conn.cursor() as cur:
                cur.execute(
                    "update fans set profile = coalesce(profile,'{}'::jsonb) || jsonb_build_object('tier', -1) where id=%s",
                    (fan_id,),
                )
        return

    if action == "paid":
        with conn.cursor() as cur:
            cur.execute(
                "update fans set profile = coalesce(profile,'{}'::jsonb) || jsonb_build_object('unpaid_strikes', 0) where id=%s",
                (fan_id,),
            )
        with conn.cursor() as cur:
            cur.execute(
                f"select coalesce(sum(price_cents) filter (where status='paid'),0)::int as ltv from ppv_offers where thread_id=%s{cast}",
                (thread_id,),
            )
            ltv = int((cur.fetchone() or {"ltv": 0})["ltv"])
        new_tier = 2 if ltv >= 50000 else (1 if ltv > 0 else 0)
        with conn.cursor() as cur:
            cur.execute(
                "update fans set profile = coalesce(profile,'{}'::jsonb) || jsonb_build_object('tier', %s) where id=%s",
                (new_tier, fan_id),
            )

def _thread_fan_and_ltv(conn, thread_id: Any) -> Tuple[Optional[Any], int, Optional[int]]:
    """Return (fan_id, ltv_cents, explicit_tier_or_none)."""
    cast = _thread_cast(thread_id)
    with conn.cursor() as cur:
        cur.execute(f"select fan_id from threads where id=%s{cast}", (thread_id,))
        r = cur.fetchone()
        fan_id = r["fan_id"] if r else None
    with conn.cursor() as cur:
        cur.execute(
            f"select coalesce(sum(price_cents) filter (where status='paid'),0)::int as ltv from ppv_offers where thread_id=%s{cast}",
            (thread_id,),
        )
        ltv = int((cur.fetchone() or {"ltv": 0})["ltv"])
    explicit_tier: Optional[int] = None
    if fan_id is not None:
        with conn.cursor() as cur:
            cur.execute("select (profile->>'tier')::int as t from fans where id=%s", (fan_id,))
            row = cur.fetchone()
            explicit_tier = row["t"] if row and row["t"] is not None else None
    return fan_id, ltv, explicit_tier

@app.post("/thread/{thread_id}/ppv/offer", response_model=OfferStartOut)
def ppv_offer_create_thread(thread_id: str, payload: OfferStartIn):
    desc = (payload.description or "").strip()
    if len(desc) > 600:
        desc = desc[:600].rstrip()
    price = payload.price_cents if isinstance(payload.price_cents, int) else 1499
    price = max(299, min(price, 100000))
    asset = (payload.asset_label or "").strip() or "Custom clip"

    cast = _thread_cast(thread_id)
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            insert into ppv_offers (thread_id, description, price_cents, asset_label, status)
            values (%s{cast}, %s, %s, %s, 'suggested')
            returning id
            """,
            (thread_id, desc, price, asset),
        )
        offer_id = cur.fetchone()["id"]
    with db() as conn:
        log_event(conn, thread_id, "ppv_offer_created", {"offer_id": str(offer_id), "price_cents": price, "asset_label": asset})
        conn.commit()
    return {"offer_id": str(offer_id)}

# legacy (also accepts optional thread_id for compatibility)
@app.post("/ppv/offer", response_model=OfferStartOut)
def ppv_offer_create_legacy(payload: OfferStartIn):
    with db() as conn:
        thread_id = payload.thread_id or ensure_test_thread(conn)
        cast = _thread_cast(thread_id)
        desc = (payload.description or "").strip()
        if len(desc) > 600:
            desc = desc[:600].rstrip()
        price = payload.price_cents if isinstance(payload.price_cents, int) else 1499
        price = max(299, min(price, 100000))
        asset = (payload.asset_label or "").strip() or "Custom clip"
        with conn.cursor() as cur:
            cur.execute(
                f"""
                insert into ppv_offers (thread_id, description, price_cents, asset_label, status)
                values (%s{cast}, %s, %s, %s, 'suggested')
                returning id
                """,
                (thread_id, desc, price, asset),
            )
            offer_id = cur.fetchone()["id"]
        log_event(conn, thread_id, "ppv_offer_created", {"offer_id": str(offer_id), "price_cents": price, "asset_label": asset})
        conn.commit()
    return {"offer_id": str(offer_id)}

@app.get("/thread/{thread_id}/ppv/offers")
def ppv_offers_recent(thread_id: str, limit: int = 10):
    cast = _thread_cast(thread_id)
    limit = max(1, min(50, limit))
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select id::text as id, status, price_cents, asset_label, left(description, 160) as preview
            from ppv_offers
            where thread_id=%s{cast}
            order by created_at desc
            limit %s
            """,
            (thread_id, limit),
        )
        return {"offers": cur.fetchall()}

@app.post("/ppv/offer/{offer_id}/mark")
def ppv_offer_mark(offer_id: UUID, payload: OfferMarkIn):
    action = payload.action
    if action not in {"sent", "not_sent", "paid", "unpaid"}:
        return {"ok": False, "error": "invalid action"}

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("select thread_id from ppv_offers where id = %s::uuid", (str(offer_id),))
            row = cur.fetchone()
            if not row:
                return {"ok": False, "error": "offer not found"}
            thread_id = row["thread_id"]

        with conn.cursor() as cur:
            cur.execute("update ppv_offers set status=%s where id=%s::uuid", (action, str(offer_id)))

        # Pattern success learning on 'paid'
        if mem is not None and action == "paid":
            try:
                mem.track_patterns_on_paid(conn, str(thread_id))
            except Exception:
                logging.exception("pattern-paid update failed")

        # Apply policy and compute up-to-date LTV/tier to return for UI
        try:
            _update_tier_on_payment(conn, thread_id, action)
        except Exception:
            logging.exception("tier update failed")

        # compute ltv & tier to return
        fan_id, ltv, explicit_tier = _thread_fan_and_ltv(conn, thread_id)
        current_tier = _compute_tier(explicit_tier, ltv)

        log_event(conn, thread_id, "ppv_offer_mark", {"offer_id": str(offer_id), "action": action})
        conn.commit()

    return {"ok": True, "ltv_cents": ltv, "tier": current_tier}

@app.post("/ppv/offer/{offer_id}/undo")
def ppv_offer_undo(offer_id: UUID):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("select thread_id from ppv_offers where id=%s::uuid", (str(offer_id),))
            row = cur.fetchone()
            if not row:
                return {"ok": False, "error": "offer not found"}
            thread_id = row["thread_id"]

        with conn.cursor() as cur:
            cur.execute("delete from ppv_offers where id=%s::uuid", (str(offer_id),))

        log_event(conn, thread_id, "ppv_offer_undo", {"offer_id": str(offer_id)})
        conn.commit()

    return {"ok": True, "deleted": 1}

# ---------------- per-thread suggest (available) ----------------
@app.post("/thread/{thread_id}/suggest", response_model=SuggestOut)
def suggest_thread(thread_id: str):
    cast = _thread_cast(thread_id)
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select role, text
            from messages
            where thread_id=%s{cast}
            order by id desc
            limit 8
            """,
            (thread_id,),
        )
        recent = cur.fetchall()[::-1]

    fan_lines = [m["text"] for m in recent if m["role"] == "fan"]
    wants_burst = len(fan_lines) >= 2 or any("?" in x for x in fan_lines)
    reply_count = min(3, 2 if wants_burst else 1)

    drafts = (
        ["mmm still thinking about what you said… tell me one more detail 😏"]
        if reply_count == 1
        else ["I can't stop picturing that 👀", "what would make tonight perfect for you? 🥰"][:reply_count]
    )
    return {"drafts": drafts}

# ---------------- Brain (RunPod) ----------------
def _build_router_input_public(conn, thread_id: str) -> dict:
    cast = _thread_cast(thread_id)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select role, text
            from messages
            where thread_id=%s{cast}
            order by id desc
            limit 8
            """,
            (thread_id,),
        )
        recent = cur.fetchall()[::-1]
    return _build_router_input(conn, thread_id, recent)

@app.get("/brain/models")
def brain_models():
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"} if RUNPOD_API_KEY else {}
    try:
        r = requests.get(f"{RUNPOD_BASE}/models", headers=headers, timeout=30)
        if "application/json" in (r.headers.get("content-type") or ""):
            return r.json()
        return {"status": r.status_code, "body": r.text}
    except Exception as e:
        return {"status": 0, "error": str(e)}

@app.get("/brain/ping")
def brain_ping():
    if not RUNPOD_API_KEY:
        raise HTTPException(500, "RUNPOD_API_KEY missing")
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": COACH_MODEL, "messages": [{"role": "user", "content": "Reply 'OK'"}], "temperature": 0.2, "max_tokens": 5}
    r = requests.post(f"{RUNPOD_BASE}/chat/completions", headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code, "error": r.text}
    data = r.json()
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return {"ok": True, "content": content}

@app.post("/brain/plan")
def brain_plan():
    if not RUNPOD_API_KEY:
        raise HTTPException(500, "RUNPOD_API_KEY missing")

    with db() as conn:
        thread_id = ensure_test_thread(conn)
        cast = _thread_cast(thread_id)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                select role, text
                from messages
                where thread_id=%s{cast}
                order by id desc
                limit 8
                """,
                (thread_id,),
            )
            recent = cur.fetchall()[::-1]

        router_input = _build_router_input(conn, thread_id, recent)
        choice = _choose_stage(router_input)
        stage_name = choice.get("stage") or DEFAULT_STAGE_CARD

        try:
            log_event(
                conn,
                thread_id,
                "coach_stage_pick",
                {
                    "stage": stage_name,
                    "reason": choice.get("reason"),
                    "signals": router_input.get("db", {}),
                },
            )
        except Exception:
            pass

    if load_card:
        try:
            sys = load_card(stage_name)
        except Exception:
            logging.exception("Failed to load card '%s'", stage_name)
            sys = (
                "You are the COACH planner. Reply only with one JSON object: "
                '{"next_action":string,"reason":string,'
                '"offer":{"suggest":boolean,"price_cents":int|null,"asset_label":string|null,"blurred_preview":string|null},'
                '"tone":string,"next_state":string}. No markdown or extra text.'
            )
    else:
        sys = (
            "You are the COACH planner. Reply only with one JSON object: "
            '{"next_action":string,"reason":string,'
            '"offer":{"suggest":boolean,"price_cents":int|null,"asset_label":string|null,"blurred_preview":string|null},'
            '"tone":string,"next_state":string}. No markdown or extra text.'
        )

    user = {
        "state": {"recent_messages": recent},
        "rules": {
            "opener_is_for_brand_new": True,
            "no_price_in_opener": True,
            "no_explicit_in_opener": True,
            "one_question_only": True,
        },
    }

    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": COACH_MODEL, "messages": [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}], "temperature": 0.2, "max_tokens": 256}

    try:
        resp = requests.post(f"{RUNPOD_BASE}/chat/completions", headers=headers, json=payload, timeout=120)
    except Exception as e:
        logging.exception("RunPod chat request failed")
        return {"ok": False, "error": f"request_error: {e.__class__.__name__}: {e}"}

    if resp.status_code != 200:
        probe = {"model": COACH_MODEL, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 8, "temperature": 0.1}
        try:
            probe_resp = requests.post(f"{RUNPOD_BASE}/chat/completions", headers=headers, json=probe, timeout=60)
            probe_info = {"probe_status": probe_resp.status_code, "probe_body": probe_resp.text}
        except Exception as pe:
            probe_info = {"probe_error": str(pe)}
        return {"ok": False, "status": resp.status_code, "error": resp.text, **probe_info}

    data = resp.json()
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    plan = _extract_json(content) if content else {}

    if not isinstance(plan, dict) or "next_action" not in plan:
        plan = {
            "next_action": "build_trust",
            "reason": "fallback",
            "offer": {"suggest": False, "price_cents": None, "asset_label": None, "blurred_preview": None},
            "tone": "playful",
            "next_state": "",
        }

    with db() as conn:
        try:
            log_event(conn, thread_id, "coach_plan_raw", {"raw": content})
        except Exception:
            pass
        log_event(conn, thread_id, "coach_plan", {"plan": plan})
        conn.commit()

    return {"plan": plan}

@app.get("/brain/diag")
def brain_diag():
    if not RUNPOD_API_KEY:
        raise HTTPException(500, "RUNPOD_API_KEY missing")
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": COACH_MODEL, "messages": [{"role": "user", "content": "Say OK"}], "temperature": 0.1, "max_tokens": 8}
    try:
        r = requests.post(f"{RUNPOD_BASE}/chat/completions", headers=headers, json=payload, timeout=60)
    except Exception as e:
        return {"ok": False, "status": 0, "error": f"request_error: {e}"}
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code, "error": r.text}
    data = r.json()
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return {"ok": True, "content": content}

@app.get("/brain/plan/test")
def brain_plan_test():
    return brain_plan()

# ---------------- Router / Memory / Scheduler diagnostics ----------------
@app.get("/brain/router/preview")
def router_preview():
    with db() as conn:
        thread_id = ensure_test_thread(conn)
        cast = _thread_cast(thread_id)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                select role, text
                from messages
                where thread_id=%s{cast}
                order by id desc
                limit 8
                """,
                (thread_id,),
            )
            recent = cur.fetchall()[::-1]

        router_input = _build_router_input(conn, thread_id, recent)

    choice = _choose_stage(router_input)
    return {
        "stage": choice.get("stage") or DEFAULT_STAGE_CARD,
        "reason": choice.get("reason"),
        "debug": choice.get("debug", {}),
        "signals": router_input,
    }

# -------- memory/profile/patterns observers --------
@app.get("/thread/{thread_id}/memory/summary")
def memory_summary(thread_id: str):
    with db() as conn:
        return _memory_summary(conn, thread_id)

@app.get("/thread/{thread_id}/profile")
def profile_get(thread_id: str):
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "select turns, summary, traits, updated_at from thread_profiles where thread_id=%s::uuid",
            (thread_id,)
        )
        row = cur.fetchone()
        if not row:
            return {"turns": 0, "summary": "", "traits": {}, "updated_at": None}
        return {
            "turns": int(row["turns"] or 0),
            "summary": row["summary"] or "",
            "traits": row["traits"] or {},
            "updated_at": row["updated_at"],
        }

@app.get("/thread/{thread_id}/patterns")
def patterns_get(thread_id: str):
    if get_thread_patterns is None:
        raise HTTPException(500, "patterns module not available")
    with db() as conn:
        return get_thread_patterns(conn, thread_id)

# -------- SCHEDULE endpoints --------
@app.get("/thread/{thread_id}/schedule/upcoming")
def schedule_upcoming(thread_id: str, hours: int = 168):
    hours = max(1, min(24*90, hours))
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select id, who, title, kind, scheduled_at, timezone, confidence, status, generated_text
              from thread_schedules
             where thread_id=%s::uuid
               and scheduled_at <= now() + (%s || ' hours')::interval
               and status in ('pending','ready')
             order by scheduled_at asc
            """,
            (thread_id, hours),
        )
        rows = cur.fetchall() or []
    return {"upcoming": rows}

@app.get("/scheduler/ready")
def scheduler_ready(limit: int = 20):
    limit = max(1, min(100, limit))
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select id, thread_id::text as thread_id, title, kind, scheduled_at, generated_text
              from thread_schedules
             where status='ready'
             order by scheduled_at asc
             limit %s
            """,
            (limit,),
        )
        rows = cur.fetchall() or []
    return {"ready": rows}

class SchedMarkIn(BaseModel):
    action: Literal["ready","sent","dismissed"]

@app.post("/scheduler/{sched_id}/mark")
def scheduler_mark(sched_id: int, payload: SchedMarkIn):
    act = payload.action
    if act not in {"ready","sent","dismissed"}:
        raise HTTPException(400, "invalid action")
    with db() as conn, conn.cursor() as cur:
        cur.execute("update thread_schedules set status=%s where id=%s returning thread_id", (act, sched_id))
        r = cur.fetchone()
        if not r:
            raise HTTPException(404, "schedule not found")
        thread_id = r["thread_id"]
    return {"ok": True, "thread_id": str(thread_id), "status": act}

# --------- Background Scheduler Loop ---------
async def _scheduler_loop():
    if not SCHEDULER_ENABLED:
        logging.info("Scheduler disabled.")
        return
    logging.info("Scheduler: starting loop (poll=%ss, grace=%ss, tz=%s)", SCHEDULER_POLL_SECS, SCHEDULER_DUE_GRACE_SECS, CREATOR_TZ)
    while True:
        try:
            due_cutoff = datetime.now(timezone.utc) + timedelta(seconds=SCHEDULER_DUE_GRACE_SECS)
            with db() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    select id, thread_id::text as thread_id, title, kind
                      from thread_schedules
                     where status='pending' and scheduled_at <= %s
                     order by scheduled_at asc
                     limit 25
                    for update skip locked
                    """,
                    (due_cutoff,),
                )
                rows = cur.fetchall() or []

                for r in rows:
                    sched_id = r["id"]
                    th_id = r["thread_id"]
                    title = r["title"]; kind = r["kind"]
                    data = _compose_reminder_text(conn, th_id, title, kind)
                    with conn.cursor() as c2:
                        c2.execute(
                            "update thread_schedules set status='ready', generated_text=%s, last_triggered_at=now() where id=%s",
                            (data.get("text") or "", sched_id),
                        )
                    log_event(conn, th_id, "scheduler_ready", {"schedule_id": sched_id, "title": title, "kind": kind})
                conn.commit()
        except Exception as e:
            logging.exception("Scheduler tick error: %s", e)
        await asyncio.sleep(SCHEDULER_POLL_SECS)

@app.on_event("startup")
async def on_startup():
    if SCHEDULER_ENABLED:
        asyncio.create_task(_scheduler_loop())

@app.get("/brain/card/{name}")
def card_resolved(name: str):
    if not load_card:
        raise HTTPException(500, "cards_loader not available")
    try:
        text = load_card(name)
        return {"name": name, "text": text}
    except Exception as e:
        raise HTTPException(404, f"card '{name}' not found or failed to load: {e}")

# ---------------- NEW: Pattern ingest & stats from history ----------------
@app.post("/patterns/ingest/{thread_id}")
def patterns_ingest(thread_id: str):
    if ingest_thread_patterns is None:
        raise HTTPException(500, "patterns module not available")
    with db() as conn:
        res = ingest_thread_patterns(conn, thread_id)
        conn.commit()
    return res

@app.post("/patterns/ingest_all")
def patterns_ingest_all(limit: int = 2000):
    if ingest_thread_patterns is None:
        raise HTTPException(500, "patterns module not available")
    updated = 0
    with db() as conn, conn.cursor() as cur:
        cur.execute("select id::text from threads where creator_id=%s order by created_at asc", (CREATOR_ID,))
        tids = [r["id"] for r in cur.fetchall()]
        for tid in tids[: max(1, limit)]:
            res = ingest_thread_patterns(conn, tid)
            updated += int(res.get("updated") or 0)
        conn.commit()
    return {"ok": True, "updated": updated}

# ---- v2 Brain Bridge (sidecar) ---------------------------------------------
# Adds /brain/suggest_v2 that forwards to the evolvable-brain sidecar.
# BRAIN_URL can be set (default: http://127.0.0.1:8001)

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import os
import httpx

class _V2SuggestPayload(BaseModel):
    messages: List[Dict[str, Any]]
    profile: Optional[Dict[str, Any]] = None
    ppv_catalog: Optional[List[Dict[str, Any]]] = None
    budget: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None

@app.post("/brain/suggest_v2")
async def brain_suggest_v2(payload: _V2SuggestPayload, view: str = "operator"):
    base = os.getenv("BRAIN_URL", "http://127.0.0.1:8001")
    params = {"view": view}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(f"{base}/suggest", params=params, json=payload.model_dump())
        r.raise_for_status()
        return r.json()
# ---------------------------------------------------------------------------

# ========================= PHASE A: OPERATOR-SAFE ONE-CALL ==================
# Pull last-8 + budgets + storybook; call brain /auto_decide; log decision.

def _budgets_from_tier(tier: int) -> dict:
    """
    Bronze(-1): super conservative
    Silver(0): default conservative
    Gold(1): moderate
    Diamond(2): higher ceiling
    """
    if tier <= -1:
        return dict(max_paid_per_24h_user=1, min_hours_between_paid=8.0, price_floor=8.99, price_ceiling=14.99, price_step=1.0, exploration_quota=0.05)
    if tier == 0:
        return dict(max_paid_per_24h_user=2, min_hours_between_paid=6.0, price_floor=9.00,  price_ceiling=19.00,  price_step=1.0, exploration_quota=0.10)
    if tier == 1:
        return dict(max_paid_per_24h_user=3, min_hours_between_paid=4.0, price_floor=11.00, price_ceiling=25.00,  price_step=1.0, exploration_quota=0.15)
    # diamond
    return dict(max_paid_per_24h_user=4, min_hours_between_paid=3.0, price_floor=13.00, price_ceiling=39.00, price_step=1.0, exploration_quota=0.20)

def _ppv_catalog_for_thread(conn, thread_id: str) -> list[dict]:
    """
    Try to pull a real catalog if table exists; otherwise empty list (Brain will skip PPV).
    Expected table (optional): ppv_catalog(creator_id uuid, ppv_asset_id text, base_price numeric, description text, media_type text, tags text[])
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select ppv_asset_id, base_price::float, description, media_type
                  from ppv_catalog
                 where creator_id=%s
                 order by created_at desc
                 limit 200
                """,
                (CREATOR_ID,),
            )
            rows = cur.fetchall() or []
            return [
                {"ppv_asset_id": r["ppv_asset_id"], "base_price": float(r["base_price"]), "description": r.get("description") or "", "media_type": r.get("media_type") or "video"}
                for r in rows
            ]
    except Exception:
        # table may not exist yet
        return []

@app.post("/thread/{thread_id}/brain/next")
async def thread_brain_next(thread_id: str, view: str = "operator"):
    """
    Pull context from DB, compute budgets, call sidecar /auto_decide.
    Returns the Brain's Decision JSON untouched.
    """
    cast = _thread_cast(thread_id)
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select role, text
              from messages
             where thread_id=%s{cast}
             order by id desc
             limit 16
            """,
            (thread_id,),
        )
        recent = cur.fetchall()[::-1]  # chronological

        # split into fan/creator for the Brain contract
        fan_last = [{"role": "fan", "text": r["text"]} for r in recent if r["role"] == "fan"][-8:]
        creator_last = [{"role": "creator", "text": r["text"]} for r in recent if r["role"] == "creator"][-8:]

        # memory summary (storybook/facts)
        memsum = _memory_summary(conn, thread_id) or {}
        storybook = ""
        try:
            counts = memsum.get("counts", {})
            storybook = f"petnames_fan={counts.get('petnames_fan_count',0)}, inside_jokes={counts.get('inside_jokes_count',0)}"
        except Exception:
            storybook = ""

        # tier/LTV -> budgets
        fan_id, ltv_cents, explicit_tier = _thread_fan_and_ltv(conn, thread_id)
        tier = _compute_tier(explicit_tier, ltv_cents)
        budgets = _budgets_from_tier(tier)

        # context
        local_hour = datetime.now(_safe_tz()).hour
        ctx = {"local_hour": int(local_hour), "consecutive_no_reply": 0}

        # optional PPV catalog
        catalog = _ppv_catalog_for_thread(conn, thread_id)

    base = os.getenv("BRAIN_URL", "http://127.0.0.1:8001")
    payload = {
        "messages": {"fan_last": fan_last, "creator_last": creator_last},
        "memory": {"storybook": storybook},
        "profile": {"fan_id": str(fan_id) if fan_id else "unknown", "tier": ["bronze","silver","gold","diamond"][max(-1,tier)+1], "relationship_age_days": 0},
        "budgets": budgets,
        "context": ctx,
        "catalog": catalog
    }

    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.post(f"{base}/auto_decide", json=payload, params={"view": view})
        r.raise_for_status()
        decision = r.json()

    # Log decision (observability)
    with db() as conn:
        try:
            log_event(conn, thread_id, "brain_decide_v2", {"decision": decision, "budgets": budgets})
            conn.commit()
        except Exception:
            logging.exception("failed to log brain_decide_v2")

    return decision

# ===================== PHASE B: PERSONA WRITER (STYLE LAYER) =================

def _writer_headers():
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

def _default_persona(conn, thread_id: str) -> dict:
    memsum = _memory_summary(conn, thread_id) or {}
    pet = "baby"
    try:
        # pick a petname if creator used any (very simple heuristic)
        if (memsum.get("counts") or {}).get("petnames_creator_count", 0) > 0:
            pet = "baby"  # you can enrich to pull last used petname from mem.items later
    except Exception:
        pass
    return {
        "tone": "playful",
        "energy": "medium",
        "emoji_budget": 2,
        "petname": pet,
        "style_rules": [
            "1 question max",
            "short lines",
            "no giant paragraphs",
            "consensual, flirty, warm",
        ],
    }

def _heuristic_style(bubbles: List[dict], persona: dict) -> List[dict]:
    """
    Offline fallback: rewrite texts quickly to approximate persona.
    No network; keeps same bubble count.
    """
    tone = (persona.get("tone") or "playful").lower()
    pet = persona.get("petname") or "baby"
    emoji_budget = int(persona.get("emoji_budget") or 2)

    def add_emojis(s: str) -> str:
        if emoji_budget <= 0: 
            return s
        emojis = {
            "playful": ["😉","😏","👀","😜","🥰"],
            "soft": ["☺️","✨","💕","😊","🌸"],
            "flirty": ["😏","🔥","💋","😉","👀"],
            "warm": ["😊","💖","🤍","✨","☺️"],
        }.get(tone, ["😊"])
        used = 0
        out = []
        for tok in s.split():
            out.append(tok)
            if used < emoji_budget and tok.endswith("?"):
                out[-1] = tok + emojis[used % len(emojis)]
                used += 1
        s2 = " ".join(out)
        # If we still have budget, tack one at the end
        if used < emoji_budget:
            s2 = (s2 + " " + emojis[used % len(emojis)]).strip()
        return s2

    styled = []
    for b in bubbles:
        t = (b.get("text") or "").strip()
        if not t:
            styled.append({"text": t})
            continue
        # inject petname once if not present
        if pet.lower() not in t.lower() and len(t.split()) > 3:
            # insert petname after first word
            parts = t.split()
            t = parts[0] + f" {pet}," + " " + " ".join(parts[1:])
        # keep one question max (very light touch)
        qs = [i for i,c in enumerate(t) if c == "?"]
        if len(qs) > 1:
            # drop later '?' chars
            idxs = qs[1:]
            t = "".join(c for i,c in enumerate(t) if i not in idxs)
        # cap length
        if len(t) > 220:
            t = t[:220].rstrip() + "…"
        t = add_emojis(t)
        styled.append({"text": t})
    return styled

async def _call_writer_api(bubbles: List[dict], persona: dict) -> Optional[List[dict]]:
    if not RUNPOD_API_KEY:
        return None
    sys = (
        "You are a persona writer. Rewrite the given chat bubbles into the creator's voice. "
        "Respect tone/energy/emoji budget and style rules. Keep the same number of bubbles. "
        "Return JSON only: {\"bubbles\":[{\"text\":string}, ...]}."
    )
    user = {"persona": persona, "bubbles": bubbles}
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            r = await client.post(
                f"{RUNPOD_WRITER_BASE}/chat/completions",
                headers=_writer_headers(),
                json={
                    "model": RUNPOD_WRITER_MODEL,
                    "messages": [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                    ],
                    "temperature": 0.5,
                    "max_tokens": 300,
                },
            )
            if r.status_code != 200:
                return None
            data = r.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            j = _extract_json(content)
            if isinstance(j, dict) and isinstance(j.get("bubbles"), list):
                return [{"text": (x.get("text") or "").strip()} for x in j["bubbles"]]
            return None
    except Exception:
        return None

class PersonaIn(BaseModel):
    tone: Optional[str] = None
    energy: Optional[str] = None
    emoji_budget: Optional[int] = None
    petname: Optional[str] = None
    style_rules: Optional[List[str]] = None

@app.post("/persona/default/{thread_id}")
def persona_default(thread_id: str):
    with db() as conn:
        return _default_persona(conn, thread_id)

@app.post("/thread/{thread_id}/brain/next_styled")
async def thread_brain_next_styled(thread_id: str, persona: Optional[PersonaIn] = None):
    """
    Compose -> decide (Brain) -> style (Writer or heuristic) -> log.
    """
    # 1) get neutral decision from Phase A endpoint
    decision = await thread_brain_next(thread_id, view="operator")

    # 2) build persona (from input or default)
    with db() as conn:
        per = (persona.model_dump() if persona else {}) if persona else {}
        defaults = _default_persona(conn, thread_id)
        # merge defaults where missing
        persona_used = {**defaults, **{k:v for k,v in per.items() if v is not None}}

    bubbles = (decision.get("message_pack") or {}).get("bubbles") or []
    # Normalize to [{text:...}]
    bubbles = [{"text": b.get("text") if isinstance(b, dict) else str(b)} for b in bubbles]

    # 3) try writer API; fallback to heuristic
    styled = await _call_writer_api(bubbles, persona_used)
    if not styled:
        styled = _heuristic_style(bubbles, persona_used)

    # 4) swap bubbles + log
    decision.setdefault("message_pack", {})["bubbles"] = styled

    with db() as conn:
        try:
            log_event(conn, thread_id, "writer_style", {"persona": persona_used, "bubbles": styled})
            conn.commit()
        except Exception:
            logging.exception("failed to log writer_style")

    # include persona so operator UI can preview rules if needed
    decision["persona_used"] = persona_used
    return decision

# ===================== PHASE C/D add-ons: decisions feed ====================
@app.get("/thread/{thread_id}/brain/last_decisions")
def brain_last_decisions(thread_id: str, limit: int = 20):
    limit = max(1, min(100, limit))
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select created_at, data
              from events
             where thread_id=%s::uuid
               and type in ('brain_decide_v2','writer_style')
             order by created_at desc
             limit %s
            """,
            (thread_id, limit),
        )
        rows = cur.fetchall() or []
        return {"events": [{"created_at": r["created_at"], "data": r["data"]} for r in rows]}
# ===================== END PATCHES ==========================================
