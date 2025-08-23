# New router that gathers the last 16 messages, computes budgets, builds persona-friendly payload,
# and calls the sidecar brain's /auto_decide. Drop this file into app/ and include the router
# from your main.py with:
#   from app.sidecar_bridge import router as sidecar_router
#   app.include_router(sidecar_router)
from __future__ import annotations

import os, json
from typing import List, Dict, Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException
import httpx
import psycopg
from psycopg.rows import dict_row

router = APIRouter(prefix="/thread", tags=["brain-bridge"])

DB_URL = os.getenv("DATABASE_URL")
BRAIN_URL = os.getenv("BRAIN_URL", "http://127.0.0.1:8001")

def _db():
    return psycopg.connect(DB_URL, row_factory=dict_row)

def _recent_dialog(conn, thread_id: str, limit: int = 16) -> List[Dict[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select role, text
            from messages
            where thread_id=%s::uuid
            order by id desc
            limit %s
            """,
            (thread_id, limit),
        )
        rows = cur.fetchall() or []
    # return ascending
    rows = rows[::-1]
    return [{"role": r["role"], "text": r["text"]} for r in rows]

def _split_roles(dialog: List[Dict[str,str]]) -> Dict[str, List[Dict[str,str]]]:
    fan = [{"text": d["text"]} for d in dialog if d["role"]=="fan"]
    creator = [{"text": d["text"]} for d in dialog if d["role"]=="creator"]
    # keep only the last 8 of each to bound size
    return {"fan_last": fan[-8:], "creator_last": creator[-8:]}

def _thread_profile(conn, thread_id: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("select fan_id from threads where id=%s::uuid", (thread_id,))
        r = cur.fetchone()
        if not r:
            raise HTTPException(404, "thread not found")
        fan_id = r["fan_id"]
        cur.execute("select profile from fans where id=%s", (fan_id,))
        f = cur.fetchone() or {"profile": {}}
        prof = f["profile"] or {}
        explicit_tier = prof.get("tier") if isinstance(prof, dict) else None

        # LTV
        cur.execute("select coalesce(sum(price_cents) filter (where status='paid'),0)::int as ltv from ppv_offers where thread_id=%s::uuid", (thread_id,))
        ltv = int((cur.fetchone() or {"ltv": 0})["ltv"])

    # map to label
    if isinstance(explicit_tier, int):
        label = { -1:"bronze", 0:"silver", 1:"gold", 2:"diamond" }.get(explicit_tier, "silver")
    else:
        label = "diamond" if ltv >= 50000 else ("gold" if ltv > 0 else "silver")
    # "emerald" is opt-in via profile flag
    if isinstance(prof, dict) and prof.get("relationship") == "emerald":
        label = "emerald"

    tz = None
    if isinstance(prof, dict):
        tz = prof.get("tz") or prof.get("timezone")
    tz = tz or "America/New_York"
    try:
        local_hour = datetime.now(ZoneInfo(tz)).hour
    except Exception:
        tz = "UTC"
        local_hour = datetime.utcnow().hour

    return {"fan_id": str(fan_id), "tier": label, "relationship_age_days": int(prof.get("relationship_age_days") or 0), "thread_timezone": tz, "local_hour": local_hour, "ltv_cents": ltv}

def _budgets(label: str, ltv_cents: int) -> Dict[str, Any]:
    # More permissive for deeper tiers (script flows can need multiple PPVs)
    base = {
        "silver":  {"max_paid_per_24h_user": 3, "min_hours_between_paid": 1.0,  "price_floor": 9,  "price_ceiling": 60,  "price_step": 1.0},
        "gold":    {"max_paid_per_24h_user": 4, "min_hours_between_paid": 0.75, "price_floor": 12, "price_ceiling": 120, "price_step": 1.0},
        "diamond": {"max_paid_per_24h_user": 5, "min_hours_between_paid": 0.5,  "price_floor": 18, "price_ceiling": 240, "price_step": 1.0},
        "emerald": {"max_paid_per_24h_user": 6, "min_hours_between_paid": 0.33, "price_floor": 25, "price_ceiling": 400, "price_step": 1.0},
    }[label]
    # light lift if LTV is high inside the label
    if ltv_cents >= 200000:
        base["price_ceiling"] = max(base["price_ceiling"], 600)
        base["max_paid_per_24h_user"] += 1
    base["exploration_quota"] = 0.25
    base["compute_tier"] = "balanced"
    return base

def _fake_catalog() -> List[Dict[str, Any]]:
    return [
        {"ppv_asset_id":"ppv_1001","title":"Mirror tease set","description":"Playful mirror set in black lace—smiles & curves.","media_type":"photo","tags":["tease","lingerie","mirror"],"base_price":10.0},
        {"ppv_asset_id":"ppv_2001","title":"Flirty bedroom mini","description":"Short playful clip, cozy vibe, sweet & suggestive.","media_type":"video","tags":["tease","cozy"],"base_price":18.0},
        {"ppv_asset_id":"ppv_2500","title":"Bedroom slow dance","description":"Tasteful slow dance by the window, soft lighting.","media_type":"video","tags":["dance","soft"],"base_price":24.0},
        {"ppv_asset_id":"ppv_3001","title":"Cute voice note","description":"Soft voice note saying hi and asking about your day.","media_type":"voice","tags":["voice","soft"],"base_price":12.0},
        {"ppv_asset_id":"ppv_9001","title":"Bundle: weekend set","description":"Mixed bundle of tasteful photos & a short playful clip.","media_type":"bundle","tags":["bundle","weekend"],"base_price":25.0},
    ]

@router.post("/{thread_id}/brain/decide")
async def thread_brain_decide(thread_id: str):
    if not DB_URL:
        raise HTTPException(500, "DATABASE_URL missing")
    dialog: List[Dict[str,str]]
    with _db() as conn:
        dialog = _recent_dialog(conn, thread_id, limit=16)
        prof = _thread_profile(conn, thread_id)
    split = _split_roles(dialog)
    payload = {
        "messages": {**split},
        "memory": {"storybook": ""},  # you can enrich later with your memory module summary
        "profile": {"fan_id": prof["fan_id"], "tier": prof["tier"], "relationship_age_days": prof["relationship_age_days"]},
        "budgets": _budgets(prof["tier"], prof["ltv_cents"]),
        "context": {"local_hour": prof["local_hour"], "consecutive_no_reply": 0, "thread_timezone": prof["thread_timezone"], "recent_dialog": dialog},
        "catalog": _fake_catalog()
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(f"{BRAIN_URL}/auto_decide", json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"brain error: {r.text[:200]}")
        return r.json()
