from copy import deepcopy
from typing import Any, Dict

from app.constants import ALLOWED_NEXT, ALLOWED_TONES, DEFAULT_PLAN

def _as_bool(v: Any, fallback: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low in {"true", "1", "yes", "y"}:
            return True
        if low in {"false", "0", "no", "n"}:
            return False
    return fallback

def _clamp_tone(v: Any) -> str:
    s = str(v or "").strip().lower()
    return s if s in ALLOWED_TONES else "warm"

def _clamp_next(v: Any) -> str:
    s = str(v or "").strip()
    return s if s in ALLOWED_NEXT else DEFAULT_PLAN["next_state"]

def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and harden a model-produced plan for the opener stage.
    - Accepts flat or nested keys.
    - Forces opener safety (no price/explicit, no fabrication, one question).
    - Clamps tone/next_state to allowed sets.
    - Fills all missing fields with DEFAULT_PLAN.
    """
    out = deepcopy(DEFAULT_PLAN)

    # --- accept both flat and nested shapes -------------------------------
    # Some models might output {tone, next_state, emoji_max...} at top-level.
    flat_tone = plan.get("tone")
    flat_next = plan.get("next_state")
    flat_emoji = plan.get("emoji_max")

    # Nested message_plan from model
    mp_in = (plan.get("message_plan") or {})
    msg_plan = {**out["message_plan"], **mp_in}

    # Merge flat hints into message_plan if present
    if flat_tone is not None:
        msg_plan["tone"] = flat_tone
    if flat_emoji is not None:
        msg_plan["emoji_max"] = flat_emoji

    # Clamp message_plan fields
    msg_plan["tone"] = _clamp_tone(msg_plan.get("tone"))
    try:
        msg_plan["emoji_max"] = max(0, min(2, int(msg_plan.get("emoji_max", out["message_plan"]["emoji_max"]))))
    except Exception:
        msg_plan["emoji_max"] = out["message_plan"]["emoji_max"]

    out["message_plan"] = msg_plan

    # Safety: opener is always strict
    s_in = (plan.get("safety") or {})
    s = {**out["safety"], **s_in}
    s["no_price"] = True
    s["no_explicit"] = True
    s["no_fabrication"] = True
    s["one_question_only"] = True
    out["safety"] = s

    # State / accelerate / reason
    next_state_in = plan.get("next_state", flat_next)
    out["next_state"] = _clamp_next(next_state_in)
    out["accelerate"] = _as_bool(plan.get("accelerate", out["accelerate"]), out["accelerate"])

    # state_updates (optional dict)
    if isinstance(plan.get("state_updates"), dict):
        out["state_updates"] = {**out["state_updates"], **plan["state_updates"]}

    # human reason string
    if plan.get("reason") is not None:
        out["reason"] = str(plan["reason"])

    return out