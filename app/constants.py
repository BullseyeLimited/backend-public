from typing import Final, Set, Dict, Any

# What the planner is allowed to output / we will clamp to.
ALLOWED_NEXT: Final[Set[str]] = {
    "open_curiosity",  # short, easy-reply opener
    "build_trust",
    "discovery",
    "tease",
    "offer",
    "cooldown",
}

ALLOWED_TONES: Final[Set[str]] = {
    "warm",
    "playful",
    "soft",
    "neutral",
    "flirty_low",
}

# Safe default plan for the opener stage
DEFAULT_PLAN: Final[Dict[str, Any]] = {
    "message_plan": {
        # which micro-hook to use in the first line
        "hook": "open_curiosity",
        # tone/energy guidance for the writer/agent
        "tone": "warm",
        "energy_match": "med",      # low / med / high
        "anchor": None,             # e.g. {"type":"time_window","value":"evening"}
        "emoji_max": 1,             # 0–2; opener should be light
    },
    "state_updates": {},            # brain can stash tiny flags here
    "safety": {
        "no_price": True,           # never mention price in opener
        "no_explicit": True,        # keep it SFW/flirty-light
        "no_fabrication": True,     # do not invent hard facts
        "one_question_only": True,  # reduce friction
    },
    # where the conversation should go after a successful opener
    "next_state": "discovery",
    # set True if user clearly wants to skip ahead (eager)
    "accelerate": False,
    "reason": "",                   # short natural-language why
}