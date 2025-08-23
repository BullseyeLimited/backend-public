# app/telemetry/router.py
from __future__ import annotations

from typing import Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .loader import CFG, detect_signals, pick_stage, score_stages, reload_config

router = APIRouter(tags=["telemetry"])

class ClassifyIn(BaseModel):
    tier: str  # 'silver' | 'gold' | 'diamond' | 'emerald'
    text: str
    conversation_id: str | None = None
    turn_idx: int | None = None
    speaker: str | None = "fan"

class ClassifyOut(BaseModel):
    tier: str
    text: str
    signals: List[str]
    stage_scores: Dict[str, int]
    stage_pred: str
    catalog: Dict[str, Dict[str, str]] | None = None
    conversation_id: str | None = None
    turn_idx: int | None = None
    speaker: str | None = None

@router.get("/telemetry/config")
def telemetry_config():
    wm = CFG.weight_meta
    if not wm:
        raise HTTPException(500, "weight meta not loaded")
    return {
        "stages_by_tier": wm.stages_by_tier,
        "fallback_stage_by_tier": wm.fallback_stage_by_tier,
        "unknown_threshold": wm.unknown_threshold,
        "signal_count": len(CFG.signals),
        "stage_count": len(CFG.stages),
    }

@router.post("/telemetry/reload")
def telemetry_reload():
    return reload_config()

@router.post("/telemetry/classify", response_model=ClassifyOut)
def telemetry_classify(payload: ClassifyIn):
    tier = (payload.tier or "").lower()
    if tier not in ("silver", "gold", "diamond", "emerald"):
        raise HTTPException(400, "tier must be 'silver' | 'gold' | 'diamond' | 'emerald'")
    sigs = detect_signals(payload.text)
    stage_id, scores = pick_stage(tier, sigs)

    tiny = {}
    ids = CFG.weight_meta.stages_by_tier.get(tier, []) if CFG.weight_meta else []
    for sid in ids:
        st = CFG.stages.get(sid)
        if st:
            tiny[sid] = {"name": st.name, "definition": st.definition}

    return ClassifyOut(
        tier=tier,
        text=payload.text,
        signals=sigs,
        stage_scores=scores,
        stage_pred=stage_id,
        catalog=tiny,
        conversation_id=payload.conversation_id,
        turn_idx=payload.turn_idx,
        speaker=payload.speaker,
    )
