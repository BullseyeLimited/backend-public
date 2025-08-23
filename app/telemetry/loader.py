# app/telemetry/loader.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

CFG_ROOT = Path(__file__).resolve().parent.parent / "telemetry_cfg"

def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on {path.name}:{i} → {line[:120]} … ({e})")
    return out


# ---- regex guard: exact word/phrase, no substrings, no "'s"
def _strict_word_regex(token: str) -> str:
    # escapes token, allows spaces (phrases), disallows word-char neighbors and "'s"
    esc = re.escape(token)
    return rf"(?<!\w){esc}(?!\w|['’]s\b)"

@dataclass
class StageDef:
    id: str
    name: str
    definition: str
    tier: str

@dataclass
class SignalDef:
    id: str
    any_regex: List[str] = field(default_factory=list)
    none_regex: List[str] = field(default_factory=list)
    weights: Dict[str, int] = field(default_factory=dict)
    compiled_any: List[re.Pattern] = field(default_factory=list, init=False)
    compiled_none: List[re.Pattern] = field(default_factory=list, init=False)

    def compile(self):
        self.compiled_any = [re.compile(r, re.I) for r in self.any_regex]
        self.compiled_none = [re.compile(r, re.I) for r in self.none_regex]

    def fires_on(self, text: str) -> bool:
        t = text or ""
        if not t:
            return False
        if self.none_regex and any(rx.search(t) for rx in self.compiled_none):
            return False
        return any(rx.search(t) for rx in self.compiled_any)

@dataclass
class WeightMeta:
    stages_by_tier: Dict[str, List[str]]
    fallback_stage_by_tier: Dict[str, str]
    unknown_threshold: int = 1

@dataclass
class TelemetryConfig:
    stages: Dict[str, StageDef] = field(default_factory=dict)
    signals: List[SignalDef] = field(default_factory=list)
    weight_meta: WeightMeta | None = None

    def all_stage_ids(self, tier: str) -> List[str]:
        if not self.weight_meta:
            return []
        return self.weight_meta.stages_by_tier.get(tier, [])

    def tiny_catalog(self, tier: str) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        for sid in self.all_stage_ids(tier):
            st = self.stages.get(sid)
            if st:
                out[sid] = {"name": st.name, "definition": st.definition}
        return out

# --- defaults (used only if files missing) ---
_DEFAULT_SILVER = [
    StageDef("S1","Icebreaker","Polite first contact; 'hi/hey/hello', light compliment.","silver"),
    StageDef("S2","Curiosity","Info‑seeking about you/content; authenticity/boundary checks.","silver"),
    StageDef("S3","Rapport Seeds","Two‑way rapport; fan shares; warmth & callbacks.","silver"),
    StageDef("S4","Flirt & Light Tease","Playful interest; light innuendo; emojis.","silver"),
    StageDef("S5","Consideration","Price/menu/how‑it‑works; mild hesitation.","silver"),
    StageDef("S6","Objection / Deferral","Pushback on value, free sample asks, 'maybe later'.","silver"),
    StageDef("S7","Decision (In Progress)","Green‑light language: 'let’s do this / i’m going to'.","silver"),
    StageDef("S8","First Commitment Confirmed","Confirmed subscribe/tip/unlock.","silver"),
    StageDef("S9","Boundary / Repair","Discomfort/overstep; needs reset.","silver"),
]
_DEFAULT_GOLD = [
    StageDef("G1","Afterglow Welcome","Immediately post‑payment; thanks/access checks.","gold"),
    StageDef("G2","Ongoing Engagement","Routine paid chat; positive vibes; no price talk.","gold"),
    StageDef("G3","Upgrade Curiosity","Asks about customs/bundles/inclusions.","gold"),
    StageDef("G4","Objection / Scope Clarify","Clarifies expectations, samples, value friction.","gold"),
    StageDef("G5","Upsell Decision (In Progress)","Green‑light for extra tip/custom/unlock now.","gold"),
    StageDef("G6","Repeat Commitment Confirmed","Another tip/unlock/renewal confirmed.","gold"),
    StageDef("G7","Renewal Planning","Staying next month; scheduling future content.","gold"),
    StageDef("G8","Churn Risk / Recovery","Cancel/refund/‘not worth’ talk.","gold"),
    StageDef("G9","Reconnection / Repair","Apology/acknowledgment; bridge back.","gold"),
]
_DEFAULT_META = WeightMeta(
    stages_by_tier={"silver":[s.id for s in _DEFAULT_SILVER], "gold":[g.id for g in _DEFAULT_GOLD]},
    fallback_stage_by_tier={"silver":"S2","gold":"G2"},
    unknown_threshold=1,
)
_DEFAULT_SIGNALS: List[SignalDef] = []  # keep minimal; real ones live on disk

def _to_stage_map(stages: List[StageDef]) -> Dict[str, StageDef]:
    return {s.id: s for s in stages}

def _expand_keywords_to_regex(keywords: List[str]) -> List[str]:
    # build strict word/phrase regex for each keyword
    out: List[str] = []
    for kw in (keywords or []):
        kw = kw.strip()
        if not kw:
            continue
        out.append(_strict_word_regex(kw))
    return out

def _load_stages_file(path: Path) -> List[StageDef]:
    items = _read_jsonl(path)
    out: List[StageDef] = []
    for obj in items:
        out.append(StageDef(
            id=obj["id"], name=obj["name"], definition=obj["definition"], tier=obj["tier"].lower()
        ))
    return out

def _load_signals_file(path: Path) -> List[SignalDef]:
    raw = _read_jsonl(path)
    sigs: List[SignalDef] = []
    for o in raw:
        any_regex = list(o.get("any_regex", []))
        none_regex = list(o.get("none_regex", []))
        # NEW: keyword helpers
        if "keywords" in o:
            any_regex.extend(_expand_keywords_to_regex(o.get("keywords") or []))
        if "none_keywords" in o:
            none_regex.extend(_expand_keywords_to_regex(o.get("none_keywords") or []))
        s = SignalDef(
            id=o["id"],
            any_regex=any_regex,
            none_regex=none_regex,
            weights={k: int(v) for k, v in (o.get("weights") or {}).items()},
        )
        s.compile()
        sigs.append(s)
    return sigs

def _load_weight_meta(path: Path) -> WeightMeta:
    obj = _read_json(path)
    return WeightMeta(
        stages_by_tier={k: list(v) for k, v in (obj.get("stages_by_tier") or {}).items()},
        fallback_stage_by_tier={k: str(v) for k, v in (obj.get("fallback_stage_by_tier") or {}).items()},
        unknown_threshold=int(obj.get("unknown_threshold", 1)),
    )

def load_config() -> 'TelemetryConfig':
    stages_silver_path = CFG_ROOT / "stages_silver.jsonl"
    stages_gold_path   = CFG_ROOT / "stages_gold.jsonl"
    signals_path       = CFG_ROOT / "signals.jsonl"
    meta_path          = CFG_ROOT / "weight_meta.json"

    have_all = all(p.exists() for p in [stages_silver_path, stages_gold_path, signals_path, meta_path])

    if have_all:
        silver  = _load_stages_file(stages_silver_path)
        gold    = _load_stages_file(stages_gold_path)
        signals = _load_signals_file(signals_path)
        meta    = _load_weight_meta(meta_path)
        return TelemetryConfig(stages=_to_stage_map(silver + gold), signals=signals, weight_meta=meta)

    # defaults
    return TelemetryConfig(
        stages=_to_stage_map(_DEFAULT_SILVER + _DEFAULT_GOLD),
        signals=_DEFAULT_SIGNALS,
        weight_meta=_DEFAULT_META,
    )

CFG = load_config()

def detect_signals(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    hits: List[str] = []
    for sig in CFG.signals:
        try:
            if sig.fires_on(t):
                hits.append(sig.id)
        except re.error:
            continue
    return hits

def score_stages(tier: str, signals: List[str]) -> Dict[str, int]:
    stage_ids = CFG.weight_meta.stages_by_tier.get(tier.lower(), []) if CFG.weight_meta else []
    scores: Dict[str, int] = {sid: 0 for sid in stage_ids}
    for sid in signals:
        sig = next((s for s in CFG.signals if s.id == sid), None)
        if not sig:
            continue
        for stg in stage_ids:
            scores[stg] = scores.get(stg, 0) + int(sig.weights.get(stg, 0))
    return scores

def pick_stage(tier: str, signals: List[str]) -> Tuple[str, Dict[str, int]]:
    tier = tier.lower()
    scores = score_stages(tier, signals)
    if not scores:
        fb = CFG.weight_meta.fallback_stage_by_tier.get(tier, "") if CFG.weight_meta else ""
        return fb, {}
    best = max(scores.items(), key=lambda kv: kv[1])
    thr = CFG.weight_meta.unknown_threshold if CFG.weight_meta else 1
    if best[1] < thr:
        fb = CFG.weight_meta.fallback_stage_by_tier.get(tier, best[0]) if CFG.weight_meta else best[0]
        return fb, scores
    return best[0], scores

def reload_config() -> Dict[str, Any]:
    global CFG
    CFG = load_config()
    return {
        "loaded": True,
        "stage_count": len(CFG.stages),
        "signal_count": len(CFG.signals),
        "tiers": list(CFG.weight_meta.stages_by_tier.keys()) if CFG.weight_meta else [],
    }
