# app/cards_loader.py
# Minimal card loader: find a card, inline its includes (front-matter + inline),
# dedupe, cycle-guard, and return the fully inlined text.

from pathlib import Path
from functools import lru_cache
import re
from typing import List, Set

BASE = Path(__file__).resolve().parent
CARD_DIRS = [
    BASE / "coach_cards" / "stage",
    BASE / "coach_cards" / "tactic",
    BASE / "coach_cards" / "pricing",
    BASE / "coach_cards" / "objection",
    BASE / "coach_cards",           # fallback root
    BASE / "safety",                # shared safety snippets
]

INCLUDE_LINE_RE = re.compile(r"(?m)^\s*include:\s*([A-Za-z0-9_.\-/]+)\s*$")
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
FM_INLINE_RE     = re.compile(r"(?m)^\s*includes?\s*:\s*\[(.*?)\]\s*$")
FM_BULLET_RE     = re.compile(r"(?m)^\s*-\s*([A-Za-z0-9_.\-/]+)\s*$")

def _find_file(name: str) -> Path:
    if not name:
        raise FileNotFoundError("Empty card name")
    candidates = [name] if name.endswith(".md") else [name, f"{name}.md"]

    for root in CARD_DIRS:
        for cand in candidates:
            p = root / cand
            if p.exists():
                return p
    for root in CARD_DIRS:
        for cand in candidates:
            hits = list(root.rglob(cand))
            if hits:
                return hits[0]
    p = BASE / candidates[-1]
    if p.exists():
        return p
    raise FileNotFoundError(f"Card not found: {name}")

def _front_matter(text: str) -> str | None:
    m = FRONT_MATTER_RE.match(text)
    return m.group(1) if m else None

def _front_matter_includes(text: str) -> List[str]:
    fm = _front_matter(text)
    if not fm:
        return []
    incs: List[str] = []
    m_inline = FM_INLINE_RE.search(fm)
    if m_inline:
        inside = m_inline.group(1)
        for raw in inside.split(","):
            name = raw.strip().strip("'\"")
            if name:
                incs.append(name)
    for mb in FM_BULLET_RE.finditer(fm):
        incs.append(mb.group(1).strip())
    # de-dup preserve order
    seen, out = set(), []
    for x in incs:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out

def _inline_include_directives(text: str) -> List[str]:
    return [m.group(1).strip() for m in INCLUDE_LINE_RE.finditer(text)]

def _strip_include_lines(text: str) -> str:
    return INCLUDE_LINE_RE.sub("", text)

def _resolve(name: str, visited: Set[str]) -> str:
    key = name if name.endswith(".md") else f"{name}.md"
    if key in visited:
        return ""  # cycle guard
    visited.add(key)

    p = _find_file(name)
    raw = p.read_text(encoding="utf-8")

    incs = _front_matter_includes(raw) + _inline_include_directives(raw)
    # de-dup
    seen, uniq = set(), []
    for inc in incs:
        if inc and inc not in seen:
            uniq.append(inc); seen.add(inc)

    parts = [_resolve(inc, visited) for inc in uniq]
    parts.append(_strip_include_lines(raw))
    return "\n\n".join(t for t in parts if t.strip())

@lru_cache(maxsize=256)
def load_card(name: str) -> str:
    # cache for speed; restart the app to pick up edits during dev
    return _resolve(name, set())