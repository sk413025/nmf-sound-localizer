"""Runtime loader for the embedded figure layout contract.

The canonical contract lives in ``figures/conf/layout_spec.md`` so the human-
readable design spec and the machine-consumed values cannot silently drift.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYOUT_SPEC = REPO_ROOT / "figures" / "conf" / "layout_spec.md"
START_MARKER = "<!-- runtime-layout-contract:start -->"
END_MARKER = "<!-- runtime-layout-contract:end -->"


def _extract_contract_yaml(text: str) -> str:
    pattern = re.compile(
        rf"{re.escape(START_MARKER)}\s*```yaml\s*(.*?)\s*```\s*{re.escape(END_MARKER)}",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(
            f"Embedded runtime layout contract not found in {LAYOUT_SPEC}"
        )
    return match.group(1)


@lru_cache(maxsize=1)
def load_layout_contract() -> dict[str, Any]:
    text = LAYOUT_SPEC.read_text(encoding="utf-8")
    payload = yaml.safe_load(_extract_contract_yaml(text)) or {}
    if "fonts" not in payload or "figures" not in payload:
        raise RuntimeError(
            f"Runtime layout contract in {LAYOUT_SPEC} is missing required keys"
        )
    return payload


def contract_version() -> int:
    return int(load_layout_contract().get("version", 0))


def source_layout_spec() -> str:
    return str(load_layout_contract().get("source_layout_spec", "figures/conf/layout_spec.md"))


def font_tokens() -> dict[str, float]:
    return {
        key: float(value)
        for key, value in dict(load_layout_contract()["fonts"]).items()
    }


def semantic_palette() -> dict[str, str]:
    return {
        key: str(value)
        for key, value in dict(load_layout_contract()["semantic_palette"]).items()
    }


def semantic_color(name: str) -> str:
    palette = semantic_palette()
    if name not in palette:
        raise KeyError(f"Unknown semantic color token: {name}")
    return palette[name]


def stroke_tokens() -> dict[str, float]:
    return {
        key: float(value)
        for key, value in dict(load_layout_contract()["stroke_tokens"]).items()
    }


def stroke_pt(name: str) -> float:
    strokes = stroke_tokens()
    if name not in strokes:
        raise KeyError(f"Unknown stroke token: {name}")
    return strokes[name]


def family_style() -> dict[str, float]:
    return {
        key: float(value)
        for key, value in dict(load_layout_contract()["family_style"]).items()
    }


def family_style_value(name: str) -> float:
    profile = family_style()
    if name not in profile:
        raise KeyError(f"Unknown family style token: {name}")
    return profile[name]


def style_colors() -> dict[str, str]:
    return {
        key: str(value)
        for key, value in dict(load_layout_contract()["style_colors"]).items()
    }


def style_color(name: str) -> str:
    colors = style_colors()
    if name not in colors:
        raise KeyError(f"Unknown style color token: {name}")
    return colors[name]


def font_pt(name: str) -> float:
    fonts = font_tokens()
    if name not in fonts:
        raise KeyError(f"Unknown font token: {name}")
    return fonts[name]


def figure_typography(figure_id: str) -> dict[str, float]:
    typography = font_tokens()
    figure_cfg = figure_contract(figure_id)
    if "typography" in figure_cfg:
        typography.update(
            {
                key: float(value)
                for key, value in dict(figure_cfg["typography"]).items()
            }
        )
    return typography


def figure_contract(figure_id: str) -> dict[str, Any]:
    figures = load_layout_contract()["figures"]
    if figure_id not in figures:
        raise KeyError(f"Unknown figure contract: {figure_id}")
    return dict(figures[figure_id])


def figure_section(figure_id: str, section: str) -> dict[str, Any]:
    figure_cfg = figure_contract(figure_id)
    if section not in figure_cfg:
        raise KeyError(f"Figure {figure_id} has no '{section}' section in layout contract")
    return dict(figure_cfg[section])


def pt_to_px(pt: float, dpi: float) -> int:
    return max(1, int(round(float(pt) / 72.0 * float(dpi))))
