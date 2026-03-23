"""Shared figure-facing naming helpers for panel labels and manifests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CROSSWALK_PATH = (
    Path(__file__).resolve().parent / "conf" / "model_display_crosswalk.yaml"
)


@lru_cache(maxsize=None)
def load_display_crosswalk(yaml_path: str | Path | None = None) -> dict[str, Any]:
    """Load the active figure-display crosswalk."""
    path = DEFAULT_CROSSWALK_PATH if yaml_path is None else Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Display crosswalk not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def get_role(role_id: str, yaml_path: str | Path | None = None) -> dict[str, Any]:
    """Return the role payload for a semantic display role."""
    roles = load_display_crosswalk(yaml_path).get("roles", {})
    if role_id not in roles:
        raise KeyError(f"Unknown display role: {role_id}")
    return roles[role_id]


def get_role_label(
    role_id: str,
    label_type: str = "full",
    yaml_path: str | Path | None = None,
) -> str:
    """Return the full or short label for a semantic display role."""
    if label_type not in {"full", "short"}:
        raise ValueError(f"Unsupported label_type: {label_type}")
    role = get_role(role_id, yaml_path)
    key = "display_full" if label_type == "full" else "display_short"
    label = role.get(key)
    if not label:
        raise KeyError(f"Missing {key} for display role: {role_id}")
    return str(label)


def get_panel_binding(
    figure_id: str,
    panel_id: str,
    binding_id: str,
    yaml_path: str | Path | None = None,
) -> str:
    """Resolve a panel-local binding id to a semantic display role id."""
    figures = load_display_crosswalk(yaml_path).get("figures", {})
    try:
        bindings = figures[figure_id]["panels"][panel_id]["bindings"]
    except KeyError as exc:
        raise KeyError(
            f"Missing panel binding scope: {figure_id}.{panel_id}"
        ) from exc
    if binding_id not in bindings:
        raise KeyError(f"Missing binding {binding_id} in {figure_id}.{panel_id}")
    return str(bindings[binding_id])


def get_bound_label(
    figure_id: str,
    panel_id: str,
    binding_id: str,
    label_type: str = "full",
    yaml_path: str | Path | None = None,
) -> str:
    """Resolve a panel-local binding id to a full or short display label."""
    role_id = get_panel_binding(figure_id, panel_id, binding_id, yaml_path)
    return get_role_label(role_id, label_type=label_type, yaml_path=yaml_path)
