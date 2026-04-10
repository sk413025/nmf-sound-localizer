#!/usr/bin/env python3
"""Validate machine-readable governance semantics for a high-risk paper round."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


STATUSES = [
    "core-only",
    "second-layer earned",
    "branch earned",
    "leaf allowed",
]
STATUS_RANK = {status: idx for idx, status in enumerate(STATUSES)}

ARCHITECTURE_SCOPES = {
    "local-salience",
    "cross-section",
    "whole-manuscript",
}

ROLE_MODES = {"separated", "compressed"}
YES_NO = {"yes", "no"}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("top-level YAML payload must be a mapping")
    return data


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value.strip()
    return value


def _expect_nonempty(errors: list[str], mapping: dict[str, Any], key: str, context: str) -> None:
    if not _is_nonempty_string(mapping.get(key)):
        errors.append(f"{context}.{key} must be a non-empty string")


def _expect_enum(
    errors: list[str],
    mapping: dict[str, Any],
    key: str,
    allowed: set[str] | list[str],
    context: str,
) -> None:
    value = _normalized_value(mapping.get(key))
    if value not in allowed:
        errors.append(f"{context}.{key} must be one of {sorted(allowed)!r}; got {value!r}")


def _expect_empty(errors: list[str], mapping: dict[str, Any], key: str, context: str) -> None:
    value = mapping.get(key)
    if isinstance(value, list):
        if value:
            errors.append(f"{context}.{key} must be empty")
        return
    if _is_nonempty_string(value):
        errors.append(f"{context}.{key} must be empty")


def _validate_top_level(data: dict[str, Any], errors: list[str]) -> None:
    required = {
        "risk_level",
        "architecture_scope",
        "broader_significance_in_scope",
        "role_mode",
        "role_compression_reason",
        "task_packet",
        "review_verdict",
        "verification_verdict",
        "closeout",
    }
    missing = sorted(required - set(data))
    if missing:
        errors.append(f"top-level keys missing: {missing}")
        return

    if data["risk_level"] != "high-risk":
        errors.append("risk_level must be 'high-risk'; use this gate only for high-risk rounds")
    if data["architecture_scope"] not in ARCHITECTURE_SCOPES:
        errors.append(
            f"architecture_scope must be one of {sorted(ARCHITECTURE_SCOPES)!r}; "
            f"got {data['architecture_scope']!r}"
        )
    if data["broader_significance_in_scope"] is not True:
        errors.append(
            "broader_significance_in_scope must be true; this gate is only for "
            "high-risk rounds with broader significance in scope"
        )
    if data["role_mode"] not in ROLE_MODES:
        errors.append("role_mode must be 'separated' or 'compressed'")
    if "role_compression_reason" not in data:
        errors.append("role_compression_reason must be present")

    for key in ["task_packet", "review_verdict", "verification_verdict", "closeout"]:
        if not isinstance(data.get(key), dict):
            errors.append(f"{key} must be a mapping")


def _validate_task_packet(task_packet: dict[str, Any], errors: list[str]) -> None:
    context = "task_packet"
    _expect_nonempty(errors, task_packet, "core_discovery", context)
    _expect_enum(errors, task_packet, "proposed_status", STATUSES, context)
    status = _normalized_value(task_packet.get("proposed_status"))

    if status == "core-only":
        for key in [
            "second_layer_discovery",
            "trunk",
            "branch",
            "leaf",
            "front_door_preload_sentence",
            "no_bolt_on_test",
        ]:
            _expect_empty(errors, task_packet, key, context)
        readout = task_packet.get("two_takeaway_editor_readout")
        if readout not in (None, [], ""):
            errors.append(f"{context}.two_takeaway_editor_readout must be empty for core-only")
        return

    for key in ["second_layer_discovery", "trunk", "front_door_preload_sentence"]:
        _expect_nonempty(errors, task_packet, key, context)
    readout = task_packet.get("two_takeaway_editor_readout")
    if not isinstance(readout, list) or len(readout) != 2 or not all(
        _is_nonempty_string(item) for item in readout
    ):
        errors.append(f"{context}.two_takeaway_editor_readout must be a two-item non-empty list")

    if status == "second-layer earned":
        _expect_empty(errors, task_packet, "branch", context)
        _expect_empty(errors, task_packet, "leaf", context)
        _expect_empty(errors, task_packet, "no_bolt_on_test", context)
        return

    _expect_nonempty(errors, task_packet, "branch", context)
    _expect_nonempty(errors, task_packet, "no_bolt_on_test", context)

    if status == "branch earned":
        _expect_empty(errors, task_packet, "leaf", context)
        return

    _expect_nonempty(errors, task_packet, "leaf", context)


def _validate_review_verdict(review: dict[str, Any], errors: list[str]) -> None:
    context = "review_verdict"
    _expect_nonempty(errors, review, "reviewer_owner", context)
    _expect_enum(errors, review, "reviewed_status", STATUSES, context)
    if "status_change_note" not in review:
        errors.append(f"{context}.status_change_note must be present")


def _validate_verification(verifier: dict[str, Any], errors: list[str]) -> None:
    context = "verification_verdict"
    _expect_nonempty(errors, verifier, "verifier_owner", context)
    _expect_enum(errors, verifier, "verified_status", STATUSES, context)
    _expect_enum(errors, verifier, "independent_verification_complete", YES_NO, context)
    _expect_enum(errors, verifier, "closeout_ready", YES_NO, context)


def _validate_closeout(closeout: dict[str, Any], errors: list[str]) -> None:
    context = "closeout"
    _expect_nonempty(errors, closeout, "implementer_owner", context)
    _expect_enum(errors, closeout, "final_status", STATUSES, context)
    if "status_change_note" not in closeout:
        errors.append(f"{context}.status_change_note must be present")
    if "remaining_gap_disclosed" not in closeout:
        errors.append(f"{context}.remaining_gap_disclosed must be present")


def _require_status_change_note(errors: list[str], note: Any, context: str, message: str) -> None:
    normalized = _normalized_value(note)
    if not _is_nonempty_string(note) or normalized == "none":
        errors.append(f"{context}.status_change_note must explain the downgrade when {message}")


def _validate_owner_mode(data: dict[str, Any], review: dict[str, Any], verifier: dict[str, Any], closeout: dict[str, Any], errors: list[str]) -> None:
    role_mode = _normalized_value(data["role_mode"])
    role_reason = _normalized_value(data["role_compression_reason"])
    owners = {
        _normalized_value(closeout["implementer_owner"]),
        _normalized_value(review["reviewer_owner"]),
        _normalized_value(verifier["verifier_owner"]),
    }
    if role_mode == "separated":
        if len(owners) != 3:
            errors.append("role_mode='separated' requires distinct implementer, reviewer, and verifier owners")
        if role_reason not in {"", "none", None}:
            errors.append("role_compression_reason must be empty or 'none' when role_mode is separated")
        return

    if len(owners) == 3:
        errors.append("role_mode='compressed' requires at least one shared owner")
    if not _is_nonempty_string(data.get("role_compression_reason")) or role_reason == "none":
        errors.append("compressed role mode requires a non-empty role_compression_reason")


def _validate_cross_field_consistency(data: dict[str, Any], errors: list[str]) -> None:
    task_packet = data["task_packet"]
    review = data["review_verdict"]
    verifier = data["verification_verdict"]
    closeout = data["closeout"]

    _validate_owner_mode(data, review, verifier, closeout, errors)

    proposed = _normalized_value(task_packet["proposed_status"])
    reviewed = _normalized_value(review["reviewed_status"])
    final = _normalized_value(closeout["final_status"])
    verified = _normalized_value(verifier["verified_status"])

    if STATUS_RANK[reviewed] > STATUS_RANK[proposed]:
        errors.append("review_verdict.reviewed_status cannot exceed task_packet.proposed_status")
    if STATUS_RANK[final] > STATUS_RANK[reviewed]:
        errors.append("closeout.final_status cannot exceed review_verdict.reviewed_status")
    if verified != final:
        errors.append("verification_verdict.verified_status must match closeout.final_status")

    if STATUS_RANK[reviewed] < STATUS_RANK[proposed]:
        _require_status_change_note(
            errors,
            review.get("status_change_note"),
            "review_verdict",
            "reviewed_status is lower than task_packet.proposed_status",
        )
    if STATUS_RANK[final] < STATUS_RANK[reviewed]:
        _require_status_change_note(
            errors,
            closeout.get("status_change_note"),
            "closeout",
            "final_status is lower than review_verdict.reviewed_status",
        )

    if _normalized_value(verifier["independent_verification_complete"]) != "yes":
        errors.append("verification_verdict.independent_verification_complete must be 'yes'")
    if _normalized_value(verifier["closeout_ready"]) != "yes":
        errors.append("verification_verdict.closeout_ready must be 'yes'")


def validate_round(round_dir: Path) -> list[str]:
    errors: list[str] = []
    artifact = round_dir / "governance_round.yaml"
    if not artifact.exists():
        return [f"missing required artifact: {artifact}"]

    try:
        data = _load_yaml(artifact)
    except Exception as exc:  # pragma: no cover
        return [f"failed to load {artifact}: {exc}"]

    _validate_top_level(data, errors)
    if errors:
        return errors

    _validate_task_packet(data["task_packet"], errors)
    _validate_review_verdict(data["review_verdict"], errors)
    _validate_verification(data["verification_verdict"], errors)
    _validate_closeout(data["closeout"], errors)

    if not errors:
        _validate_cross_field_consistency(data, errors)

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate machine-readable governance semantics for a high-risk round."
    )
    parser.add_argument(
        "--round-dir",
        required=True,
        type=Path,
        help="Results round directory that must contain governance_round.yaml",
    )
    args = parser.parse_args(argv)

    errors = validate_round(args.round_dir.resolve())
    if errors:
        print("Governance round semantic check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Governance round semantic check passed for {args.round_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
