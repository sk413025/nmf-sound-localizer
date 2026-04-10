from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "paper"
    / "check_round_governance_semantics.py"
)


def _base_round(status: str = "branch earned") -> dict:
    return {
        "risk_level": "high-risk",
        "architecture_scope": "cross-section",
        "broader_significance_in_scope": True,
        "role_mode": "separated",
        "role_compression_reason": "none",
        "task_packet": {
            "core_discovery": "The system carries a recurring physical code.",
            "second_layer_discovery": (
                "The code relocates part of sensing into passive structure."
                if status != "core-only"
                else ""
            ),
            "trunk": (
                "Passive structure itself becomes part of the sensing substrate."
                if status != "core-only"
                else ""
            ),
            "branch": (
                "Coarse neighborhoods support bounded front-end filtering."
                if status in {"branch earned", "leaf allowed"}
                else ""
            ),
            "leaf": "A weaker farther implication." if status == "leaf allowed" else "",
            "proposed_status": status,
            "front_door_preload_sentence": (
                "This discovery relocates part of sensing into passive structure."
                if status != "core-only"
                else ""
            ),
            "two_takeaway_editor_readout": (
                [
                    "The system carries a recurring physical code.",
                    "Passive structure itself becomes part of the sensing substrate.",
                ]
                if status != "core-only"
                else []
            ),
            "no_bolt_on_test": (
                "Remove the trunk and the branch collapses."
                if status in {"branch earned", "leaf allowed"}
                else ""
            ),
        },
        "review_verdict": {
            "reviewer_owner": "Reviewer",
            "reviewed_status": status,
            "status_change_note": "none",
        },
        "verification_verdict": {
            "verifier_owner": "Verifier",
            "verified_status": status,
            "independent_verification_complete": "yes",
            "closeout_ready": "yes",
        },
        "closeout": {
            "implementer_owner": "Implementer",
            "final_status": status,
            "status_change_note": "none",
            "remaining_gap_disclosed": "none",
        },
    }


def _run(round_dir: Path, payload: dict) -> subprocess.CompletedProcess[str]:
    artifact = round_dir / "governance_round.yaml"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--round-dir", str(round_dir)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_branch_earned_round_passes(tmp_path: Path) -> None:
    result = _run(tmp_path / "branch-earned", _base_round("branch earned"))
    assert result.returncode == 0, result.stderr
    assert "passed" in result.stdout.lower()


def test_leaf_allowed_round_passes(tmp_path: Path) -> None:
    result = _run(tmp_path / "leaf-allowed", _base_round("leaf allowed"))
    assert result.returncode == 0, result.stderr


def test_core_only_cannot_carry_trunk_fields(tmp_path: Path) -> None:
    payload = _base_round("core-only")
    payload["task_packet"]["second_layer_discovery"] = "Should not be here."
    result = _run(tmp_path / "core-only-invalid", payload)
    assert result.returncode != 0
    assert "task_packet.second_layer_discovery" in result.stderr


def test_branch_status_requires_branch_field(tmp_path: Path) -> None:
    payload = _base_round("branch earned")
    payload["task_packet"]["branch"] = ""
    result = _run(tmp_path / "missing-branch", payload)
    assert result.returncode != 0
    assert "task_packet.branch" in result.stderr


def test_review_demotion_requires_note(tmp_path: Path) -> None:
    payload = _base_round("leaf allowed")
    payload["review_verdict"]["reviewed_status"] = "branch earned"
    payload["closeout"]["final_status"] = "branch earned"
    payload["verification_verdict"]["verified_status"] = "branch earned"
    result = _run(tmp_path / "review-demotion-no-note", payload)
    assert result.returncode != 0
    assert "review_verdict.status_change_note" in result.stderr


def test_removed_reviewer_and_closeout_projection_fields_are_optional(tmp_path: Path) -> None:
    payload = _base_round("branch earned")
    result = _run(tmp_path / "minimal-canonical-yaml", payload)
    assert result.returncode == 0, result.stderr


def test_review_demotion_with_note_passes(tmp_path: Path) -> None:
    payload = _base_round("leaf allowed")
    payload["review_verdict"]["reviewed_status"] = "branch earned"
    payload["review_verdict"]["status_change_note"] = "Drop the optional leaf to preserve trunk memory."
    payload["closeout"]["final_status"] = "branch earned"
    payload["verification_verdict"]["verified_status"] = "branch earned"
    result = _run(tmp_path / "review-demotion-pass", payload)
    assert result.returncode == 0, result.stderr


def test_closeout_demotion_requires_note(tmp_path: Path) -> None:
    payload = _base_round("branch earned")
    payload["closeout"]["final_status"] = "second-layer earned"
    payload["verification_verdict"]["verified_status"] = "second-layer earned"
    result = _run(tmp_path / "closeout-demotion-no-note", payload)
    assert result.returncode != 0
    assert "closeout.status_change_note" in result.stderr


def test_compressed_role_mode_requires_reason(tmp_path: Path) -> None:
    payload = _base_round("branch earned")
    payload["role_mode"] = "compressed"
    payload["role_compression_reason"] = "none"
    payload["review_verdict"]["reviewer_owner"] = "Implementer"
    payload["verification_verdict"]["verifier_owner"] = "Implementer"
    result = _run(tmp_path / "compressed-role-invalid", payload)
    assert result.returncode != 0
    assert "role_compression_reason" in result.stderr


def test_separated_role_mode_requires_distinct_owners(tmp_path: Path) -> None:
    payload = _base_round("branch earned")
    payload["review_verdict"]["reviewer_owner"] = "Implementer"
    result = _run(tmp_path / "separated-role-invalid", payload)
    assert result.returncode != 0
    assert "role_mode='separated'" in result.stderr


def test_verifier_must_match_final_status(tmp_path: Path) -> None:
    payload = _base_round("second-layer earned")
    payload["verification_verdict"]["verified_status"] = "branch earned"
    result = _run(tmp_path / "verifier-mismatch", payload)
    assert result.returncode != 0
    assert "verified_status" in result.stderr
