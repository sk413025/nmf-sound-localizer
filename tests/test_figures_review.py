from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from figures.style import collect_layout_metadata, save_outputs


def test_save_outputs_writes_layout_sidecar(tmp_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot([0, 1], [0, 1])
    ax.set_title("Example")

    outputs = save_outputs(fig, tmp_path / "example")
    plt.close(fig)

    assert len(outputs) == 2
    layout_path = tmp_path / "example.layout.json"
    assert layout_path.exists()

    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    assert payload["figure_mm"]["width"] > 0
    assert payload["figure_mm"]["height"] > 0
    assert len(payload["axes"]) == 1
    assert payload["axes"][0]["title"] == "Example"


def test_collect_layout_metadata_reports_axes_boxes() -> None:
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot([0, 1], [1, 0])
    ax2.imshow([[0, 1], [1, 0]])

    payload = collect_layout_metadata(fig)
    plt.close(fig)

    assert payload["figure_mm"]["width"] > 0
    assert payload["figure_mm"]["height"] > 0
    assert len(payload["axes"]) == 2
    assert payload["axes"][0]["bbox_mm"]["width"] > 0
    assert payload["axes"][1]["bbox_mm"]["height"] > 0
