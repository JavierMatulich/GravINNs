"""Regression tests for the dissipative trainer wiring."""
from __future__ import annotations

import ast
from pathlib import Path


def test_dissipative_plotter_does_not_reference_standard_orbit_cadence():
    """The dissipative protocol records energy snapshots, not orbit snapshots.

    A previous refactor passed an undefined ``record_every_orbit`` variable to
    ``TrainingPlotter``, causing D1 and all other dissipative runs to fail before
    Stage 1.  Keep the dissipative call independent of that standard-trainer
    cadence.
    """
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "gravinns" / "training" / "dissipative_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TrainingPlotter"
    ]
    assert len(calls) == 1
    kw_names = {kw.arg for kw in calls[0].keywords if kw.arg is not None}
    assert "record_every_orbit" not in kw_names
