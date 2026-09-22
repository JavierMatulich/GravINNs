#!/usr/bin/env python3
"""Lightweight release audit for publication archives."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []

# Generated Python bytecode must not ship.
for pat in ("__pycache__", "*.pyc", "*.pyo"):
    hits = list(ROOT.rglob(pat))
    if hits:
        errors.append(f"generated Python artifacts found for {pat}: {len(hits)}")

# The PN series should exist only in the canonical physics module.
canonical = ROOT / "src/gravinns/physics/hamiltonian.py"
for path in (ROOT / "src/gravinns").rglob("*.py"):
    if path == canonical:
        continue
    text = path.read_text(encoding="utf-8")
    if "H1 =" in text or "H2 =" in text or "H3 =" in text:
        errors.append(f"possible duplicated PN Hamiltonian formula: {path.relative_to(ROOT)}")

# Publication metadata should not contain the temporary generic author value.
for name in ("pyproject.toml", "CITATION.cff", "LICENSE"):
    path = ROOT / name
    if "GravINNs contributors" in path.read_text(encoding="utf-8"):
        errors.append(f"generic author metadata remains in {name}")

if errors:
    print("RELEASE AUDIT FAILED")
    for err in errors:
        print(f" - {err}")
    sys.exit(1)

print("Release audit passed.")
