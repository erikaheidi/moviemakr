"""Persisted per-scene fingerprints, timings and probe results.

Takes a Path rather than a Script so it stays a stdlib-only leaf: `status` can
read state without pulling in Docker.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EMPTY: dict[str, Any] = {"scenes": {}}


def load_state(state_file: Path) -> dict:
    """Read state.json, falling back to empty when absent or corrupt.

    A truncated file (an interrupted write) must not block a whole run - losing
    the fingerprints costs a re-render, raising here would cost the same and
    leave nothing usable.
    """
    if state_file.is_file():
        try:
            data = json.loads(state_file.read_text())
        except json.JSONDecodeError:
            return {"scenes": {}}
        if isinstance(data, dict):
            data.setdefault("scenes", {})
            return data
    return {"scenes": {}}


def save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2) + "\n")


def scene_entry(state: dict, scene_id: str) -> dict:
    return (state.get("scenes") or {}).get(scene_id) or {}
