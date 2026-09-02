"""Per-scene state, as data.

Answers "what would a render actually redo?" by recomputing each scene's
fingerprint and comparing it to the one stored in state.json. A scene whose
script changed since it rendered is **stale**: the clip is on disk, but it no
longer matches the script.

This is the shared source of truth for the `status` command and the web view, so
neither can drift from the other. It sits below `cli` and above `render` /
`docker` / `state` / `media` in the import graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends import comfy as comfy_backend
from .backends.sdcpp import fingerprint
from .config import Script
from .media import probe_clip
from .render import resolve_refs
from .state import load_state, scene_entry


def _current_fingerprint(scene, script: Script, prev_chain: Path | None):
    """Recompute a scene's fingerprint the way its backend would.

    Each backend hashes a different thing, so asking the wrong one would report
    every scene of a comfy run as stale and re-render the lot.
    """
    layout = script.layout
    if script.backend == "comfy":
        # Names only: this must not cut a tail clip just to answer `status`.
        name, host, _ = comfy_backend.prepare_chain(scene, script, prev_chain, dry_run=True)
        return comfy_backend.fingerprint(
            scene, script, [host] if host is not None else [], overlap_clip=name
        )
    refs, _ = resolve_refs(scene, prev_chain, dry_run=False)
    dirs = [
        layout.refvideo_dir(src, scene.settings.width, scene.settings.height)
        for src in scene.ref_videos
    ]
    return fingerprint(scene, script, refs, dirs)


def scene_rows(script: Script) -> list[dict[str, Any]]:
    """One row per scene: {"scene", "state", "probe", "elapsed"}.

    `state` is one of pending / rendered / stale / failed. Rows are in scene
    order, and the walk carries `prev_frame` forward because a chained scene's
    fingerprint depends on the previous scene's extracted last frame.
    """
    layout = script.layout
    state = load_state(layout.state_file)

    rows: list[dict[str, Any]] = []
    prev_chain: Path | None = None
    for scene in script.scenes:
        clip = layout.clip(scene.slug)
        entry = scene_entry(state, scene.id)

        if clip.is_file() and clip.stat().st_size > 0:
            probe = entry.get("probe") or probe_clip(clip)
            stored = entry.get("fingerprint")
            if stored is None:
                scene_state = entry.get("state", "rendered")
            else:
                current = _current_fingerprint(scene, script, prev_chain)
                scene_state = "rendered" if stored == current else "stale"
        else:
            probe = {}
            scene_state = entry.get("state", "pending")

        rows.append({
            "scene": scene,
            "state": scene_state,
            "probe": probe,
            "elapsed": entry.get("elapsed"),
        })
        if script.backend == "comfy":
            prev_chain = clip if clip.is_file() else None
        else:
            frame = layout.frame(scene.slug)
            prev_chain = frame if frame.is_file() else None

    return rows
