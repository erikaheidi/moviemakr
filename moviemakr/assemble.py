"""Stitch the rendered clips into one movie.

Normalize first, concat second. The generator writes PCM audio into WebM, which
is off-spec and does not stream-copy reliably, and scenes may differ in size -
so each clip is re-encoded to uniform codecs/resolution/fps, which makes the
concat itself a cheap stream copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .config import Script
from .media import (
    NormalizeSpec,
    concat_cmd,
    concat_list_text,
    music_mix_cmd,
    normalize_clip,
    run_ffmpeg,
)
from .state import load_state


def overlap_trim(scene, state: dict) -> float:
    """Seconds to drop from the head of a scene, because it was anchored there.

    An overlap-chained scene regenerates the tail of the one before it, so those
    frames exist twice and the movie must show them once. The count comes from
    what the render actually anchored - recorded in state.json - not from the
    script's current `overlap_frames`, which may have been edited since. Trimming
    by today's value would cut the wrong amount out of yesterday's clip.
    """
    entry = (state.get("scenes") or {}).get(scene.id) or {}
    frames = entry.get("overlap_frames") or 0
    if frames <= 0:
        return 0.0
    return frames / scene.settings.fps


def normalize_spec(script: Script) -> NormalizeSpec:
    width, height = script.primary_size
    return NormalizeSpec(
        width=width,
        height=height,
        fps=script.fps,
        container=script.output.container,
        keep_audio=script.output.keep_audio,
    )


def assemble(script: Script, scenes: Sequence) -> Path:
    layout = script.layout
    layout.ensure_dirs()
    spec = normalize_spec(script)

    print(f"\n=== assembling {len(scenes)} scene(s) ===")

    # Editing the script can change size/fps/audio handling, so a stale
    # intermediate must be rebuilt even when its source clip is untouched.
    script_mtime = script.path.stat().st_mtime
    state = load_state(layout.state_file)
    normalized: list[Path] = []
    for scene in scenes:
        clip = layout.clip(scene.slug)
        dest = layout.normalized(scene.slug)
        trim = overlap_trim(scene, state)
        if (not dest.is_file()
                or dest.stat().st_mtime < max(clip.stat().st_mtime, script_mtime)):
            note = f" (trimming {trim:.2f}s of overlap)" if trim else ""
            print(f"  normalizing {scene.slug}{note}")
            normalize_clip(clip, dest, spec, skip_seconds=trim)
        normalized.append(dest)

    layout.concat_file.write_text(concat_list_text(normalized))

    movie = layout.movie
    music = script.output.music
    mixing = bool(music) and script.output.keep_audio
    concat_target = layout.concat_tmp if mixing else movie

    print("  concatenating")
    run_ffmpeg(concat_cmd(layout.concat_file, concat_target))

    if mixing:
        print("  mixing music bed")
        run_ffmpeg(music_mix_cmd(
            concat_target, music, movie,
            script.output.music_gain_db, spec.codecs["acodec"],
        ))
        concat_target.unlink(missing_ok=True)

    return movie
