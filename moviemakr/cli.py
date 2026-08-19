"""Command line entry point."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .assemble import assemble
from .config import Script, load_script
from .docker import fingerprint, format_argv
from .errors import ConfigError
from .media import probe_clip
from .render import RenderOptions, render, resolve_refs
from .report import print_summary
from .state import load_state, scene_entry

# The package lives one level below the project root, which owns assets/ and
# renders/ and anchors every relative path in a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moviemakr",
        description="Render a multi-scene movie script with stable-diffusion.cpp.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render", help="render scenes then assemble the movie")
    p_render.add_argument("script", type=Path)
    group = p_render.add_mutually_exclusive_group()
    group.add_argument("--only", help="scene indices, e.g. 2,4-6")
    group.add_argument("--scene", help="scene ids, comma separated")
    p_render.add_argument("--force", action="store_true", help="re-render even if up to date")
    p_render.add_argument("--retries", type=int, default=2, help="retries per scene (default 2)")
    p_render.add_argument("--halt-on-failure", action="store_true")
    p_render.add_argument("--no-assemble", action="store_true")
    p_render.add_argument("--dry-run", action="store_true", help="print commands, render nothing")
    p_render.add_argument("--allow-cpu", action="store_true",
                          help="skip the GPU preflight check and render even without a GPU")

    p_asm = sub.add_parser("assemble", help="re-assemble from existing clips")
    p_asm.add_argument("script", type=Path)

    p_status = sub.add_parser("status", help="show per-scene state")
    p_status.add_argument("script", type=Path)

    return parser


def cmd_status(script: Script) -> int:
    """Per-scene state, including whether a render would actually redo anything."""
    layout = script.layout
    state = load_state(layout.state_file)

    results = []
    prev_frame: Path | None = None
    for scene in script.scenes:
        clip = layout.clip(scene.slug)
        entry = scene_entry(state, scene.id)

        if clip.is_file() and clip.stat().st_size > 0:
            probe = entry.get("probe") or probe_clip(clip)
            stored = entry.get("fingerprint")
            if stored is None:
                scene_state = entry.get("state", "rendered")
            else:
                refs, _ = resolve_refs(scene, prev_frame, dry_run=False)
                dirs = [
                    layout.refvideo_dir(src, scene.settings.width, scene.settings.height)
                    for src in scene.ref_videos
                ]
                current = fingerprint(scene, script, refs, dirs)
                scene_state = "rendered" if stored == current else "stale"
        else:
            probe = {}
            scene_state = entry.get("state", "pending")

        results.append({
            "scene": scene,
            "state": scene_state,
            "probe": probe,
            "elapsed": entry.get("elapsed"),
        })
        frame = layout.frame(scene.slug)
        prev_frame = frame if frame.is_file() else None

    movie = layout.movie
    print_summary(results, layout.logs_dir, movie if movie.is_file() else None)
    return 0


def cmd_assemble(script: Script) -> int:
    layout = script.layout
    layout.ensure_dirs()
    scenes = [
        s for s in script.scenes
        if layout.clip(s.slug).is_file() and layout.clip(s.slug).stat().st_size > 0
    ]
    if not scenes:
        print("error: no rendered clips found", file=sys.stderr)
        return 1
    movie = assemble(script, scenes)
    print(f"\nmovie: {movie}")
    return 0


def check_tools(command: str, dry_run: bool) -> int | None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            print(f"error: {tool} not found on PATH", file=sys.stderr)
            return 2
    if command == "render" and not dry_run and shutil.which("docker") is None:
        print("error: docker not found on PATH", file=sys.stderr)
        return 2
    return None


def main(argv: Sequence[str] | None = None, project_root: Path | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = project_root or PROJECT_ROOT

    code = check_tools(args.command, getattr(args, "dry_run", False))
    if code is not None:
        return code

    try:
        script = load_script(args.script.resolve(), project_root)

        if args.command == "render":
            return render(script, RenderOptions.from_args(args))
        if args.command == "assemble":
            return cmd_assemble(script)
        if args.command == "status":
            return cmd_status(script)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        program = Path(exc.cmd[0]).name if exc.cmd else "command"
        detail = (exc.stderr or "").strip() or (exc.stdout or "").strip()
        print(f"error: {program} failed (exit {exc.returncode})", file=sys.stderr)
        if detail:
            print(detail, file=sys.stderr)
        print(f"  command: {format_argv(exc.cmd)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    return 0
