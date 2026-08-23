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
from .docker import format_argv
from .errors import ConfigError
from .layout import WORKSPACE_ENV, Workspace
from .render import RenderOptions, render
from .report import print_summary
from .status import scene_rows

# Where the data used to live, back when it sat inside the checkout. Kept only
# as the last fallback for `Workspace.resolve`, so an invocation with neither
# --workspace nor $MOVIEMAKR_WORKSPACE behaves exactly as it did before.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moviemakr",
        description="Render a multi-scene movie script with stable-diffusion.cpp.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared by every subcommand: which data root to read scripts/assets from
    # and write renders to.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--workspace", type=Path, default=None,
        help=f"data root holding scripts/, assets/, drafts/ and renders/ "
             f"(default: ${WORKSPACE_ENV}, else the moviemakr checkout)",
    )

    p_render = sub.add_parser("render", parents=[common],
                              help="render scenes then assemble the movie")
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

    p_asm = sub.add_parser("assemble", parents=[common],
                           help="re-assemble from existing clips")
    p_asm.add_argument("script", type=Path)

    p_status = sub.add_parser("status", parents=[common], help="show per-scene state")
    p_status.add_argument("script", type=Path)

    p_serve = sub.add_parser("serve", parents=[common],
                             help="browse the workspace over HTTP")
    p_serve.add_argument("--host", default="127.0.0.1",
                         help="bind address (default 127.0.0.1; use 0.0.0.0 behind tailscale serve)")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--reload", action="store_true", help="auto-reload on code changes")

    return parser


def cmd_status(script: Script) -> int:
    """Per-scene state, including whether a render would actually redo anything."""
    layout = script.layout
    movie = layout.movie
    print_summary(scene_rows(script), layout.logs_dir, movie if movie.is_file() else None)
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


def cmd_serve(args: argparse.Namespace, workspace: Workspace) -> int:
    """Imported lazily: the core CLI must work without FastAPI installed."""
    try:
        from .web import run_server
    except ImportError as exc:
        print(f"error: the web extra is not installed ({exc})", file=sys.stderr)
        print("  pip install 'moviemakr[web]'", file=sys.stderr)
        return 2
    return run_server(workspace, host=args.host, port=args.port, reload=args.reload)


def main(argv: Sequence[str] | None = None, project_root: Path | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = project_root or PROJECT_ROOT

    code = check_tools(args.command, getattr(args, "dry_run", False))
    if code is not None:
        return code

    try:
        workspace = Workspace.resolve(args.workspace, default=project_root)

        if args.command == "serve":
            return cmd_serve(args, workspace)

        script = load_script(args.script.resolve(), workspace)

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
