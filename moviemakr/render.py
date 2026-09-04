"""The render loop.

Scenes are rendered in order. Each one is fingerprinted first, so an unchanged
scene is skipped rather than re-rendered - a render costs minutes to hours.

The pure helpers at the top (`parse_index_spec`, `select_scenes`, `resolve_refs`,
`is_up_to_date`) hold the decisions; the I/O helpers below them hold the effects.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assemble import assemble
from .backends import comfy as comfy_backend
from .backends.sdcpp import (
    check_gpu,
    container_name,
    docker_argv,
    fingerprint,
    format_argv,
    run_scene,
)
from .config import Scene, Script
from .errors import ConfigError
from .media import extract_last_frame, prepare_ref_video, probe_clip
from .report import fmt_duration, print_summary
from .state import load_state, save_state, scene_entry

RETRY_BACKOFF_SECONDS = 5


@dataclass(frozen=True, slots=True)
class RenderOptions:
    """Render behaviour, decoupled from argparse so the pipeline is callable."""

    only: str | None = None
    scene: str | None = None
    force: bool = False
    retries: int = 2
    halt_on_failure: bool = False
    no_assemble: bool = False
    dry_run: bool = False
    allow_cpu: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RenderOptions:
        return cls(
            only=args.only,
            scene=args.scene,
            force=args.force,
            retries=args.retries,
            halt_on_failure=args.halt_on_failure,
            no_assemble=args.no_assemble,
            dry_run=args.dry_run,
            allow_cpu=args.allow_cpu,
        )

    @property
    def attempts(self) -> int:
        return max(1, self.retries + 1)


@dataclass(frozen=True, slots=True)
class SceneJob:
    """Everything resolved for one scene before the render/skip decision."""

    scene: Scene
    refs: tuple[Path, ...]
    ref_video_dirs: tuple[Path, ...]
    fingerprint: str
    # comfy only: the API graph, the anchor clip's name inside ComfyUI's input
    # dir, and how many frames it actually anchors (already snapped to the grid).
    graph: dict | None = None
    overlap_clip: str | None = None
    overlap_frames: int = 0


# --------------------------------------------------------------------------
# pure decisions
# --------------------------------------------------------------------------


def parse_index_spec(spec: str) -> set[int]:
    """Parse `--only 2,4-6` into a set of scene indices."""
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, _, hi = part.partition("-")
            selected.update(range(_index(lo, spec), _index(hi, spec) + 1))
        else:
            selected.add(_index(part, spec))
    return selected


def _index(text: str, spec: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        raise ConfigError(
            f"--only expects scene indices like '2,4-6', got {spec!r}"
        ) from None


def select_scenes(scenes: Sequence[Scene], opts: RenderOptions) -> set[int]:
    """Indices to render. Validates that the selection actually matches scenes."""
    if opts.scene:
        wanted = {s.strip() for s in opts.scene.split(",") if s.strip()}
        missing = wanted - {sc.id for sc in scenes}
        if missing:
            raise ConfigError(f"no such scene id(s): {', '.join(sorted(missing))}")
        return {sc.index for sc in scenes if sc.id in wanted}

    if opts.only:
        selected = parse_index_spec(opts.only)
        valid = {sc.index for sc in scenes}
        unknown = sorted(selected - valid)
        if unknown:
            raise ConfigError(
                f"--only refers to scene(s) that do not exist: "
                f"{', '.join(str(i) for i in unknown)} "
                f"(the script has {len(scenes)} scene(s))"
            )
        if not selected:
            raise ConfigError(f"--only selected no scenes: {opts.only!r}")
        return selected

    return {sc.index for sc in scenes}


def resolve_refs(scene: Scene, prev_frame: Path | None, *,
                 dry_run: bool) -> tuple[tuple[Path, ...], str | None]:
    """Reference images for a scene, chained frame first.

    Returns the refs and an optional warning, rather than printing, so the four
    chaining cases are testable without a filesystem.

    In a dry run the frame does not exist yet but will by the time this scene
    really runs, so the wiring the real run would use is shown.
    """
    if not scene.chain_from_previous:
        return tuple(scene.ref_images), None
    if prev_frame and (prev_frame.is_file() or dry_run):
        return (prev_frame, *scene.ref_images), None
    return tuple(scene.ref_images), (
        f"  ! {scene.slug}: chain_from_previous requested but no previous "
        f"frame is available - rendering without it"
    )


def is_up_to_date(clip: Path, entry: Mapping[str, Any], fp: str, *, force: bool) -> bool:
    return (
        not force
        and clip.is_file()
        and clip.stat().st_size > 0
        and entry.get("fingerprint") == fp
    )


# --------------------------------------------------------------------------
# effects
# --------------------------------------------------------------------------


def preflight(script: Script, opts: RenderOptions) -> int | None:
    """None to proceed; an exit code to return instead.

    Both backends answer the same question - "will this actually render?" - but
    from different evidence: sdcpp asks the container which compute backends it
    can see, comfy asks the server whether it holds the named models.
    """
    if script.backend == "comfy":
        # Pure and offline, so it runs on a dry run too - a mismatched checkpoint
        # is exactly what a dry run is for catching. It warns, never blocks.
        for warning in comfy_backend.pairing_warnings(script):
            print(f"warning: {warning}", file=sys.stderr)

    if opts.dry_run:
        return None  # a dry run must stay free, and must work with nothing running

    if script.backend == "comfy":
        ok, message = comfy_backend.check_server(script)
        hint = "  Start ComfyUI, or fix the model names under comfy:."
    else:
        if opts.allow_cpu:
            return None
        ok, message = check_gpu(script)
        hint = "  Pass --allow-cpu to render anyway."
    print(f"{'' if ok else 'error: '}{message}", file=sys.stdout if ok else sys.stderr)
    if ok:
        return None
    print(hint, file=sys.stderr)
    return 2


def build_job(scene: Scene, script: Script, prev_frame: Path | None, *,
              dry_run: bool) -> SceneJob:
    """Resolve refs and ref videos, then fingerprint.

    A dry run only needs the frame directory *path* to print an accurate command,
    so it must not transcode anything.
    """
    if script.backend == "comfy":
        return build_comfy_job(scene, script, prev_frame, dry_run=dry_run)

    refs, warning = resolve_refs(scene, prev_frame, dry_run=dry_run)
    if warning:
        print(warning, file=sys.stderr)

    layout = script.layout
    ref_video_dirs = []
    for src in scene.ref_videos:
        frame_dir = layout.refvideo_dir(src, scene.settings.width, scene.settings.height)
        if not dry_run:
            prepare_ref_video(src, frame_dir, scene.settings.width, scene.settings.height)
        ref_video_dirs.append(frame_dir)

    return SceneJob(
        scene=scene,
        refs=refs,
        ref_video_dirs=tuple(ref_video_dirs),
        fingerprint=fingerprint(scene, script, refs, ref_video_dirs),
    )


def chain_artefact(scene: Scene, script: Script, *, refresh: bool = False,
                   dry_run: bool = False) -> Path | None:
    """What this scene hands to the next one to chain from.

    sdcpp chains on a still, so the artefact is an extracted PNG. comfy chains on
    a segment, so it is the clip itself and the tail is cut later. A dry run
    returns the clip path even when nothing has been rendered, so the plan can
    show what *would* be anchored.
    """
    if script.backend == "comfy":
        clip = script.layout.clip(scene.slug)
        return clip if (dry_run or clip.is_file()) else None
    return chain_frame(scene, script, refresh=refresh)


def build_comfy_job(scene: Scene, script: Script, prev_clip: Path | None, *,
                    dry_run: bool) -> SceneJob:
    """Resolve the anchor clip and build the graph, then fingerprint it.

    `prev_clip` is the previous scene's rendered clip, not a still: the comfy
    backend chains on a segment, so the artefact handed forward is the clip
    itself and the tail is cut from it here.
    """
    name, host, frames = comfy_backend.prepare_chain(scene, script, prev_clip, dry_run=dry_run)
    placed = comfy_backend.prepare_refs(script, scene.ref_images, dry_run=dry_run)
    ref_names = [n for n, _ in placed]
    graph = comfy_backend.build_graph(scene, script, overlap_clip=name, refs=ref_names)
    # Every host file the graph reads, content-hashed: the anchor clip and each
    # reference image. Swapping an anchor must invalidate the scenes using it.
    inputs = ([host] if host is not None else []) + [p for _, p in placed]
    return SceneJob(
        scene=scene,
        refs=tuple(scene.ref_images),
        ref_video_dirs=(),
        fingerprint=comfy_backend.fingerprint(
            scene, script, inputs, overlap_clip=name, refs=ref_names),
        graph=graph,
        overlap_clip=name,
        overlap_frames=frames,
    )


def chain_frame(scene: Scene, script: Script, *, refresh: bool = False) -> Path | None:
    """Ensure the scene's last frame exists on disk, and return it if it does."""
    layout = script.layout
    frame = layout.frame(scene.slug)
    clip = layout.clip(scene.slug)
    if clip.is_file() and (refresh or not frame.is_file()):
        extract_last_frame(clip, frame)
    return frame if frame.is_file() else None


def attempt_scene(job: SceneJob, script: Script, *, attempts: int,
                  total: int) -> tuple[bool, float, int]:
    """The retry loop for one scene. Returns (ok, elapsed, last exit code).

    Propagates KeyboardInterrupt: `run_scene` has already killed the container by
    then, and the caller owns discarding the partial clip and saving state.
    """
    scene = job.scene
    layout = script.layout
    clip = layout.clip(scene.slug)
    is_comfy = script.backend == "comfy"
    argv = None if is_comfy else docker_argv(scene, script, job.refs, job.ref_video_dirs)

    elapsed = 0.0
    code = -1
    for attempt in range(1, attempts + 1):
        label = f"attempt {attempt}/{attempts}" if attempts > 1 else "rendering"
        print(f"\n=== [{scene.index}/{total}] {scene.slug}: {label} ===")
        print(f"    {scene.full_prompt()[:100]}")
        if job.overlap_frames:
            print(f"    anchored on {job.overlap_frames} frames of the previous scene")
        if clip.exists():
            clip.unlink()

        log_path = layout.log(scene.slug, attempt)
        start = time.time()
        if is_comfy:
            assert job.graph is not None
            code = comfy_backend.run_scene(job.graph, script, clip, log_path,
                                           label=f"[{scene.index}/{total}] ")
        else:
            code = run_scene(argv, log_path, container_name(scene, script))
        elapsed = time.time() - start

        if code == 0 and clip.is_file() and clip.stat().st_size > 0:
            return True, elapsed, code

        print(f"  ! {scene.slug}: attempt {attempt} failed (exit {code}); see {log_path}",
              file=sys.stderr)
        if attempt < attempts:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return False, elapsed, code


def record_success(state: dict, job: SceneJob, clip: Path, probe: dict,
                   elapsed: float) -> None:
    entry = {
        "fingerprint": job.fingerprint,
        "state": "rendered",
        "clip": str(clip),
        "probe": probe,
        "elapsed": elapsed,
        "rendered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if job.overlap_frames:
        # Recorded because assembly has to trim exactly what was anchored here,
        # even if the script's overlap_frames is edited afterwards.
        entry["overlap_frames"] = job.overlap_frames
    state["scenes"][job.scene.id] = entry


def finish(script: Script, opts: RenderOptions, results: list[dict], failed: bool) -> int:
    layout = script.layout
    movie = None
    if not opts.no_assemble:
        # Assembly uses every clip that exists, not just the selected scenes, so
        # `--only 2` still re-stitches the whole movie.
        renderable = [
            s for s in script.scenes
            if layout.clip(s.slug).is_file() and layout.clip(s.slug).stat().st_size > 0
        ]
        if renderable:
            try:
                movie = assemble(script, renderable)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                print(f"\nassembly failed: {exc}", file=sys.stderr)
                failed = True
        else:
            print("\nnothing to assemble - no clips were produced", file=sys.stderr)

    print_summary(results, layout.logs_dir, movie)
    return 1 if failed else 0


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def render(script: Script, opts: RenderOptions) -> int:
    layout = script.layout
    layout.ensure_dirs()

    code = preflight(script, opts)
    if code is not None:
        return code

    state = load_state(layout.state_file)
    selected = select_scenes(script.scenes, opts)
    total = len(script.scenes)

    results: list[dict] = []
    prev_chain: Path | None = None
    failed = False

    for scene in script.scenes:
        # Scenes outside the selection still contribute their chain artefact, so
        # a filtered run does not break continuity downstream.
        if scene.index not in selected:
            prev_chain = chain_artefact(scene, script, dry_run=opts.dry_run)
            continue

        job = build_job(scene, script, prev_chain, dry_run=opts.dry_run)
        clip = layout.clip(scene.slug)
        up_to_date = is_up_to_date(
            clip, scene_entry(state, scene.id), job.fingerprint, force=opts.force
        )

        if opts.dry_run:
            print(f"\n=== [{scene.index}/{total}] {scene.slug} "
                  f"({'skip' if up_to_date else 'render'}) ===")
            if job.graph is not None:
                if job.overlap_frames:
                    print(f"# anchored on {job.overlap_frames} frames of "
                          f"{job.overlap_clip}")
                print(comfy_backend.format_graph(job.graph))
            else:
                print(format_argv(docker_argv(scene, script, job.refs, job.ref_video_dirs)))
            results.append({"scene": scene, "state": "dry-run", "probe": {}, "elapsed": None})
            prev_chain = chain_artefact(scene, script, dry_run=True)
            continue

        if up_to_date:
            print(f"\n=== [{scene.index}/{total}] {scene.slug}: up to date, skipping")
            entry = scene_entry(state, scene.id)
            results.append({
                "scene": scene,
                "state": "skipped",
                "probe": entry.get("probe") or probe_clip(clip),
                "elapsed": entry.get("elapsed"),
            })
            prev_chain = chain_artefact(scene, script)
            continue

        try:
            ok, elapsed, _ = attempt_scene(job, script, attempts=opts.attempts, total=total)
        except KeyboardInterrupt:
            if clip.exists():
                clip.unlink()  # discard the partial clip
            print("\ninterrupted - partial clip discarded", file=sys.stderr)
            save_state(layout.state_file, state)
            return 130

        if not ok:
            failed = True
            state["scenes"][scene.id] = {"fingerprint": None, "state": "failed"}
            save_state(layout.state_file, state)
            results.append({"scene": scene, "state": "failed", "probe": {}, "elapsed": elapsed})
            if opts.halt_on_failure:
                print("\nhalting: --halt-on-failure is set", file=sys.stderr)
                break
            prev_chain = None
            continue

        probe = probe_clip(clip)
        record_success(state, job, clip, probe, elapsed)
        save_state(layout.state_file, state)
        results.append({"scene": scene, "state": "rendered", "probe": probe, "elapsed": elapsed})
        print(f"  -> {clip.name}  {probe.get('frames')} frames  "
              f"{fmt_duration(probe.get('duration'))}  (took {fmt_duration(elapsed)})")
        prev_chain = chain_artefact(scene, script, refresh=True)

    if opts.dry_run:
        return 0

    return finish(script, opts, results, failed)
