"""Building and running the per-scene container.

`fingerprint` lives here, directly below `sd_args`, because their contract is
"the hash is exactly this argv plus reference content". They have to be edited
together: adding a flag to `sd_args` invalidates every stored fingerprint, and
that consequence should be visible at the edit site.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import Script
from .layout import CONTAINER_ASSETS, CONTAINER_MODELS, CONTAINER_OUT, slugify

GPU_CHECK_TIMEOUT = 300


# --------------------------------------------------------------------------
# the sd-cli argv
# --------------------------------------------------------------------------


def sd_args(scene, script: Script, refs: Sequence[Path],
            ref_video_dirs: Sequence[Path] | None = None) -> list[str]:
    """The sd-cli portion of the command - this is what the fingerprint covers.

    Argument order is load-bearing twice: it is the CLI contract, and it is the
    byte sequence the hash is built from.
    """
    settings = scene.settings
    layout = script.layout
    ref_video_dirs = ref_video_dirs or []

    args = ["--mode", "vid_gen"]
    for key, flag in (
        ("diffusion_model", "--diffusion-model"),
        ("llm", "--llm"),
        ("vae", "--vae"),
        ("audio_vae", "--audio-vae"),
    ):
        if key in script.model_files:
            args += [flag, layout.to_container(script.model_files[key])]

    args += [
        "-W", str(settings.width),
        "-H", str(settings.height),
        "--fps", str(settings.fps),
        "--video-frames", str(settings.video_frames),
        "--cfg-scale", str(settings.cfg_scale),
        "-s", str(settings.seed),
    ]
    if settings.steps is not None:
        args += ["--steps", str(settings.steps)]
    if settings.sampling_method:
        args += ["--sampling-method", str(settings.sampling_method)]

    args += ["--prompt", scene.full_prompt()]
    if settings.negative_prompt.strip():
        args += ["--negative-prompt", settings.negative_prompt]

    # Ref order is the <Picture N> contract for H3 prompts: chained frame first
    # (inserted at index 0 by the render loop), then anchors, then the scene's own.
    for ref in refs:
        args += ["-r", layout.to_container(ref)]
    for vdir in ref_video_dirs:
        args += ["--ref-video", layout.to_container(vdir)]
    if len(refs) + len(ref_video_dirs) > 1:
        args.append("--increase-ref-index")

    args += [str(a) for a in settings.extra_args]
    args += ["--output", layout.to_container(layout.clip(scene.slug))]
    return args


# --------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------


def ref_token(ref: Path) -> str:
    """Content digest for a reference image, or its path when it does not exist.

    Content hashing is what makes chaining correct: when scene N re-renders,
    scene N+1's chained frame changes on disk under an unchanged path, and only
    hashing the bytes invalidates N+1 too.

    The missing-file branch is reachable only in a dry run, where the previous
    scene's frame has not been extracted yet.
    """
    if ref.is_file():
        return hashlib.sha256(ref.read_bytes()).hexdigest()
    return str(ref)


def refvideo_token(vdir: Path) -> str:
    stamp = vdir / ".source"
    return stamp.read_text() if stamp.is_file() else str(vdir)


def digest(args: Sequence[str], refvideo_tokens: Sequence[str],
           ref_tokens: Sequence[str]) -> str:
    """Pure hash over the argv and the reference tokens.

    Byte order is fixed: every arg followed by a NUL, then the ref-video tokens,
    then the ref tokens. Changing it re-renders every scene of every existing run.
    """
    h = hashlib.sha256()
    for arg in args:
        h.update(arg.encode())
        h.update(b"\0")
    for token in refvideo_tokens:
        h.update(b"refvideo:")
        h.update(token.encode())
    for token in ref_tokens:
        h.update(b"ref:")
        h.update(token.encode())
    return h.hexdigest()


def fingerprint(scene, script: Script, refs: Sequence[Path],
                ref_video_dirs: Sequence[Path] | None = None) -> str:
    ref_video_dirs = ref_video_dirs or []
    return digest(
        sd_args(scene, script, refs, ref_video_dirs),
        [refvideo_token(v) for v in ref_video_dirs],
        [ref_token(r) for r in refs],
    )


# --------------------------------------------------------------------------
# the docker argv
# --------------------------------------------------------------------------


def device_gids(devices: Sequence[str]) -> list[int]:
    """Groups owning the GPU device nodes (typically render and video).

    A non-root container must join these or Vulkan enumerates no devices and
    ggml silently falls back to CPU - a very slow, very quiet failure.
    """
    gids: list[int] = []
    for dev in devices:
        try:
            gid = os.stat(dev).st_gid
        except OSError:
            continue
        if gid not in gids:
            gids.append(gid)
    return gids


def docker_base_argv(script: Script) -> list[str]:
    """docker flags shared by the render and the GPU preflight check."""
    argv = ["docker", "run", "--rm"]
    devices = script.docker.devices
    if script.docker.run_as_current_user:
        argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        for gid in device_gids(devices):
            argv += ["--group-add", str(gid)]
    for dev in devices:
        argv += ["--device", dev]
    return argv


def container_name(scene, script: Script) -> str:
    return f"moviemakr-{slugify(script.name)}-{scene.slug}"


def docker_argv(scene, script: Script, refs: Sequence[Path],
                ref_video_dirs: Sequence[Path] | None = None) -> list[str]:
    layout = script.layout
    argv = docker_base_argv(script)
    # Naming the container makes it addressable: interrupts can stop it, and
    # `docker ps` shows which scene is rendering.
    argv += ["--name", container_name(scene, script)]
    # Allocate a TTY so sd-cli line-buffers and progress streams live instead of
    # arriving in one block at the end.
    argv += ["-t"]
    argv += [
        "-v", f"{layout.model_root}:{CONTAINER_MODELS}:ro",
        "-v", f"{layout.assets_dir}:{CONTAINER_ASSETS}:ro",
        "-v", f"{layout.run_dir}:{CONTAINER_OUT}",
        script.docker.image,
    ]
    return argv + sd_args(scene, script, refs, ref_video_dirs)


def format_argv(argv: Sequence[str]) -> str:
    """Shell-quoted command line, for --dry-run and the log header."""
    return " ".join(shlex.quote(a) for a in argv)


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def kill_container(name: str) -> None:
    """Stop a container by name, ignoring 'no such container'.

    `docker run` is a client: killing it leaves the container running under the
    daemon. Every interrupt and retry path has to reach for the daemon directly.
    """
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)


def check_gpu(script: Script) -> tuple[bool, str]:
    """Ask the container which backends it can see, before committing to a long run.

    ggml falls back to CPU without complaint when Vulkan finds no device, which
    costs hours before it becomes obvious. Catch it in seconds instead.
    """
    argv = docker_base_argv(script) + [script.docker.image, "--list-devices"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=GPU_CHECK_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError) as exc:
        # Docker being slow or absent is not evidence of a CPU fallback.
        return True, f"could not verify GPU access ({exc}); continuing"

    devices = [
        line.split("\t")[0]
        for line in proc.stdout.splitlines()
        if "\t" in line and not line.startswith("load_backend")
    ]
    accel = [d for d in devices if d.strip().upper() != "CPU"]
    if accel:
        return True, f"GPU backend: {', '.join(accel)}"

    if proc.returncode != 0:
        # The container never got far enough to enumerate anything; report what
        # actually went wrong rather than blaming the GPU.
        output = (proc.stderr or proc.stdout).strip()
        tail = output.splitlines()[-1] if output else "no output"
        return False, (
            f"GPU preflight failed: `docker run ... --list-devices` exited "
            f"{proc.returncode}\n  {tail}"
        )

    return False, (
        "no GPU backend visible inside the container - ggml would fall back to "
        "CPU and take many hours per scene.\n"
        "  Check that docker.devices matches your hardware, or set "
        "docker.run_as_current_user: false to run the container as root."
    )


def run_scene(argv: Sequence[str], log_path: Path, name: str) -> int:
    """Stream the container's output to both the terminal and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kill_container(name)  # clear a stale container left by an earlier crash
    with log_path.open("w") as log:
        log.write(format_argv(argv) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            list(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            return proc.wait()
        except KeyboardInterrupt:
            # Stop the container first; the client exits on its own once the
            # container is gone. Terminating only the client orphans a job that
            # then burns CPU and RAM indefinitely.
            print(f"\nstopping container {name} ...", file=sys.stderr)
            kill_container(name)
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
