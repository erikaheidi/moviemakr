"""The ComfyUI backend: building the API-format graph for one scene.

ComfyUI is a long-running server, not a per-scene container, so this backend
submits a graph instead of an argv. The graph is written directly rather than
loaded from one of ComfyUI's shipped `.json` templates: those are in UI format
(nodes/links/subgraphs) and only the browser can flatten them - there is no
converter on the Python side.

`fingerprint` lives directly below `build_graph` for the same reason it sits
below `sd_args` in the sdcpp backend: the hash is exactly this graph plus the
content of the images it references, so the two have to be edited together.

Node ids are stable strings rather than numbers. ComfyUI accepts any string, and
readable ids keep the golden graph in the tests diffable.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import signal
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Sequence
from pathlib import Path

from ..config import Script, align_canvas
from ..errors import ConfigError

# Polled rather than driven from the websocket: /history is enough to know when a
# prompt finished and whether it worked, and polling keeps the package's only
# dependency PyYAML. A render takes minutes, so a 2s interval costs nothing.
POLL_SECONDS = 2.0
HEARTBEAT_SECONDS = 30.0
HTTP_TIMEOUT = 30.0

# MiniMax H3 samples on a 17k+5 frame grid at 24 fps: 5, 22, 39, 56 ... A length
# off the grid is silently snapped by the node, so do it here where it is visible.
FRAME_GRID = 17
FRAME_BASE = 5
FPS = 24

# ComfyUI writes SaveVideo output under this prefix inside its own output dir.
OUTPUT_PREFIX = "moviemakr"


def align_up(frames: int) -> int:
    """Round a target length up onto the model's frame grid."""
    n = max(FRAME_BASE, int(frames))
    while n % FRAME_GRID != FRAME_BASE:
        n += 1
    return n


def align_down(frames: int) -> int:
    """Round a guide clip length *down* onto the grid.

    Guides snap down and targets snap up. Getting the direction wrong is the
    easiest silent bug here: a guide one frame too long is cropped by the node
    without a word, and the assembly trim then removes the wrong count.
    """
    n = int(frames)
    if n < FRAME_BASE:
        return 0
    while n % FRAME_GRID != FRAME_BASE:
        n -= 1
    return n


def duration_seconds(frames: int, fps: int = FPS) -> float:
    return frames / fps


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------


def output_prefix(script: Script, scene) -> str:
    """Where SaveVideo writes, relative to ComfyUI's output directory."""
    return f"{OUTPUT_PREFIX}/{script.layout.name_slug}/{scene.slug}"


def build_graph(scene, script: Script, *, first_frame: str | None = None,
                overlap_clip: str | None = None,
                refs: Sequence[str] = ()) -> dict[str, dict]:
    """The API-format prompt for one scene.

    `first_frame`, `overlap_clip` and `refs` are paths relative to ComfyUI's
    input directory, already placed there by the caller - the graph only names
    them.

    `first_frame` and `overlap_clip` are alternative ways to chain. `overlap_clip`
    anchors a whole tail segment of the previous scene, its soundtrack included,
    so motion and the soundscape continue across the seam; `first_frame` anchors a
    single still, which restarts both. When both are given the overlap wins.

    `refs` are out-of-timeline reference images - character sheets and anchors.
    They select a different conditioning node, and their order is the `<Picture N>`
    contract the prompt refers to. Refs and an overlap anchor compose: the model
    reads `minimax_refs` and `minimax_keyframes` independently and concatenates
    their latents, so a scene can hold a character *and* continue a movement.
    """
    comfy = script.comfy
    if comfy is None:
        raise ValueError("build_graph needs a script with a comfy: block")

    settings = scene.settings
    length = align_up(settings.video_frames)
    width = align_canvas(settings.width)
    height = align_canvas(settings.height)
    steps = settings.steps if settings.steps is not None else comfy.steps

    graph: dict[str, dict] = {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": comfy.diffusion_model, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": comfy.text_encoder,
                "type": "minimax",
                # The 48 GiB encoder must stay off the GPU: on a unified-memory
                # APU it otherwise competes with the diffusion model for the
                # same pool and the GPU faults mid-sample.
                "device": "cpu",
            },
        },
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": comfy.video_vae}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": comfy.audio_vae}},
    }

    model_src = ["unet", 0]
    if comfy.lora:
        graph["lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_src,
                "lora_name": comfy.lora,
                "strength_model": comfy.lora_strength,
            },
        }
        model_src = ["lora", 0]

    graph["shift"] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {
            "model": model_src,
            "shift_video": comfy.shift_video,
            "shift_audio": comfy.shift_audio,
        },
    }

    cond_inputs: dict = {
        "clip": ["clip", 0],
        "vae": ["vae", 0],
        "prompt": scene.full_prompt(),
        "width": width,
        "height": height,
        "length": length,
    }

    if refs:
        # ReferenceToVideo takes no first_frame/last_frame - in-timeline anchoring
        # is AddGuide's job, and the two compose.
        cond_inputs["audio_vae"] = ["avae", 0]
        cond_inputs["ref_image_size"] = comfy.ref_image_size
        # Autogrow inputs are ONE nested input, not flat ref_image_N keys. Flat
        # keys reach execute() as unexpected kwargs and raise there - not at
        # validation, which accepts them. The node iterates .values(), so this
        # order is the prompt's <Picture N> contract.
        ref_images: dict[str, list] = {}
        for i, name in enumerate(refs, start=1):
            node = f"ref{i}"
            graph[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
            ref_images[f"ref_image_{i}"] = [node, 0]
        cond_inputs["ref_images"] = ref_images
        graph["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond_inputs}
    else:
        if first_frame is not None and overlap_clip is None:
            graph["first"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
            cond_inputs["first_frame"] = ["first", 0]
        graph["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs}

    conditioning = ["cond", 0]
    if overlap_clip is not None:
        # One file carries both streams, so the frames and their soundtrack stay
        # aligned without re-syncing a frame dump against a separate wav.
        graph["tail"] = {"class_type": "LoadVideo", "inputs": {"file": overlap_clip}}
        graph["comp"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["tail", 0]}}
        graph["guide"] = {
            "class_type": "MiniMaxH3AddGuide",
            "inputs": {
                "positive": ["cond", 0],
                "latent": ["cond", 1],
                # Anchored at the head; the assembly trim removes it again.
                "frame_idx": 0,
                "vae": ["vae", 0],
                "audio_vae": ["avae", 0],
                "image": ["comp", 0],
                "audio": ["comp", 1],
            },
        }
        conditioning = ["guide", 0]

    graph.update({
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": settings.seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": comfy.sampler}},
        "sigmas": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["shift", 0],
                "scheduler": comfy.scheduler,
                "steps": steps,
                "denoise": 1.0,
            },
        },
        # H3 has no negative conditioning at CFG 1, so it is guided, not CFG-sampled.
        "guider": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["shift", 0], "conditioning": conditioning},
        },
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["sampler", 0],
                "sigmas": ["sigmas", 0],
                "latent_image": ["cond", 1],
            },
        },
        # One joint AV latent, split by two decoders: the video VAE takes the
        # first nested tensor, the audio VAE the last.
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "deca": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]},
        },
        "video": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["dec", 0], "audio": ["deca", 0], "fps": float(settings.fps)},
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["video", 0],
                "filename_prefix": output_prefix(script, scene),
                "format": "auto",
                "format.codec": "auto",
                "codec": "auto",
            },
        },
    })
    return graph


def format_graph(graph: dict[str, dict]) -> str:
    """Readable JSON, for --dry-run and the log header."""
    return json.dumps(graph, indent=2, sort_keys=True)


# --------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------

# Keys that say *where* the render happened, not *what* it produced. Moving the
# workspace or pointing at another ComfyUI must not invalidate a stored render,
# which is the same promise the sdcpp backend keeps by hashing container paths.
VOLATILE_INPUTS = frozenset({"filename_prefix"})


def canonical(graph: dict[str, dict]) -> str:
    """Stable text for the graph, with the volatile bits removed."""
    stripped = {
        node_id: {
            "class_type": node["class_type"],
            "inputs": {k: v for k, v in node["inputs"].items() if k not in VOLATILE_INPUTS},
        }
        for node_id, node in graph.items()
    }
    return json.dumps(stripped, sort_keys=True, separators=(",", ":"))


def ref_token(ref: Path) -> str:
    """Content digest for an input image, or its path when it does not exist.

    Content hashing is what makes chaining correct: when scene N re-renders,
    scene N+1's chained frame changes on disk under an unchanged name, and only
    hashing the bytes invalidates N+1 too. The missing-file branch is reachable
    only in a dry run, before the previous scene has been rendered.
    """
    if ref.is_file():
        return hashlib.sha256(ref.read_bytes()).hexdigest()
    return str(ref)


def digest(graph_text: str, ref_tokens: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(b"comfy:")
    h.update(graph_text.encode())
    for token in ref_tokens:
        h.update(b"\0ref:")
        h.update(token.encode())
    return h.hexdigest()


def fingerprint(scene, script: Script, inputs: Sequence[Path] = (), *,
                first_frame: str | None = None, overlap_clip: str | None = None,
                refs: Sequence[str] = ()) -> str:
    """Hash of the graph plus the content of every host file it reads.

    `inputs` are the host-side paths of those files. Their *content* is hashed,
    not their names, so re-rendering scene N invalidates scene N+1 whose anchor
    changed underneath an unchanged filename, and swapping an anchor image
    invalidates every scene that used it.
    """
    return digest(
        canonical(build_graph(scene, script, first_frame=first_frame,
                              overlap_clip=overlap_clip, refs=refs)),
        [ref_token(p) for p in inputs],
    )


# --------------------------------------------------------------------------
# talking to the server (pure parts)
# --------------------------------------------------------------------------


def prompt_payload(graph: dict, prompt_id: str, client_id: str) -> dict:
    return {"prompt": graph, "prompt_id": prompt_id, "client_id": client_id}


def history_entry(history: dict, prompt_id: str) -> dict | None:
    """ComfyUI keys /history by prompt id, even when asked for just one."""
    return (history or {}).get(prompt_id)


def is_finished(entry: dict | None) -> bool:
    """Has the prompt stopped, either way?

    Not `completed`: that flag means *succeeded*, and a prompt that raised has
    `{"status_str": "error", "completed": false}`. Waiting on `completed` alone
    hangs forever on exactly the failures the retry loop exists to handle - and
    the vanished-prompt guard does not help, because the entry *is* there.

    An entry only appears in /history once the prompt is done, so its presence
    with any status is itself terminal.
    """
    status = (entry or {}).get("status") or {}
    return bool(status.get("completed")) or status.get("status_str") is not None


def failure_reason(entry: dict | None) -> str | None:
    """None when the prompt succeeded, else a one-line reason."""
    status = (entry or {}).get("status") or {}
    if status.get("status_str") == "success":
        return None
    for kind, payload in reversed(status.get("messages") or []):
        if kind == "execution_error" and isinstance(payload, dict):
            node = payload.get("node_type") or payload.get("node_id") or "?"
            return f"{node}: {payload.get('exception_message') or payload.get('exception_type')}"
    return status.get("status_str") or "unknown failure"


def queue_has(queue: dict, prompt_id: str) -> bool:
    """Is the prompt still running or waiting?

    /history only gains an entry once a prompt *finishes*, so "absent from
    history" is the normal state while rendering. The queue is what separates
    still-working from vanished - without this check a prompt that the server
    dropped would be waited on forever.
    """
    for key in ("queue_running", "queue_pending"):
        for item in (queue or {}).get(key) or []:
            if len(item) > 1 and item[1] == prompt_id:
                return True
    return False


def saved_files(entry: dict | None) -> list[dict]:
    """Files the prompt wrote, newest node last.

    SaveVideo reports its mp4 under the key `images`, not `video` - reading
    `video` here would silently find nothing and look like a failed render.
    """
    files: list[dict] = []
    for node_output in ((entry or {}).get("outputs") or {}).values():
        for key in ("images", "gifs", "video", "audio"):
            for item in node_output.get(key) or []:
                if isinstance(item, dict) and item.get("filename"):
                    files.append(item)
    return files


def output_path(output_dir: Path, item: dict) -> Path:
    """Where a reported output lives on the host side of ComfyUI's output dir."""
    return Path(output_dir) / (item.get("subfolder") or "") / item["filename"]


def view_url(base_url: str, item: dict) -> str:
    from urllib.parse import urlencode

    query = urlencode({
        "filename": item["filename"],
        "subfolder": item.get("subfolder") or "",
        "type": item.get("type") or "output",
    })
    return f"{base_url}/view?{query}"


# --------------------------------------------------------------------------
# talking to the server (effects)
# --------------------------------------------------------------------------


def _request(url: str, data: bytes | None = None, *, timeout: float = HTTP_TIMEOUT):
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url: str, *, timeout: float = HTTP_TIMEOUT) -> dict:
    with _request(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def post_json(url: str, payload: dict, *, timeout: float = HTTP_TIMEOUT) -> dict:
    body = json.dumps(payload).encode()
    try:
        with _request(url, body, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        # A 400 here is a rejected graph, and its node_errors say exactly which
        # input the server disliked. Surfacing that beats "HTTP 400".
        detail = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(detail)
            detail = json.dumps(parsed.get("node_errors") or parsed, indent=2)
        except json.JSONDecodeError:
            pass
        raise ConfigError(f"ComfyUI rejected the prompt ({exc.code}):\n{detail}") from None


def pairing_warnings(script: Script) -> list[str]:
    """Warn when the checkpoint looks wrong for how the scenes are conditioned.

    MiniMax H3 ships as two task variants. `ref2va` reads reference images;
    `fl2va` reads first/last keyframes. Crossing them does not error - ComfyUI
    runs the graph and the model produces something, just not what the script
    describes - and the same is true of a turbo LoRA trained for the other one.
    That silence is the whole reason for guessing here.

    Heuristic on purpose, by filename, so it warns rather than refuses: the names
    follow Comfy-Org's convention, but a renamed file is not an error.
    """
    comfy = script.comfy
    if comfy is None:
        return []
    uses_refs = any(scene.ref_images for scene in script.scenes)
    model, lora = comfy.diffusion_model.lower(), (comfy.lora or "").lower()
    out = []

    if uses_refs and "fl2va" in model:
        out.append(
            f"{comfy.diffusion_model} looks like an fl2va (first/last frame) model, "
            f"but scenes carry reference images, which need a ref2va model."
        )
    if not uses_refs and "ref2va" in model:
        out.append(
            f"{comfy.diffusion_model} looks like a ref2va (reference) model, but no "
            f"scene has reference images; an fl2va model fits keyframe chaining."
        )
    if lora:
        if "ref2v" in lora and "fl2va" in model:
            out.append(f"{comfy.lora} is a ref2v LoRA on an fl2va model.")
        if "fl2v" in lora and "ref2va" in model:
            out.append(f"{comfy.lora} is an fl2v LoRA on a ref2va model.")

    # Settings that mean something to sd-cli and nothing here. A ported script
    # carries them unchanged, and a key that is silently inert is exactly what
    # `check_keys` exists to prevent - it just cannot see this, because the keys
    # are valid, only inert on this backend.
    inert: list[str] = []
    for scene in script.scenes:
        st = scene.settings
        if st.cfg_scale != 1.0 and "cfg_scale" not in inert:
            # H3 has no negative conditioning: ComfyUI drives it through
            # BasicGuider, which takes no CFG at all.
            inert.append("cfg_scale")
        if st.negative_prompt.strip() and "negative_prompt" not in inert:
            inert.append("negative_prompt")
        if st.sampling_method and "sampling_method" not in inert:
            inert.append("sampling_method")
        if st.extra_args and "extra_args" not in inert:
            inert.append("extra_args")
    if inert:
        out.append(
            f"these settings do nothing on the comfy backend and are ignored: "
            f"{', '.join(inert)}"
            + ("\n  sampling_method is sdcpp's; set comfy.sampler instead."
               if "sampling_method" in inert else "")
        )
    return out


AUTOGROW_TYPE = "COMFY_AUTOGROW_V3"


def validate_graph(graph: dict[str, dict], info: dict) -> list[str]:
    """Check a graph against the server's own schema, before spending GPU time.

    /prompt's own validation is permissive - it accepted flat `ref_image_1` keys
    that then raised inside execute() as unexpected kwargs, after the text encoder
    had loaded. So the shapes are checked here instead.

    Autogrow inputs are the subtle one: they are declared as a single input whose
    value is a *dict* of prefixed entries. A flat key is not an input at all.
    """
    problems: list[str] = []
    for node_id, node in graph.items():
        class_type = node.get("class_type")
        spec = info.get(class_type)
        if spec is None:
            problems.append(f"{node_id}: no such node type {class_type!r}")
            continue
        required = (spec.get("input") or {}).get("required") or {}
        optional = (spec.get("input") or {}).get("optional") or {}
        declared = {**required, **optional}
        given = node.get("inputs") or {}

        autogrow = {
            name for name, decl in declared.items()
            if isinstance(decl, list) and decl and decl[0] == AUTOGROW_TYPE
        }

        for key, value in given.items():
            # A dynamic combo serialises as extra dotted keys (`format.codec`).
            if key in declared or "." in key:
                if key in autogrow and not isinstance(value, dict):
                    problems.append(
                        f"{node_id} ({class_type}): {key} is an autogrow input and "
                        f"needs a dict of prefixed entries, got {type(value).__name__}"
                    )
                continue
            hint = ""
            for name in autogrow:
                prefix = ((declared[name][1] or {}).get("template") or {}).get("prefix")
                if prefix and key.startswith(prefix):
                    hint = f" - it belongs inside the {name!r} dict"
                    break
            problems.append(f"{node_id} ({class_type}): unknown input {key!r}{hint}")

        for key in set(required) - set(given):
            problems.append(f"{node_id} ({class_type}): missing required input {key!r}")
    return problems


def check_server(script: Script) -> tuple[bool, str]:
    """Preflight: the server is up, and it has the models the script names.

    Checked before committing to a long render, for the same reason sdcpp asks
    the container which backends it can see: the alternative is finding out after
    the queue has accepted the job.
    """
    comfy = script.comfy
    assert comfy is not None
    try:
        info = get_json(f"{comfy.url}/object_info", timeout=10)
    except (urllib.error.URLError, OSError) as exc:
        return False, f"cannot reach ComfyUI at {comfy.url}: {exc}"

    wanted = [
        ("UNETLoader", "unet_name", comfy.diffusion_model),
        ("CLIPLoader", "clip_name", comfy.text_encoder),
        ("VAELoader", "vae_name", comfy.video_vae),
        ("VAELoader", "vae_name", comfy.audio_vae),
    ]
    if comfy.lora:
        wanted.append(("LoraLoaderModelOnly", "lora_name", comfy.lora))

    missing = []
    for node, field, name in wanted:
        spec = ((info.get(node) or {}).get("input") or {}).get("required") or {}
        options = (spec.get(field) or [None])[0]
        if isinstance(options, list) and name not in options:
            missing.append(f"{name} (not offered by {node}.{field})")
    if missing:
        return False, "ComfyUI does not have:\n  " + "\n  ".join(missing)

    # Check the shape of a real graph too, not just the model names. A scene with
    # references exercises the most error-prone node, so prefer one of those.
    scene = next((s for s in script.scenes if s.ref_images), script.scenes[0])
    probe = build_graph(scene, script, refs=[f"probe{i}" for i in range(len(scene.ref_images))])
    if problems := validate_graph(probe, info):
        return False, "the graph does not match this server's schema:\n  " + "\n  ".join(problems)

    return True, f"ComfyUI ready at {comfy.url}"


def interrupt(url: str) -> None:
    """Stop the running prompt; the analogue of killing the sdcpp container."""
    # Best effort on the way out: the caller is already tearing the render down,
    # and a server that will not answer is not a reason to fail louder.
    with contextlib.suppress(Exception):
        post_json(f"{url}/interrupt", {})


def effective_overlap(scene, *, has_previous: bool) -> int:
    """Frames this scene will actually be anchored on, snapped to the grid.

    Snapped here, once, because the assembly trim must remove exactly what the
    node anchored - and the node snaps *down* without saying so.
    """
    if not has_previous or not scene.chain_from_previous:
        return 0
    return align_down(scene.settings.overlap_frames)


def ref_input_name(script: Script, ref: Path) -> str:
    """Path of a reference image, relative to ComfyUI's input directory.

    Named by content digest, not by the source filename: two anchors called
    `josy.jpg` in different asset folders would otherwise collide, and a replaced
    asset under an unchanged name would keep the stale copy.
    """
    digest = hashlib.sha256(ref.read_bytes()).hexdigest()[:16] if ref.is_file() else "missing"
    return f"{OUTPUT_PREFIX}/{script.layout.name_slug}/refs/{digest}{ref.suffix.lower()}"


def prepare_refs(script: Script, refs: Sequence[Path], *,
                 dry_run: bool = False) -> list[tuple[str, Path]]:
    """Copy reference images where ComfyUI can read them.

    LoadImage resolves names inside ComfyUI's own input directory and refuses to
    escape it, so an asset living in the workspace has to be copied in. Returns
    (name for the graph, host path for hashing) in `<Picture N>` order.
    """
    if not refs:
        return []
    comfy = script.comfy
    assert comfy is not None
    if comfy.input_dir is None:
        raise ConfigError(
            "comfy.input_dir is required for reference images: they have to be "
            "copied where ComfyUI can read them"
        )

    out: list[tuple[str, Path]] = []
    for ref in refs:
        name = ref_input_name(script, ref)
        dest = comfy.input_dir / name
        if not dry_run and ref.is_file():  # noqa: SIM102 - two separate concerns
            # Content-addressed, so an existing file of the same name is the same
            # image and re-copying it would only churn the disk.
            if not dest.is_file() or dest.stat().st_size != ref.stat().st_size:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ref, dest)
        out.append((name, ref))
    return out


def input_name(script: Script, prev_slug: str) -> str:
    """Path of a scene's anchor clip, relative to ComfyUI's input directory.

    A subfolder is fine even though LoadVideo's dropdown only lists top-level
    files: the node validates with `exists_annotated_filepath`, and declaring its
    own validator makes ComfyUI skip the combo-membership check.
    """
    return f"{OUTPUT_PREFIX}/{script.layout.name_slug}/{prev_slug}.tail.mp4"


def prepare_chain(scene, script: Script, prev_clip: Path | None, *,
                  dry_run: bool = False) -> tuple[str | None, Path | None, int]:
    """Cut the previous scene's tail into ComfyUI's input dir.

    Returns (name the graph will use, host path for hashing, frames anchored).
    A dry run computes the names but writes nothing, so it stays free.
    """
    from ..media import extract_tail_clip

    frames = effective_overlap(scene, has_previous=prev_clip is not None)
    if frames <= 0 or prev_clip is None:
        return None, None, 0

    comfy = script.comfy
    assert comfy is not None
    if comfy.input_dir is None:
        raise ConfigError(
            "comfy.input_dir is required to chain scenes: the anchor clip has to "
            "be written where ComfyUI can read it"
        )

    name = input_name(script, prev_clip.stem)
    host = comfy.input_dir / name
    if dry_run:
        return name, host, frames
    if not prev_clip.is_file():
        return None, None, 0
    if not extract_tail_clip(prev_clip, host, frames, scene.settings.fps):
        return None, None, 0
    return name, host, frames


def collect(entry: dict, dest: Path, comfy) -> bool:
    """Bring the rendered clip back to the workspace.

    A local ComfyUI shares its output directory with us, so the file is copied.
    Falling back to /view keeps a remote server workable, at the cost of pushing
    a few hundred MB through HTTP.
    """
    files = saved_files(entry)
    if not files:
        return False
    item = files[-1]

    dest.parent.mkdir(parents=True, exist_ok=True)
    if comfy.output_dir is not None:
        src = output_path(comfy.output_dir, item)
        if src.is_file():
            shutil.copyfile(src, dest)
            return dest.is_file() and dest.stat().st_size > 0

    try:
        with _request(view_url(comfy.url, item), timeout=HTTP_TIMEOUT * 10) as resp, \
                dest.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except (urllib.error.URLError, OSError):
        return False
    return dest.is_file() and dest.stat().st_size > 0


class _TermAsInterrupt:
    """Make SIGTERM take the KeyboardInterrupt path while a prompt is in flight.

    `docker run` is a client whose death the daemon notices; an HTTP POST is not.
    A killed moviemakr leaves ComfyUI sampling a prompt nobody will collect - it
    holds the GPU for the rest of the run and the next attempt queues behind it.
    Ctrl-C already routed through `interrupt`; SIGTERM did not, which is how a
    timed-out wrapper orphaned a 25-minute render.

    Restores the previous handler on the way out, and does nothing at all off the
    main thread, where `signal.signal` is not allowed.
    """

    def __enter__(self):
        self._prev = None
        try:
            self._prev = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._raise)
        except (ValueError, OSError):  # not the main thread
            self._prev = None
        return self

    @staticmethod
    def _raise(_signum, _frame):
        raise KeyboardInterrupt

    def __exit__(self, *_exc):
        if self._prev is not None:
            with contextlib.suppress(ValueError, OSError):  # not the main thread
                signal.signal(signal.SIGTERM, self._prev)
        return False


def run_scene(graph: dict, script: Script, dest: Path, log_path: Path, *,
              label: str = "") -> int:
    """Submit one graph and wait for it. Returns a shell-style exit code.

    Non-zero on any failure so the render loop's existing retry and backoff
    handle ComfyUI exactly as they handle a container that exited badly.
    """
    comfy = script.comfy
    assert comfy is not None
    prompt_id = str(uuid.uuid4())
    client_id = uuid.uuid4().hex

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _TermAsInterrupt(), log_path.open("w") as log:
        log.write(f"POST {comfy.url}/prompt  prompt_id={prompt_id}\n\n")
        log.write(format_graph(graph) + "\n\n")
        log.flush()

        try:
            post_json(f"{comfy.url}/prompt", prompt_payload(graph, prompt_id, client_id))
        except ConfigError as exc:
            print(f"  ! {exc}", file=sys.stderr)
            log.write(f"{exc}\n")
            return 2
        except (urllib.error.URLError, OSError) as exc:
            print(f"  ! cannot reach ComfyUI: {exc}", file=sys.stderr)
            log.write(f"unreachable: {exc}\n")
            return 3

        started = time.time()
        last_beat = started
        try:
            while True:
                time.sleep(POLL_SECONDS)
                try:
                    entry = history_entry(
                        get_json(f"{comfy.url}/history/{prompt_id}"), prompt_id)
                except (urllib.error.URLError, OSError) as exc:
                    # A blip while the server is busy is not a failed render.
                    log.write(f"poll error (continuing): {exc}\n")
                    continue

                if is_finished(entry):
                    break

                # Absent from history is normal while rendering, but absent from
                # the queue too means the server dropped it - stop rather than
                # poll forever.
                if entry is None:
                    try:
                        if not queue_has(get_json(f"{comfy.url}/queue"), prompt_id):
                            msg = f"prompt {prompt_id} is in neither the queue nor the history"
                            print(f"  ! {msg}", file=sys.stderr)
                            log.write(msg + "\n")
                            return 5
                    except (urllib.error.URLError, OSError):
                        pass  # a blip; the next poll will settle it

                now = time.time()
                if now - last_beat >= HEARTBEAT_SECONDS:
                    last_beat = now
                    line = f"  {label}still rendering ({fmt_elapsed(now - started)})"
                    print(line, flush=True)
                    log.write(line + "\n")
                    log.flush()
        except KeyboardInterrupt:
            print(f"\ninterrupting ComfyUI prompt {prompt_id} ...", file=sys.stderr)
            interrupt(comfy.url)
            raise

        reason = failure_reason(entry)
        if reason:
            print(f"  ! {reason}", file=sys.stderr)
            log.write(f"failed: {reason}\n")
            return 1

        if not collect(entry, dest, comfy):
            msg = "prompt succeeded but no output file could be collected"
            print(f"  ! {msg}", file=sys.stderr)
            log.write(msg + "\n" + json.dumps(entry.get("outputs") or {}, indent=2) + "\n")
            return 4

        log.write(f"collected -> {dest}\n")
        return 0


def fmt_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"
