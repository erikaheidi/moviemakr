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

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from ..config import Script

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


def build_graph(scene, script: Script, *, first_frame: str | None = None) -> dict[str, dict]:
    """The API-format prompt for one scene.

    `first_frame` is a path relative to ComfyUI's input directory, already
    placed there by the caller - the graph only ever names it.
    """
    comfy = script.comfy
    if comfy is None:
        raise ValueError("build_graph needs a script with a comfy: block")

    settings = scene.settings
    length = align_up(settings.video_frames)
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
        "width": settings.width,
        "height": settings.height,
        "length": length,
    }
    if first_frame is not None:
        graph["first"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        cond_inputs["first_frame"] = ["first", 0]
    graph["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs}

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
            "inputs": {"model": ["shift", 0], "conditioning": ["cond", 0]},
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
                first_frame: str | None = None) -> str:
    """Hash of the graph plus the content of every host file it reads."""
    return digest(
        canonical(build_graph(scene, script, first_frame=first_frame)),
        [ref_token(p) for p in inputs],
    )
