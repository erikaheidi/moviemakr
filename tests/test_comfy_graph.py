"""Golden API graph for the ComfyUI backend.

This is the anchor for `comfy`, the way test_sd_args.py is for `sdcpp`. A scene
is skipped when its stored fingerprint matches, so if the graph drifts, every
scene of every existing comfy run silently re-renders at ~14 minutes apiece.
Treat a golden failure as a real regression, not a value to update.
"""

from __future__ import annotations

import pytest

from moviemakr.backends.comfy import (
    align_down,
    align_up,
    build_graph,
    canonical,
    fingerprint,
    format_graph,
)
from moviemakr.errors import ConfigError

MINIMAL_GRAPH = {
    "unet": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "minimax_h3_fl2va_pruned_bf16.safetensors",
            "weight_dtype": "default",
        },
    },
    "clip": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_bf16.safetensors",
            "type": "minimax",
            "device": "cpu",
        },
    },
    "vae": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
    },
    "avae": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
    },
    "shift": {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {"model": ["unet", 0], "shift_video": 12.0, "shift_audio": 3.0},
    },
    "cond": {
        "class_type": "MiniMaxH3ImageToVideo",
        "inputs": {
            "clip": ["clip", 0],
            "vae": ["vae", 0],
            "prompt": "A test scene.",
            "width": 540,
            "height": 960,
            "length": 124,
        },
    },
    "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
    "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
    "sigmas": {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": ["shift", 0],
            "scheduler": "simple",
            "steps": 8,
            "denoise": 1.0,
        },
    },
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
    "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
    "deca": {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]},
    },
    "video": {
        "class_type": "CreateVideo",
        "inputs": {"images": ["dec", 0], "audio": ["deca", 0], "fps": 24.0},
    },
    "save": {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["video", 0],
            "filename_prefix": "moviemakr/test-movie/001-opening",
            "format": "auto",
            "format.codec": "auto",
            "codec": "auto",
        },
    },
}

MINIMAL_FP = "48dba1f2dd3ebd2a0cbf6eea03ccce2afafef8cb0dd1a0897bc5adb8bd297c84"


@pytest.fixture
def scene_and_script(load_comfy):
    script = load_comfy()
    return script.scenes[0], script


# --- the golden -----------------------------------------------------------


def test_minimal_graph_is_golden(scene_and_script):
    scene, script = scene_and_script
    assert build_graph(scene, script) == MINIMAL_GRAPH


def test_minimal_fingerprint_is_golden(scene_and_script):
    scene, script = scene_and_script
    assert fingerprint(scene, script) == MINIMAL_FP


# --- invariants that the golden alone would not explain -------------------


def test_text_encoder_stays_on_the_cpu(scene_and_script):
    """The 48 GiB encoder on the GPU faults an APU mid-sample. Non-negotiable."""
    scene, script = scene_and_script
    assert build_graph(scene, script)["clip"]["inputs"]["device"] == "cpu"


def test_audio_is_decoded_and_muxed(scene_and_script):
    """Video and audio come out of one joint latent, split by two decoders."""
    scene, script = scene_and_script
    graph = build_graph(scene, script)
    assert graph["dec"]["inputs"]["samples"] == graph["deca"]["inputs"]["samples"]
    assert graph["deca"]["inputs"]["vae"] == ["avae", 0]
    assert graph["video"]["inputs"]["audio"] == ["deca", 0]


def test_sampling_is_guided_not_cfg(scene_and_script):
    """H3 has no negative conditioning at CFG 1, so KSampler must not appear."""
    scene, script = scene_and_script
    classes = {n["class_type"] for n in build_graph(scene, script).values()}
    assert "KSampler" not in classes
    assert {"BasicGuider", "SamplerCustomAdvanced"} <= classes


def test_no_lora_means_the_unet_feeds_the_shift(scene_and_script):
    scene, script = scene_and_script
    graph = build_graph(scene, script)
    assert "lora" not in graph
    assert graph["shift"]["inputs"]["model"] == ["unet", 0]


def test_a_lora_is_spliced_between_unet_and_shift(load_comfy):
    script = load_comfy({"comfy": {"lora": "turbo.safetensors", "lora_strength": 0.8}})
    graph = build_graph(script.scenes[0], script)
    assert graph["lora"]["inputs"]["model"] == ["unet", 0]
    assert graph["lora"]["inputs"]["strength_model"] == 0.8
    assert graph["shift"]["inputs"]["model"] == ["lora", 0]


def test_a_first_frame_adds_a_loader_and_wires_it(load_comfy):
    script = load_comfy()
    graph = build_graph(script.scenes[0], script, first_frame="moviemakr/x/prev.png")
    assert graph["first"] == {
        "class_type": "LoadImage",
        "inputs": {"image": "moviemakr/x/prev.png"},
    }
    assert graph["cond"]["inputs"]["first_frame"] == ["first", 0]


def test_scene_steps_override_the_comfy_default(load_comfy):
    script = load_comfy({"scenes": [{"id": "opening", "prompt": "A test scene.", "steps": 20}]})
    assert build_graph(script.scenes[0], script)["sigmas"]["inputs"]["steps"] == 20


# --- the frame grid -------------------------------------------------------


@pytest.mark.parametrize("given,expected", [
    (1, 5), (4, 5), (5, 5), (6, 22), (22, 22), (23, 39), (39, 39),
    (100, 107), (120, 124), (124, 124), (125, 141),
])
def test_align_up(given, expected):
    assert align_up(given) == expected


@pytest.mark.parametrize("given,expected", [
    (0, 0), (4, 0), (5, 5), (21, 5), (22, 22), (38, 22), (39, 39), (56, 56),
])
def test_align_down(given, expected):
    """Guides snap down; the wrong direction crops the anchor silently."""
    assert align_down(given) == expected


def test_every_aligned_length_is_on_the_grid():
    for n in range(1, 200):
        assert align_up(n) % 17 == 5
        assert align_down(n) == 0 or align_down(n) % 17 == 5


def test_video_frames_is_snapped_up_in_the_graph(load_comfy):
    script = load_comfy({"defaults": {"video_frames": 100}})
    assert build_graph(script.scenes[0], script)["cond"]["inputs"]["length"] == 107


# --- fingerprint behaviour ------------------------------------------------


def test_output_location_is_not_hashed(scene_and_script):
    """Moving the workspace must not invalidate a stored render."""
    scene, script = scene_and_script
    graph = build_graph(scene, script)
    moved = {**graph, "save": {**graph["save"], "inputs": {
        **graph["save"]["inputs"], "filename_prefix": "somewhere/else"}}}
    assert canonical(graph) == canonical(moved)


def test_the_prompt_is_hashed(load_comfy):
    a = load_comfy()
    b = load_comfy({"scenes": [{"id": "opening", "prompt": "A different scene."}]})
    assert fingerprint(a.scenes[0], a) != fingerprint(b.scenes[0], b)


def test_the_model_names_are_hashed(load_comfy):
    a = load_comfy()
    b = load_comfy({"comfy": {"diffusion_model": "minimax_h3_ref2va_pruned_bf16.safetensors"}})
    assert fingerprint(a.scenes[0], a) != fingerprint(b.scenes[0], b)


def test_input_content_is_hashed_not_its_name(load_comfy, tmp_path):
    """A chained frame changes on disk under an unchanged name."""
    script = load_comfy()
    frame = tmp_path / "prev.png"
    frame.write_bytes(b"one")
    first = fingerprint(script.scenes[0], script, [frame], first_frame="prev.png")
    frame.write_bytes(b"two")
    second = fingerprint(script.scenes[0], script, [frame], first_frame="prev.png")
    assert first != second


def test_a_missing_input_falls_back_to_its_path(load_comfy, tmp_path):
    """Reachable only in a dry run, before the previous scene has rendered."""
    script = load_comfy()
    fingerprint(script.scenes[0], script, [tmp_path / "absent.png"])


# --- dry-run rendering ----------------------------------------------------


def test_format_graph_round_trips(scene_and_script):
    import json

    scene, script = scene_and_script
    graph = build_graph(scene, script)
    assert json.loads(format_graph(graph)) == graph


# --- config guards --------------------------------------------------------


def test_comfy_requires_its_model_names(write_comfy, workspace, DELETE):
    from moviemakr.config import load_script

    with pytest.raises(ConfigError, match="comfy.text_encoder is required"):
        load_script(write_comfy({"comfy": {"text_encoder": DELETE}}), workspace)


def test_comfy_rejects_a_docker_block(write_comfy, workspace):
    from moviemakr.config import load_script

    with pytest.raises(ConfigError, match="not used by the comfy backend"):
        load_script(write_comfy({"docker": {"image": "x"}}), workspace)


def test_comfy_rejects_ref_images_rather_than_dropping_them(write_comfy, workspace, make_asset):
    """Silently ignoring anchors would lose a character across a whole script."""
    from moviemakr.config import load_script

    make_asset("anchor.png")
    script = write_comfy({"scenes": [
        {"id": "opening", "prompt": "A test scene.", "ref_images": ["anchor.png"]},
    ]})
    with pytest.raises(ConfigError, match="does not support ref_images"):
        load_script(script, workspace)


def test_comfy_has_no_model_root(load_comfy):
    """There is no per-scene container, so there is no /models mount."""
    assert load_comfy().layout.model_root is None
