"""`sd_args` is both the sd-cli contract and the fingerprint input.

Argument *order* is load-bearing twice over: it is what the model receives, and
it is the byte sequence the hash is built from. These goldens were captured from
the pre-refactor single-file `moviemakr.py`.
"""

from __future__ import annotations

import moviemakr as M

MINIMAL_ARGV = [
    "--mode", "vid_gen",
    "--diffusion-model", "/models/h3/diffusion.gguf",
    "--llm", "/models/h3/llm.gguf",
    "--vae", "/models/h3/video_vae.safetensors",
    "--audio-vae", "/models/h3/audio_vae.safetensors",
    "-W", "540",
    "-H", "960",
    "--fps", "24",
    "--video-frames", "120",
    "--cfg-scale", "1.0",
    "-s", "42",
    "--prompt", "A test scene.",
    "--output", "/out/scenes/001-opening.webm",
]

FULL_ARGV = [
    "--mode", "vid_gen",
    "--diffusion-model", "/models/h3/diffusion.gguf",
    "--llm", "/models/h3/llm.gguf",
    "--vae", "/models/h3/video_vae.safetensors",
    "--audio-vae", "/models/h3/audio_vae.safetensors",
    "-W", "540",
    "-H", "960",
    "--fps", "24",
    "--video-frames", "86",
    "--cfg-scale", "1.0",
    "-s", "1234",
    "--steps", "30",
    "--sampling-method", "euler",
    "--prompt", "A test scene. cinematic.",
    "--negative-prompt", "blurry",
    "-r", "/assets/anchor.png",
    "-r", "/assets/second.png",
    "--increase-ref-index",
    "--mmap", "--diffusion-fa",
    "--output", "/out/scenes/001-opening.webm",
]

FULL_OVERRIDES = {
    "defaults": {
        "steps": 30,
        "sampling_method": "euler",
        "negative_prompt": "blurry",
        "style_suffix": "cinematic.",
        "extra_args": ["--mmap", "--diffusion-fa"],
    },
    "continuity": {"anchors": ["anchor.png", "second.png"]},
    "scenes": [
        {"id": "opening", "prompt": "A test scene", "seed": 1234, "video_frames": 86}
    ],
}


def argv(script, scene=None, refs=None, ref_video_dirs=None):
    scene = scene or script.scenes[0]
    refs = list(scene.ref_images) if refs is None else refs
    return M.sd_args(scene, script, refs, ref_video_dirs or [])


def test_golden_minimal(load):
    assert argv(load()) == MINIMAL_ARGV


def test_golden_full(load, make_asset):
    make_asset("anchor.png")
    make_asset("second.png", b"second-image-bytes")
    assert argv(load(FULL_OVERRIDES)) == FULL_ARGV


def test_audio_vae_omitted_when_absent(load, DELETE):
    args = argv(load({"model": {"audio_vae": DELETE}}))
    assert "--audio-vae" not in args
    # The other three model flags stay, in order.
    assert args[:8] == MINIMAL_ARGV[:8]


def test_optional_flags_omitted_by_default(load):
    args = argv(load())
    assert "--steps" not in args
    assert "--sampling-method" not in args
    assert "--negative-prompt" not in args
    assert "--increase-ref-index" not in args


def test_whitespace_negative_prompt_is_omitted(load):
    assert "--negative-prompt" not in argv(load({"defaults": {"negative_prompt": "   "}}))


def test_increase_ref_index_threshold(load, make_asset):
    make_asset("a.png", b"aaa")
    make_asset("b.png", b"bbb")

    one = argv(load({"continuity": {"anchors": ["a.png"]}}))
    assert "--increase-ref-index" not in one

    two = argv(load({"continuity": {"anchors": ["a.png", "b.png"]}}))
    assert "--increase-ref-index" in two


def test_ref_video_counts_toward_increase_ref_index(load, make_asset, tmp_path):
    make_asset("a.png", b"aaa")
    script = load({"continuity": {"anchors": ["a.png"]}})
    vdir = script.run_dir / "refvideos" / "clip-540x960"
    args = argv(script, ref_video_dirs=[vdir])
    assert "--increase-ref-index" in args


def test_ref_videos_come_after_ref_images(load, make_asset):
    make_asset("a.png", b"aaa")
    script = load({"continuity": {"anchors": ["a.png"]}})
    vdir = script.run_dir / "refvideos" / "clip-540x960"
    args = argv(script, ref_video_dirs=[vdir])
    assert args.index("-r") < args.index("--ref-video")


def test_extra_args_immediately_precede_output(load):
    args = argv(load({"defaults": {"extra_args": ["--mmap", "--backend", "te=cpu"]}}))
    assert args[-5:] == ["--mmap", "--backend", "te=cpu", "--output",
                         "/out/scenes/001-opening.webm"]


def test_output_is_always_webm_regardless_of_container(load):
    """sd-cli writes WebM; only the assembled movie follows output.container."""
    for container in ("mp4", "webm"):
        args = argv(load({"output": {"container": container}}))
        assert args[-1] == "/out/scenes/001-opening.webm"
