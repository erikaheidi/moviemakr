"""The anchor of the refactor.

A scene is skipped when its stored fingerprint matches the computed one. If the
hash ever drifts, every scene in every existing run silently re-renders - hours
of GPU time each. These goldens were captured from the pre-refactor single-file
`moviemakr.py` and must survive every step of the move.

The goldens are hermetic: `sd_args` maps every path through `to_container`
first, so the hash input holds only `/models`, `/assets`, `/out` paths, and
present refs contribute a content digest. The one exception is a *missing* ref,
which hashes its host path - see `test_missing_ref_hashes_its_path`.
"""

from __future__ import annotations

import pytest

import moviemakr as M

MINIMAL_FP = "9ff202e49772729a6a2575e93a71bd661037b11586a2be2e459580350fd26d46"
ONE_REF_FP = "030727e7665615680b22484d7813c2d85c56e0d0347ba6fffa9fba51aeb8f184"
FULL_FP = "65aba240c8652703c7bd18891c5b685da58ad3310db6c48e3ec0f77032257045"
ANCHOR_CHANGED_FP = "b4bf3a60a9a9069f4ed32db0a30bdc96f15b66027e3150cbbb75f74fc3999e99"

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


def fp(script, scene=None, refs=None, ref_video_dirs=None):
    scene = scene or script.scenes[0]
    refs = list(scene.ref_images) if refs is None else refs
    return M.fingerprint(scene, script, refs, ref_video_dirs or [])


# --------------------------------------------------------------------------
# Golden values
# --------------------------------------------------------------------------


def test_golden_minimal(load):
    assert fp(load()) == MINIMAL_FP


def test_golden_one_ref(load, make_asset):
    make_asset("anchor.png")
    assert fp(load({"continuity": {"anchors": ["anchor.png"]}})) == ONE_REF_FP


def test_golden_full(load, make_asset):
    make_asset("anchor.png")
    make_asset("second.png", b"second-image-bytes")
    assert fp(load(FULL_OVERRIDES)) == FULL_FP


def test_missing_ref_hashes_its_path(load):
    """A ref that does not exist yet hashes its path, not its content.

    This is why a chained scene always reports 'render' in a dry run: the
    previous scene's frame has not been extracted at that point.

    No golden value here, deliberately. The present-file branch hashes content
    and the argv only ever holds container paths, so those are reproducible
    anywhere - but this branch hashes `str(ref)`, the absolute *host* path, so
    the digest depends on where the project lives. It is only reachable in a
    dry run (a real render has already extracted the previous frame), so it
    never reaches state.json and cannot cause a spurious re-render.
    """
    script = load()
    missing = script.run_dir / "frames" / "000-prev.last.png"
    assert not missing.exists()

    with_missing = fp(script, refs=[missing])
    assert with_missing == fp(script, refs=[missing])  # stable
    assert with_missing != fp(script, refs=[])  # the ref still participates
    other = script.run_dir / "frames" / "000-other.last.png"
    assert with_missing != fp(script, refs=[other])


def test_golden_anchor_content_change(load, make_asset):
    make_asset("anchor.png", b"DIFFERENT-bytes")
    assert fp(load({"continuity": {"anchors": ["anchor.png"]}})) == ANCHOR_CHANGED_FP


# --------------------------------------------------------------------------
# Invalidation behaviour
# --------------------------------------------------------------------------


def test_stable_across_calls(load):
    script = load()
    assert fp(script) == fp(script)


def test_content_hashed_not_path(load, make_asset):
    """The load-bearing property for chaining.

    When scene N re-renders, scene N+1's chained frame changes on disk under an
    unchanged path. Hashing content, not the path, is what invalidates N+1 too.
    """
    anchor = make_asset("anchor.png", b"before")
    overrides = {"continuity": {"anchors": ["anchor.png"]}}
    before = fp(load(overrides))

    anchor.write_bytes(b"after")
    assert fp(load(overrides)) != before


def test_unrelated_file_does_not_invalidate(load, make_asset):
    make_asset("anchor.png")
    overrides = {"continuity": {"anchors": ["anchor.png"]}}
    before = fp(load(overrides))
    make_asset("unused.png", b"nothing to do with the scene")
    assert fp(load(overrides)) == before


def test_docker_image_is_not_hashed(load):
    """Deliberate: the image is not part of the render identity."""
    before = fp(load())
    after = fp(load({"docker": {"image": "some/other:image"}}))
    assert before == after


def test_ref_order_matters(load, make_asset):
    make_asset("a.png", b"aaa")
    make_asset("b.png", b"bbb")
    forward = fp(load({"continuity": {"anchors": ["a.png", "b.png"]}}))
    reverse = fp(load({"continuity": {"anchors": ["b.png", "a.png"]}}))
    assert forward != reverse


def test_missing_ref_differs_from_present_ref(load, make_asset):
    make_asset("anchor.png")
    script = load({"continuity": {"anchors": ["anchor.png"]}})
    present = fp(script)

    absent = script.run_dir / "frames" / "anchor.png"
    assert fp(script, refs=[absent]) != present


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"scenes": [{"id": "opening", "prompt": "Different."}]}, id="prompt"),
        pytest.param({"defaults": {"seed": 7}}, id="seed"),
        pytest.param({"defaults": {"width": 544}}, id="width"),
        pytest.param({"defaults": {"height": 1024}}, id="height"),
        pytest.param({"defaults": {"fps": 30}}, id="fps"),
        pytest.param({"defaults": {"video_frames": 90}}, id="video_frames"),
        pytest.param({"defaults": {"cfg_scale": 2.5}}, id="cfg_scale"),
        pytest.param({"defaults": {"steps": 30}}, id="steps"),
        pytest.param({"defaults": {"sampling_method": "euler"}}, id="sampling_method"),
        pytest.param({"defaults": {"negative_prompt": "blurry"}}, id="negative_prompt"),
        pytest.param({"defaults": {"style_suffix": "cinematic."}}, id="style_suffix"),
        pytest.param({"defaults": {"extra_args": ["--mmap"]}}, id="extra_args"),
    ],
)
def test_setting_change_invalidates(load, overrides):
    assert fp(load(overrides)) != MINIMAL_FP
