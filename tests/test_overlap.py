"""Overlap continuation: the anchor going in, and the trim coming out.

The two halves have to agree exactly. A scene anchored on 22 frames regenerates
those 22 frames at its head, so assembly must remove 22 - no more, no less. Off
by one and every seam in the movie stutters or jumps.
"""

from __future__ import annotations

import pytest

from moviemakr.backends.comfy import align_down, build_graph, effective_overlap, fingerprint
from moviemakr.assemble import overlap_trim
from moviemakr.media import NormalizeSpec, normalize_cmd, tail_clip_cmd, tail_start

SPEC = NormalizeSpec(width=540, height=960, fps=24, container="mp4", keep_audio=True)


# --- the anchor going in --------------------------------------------------


def test_overlap_adds_the_video_loader_chain(load_comfy):
    script = load_comfy()
    graph = build_graph(script.scenes[0], script, overlap_clip="moviemakr/m/001.tail.mp4")
    assert graph["tail"]["inputs"]["file"] == "moviemakr/m/001.tail.mp4"
    assert graph["comp"]["inputs"]["video"] == ["tail", 0]


def test_overlap_anchors_both_streams(load_comfy):
    """Frames without their soundtrack would restart the soundscape anyway."""
    script = load_comfy()
    guide = build_graph(script.scenes[0], script, overlap_clip="x.mp4")["guide"]
    assert guide["class_type"] == "MiniMaxH3AddGuide"
    assert guide["inputs"]["image"] == ["comp", 0]
    assert guide["inputs"]["audio"] == ["comp", 1]
    assert guide["inputs"]["vae"] == ["vae", 0]
    assert guide["inputs"]["audio_vae"] == ["avae", 0]


def test_overlap_is_anchored_at_the_head(load_comfy):
    script = load_comfy()
    graph = build_graph(script.scenes[0], script, overlap_clip="x.mp4")
    assert graph["guide"]["inputs"]["frame_idx"] == 0


def test_the_guide_replaces_the_conditioning_into_the_guider(load_comfy):
    """If the guider still read ['cond', 0] the anchor would be built and ignored."""
    script = load_comfy()
    plain = build_graph(script.scenes[0], script)
    chained = build_graph(script.scenes[0], script, overlap_clip="x.mp4")
    assert plain["guider"]["inputs"]["conditioning"] == ["cond", 0]
    assert chained["guider"]["inputs"]["conditioning"] == ["guide", 0]
    assert chained["guide"]["inputs"]["positive"] == ["cond", 0]
    assert chained["guide"]["inputs"]["latent"] == ["cond", 1]


def test_overlap_wins_over_a_first_frame(load_comfy):
    """Both would anchor frame 0; the segment carries strictly more than the still."""
    script = load_comfy()
    graph = build_graph(script.scenes[0], script, first_frame="p.png", overlap_clip="x.mp4")
    assert "first" not in graph
    assert "first_frame" not in graph["cond"]["inputs"]
    assert "guide" in graph


def test_no_overlap_leaves_the_graph_alone(load_comfy):
    script = load_comfy()
    graph = build_graph(script.scenes[0], script)
    assert {"tail", "comp", "guide"} & set(graph) == set()


def test_overlap_changes_the_fingerprint(load_comfy):
    script = load_comfy()
    scene = script.scenes[0]
    assert fingerprint(scene, script) != fingerprint(scene, script, overlap_clip="x.mp4")


# --- how many frames ------------------------------------------------------


def test_effective_overlap_needs_a_previous_scene(load_comfy):
    """Scene 1 has nothing to anchor on, however it is configured."""
    script = load_comfy({"continuity": {"chain_from_previous": True}})
    assert effective_overlap(script.scenes[0], has_previous=False) == 0


def test_effective_overlap_needs_chaining(load_comfy):
    script = load_comfy()
    scene = script.scenes[0]
    assert scene.chain_from_previous is False
    assert effective_overlap(scene, has_previous=True) == 0


def test_effective_overlap_snaps_down(load_comfy):
    """30 is not on the grid; the node would crop to 22 without saying so."""
    script = load_comfy({
        "continuity": {"chain_from_previous": True},
        "defaults": {"overlap_frames": 30},
    })
    assert effective_overlap(script.scenes[0], has_previous=True) == 22


def test_overlap_frames_zero_is_a_hard_cut(load_comfy):
    script = load_comfy({
        "continuity": {"chain_from_previous": True},
        "defaults": {"overlap_frames": 0},
    })
    assert effective_overlap(script.scenes[0], has_previous=True) == 0


def test_overlap_frames_is_a_per_scene_override(load_comfy):
    script = load_comfy({
        "continuity": {"chain_from_previous": True},
        "defaults": {"overlap_frames": 22},
        "scenes": [
            {"id": "a", "prompt": "one."},
            {"id": "b", "prompt": "two.", "overlap_frames": 39},
            {"id": "c", "prompt": "three.", "overlap_frames": 0},
        ],
    })
    got = [effective_overlap(s, has_previous=True) for s in script.scenes]
    assert got == [22, 39, 0]


def test_overlap_frames_rejects_a_negative(load_comfy):
    from moviemakr.errors import ConfigError

    with pytest.raises(ConfigError, match="must be 0 or more"):
        load_comfy({"defaults": {"overlap_frames": -1}})


# --- cutting the tail -----------------------------------------------------


def test_tail_start_leaves_exactly_the_requested_frames():
    assert tail_start(5.0, 24, 24) == pytest.approx(4.0)
    assert tail_start(5.166667, 22, 24) == pytest.approx(5.166667 - 22 / 24)


def test_tail_start_never_goes_negative():
    """A clip shorter than the overlap starts at 0 rather than seeking backwards."""
    assert tail_start(0.5, 22, 24) == 0.0


def test_tail_clip_seeks_before_the_input():
    """After -i, the seek would decode and discard the whole clip first."""
    cmd = tail_clip_cmd("a.webm", "b.mp4", start=4.0, frames=22)
    assert cmd.index("-ss") < cmd.index("-i")


def test_tail_clip_pins_the_frame_count():
    cmd = tail_clip_cmd("a.webm", "b.mp4", start=4.0, frames=22)
    assert cmd[cmd.index("-frames:v") + 1] == "22"


def test_tail_clip_keeps_audio():
    cmd = tail_clip_cmd("a.webm", "b.mp4", start=4.0, frames=22)
    assert "-c:a" in cmd
    assert "-an" not in cmd


def test_tail_clip_re_encodes():
    """`-c copy` seeks to a keyframe and would deliver the wrong frame count."""
    cmd = tail_clip_cmd("a.webm", "b.mp4", start=4.0, frames=22)
    assert "copy" not in cmd


# --- trimming it back off -------------------------------------------------


def test_normalize_seeks_before_the_input_so_audio_follows():
    """A filter-side trim moves only the video and slides the sound forward."""
    cmd = normalize_cmd("a.webm", "b.mp4", SPEC, has_audio=True, skip_seconds=0.9166)
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-ss") + 1] == "0.916600"


def test_normalize_without_a_skip_is_unchanged():
    plain = normalize_cmd("a.webm", "b.mp4", SPEC, has_audio=True)
    explicit = normalize_cmd("a.webm", "b.mp4", SPEC, has_audio=True, skip_seconds=0.0)
    assert plain == explicit
    assert "-ss" not in plain


def test_overlap_trim_reads_what_was_rendered(load_comfy):
    """Not the script's current value, which may have been edited since."""
    script = load_comfy({"defaults": {"overlap_frames": 39}})
    scene = script.scenes[0]
    state = {"scenes": {scene.id: {"overlap_frames": 22}}}
    assert overlap_trim(scene, state) == pytest.approx(22 / 24)


def test_overlap_trim_is_zero_without_state(load_comfy):
    """Clips rendered before overlap existed must not be retro-trimmed."""
    script = load_comfy({"defaults": {"overlap_frames": 22}})
    assert overlap_trim(script.scenes[0], {}) == 0.0
    assert overlap_trim(script.scenes[0], {"scenes": {}}) == 0.0


def test_the_trim_matches_the_anchor_exactly(load_comfy):
    """The two halves of the mechanism, checked against each other."""
    script = load_comfy({
        "continuity": {"chain_from_previous": True},
        "defaults": {"overlap_frames": 30, "fps": 24},
    })
    scene = script.scenes[0]
    anchored = effective_overlap(scene, has_previous=True)
    state = {"scenes": {scene.id: {"overlap_frames": anchored}}}
    assert overlap_trim(scene, state) == pytest.approx(anchored / 24)
    assert anchored == align_down(30)
