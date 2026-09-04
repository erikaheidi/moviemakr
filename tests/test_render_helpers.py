"""The pure decisions inside the render loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from moviemakr.errors import ConfigError
from moviemakr.render import (
    RenderOptions,
    is_up_to_date,
    parse_index_spec,
    resolve_refs,
    select_scenes,
)

# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("2", {2}),
        ("1,3", {1, 3}),
        ("4-6", {4, 5, 6}),
        ("2,4-6", {2, 4, 5, 6}),
        (" 2 , 4 - 6 ", {2, 4, 5, 6}),
        ("2,", {2}),
        ("3-3", {3}),
        ("6-4", set()),  # an inverted range selects nothing
    ],
)
def test_parse_index_spec(spec, expected):
    assert parse_index_spec(spec) == expected


@pytest.mark.parametrize("spec", ["abc", "2,x", "1-b", ""])
def test_parse_index_spec_rejects_garbage(spec):
    if spec == "":
        assert parse_index_spec(spec) == set()
        return
    with pytest.raises(ConfigError, match="--only expects scene indices"):
        parse_index_spec(spec)


def test_default_selects_every_scene(load):
    script = load({"scenes": [{"id": "a", "prompt": "p"}, {"id": "b", "prompt": "q"}]})
    assert select_scenes(script.scenes, RenderOptions()) == {1, 2}


def test_select_by_id(load):
    script = load({"scenes": [
        {"id": "a", "prompt": "p"}, {"id": "b", "prompt": "q"}, {"id": "c", "prompt": "r"},
    ]})
    assert select_scenes(script.scenes, RenderOptions(scene=" a , c ")) == {1, 3}


def test_unknown_id_raises(load):
    script = load()
    with pytest.raises(ConfigError, match="no such scene id"):
        select_scenes(script.scenes, RenderOptions(scene="nope,alsonope"))


def test_unknown_ids_are_listed_sorted(load):
    script = load()
    with pytest.raises(ConfigError, match="zzz, aaa|aaa, zzz") as exc:
        select_scenes(script.scenes, RenderOptions(scene="zzz,aaa"))
    assert "aaa, zzz" in str(exc.value)


def test_only_out_of_range_raises(load):
    """It used to silently select nothing, render nothing, then assemble."""
    script = load()
    with pytest.raises(ConfigError, match="do not exist: 99"):
        select_scenes(script.scenes, RenderOptions(only="99"))


def test_only_partially_out_of_range_raises(load):
    script = load({"scenes": [{"id": "a", "prompt": "p"}, {"id": "b", "prompt": "q"}]})
    with pytest.raises(ConfigError, match="do not exist: 5"):
        select_scenes(script.scenes, RenderOptions(only="1,5"))


def test_only_selecting_nothing_raises(load):
    script = load()
    with pytest.raises(ConfigError, match="selected no scenes"):
        select_scenes(script.scenes, RenderOptions(only="6-4"))


# --------------------------------------------------------------------------
# chaining
# --------------------------------------------------------------------------


def scene_of(script, chain: bool):
    from dataclasses import replace

    return replace(script.scenes[0], chain_from_previous=chain)


def test_no_chain_uses_only_the_scenes_own_refs(load, make_asset):
    make_asset("anchor.png")
    script = load({"continuity": {"anchors": ["anchor.png"]}})
    refs, warning = resolve_refs(scene_of(script, False), Path("/tmp/prev.png"), dry_run=False)
    assert [p.name for p in refs] == ["anchor.png"]
    assert warning is None


def test_chain_puts_the_previous_frame_first(load, make_asset, tmp_path):
    make_asset("anchor.png")
    script = load({"continuity": {"anchors": ["anchor.png"]}})
    prev = tmp_path / "prev.png"
    prev.write_bytes(b"frame")

    refs, warning = resolve_refs(scene_of(script, True), prev, dry_run=False)
    assert [p.name for p in refs] == ["prev.png", "anchor.png"]
    assert warning is None


def test_dry_run_wires_a_frame_that_does_not_exist_yet(load, tmp_path):
    """The real run will have extracted it by then, so show the real wiring."""
    script = load()
    prev = tmp_path / "not-yet.png"
    refs, warning = resolve_refs(scene_of(script, True), prev, dry_run=True)
    assert refs == (prev,)
    assert warning is None


def test_missing_frame_in_a_real_run_warns_and_drops_the_ref(load, tmp_path):
    script = load()
    prev = tmp_path / "not-there.png"
    refs, warning = resolve_refs(scene_of(script, True), prev, dry_run=False)
    assert refs == ()
    assert "chain_from_previous requested" in warning


def test_no_previous_scene_at_all_warns(load):
    script = load()
    refs, warning = resolve_refs(scene_of(script, True), None, dry_run=True)
    assert refs == ()
    assert warning is not None


# --------------------------------------------------------------------------
# skip decision
# --------------------------------------------------------------------------


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "001-a.webm"
    path.write_bytes(b"video")
    return path


def test_up_to_date_when_fingerprint_matches(clip):
    assert is_up_to_date(clip, {"fingerprint": "abc"}, "abc", force=False)


def test_force_always_rerenders(clip):
    assert not is_up_to_date(clip, {"fingerprint": "abc"}, "abc", force=True)


def test_fingerprint_mismatch_rerenders(clip):
    assert not is_up_to_date(clip, {"fingerprint": "old"}, "new", force=False)


def test_missing_clip_rerenders(tmp_path):
    absent = tmp_path / "gone.webm"
    assert not is_up_to_date(absent, {"fingerprint": "abc"}, "abc", force=False)


def test_empty_clip_rerenders(tmp_path):
    empty = tmp_path / "empty.webm"
    empty.write_bytes(b"")
    assert not is_up_to_date(empty, {"fingerprint": "abc"}, "abc", force=False)


def test_no_stored_entry_rerenders(clip):
    assert not is_up_to_date(clip, {}, "abc", force=False)


def test_failed_entry_rerenders(clip):
    assert not is_up_to_date(clip, {"fingerprint": None, "state": "failed"}, "abc", force=False)
