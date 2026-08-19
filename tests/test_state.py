"""state.json round-trips, and survives a truncated write."""

from __future__ import annotations

import json

from moviemakr.state import load_state, save_state, scene_entry

REAL_ENTRY = {
    "fingerprint": "42948790d277" + "0" * 52,
    "state": "rendered",
    "clip": "/out/scenes/001-cooking.webm",
    "probe": {"frames": 107, "duration": 4.416666, "has_audio": True,
              "width": 544, "height": 960},
    "elapsed": 3342.7052717208862,
    "rendered_at": "2026-08-16 16:24:03",
}


def test_missing_file(tmp_path):
    assert load_state(tmp_path / "nope.json") == {"scenes": {}}


def test_corrupt_json_does_not_block_a_run(tmp_path):
    """Losing fingerprints costs a re-render; raising costs the same and helps nobody."""
    path = tmp_path / "state.json"
    path.write_text('{"scenes": {"a": ')
    assert load_state(path) == {"scenes": {}}


def test_non_dict_json(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("[1, 2, 3]")
    assert load_state(path) == {"scenes": {}}


def test_missing_scenes_key_is_filled_in(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{}")
    assert load_state(path) == {"scenes": {}}


def test_round_trip(tmp_path):
    path = tmp_path / "state.json"
    state = {"scenes": {"cooking": REAL_ENTRY}}
    save_state(path, state)
    assert load_state(path) == state


def test_probe_survives_unchanged(tmp_path):
    path = tmp_path / "state.json"
    save_state(path, {"scenes": {"cooking": REAL_ENTRY}})
    assert load_state(path)["scenes"]["cooking"]["probe"] == REAL_ENTRY["probe"]


def test_written_form_is_indented_with_a_trailing_newline(tmp_path):
    path = tmp_path / "state.json"
    save_state(path, {"scenes": {"a": {"state": "rendered"}}})
    text = path.read_text()
    assert text.endswith("}\n")
    assert '\n  "scenes"' in text
    json.loads(text)


def test_save_creates_the_directory(tmp_path):
    path = tmp_path / "renders" / "movie" / "state.json"
    save_state(path, {"scenes": {}})
    assert path.is_file()


def test_scene_entry_missing_is_empty():
    assert scene_entry({"scenes": {}}, "nope") == {}
    assert scene_entry({}, "nope") == {}


def test_scene_entry_found():
    assert scene_entry({"scenes": {"a": REAL_ENTRY}}, "a") == REAL_ENTRY
