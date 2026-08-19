"""load_script: resolution, validation, and the shape of the loaded Script."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from moviemakr.config import Scene, load_script
from moviemakr.errors import ConfigError


def test_minimal_script_loads(load):
    script = load()
    assert script.name == "test-movie"
    assert len(script.scenes) == 1
    assert script.scenes[0].id == "opening"


def test_scene_slugs_are_numbered(load):
    script = load({"scenes": [
        {"id": "opening", "prompt": "a"},
        {"id": "Too Spicy", "prompt": "b"},
    ]})
    assert [s.slug for s in script.scenes] == ["001-opening", "002-too-spicy"]


def test_scene_id_defaults_to_position(load):
    script = load({"scenes": [{"prompt": "a"}, {"prompt": "b"}]})
    assert [s.id for s in script.scenes] == ["scene1", "scene2"]


def test_scenes_and_script_are_frozen(load):
    script = load()
    with pytest.raises(dataclasses.FrozenInstanceError):
        script.scenes[0].prompt = "changed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        script.name = "changed"


def test_scene_has_no_clip_field():
    """Clip paths come from the layout, not from a patched-on Scene attribute."""
    assert "clip" not in {f.name for f in dataclasses.fields(Scene)}


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_missing_prompt(load):
    with pytest.raises(ConfigError, match=r"scene 2: 'prompt' is required"):
        load({"scenes": [{"id": "a", "prompt": "p"}, {"id": "b"}]})


def test_duplicate_scene_id(load):
    with pytest.raises(ConfigError, match="duplicate scene id: a"):
        load({"scenes": [{"id": "a", "prompt": "p"}, {"id": "a", "prompt": "q"}]})


def test_no_scenes(load):
    with pytest.raises(ConfigError, match="no scenes"):
        load({"scenes": []})


def test_missing_script_file(project_root):
    with pytest.raises(ConfigError, match="script not found"):
        load_script(project_root / "nope.yaml", project_root)


def test_model_root_must_be_a_directory(load, tmp_path):
    with pytest.raises(ConfigError, match="model.root is not a directory"):
        load({"model": {"root": str(tmp_path / "missing")}})


@pytest.mark.parametrize("key", ["diffusion_model", "llm", "vae"])
def test_required_model_files(load, DELETE, key):
    with pytest.raises(ConfigError, match=f"model.{key} is required"):
        load({"model": {key: DELETE}})


def test_audio_vae_is_optional(load, DELETE):
    script = load({"model": {"audio_vae": DELETE}})
    assert "audio_vae" not in script.model_files


def test_model_file_must_exist(load):
    with pytest.raises(ConfigError, match="model.vae not found"):
        load({"model": {"vae": "h3/not-there.safetensors"}})


def test_bad_container(load):
    with pytest.raises(ConfigError, match="output.container must be"):
        load({"output": {"container": "avi"}})


def test_bad_audio_mode(load):
    with pytest.raises(ConfigError, match="output.audio must be"):
        load({"output": {"audio": "mute"}})


# --------------------------------------------------------------------------
# reference resolution
# --------------------------------------------------------------------------


def test_relative_ref_resolves_under_assets(load, make_asset, assets_dir):
    make_asset("anchor.png")
    script = load({"continuity": {"anchors": ["anchor.png"]}})
    assert script.scenes[0].ref_images == (assets_dir / "anchor.png",)


def test_missing_ref_fails_at_load(load):
    with pytest.raises(ConfigError, match="continuity anchor not found"):
        load({"continuity": {"anchors": ["nope.png"]}})


def test_ref_image_outside_the_mounts_fails_at_load(load, tmp_path):
    """It used to load fine and only blow up mid-render, inside to_container."""
    stray = tmp_path / "outside.png"
    stray.write_bytes(b"x")
    with pytest.raises(ConfigError, match="outside every mounted directory"):
        load({"continuity": {"anchors": [str(stray)]}})


def test_ref_video_outside_the_mounts_is_allowed(load, tmp_path):
    """Ref videos are transcoded into the run dir, so the source can live anywhere."""
    stray = tmp_path / "clip.webm"
    stray.write_bytes(b"x")
    script = load({"continuity": {"anchor_videos": [str(stray)]}})
    assert script.scenes[0].ref_videos == (stray.resolve(),)


def test_anchors_come_before_scene_refs(load, make_asset):
    make_asset("anchor.png", b"a")
    make_asset("own.png", b"o")
    script = load({
        "continuity": {"anchors": ["anchor.png"]},
        "scenes": [{"id": "a", "prompt": "p", "ref_images": ["own.png"]}],
    })
    assert [p.name for p in script.scenes[0].ref_images] == ["anchor.png", "own.png"]


def test_music_is_resolved(load, make_asset):
    make_asset("bed.mp3", b"m")
    script = load({"output": {"music": "bed.mp3"}})
    assert script.output.music.name == "bed.mp3"


# --------------------------------------------------------------------------
# chaining and layout
# --------------------------------------------------------------------------


def test_global_chain_applies_to_every_scene(load):
    script = load({
        "continuity": {"chain_from_previous": True},
        "scenes": [{"id": "a", "prompt": "p"}, {"id": "b", "prompt": "q"}],
    })
    assert all(s.chain_from_previous for s in script.scenes)


def test_scene_can_opt_out_of_a_global_chain(load):
    script = load({
        "continuity": {"chain_from_previous": True},
        "scenes": [{"id": "a", "prompt": "p", "chain_from_previous": False}],
    })
    assert script.scenes[0].chain_from_previous is False


def test_scene_can_opt_in(load):
    script = load({"scenes": [{"id": "a", "prompt": "p", "chain_from_previous": True}]})
    assert script.scenes[0].chain_from_previous is True


def test_run_dir_derives_from_name_not_filename(load, project_root):
    script = load({"name": "My Movie"}, filename="unrelated.yaml")
    assert script.run_dir == (project_root / "renders" / "my-movie").resolve()


def test_name_falls_back_to_filename(load, DELETE, project_root):
    script = load({"name": DELETE}, filename="fallback.yaml")
    assert script.name == "fallback"


def test_primary_size_and_fps_come_from_scene_one(load):
    script = load({
        "defaults": {"width": 540, "height": 960, "fps": 24},
        "scenes": [
            {"id": "a", "prompt": "p"},
            {"id": "b", "prompt": "q", "width": 720, "height": 1280, "fps": 30},
        ],
    })
    assert script.primary_size == (540, 960)
    assert script.fps == 24


# --------------------------------------------------------------------------
# full_prompt
# --------------------------------------------------------------------------


def test_style_suffix_is_appended(load):
    script = load({
        "defaults": {"style_suffix": "Cinematic."},
        "scenes": [{"id": "a", "prompt": "Three cats dance"}],
    })
    assert script.scenes[0].full_prompt() == "Three cats dance. Cinematic."


def test_trailing_period_is_not_doubled(load):
    script = load({
        "defaults": {"style_suffix": "Cinematic."},
        "scenes": [{"id": "a", "prompt": "Three cats dance."}],
    })
    assert script.scenes[0].full_prompt() == "Three cats dance. Cinematic."


@pytest.mark.parametrize("suffix", ["", "   "])
def test_empty_suffix_leaves_the_prompt_alone(load, suffix):
    script = load({
        "defaults": {"style_suffix": suffix},
        "scenes": [{"id": "a", "prompt": "Three cats dance."}],
    })
    assert script.scenes[0].full_prompt() == "Three cats dance."


def test_multiline_h3_prompt_is_untouched_without_a_suffix(load):
    prompt = "subject_definitions: <Subject 1> a cat\nsummary: it cooks\n"
    script = load({"scenes": [{"id": "a", "prompt": prompt}]})
    assert script.scenes[0].full_prompt() == prompt
