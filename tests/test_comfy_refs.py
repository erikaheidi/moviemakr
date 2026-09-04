"""Reference images on the ComfyUI backend.

Anchors are how a script holds a character together across a whole film, so the
things that matter here are ordering (the prompt's `<Picture N>` contract), that
they compose with the overlap anchor rather than replacing it, and that swapping
an anchor invalidates the scenes that used it.
"""

from __future__ import annotations

import pytest

from moviemakr.backends.comfy import build_graph, fingerprint, prepare_refs


@pytest.fixture
def with_refs(load_comfy, make_asset):
    """A comfy script whose single scene carries two anchors and one own ref."""
    def _make(**overrides):
        make_asset("anchor-a.png", b"anchor-a")
        make_asset("anchor-b.png", b"anchor-b")
        make_asset("scene.png", b"scene-ref")
        base = {
            "continuity": {"anchors": ["anchor-a.png", "anchor-b.png"]},
            "scenes": [{"id": "opening", "prompt": "A <Picture 1> in a room.",
                        "ref_images": ["scene.png"]}],
        }
        base.update(overrides)
        return load_comfy(base)
    return _make


# --- which node ----------------------------------------------------------


def test_refs_switch_to_the_reference_node(with_refs):
    script = with_refs()
    graph = build_graph(script.scenes[0], script, refs=["r1.png"])
    assert graph["cond"]["class_type"] == "MiniMaxH3ReferenceToVideo"


def test_no_refs_keeps_the_keyframe_node(load_comfy):
    script = load_comfy()
    assert build_graph(script.scenes[0], script)["cond"]["class_type"] == \
        "MiniMaxH3ImageToVideo"


def test_the_reference_node_gets_the_audio_vae(with_refs):
    """It is required there, unlike on the keyframe node."""
    script = with_refs()
    graph = build_graph(script.scenes[0], script, refs=["r1.png"])
    assert graph["cond"]["inputs"]["audio_vae"] == ["avae", 0]


def test_ref_image_size_is_passed_through(with_refs):
    script = with_refs()
    graph = build_graph(script.scenes[0], script, refs=["r1.png"])
    assert graph["cond"]["inputs"]["ref_image_size"] == "match"


def test_ref_image_size_max_is_configurable(load_comfy, make_asset):
    make_asset("a.png", b"a")
    script = load_comfy({
        "comfy": {"ref_image_size": "max"},
        "scenes": [{"id": "o", "prompt": "p.", "ref_images": ["a.png"]}],
    })
    graph = build_graph(script.scenes[0], script, refs=["r1.png"])
    assert graph["cond"]["inputs"]["ref_image_size"] == "max"


def test_ref_image_size_rejects_nonsense(load_comfy):
    from moviemakr.errors import ConfigError

    with pytest.raises(ConfigError, match="must be one of match, max"):
        load_comfy({"comfy": {"ref_image_size": "huge"}})


# --- the <Picture N> contract --------------------------------------------


def test_refs_are_numbered_in_order(with_refs):
    """Order is the contract: <Picture 1> is the first ref, and anchors lead."""
    script = with_refs()
    graph = build_graph(script.scenes[0], script, refs=["a.png", "b.png", "c.png"])
    refs = graph["cond"]["inputs"]["ref_images"]
    assert list(refs) == ["ref_image_1", "ref_image_2", "ref_image_3"]
    assert refs["ref_image_1"] == ["ref1", 0]
    assert refs["ref_image_3"] == ["ref3", 0]
    assert graph["ref1"]["inputs"]["image"] == "a.png"
    assert graph["ref3"]["inputs"]["image"] == "c.png"


def test_anchors_come_before_a_scenes_own_refs(with_refs):
    """Matches the sdcpp ordering, so a prompt's <Picture N> means the same thing."""
    script = with_refs()
    names = [p.name for p in script.scenes[0].ref_images]
    assert names == ["anchor-a.png", "anchor-b.png", "scene.png"]


def test_each_ref_gets_its_own_loader(with_refs):
    script = with_refs()
    graph = build_graph(script.scenes[0], script, refs=["a.png", "b.png"])
    loaders = [n for n, v in graph.items() if v["class_type"] == "LoadImage"]
    assert sorted(loaders) == ["ref1", "ref2"]


# --- composing with the overlap anchor ------------------------------------


def test_refs_and_overlap_compose(with_refs):
    """The model reads minimax_refs and minimax_keyframes independently and
    concatenates their latents, so a scene can hold a character *and* continue
    a movement."""
    script = with_refs()
    graph = build_graph(script.scenes[0], script, refs=["a.png"], overlap_clip="t.mp4")
    assert graph["cond"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert graph["guide"]["class_type"] == "MiniMaxH3AddGuide"
    assert graph["guide"]["inputs"]["positive"] == ["cond", 0]
    assert graph["guider"]["inputs"]["conditioning"] == ["guide", 0]


def test_refs_do_not_take_a_first_frame(with_refs):
    """ReferenceToVideo has no first_frame input; passing one would be rejected."""
    script = with_refs()
    graph = build_graph(script.scenes[0], script, refs=["a.png"], first_frame="f.png")
    assert "first_frame" not in graph["cond"]["inputs"]
    assert "first" not in graph


# --- placing them where ComfyUI can read them -----------------------------


def test_prepare_refs_copies_into_the_input_dir(with_refs, tmp_path):
    indir = tmp_path / "cin"
    indir.mkdir()
    script = with_refs(comfy={"input_dir": str(indir)})
    placed = prepare_refs(script, script.scenes[0].ref_images)
    assert len(placed) == 3
    for name, src in placed:
        assert (indir / name).read_bytes() == src.read_bytes()


def test_ref_names_are_content_addressed(with_refs, tmp_path):
    """Two anchors called josy.jpg in different folders must not collide."""
    indir = tmp_path / "cin"
    indir.mkdir()
    script = with_refs(comfy={"input_dir": str(indir)})
    names = [n for n, _ in prepare_refs(script, script.scenes[0].ref_images)]
    assert len(set(names)) == 3


def test_prepare_refs_writes_nothing_on_a_dry_run(with_refs, tmp_path):
    indir = tmp_path / "cin"
    indir.mkdir()
    script = with_refs(comfy={"input_dir": str(indir)})
    placed = prepare_refs(script, script.scenes[0].ref_images, dry_run=True)
    assert len(placed) == 3
    assert not any(indir.rglob("*.png"))


def test_refs_need_an_input_dir(with_refs):
    from moviemakr.errors import ConfigError

    script = with_refs()  # no input_dir configured
    with pytest.raises(ConfigError, match="input_dir is required"):
        prepare_refs(script, script.scenes[0].ref_images)


def test_no_refs_needs_no_input_dir(load_comfy):
    assert prepare_refs(load_comfy(), []) == []


# --- fingerprinting -------------------------------------------------------


def test_swapping_an_anchor_invalidates_the_scene(with_refs, tmp_path):
    """Otherwise a corrected character sheet would silently not be applied."""
    indir = tmp_path / "cin"
    indir.mkdir()
    script = with_refs(comfy={"input_dir": str(indir)})
    scene = script.scenes[0]
    placed = prepare_refs(script, scene.ref_images)
    names, paths = [n for n, _ in placed], [p for _, p in placed]

    before = fingerprint(scene, script, paths, refs=names)
    paths[0].write_bytes(b"a different anchor entirely")
    after = fingerprint(scene, script, paths, refs=names)
    assert before != after


def test_ref_order_is_hashed(with_refs, tmp_path):
    script = with_refs()
    scene = script.scenes[0]
    a = fingerprint(scene, script, refs=["x.png", "y.png"])
    b = fingerprint(scene, script, refs=["y.png", "x.png"])
    assert a != b


def test_refs_change_the_fingerprint_against_no_refs(with_refs):
    script = with_refs()
    scene = script.scenes[0]
    assert fingerprint(scene, script) != fingerprint(scene, script, refs=["x.png"])


# --- checkpoint pairing ---------------------------------------------------


def test_refs_on_an_fl2va_model_warn(load_comfy, make_asset):
    """Crossing the task variants does not error - it renders the wrong thing."""
    from moviemakr.backends.comfy import pairing_warnings

    make_asset("a.png", b"a")
    script = load_comfy({
        "comfy": {"diffusion_model": "minimax_h3_fl2va_pruned_bf16.safetensors"},
        "scenes": [{"id": "o", "prompt": "p.", "ref_images": ["a.png"]}],
    })
    warnings = pairing_warnings(script)
    assert any("fl2va" in w and "ref2va" in w for w in warnings)


def test_no_refs_on_a_ref2va_model_warns(load_comfy):
    from moviemakr.backends.comfy import pairing_warnings

    script = load_comfy({
        "comfy": {"diffusion_model": "minimax_h3_ref2va_pruned_bf16.safetensors"}})
    assert any("ref2va" in w for w in pairing_warnings(script))


def test_a_crossed_lora_warns(load_comfy, make_asset):
    from moviemakr.backends.comfy import pairing_warnings

    make_asset("a.png", b"a")
    script = load_comfy({
        "comfy": {"diffusion_model": "minimax_h3_ref2va_pruned_bf16.safetensors",
                  "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"},
        "scenes": [{"id": "o", "prompt": "p.", "ref_images": ["a.png"]}],
    })
    assert any("fl2v LoRA on a ref2va model" in w for w in pairing_warnings(script))


def test_a_matched_pairing_is_quiet(load_comfy, make_asset):
    from moviemakr.backends.comfy import pairing_warnings

    make_asset("a.png", b"a")
    script = load_comfy({
        "comfy": {"diffusion_model": "minimax_h3_ref2va_pruned_bf16.safetensors",
                  "lora": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"},
        "scenes": [{"id": "o", "prompt": "p.", "ref_images": ["a.png"]}],
    })
    assert pairing_warnings(script) == []


def test_renamed_files_are_not_an_error(load_comfy, make_asset):
    """The heuristic reads names; an unconventional one must stay silent, not fail."""
    from moviemakr.backends.comfy import pairing_warnings

    make_asset("a.png", b"a")
    script = load_comfy({
        "comfy": {"diffusion_model": "my-model.safetensors", "lora": "my-lora.safetensors"},
        "scenes": [{"id": "o", "prompt": "p.", "ref_images": ["a.png"]}],
    })
    assert pairing_warnings(script) == []


def test_refs_are_nested_not_flat(with_refs):
    """Autogrow inputs are one nested input. Flat ref_image_N keys pass
    validation and then raise inside execute() as unexpected kwargs."""
    script = with_refs()
    inputs = build_graph(script.scenes[0], script, refs=["a.png", "b.png"])["cond"]["inputs"]
    assert inputs["ref_images"] == {"ref_image_1": ["ref1", 0], "ref_image_2": ["ref2", 0]}
    assert not any(k.startswith("ref_image_") and k != "ref_image_size" for k in inputs)


# --- validating a graph against the server's schema -----------------------


def autogrow_info():
    """A cut-down /object_info carrying a real autogrow declaration."""
    return {
        "MiniMaxH3ReferenceToVideo": {"input": {
            "required": {"clip": ["CLIP", {}], "prompt": ["STRING", {}]},
            "optional": {"ref_images": ["COMFY_AUTOGROW_V3", {
                "template": {"prefix": "ref_image_", "min": 0, "max": 9}}]},
        }},
    }


def test_validate_rejects_flat_autogrow_keys():
    """The mistake that cost a render: accepted by /prompt, raises in execute()."""
    from moviemakr.backends.comfy import validate_graph

    graph = {"cond": {"class_type": "MiniMaxH3ReferenceToVideo",
                      "inputs": {"clip": ["c", 0], "prompt": "p", "ref_image_1": ["r", 0]}}}
    problems = validate_graph(graph, autogrow_info())
    assert len(problems) == 1
    assert "ref_image_1" in problems[0]
    assert "belongs inside the 'ref_images' dict" in problems[0]


def test_validate_accepts_the_nested_form():
    from moviemakr.backends.comfy import validate_graph

    graph = {"cond": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["c", 0], "prompt": "p", "ref_images": {"ref_image_1": ["r", 0]}}}}
    assert validate_graph(graph, autogrow_info()) == []


def test_validate_rejects_a_non_dict_autogrow_value():
    from moviemakr.backends.comfy import validate_graph

    graph = {"cond": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["c", 0], "prompt": "p", "ref_images": ["r", 0]}}}
    problems = validate_graph(graph, autogrow_info())
    assert "needs a dict" in problems[0]


def test_validate_catches_a_missing_required_input():
    from moviemakr.backends.comfy import validate_graph

    graph = {"cond": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["c", 0]}}}
    problems = validate_graph(graph, autogrow_info())
    assert any("missing required input 'prompt'" in p for p in problems)


def test_validate_catches_an_unknown_node_type():
    from moviemakr.backends.comfy import validate_graph

    graph = {"x": {"class_type": "NoSuchNode", "inputs": {}}}
    assert "no such node type" in validate_graph(graph, autogrow_info())[0]


def test_validate_allows_dynamic_combo_dotted_keys():
    """SaveVideo's dynamic combo serialises as `format.codec`."""
    from moviemakr.backends.comfy import validate_graph

    info = {"SaveVideo": {"input": {"required": {"format": [["auto"], {}]}}}}
    graph = {"s": {"class_type": "SaveVideo",
                   "inputs": {"format": "auto", "format.codec": "auto"}}}
    assert validate_graph(graph, info) == []
