"""Script upload: what is accepted, where it lands, and what it is named.

No FastAPI and no web server - `store_upload` takes a directory and bytes, so
the decision logic is testable on its own, the way `assets.py` is.
"""

from __future__ import annotations

import pytest
import yaml

from moviemakr.web import scripts as S


@pytest.fixture
def scripts_dir(tmp_path):
    path = tmp_path / "scripts"
    path.mkdir()
    return path


@pytest.fixture
def script_bytes(base_script):
    return yaml.safe_dump(base_script, sort_keys=False).encode()


# --- what is accepted ------------------------------------------------------


@pytest.mark.parametrize("name", ["a.yaml", "a.YAML", "a.yml"])
def test_accepted_suffixes(scripts_dir, script_bytes, name):
    dest, _ = S.store_upload(scripts_dir, name, script_bytes)
    assert dest.is_file()


@pytest.mark.parametrize("name", ["notes.txt", "beach.json", "beach.yaml.bak", "beach"])
def test_rejected_suffixes(scripts_dir, script_bytes, name):
    with pytest.raises(S.ScriptRejected, match="only "):
        S.store_upload(scripts_dir, name, script_bytes)


def test_empty_file_is_rejected(scripts_dir):
    with pytest.raises(S.ScriptRejected, match="empty"):
        S.store_upload(scripts_dir, "a.yaml", b"")


def test_oversize_is_rejected(scripts_dir):
    data = b"x" * (S.MAX_UPLOAD_BYTES + 1)
    with pytest.raises(S.ScriptRejected, match="exceeds"):
        S.store_upload(scripts_dir, "a.yaml", data)


def test_non_utf8_is_rejected(scripts_dir):
    with pytest.raises(S.ScriptRejected, match="not UTF-8"):
        S.store_upload(scripts_dir, "a.yaml", b"\xff\xfe\x00binary")


def test_broken_yaml_is_rejected(scripts_dir):
    with pytest.raises(S.ScriptRejected, match="invalid YAML"):
        S.store_upload(scripts_dir, "a.yaml", b"this: [is: not: valid\n  bad indent\n")


def test_a_yaml_that_is_not_a_mapping_is_rejected(scripts_dir):
    with pytest.raises(S.ScriptRejected, match="mapping"):
        S.store_upload(scripts_dir, "a.yaml", b"- one\n- two\n")


def test_yaml_without_scenes_is_rejected(scripts_dir):
    """Catches the wrong file - a lockfile, a CI config - before it lands."""
    with pytest.raises(S.ScriptRejected, match="no scenes"):
        S.store_upload(scripts_dir, "a.yaml", b"name: something else\n")


def test_a_rejected_upload_writes_nothing(scripts_dir):
    with pytest.raises(S.ScriptRejected):
        S.store_upload(scripts_dir, "a.yaml", b"name: nope\n")
    assert list(scripts_dir.iterdir()) == []


def test_load_failures_are_not_upload_failures(scripts_dir, base_script):
    """A script naming a not-yet-uploaded ref still lands - order matters."""
    base_script["continuity"] = {"anchors": ["not-here-yet.jpg"]}
    data = yaml.safe_dump(base_script, sort_keys=False).encode()
    dest, _ = S.store_upload(scripts_dir, "a.yaml", data)
    assert dest.is_file()


# --- naming and placement --------------------------------------------------


def test_filename_is_slugified(scripts_dir, script_bytes):
    dest, _ = S.store_upload(scripts_dir, "Josy Beach Drive.yaml", script_bytes)
    assert dest.name == "josy-beach-drive.yaml"


def test_folder_is_created(scripts_dir, script_bytes):
    dest, _ = S.store_upload(scripts_dir, "beach.yaml", script_bytes, folder="h3")
    assert dest == scripts_dir / "h3" / "beach.yaml"


def test_nested_folder(scripts_dir, script_bytes):
    dest, _ = S.store_upload(scripts_dir, "beach.yaml", script_bytes, folder="h3/kids/")
    assert dest == scripts_dir / "h3" / "kids" / "beach.yaml"


@pytest.mark.parametrize("folder", ["../outside", "/etc", "h3/../../outside"])
def test_folder_cannot_escape_scripts_dir(scripts_dir, script_bytes, folder):
    with pytest.raises(S.ScriptRejected):
        S.store_upload(scripts_dir, "beach.yaml", script_bytes, folder=folder)


def test_a_directory_in_the_filename_is_dropped(scripts_dir, script_bytes):
    """The name comes from another machine; it names a file, never a path."""
    dest, _ = S.store_upload(scripts_dir, "../../etc/passwd.yaml", script_bytes)
    assert dest == scripts_dir / "passwd.yaml"


def test_crlf_is_normalised(scripts_dir):
    data = b"name: x\r\nscenes:\r\n  - id: a\r\n    prompt: b\r\n"
    dest, _ = S.store_upload(scripts_dir, "a.yaml", data)
    assert "\r" not in dest.read_text()


# --- replacing -------------------------------------------------------------


def test_an_existing_script_is_not_silently_overwritten(scripts_dir, script_bytes):
    S.store_upload(scripts_dir, "beach.yaml", script_bytes)
    with pytest.raises(S.ScriptRejected, match="already exists"):
        S.store_upload(scripts_dir, "beach.yaml", script_bytes)


def test_replace_overwrites(scripts_dir, script_bytes, base_script):
    dest, note = S.store_upload(scripts_dir, "beach.yaml", script_bytes)
    assert note == "new"

    base_script["scenes"].append({"id": "second", "prompt": "Another."})
    updated = yaml.safe_dump(base_script, sort_keys=False).encode()
    dest, note = S.store_upload(scripts_dir, "beach.yaml", updated, replace=True)
    assert note == "replaced"
    assert "second" in dest.read_text()


def test_replace_is_scoped_to_the_same_folder(scripts_dir, script_bytes):
    """h3/beach.yaml and beach.yaml are different scripts, not a collision."""
    S.store_upload(scripts_dir, "beach.yaml", script_bytes)
    dest, note = S.store_upload(scripts_dir, "beach.yaml", script_bytes, folder="h3")
    assert note == "new"
    assert dest == scripts_dir / "h3" / "beach.yaml"


# --- load_error ------------------------------------------------------------


def test_load_error_is_none_for_a_good_script(workspace, project_root, script_bytes):
    dest, _ = S.store_upload(workspace.scripts_dir, "beach.yaml", script_bytes)
    assert S.load_error(dest, workspace) is None


def test_load_error_reports_a_missing_ref(workspace, base_script):
    base_script["continuity"] = {"anchors": ["not-here-yet.jpg"]}
    data = yaml.safe_dump(base_script, sort_keys=False).encode()
    dest, _ = S.store_upload(workspace.scripts_dir, "beach.yaml", data)
    problem = S.load_error(dest, workspace)
    assert problem and "not-here-yet.jpg" in problem
