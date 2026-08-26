"""Reading a workspace into the data the templates render.

No FastAPI needed - `browse` deliberately takes a Workspace and returns dicts.
"""

from __future__ import annotations

import pytest

from moviemakr.web import browse as B
from moviemakr.web.paths import rel_key


# --- discovery -------------------------------------------------------------


def test_finds_scripts_recursively_and_sorted(web_workspace):
    keys = [rel_key(web_workspace.scripts_dir, p) for p in B.script_files(web_workspace)]
    assert keys == ["h3/beach.yaml", "h3/broken.yaml", "simple.yaml"]


def test_no_scripts_dir_is_empty_not_an_error(workspace):
    assert B.script_files(workspace) == []
    assert B.script_rows(workspace) == []


def test_script_folders_are_the_subdirectories_in_use(web_workspace):
    """Offered to the upload form; the top level is not one of them."""
    assert B.script_folders(web_workspace) == ["h3"]


def test_no_scripts_dir_has_no_folders(workspace):
    assert B.script_folders(workspace) == []


def test_a_broken_script_is_a_row_not_an_exception(web_workspace):
    """One unloadable YAML must not take the index down."""
    rows = {r.key: r for r in B.script_rows(web_workspace)}
    broken = rows["h3/broken.yaml"]
    assert broken.error is not None
    assert "does-not-exist.jpg" in broken.error
    assert broken.progress == "invalid"
    assert rows["simple.yaml"].error is None


def test_a_syntactically_broken_script_is_a_row_too(web_workspace):
    """The other kind of broken: unparseable YAML, not a bad reference.

    `browse` only catches ConfigError, so this passes only while `load_script`
    converts YAMLError into one.
    """
    (web_workspace.scripts_dir / "unparseable.yaml").write_text(
        "this: [is: not: valid: yaml\n  bad indent\n")
    rows = {r.key: r for r in B.script_rows(web_workspace)}
    row = rows["unparseable.yaml"]
    assert row.error is not None
    assert "invalid YAML" in row.error
    assert row.progress == "invalid"
    assert rows["simple.yaml"].error is None


def test_row_name_comes_from_the_yaml_not_the_filename(web_workspace):
    rows = {r.key: r for r in B.script_rows(web_workspace)}
    assert rows["h3/beach.yaml"].name == "beach drive"


def test_movie_is_detected(web_workspace):
    rows = {r.key: r for r in B.script_rows(web_workspace)}
    assert rows["simple.yaml"].has_movie
    assert rows["simple.yaml"].movie_size > 0
    assert not rows["h3/beach.yaml"].has_movie


# --- one script ------------------------------------------------------------


def test_scene_table_states(web_workspace):
    script = B.load_script_at(web_workspace, "simple.yaml")
    table = B.scene_table(script)
    assert [row["id"] for row in table] == ["opening", "middle"]
    # Scene 1 has a clip but no stored fingerprint, so it reports its recorded
    # state; scene 2 has no clip at all.
    assert table[0]["state"] == "rendered"
    assert table[1]["state"] == "pending"
    assert table[0]["clip"] is not None
    assert table[1]["clip"] is None


def test_scene_table_uses_the_stored_probe(web_workspace):
    """Durations come from state.json, so the table costs no ffprobe calls."""
    script = B.load_script_at(web_workspace, "simple.yaml")
    table = B.scene_table(script)
    assert table[0]["duration"] == "3.8s"
    assert table[0]["elapsed"] == "15m02s"
    assert table[0]["size"] == "540x960"


def test_scene_table_reports_chaining_and_refs(web_workspace):
    script = B.load_script_at(web_workspace, "h3/beach.yaml")
    table = B.scene_table(script)
    assert table[0]["chained"] is False
    assert table[1]["chained"] is True
    assert table[0]["refs"] == ["josy-reference.jpg"]


def test_logs_are_listed(web_workspace):
    script = B.load_script_at(web_workspace, "simple.yaml")
    assert [entry["name"] for entry in B.log_files(script)] == \
        ["001-opening.attempt1.log"]


def test_no_logs_dir_is_empty(web_workspace):
    script = B.load_script_at(web_workspace, "h3/beach.yaml")
    assert B.log_files(script) == []


# --- drafts ----------------------------------------------------------------


def test_draft_rows(web_workspace):
    rows = B.draft_rows(web_workspace)
    assert [r.slug for r in rows] == ["picnic"]
    assert rows[0].title == "Beach picnic"
    assert "sandwich" in rows[0].excerpt


def test_draft_title_falls_back(tmp_path):
    assert B.draft_title("", "my-slug") == "my-slug"
    assert B.draft_title("\n\nplain first line\n", "my-slug") == "plain first line"
    assert B.draft_title("### Heading\n", "my-slug") == "Heading"


def test_no_drafts_dir_is_empty(workspace):
    assert B.draft_rows(workspace) == []


# --- assets ----------------------------------------------------------------


def test_asset_usage_is_textual(web_workspace):
    """Textual on purpose: a script that will not load still reports its refs."""
    usage = B.asset_usage(web_workspace)
    assert usage["josy-reference.jpg"] == ["h3/beach.yaml"]
    assert "unused.png" not in usage


def test_asset_rows_carry_usage(web_workspace):
    rows = {r.name: r for r in
            B.asset_rows(web_workspace, usage=B.asset_usage(web_workspace))}
    assert rows["josy-reference.jpg"].used_by == ["h3/beach.yaml"]
    assert rows["unused.png"].used_by == []


def test_non_images_are_not_listed_as_assets(web_workspace):
    (web_workspace.assets_dir / "notes.txt").write_text("not an image")
    assert "notes.txt" not in [p.name for p in B.asset_files(web_workspace)]


# --- summary ---------------------------------------------------------------


def test_workspace_summary(web_workspace):
    summary = B.workspace_summary(web_workspace)
    assert summary["script_count"] == 3
    assert summary["invalid_count"] == 1
    assert summary["movie_count"] == 1
    assert summary["asset_count"] == 2
    assert len(summary["drafts"]) == 1


@pytest.mark.parametrize("num,expected", [
    (0, "0 B"),
    (999, "999 B"),
    (1536, "1.5 KB"),
    (20 * 1024 * 1024, "20.0 MB"),
    (3 * 1024 ** 3, "3.0 GB"),
])
def test_human_size(num, expected):
    assert B.human_size(num) == expected
