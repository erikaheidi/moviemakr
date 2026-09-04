"""End-to-end route tests through Starlette's TestClient.

Skipped when the `web` extra is not installed - the core suite must stay
runnable with nothing but pytest and PyYAML.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="install the 'web' extra to run these")
pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from moviemakr.web import create_app  # noqa: E402


@pytest.fixture
def client(web_workspace):
    return TestClient(create_app(web_workspace))


# --- index -----------------------------------------------------------------


def test_index_lists_every_script(client):
    body = client.get("/").text
    assert "simple" in body
    assert "beach drive" in body        # the YAML name, not the filename
    assert "h3/broken.yaml" in body


def test_index_flags_the_unloadable_script_instead_of_500ing(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "card-bad" in response.text
    assert "1 invalid" in response.text


def test_unparseable_yaml_does_not_500_the_index(client, web_workspace):
    """A syntax error anywhere under scripts/ used to take the whole index out."""
    (web_workspace.scripts_dir / "unparseable.yaml").write_text(
        "this: [is: not: valid: yaml\n  bad indent\n")
    response = client.get("/")
    assert response.status_code == 200
    assert "unparseable.yaml" in response.text
    assert "2 invalid" in response.text


def test_unparseable_yaml_detail_is_422(client, web_workspace):
    (web_workspace.scripts_dir / "unparseable.yaml").write_text(
        "this: [is: not: valid: yaml\n  bad indent\n")
    response = client.get("/scripts/unparseable.yaml")
    assert response.status_code == 422
    assert "invalid YAML" in response.text


def test_unparseable_yaml_raw_still_serves(client, web_workspace):
    """You have to be able to read the file to see what you broke."""
    (web_workspace.scripts_dir / "unparseable.yaml").write_text("a: [1, 2\n")
    response = client.get("/scripts/unparseable.yaml/raw")
    assert response.status_code == 200
    assert "a: [1, 2" in response.text


def test_empty_workspace_renders(workspace):
    """A brand new workspace with no subdirectories at all."""
    response = TestClient(create_app(workspace)).get("/")
    assert response.status_code == 200
    assert "No scripts" in response.text


# --- script detail ---------------------------------------------------------


def test_script_detail(client):
    response = client.get("/scripts/simple.yaml")
    assert response.status_code == 200
    assert "opening" in response.text
    assert "Scene one." in response.text
    assert "/scripts/simple.yaml/movie" in response.text


def test_nested_script_key(client):
    assert client.get("/scripts/h3/beach.yaml").status_code == 200


def test_missing_script_is_404(client):
    assert client.get("/scripts/nope.yaml").status_code == 404


def test_unloadable_script_is_422(client):
    response = client.get("/scripts/h3/broken.yaml")
    assert response.status_code == 422
    assert "does-not-exist.jpg" in response.text


def test_raw_yaml(client):
    response = client.get("/scripts/h3/beach.yaml/raw")
    assert response.status_code == 200
    assert "name: beach drive" in response.text
    assert response.headers["content-type"].startswith("text/plain")


def test_raw_is_not_swallowed_by_the_catch_all(client):
    """`{key:path}` is greedy; /raw must still route to the raw handler."""
    assert "<html" not in client.get("/scripts/h3/beach.yaml/raw").text


def test_scenes_fragment_is_a_bare_fragment(client):
    """The progress bar and the table, with no page around them."""
    response = client.get("/scripts/simple.yaml/scenes")
    assert response.status_code == 200
    assert "<table" in response.text
    assert "progress-track" in response.text
    assert "<html" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_the_script_page_shows_how_far_the_render_got(client):
    body = client.get("/scripts/simple.yaml").text
    assert "1/2</b> rendered" in body
    assert "width: 50%" in body


def test_an_idle_run_says_so_and_does_not_poll_on_load(client):
    body = client.get("/scripts/simple.yaml").text
    assert "no render running" in body
    assert 'id="live-toggle" checked' not in body


def test_a_live_render_is_named_and_starts_the_page_polling(client, web_workspace):
    """A log being written right now, for a scene with no clip yet."""
    logs = web_workspace.renders_dir / "simple" / "logs"
    (logs / "002-middle.attempt1.log").write_text("  |===>  | 3/20 - 175.60s/it\n")

    body = client.get("/scripts/simple.yaml").text
    assert "rendering <b>2. middle</b>" in body
    assert "pass 3/20" in body
    assert "pill-rendering" in body
    assert 'id="live-toggle" checked' in body   # the live toggle, pre-armed

    # And the polled fragment carries the same thing, so it keeps up to date.
    assert "pass 3/20" in client.get("/scripts/simple.yaml/scenes").text


# --- media -----------------------------------------------------------------


def test_movie_is_served_inline(client):
    response = client.get("/scripts/simple.yaml/movie")
    assert response.status_code == 200
    assert response.content == b"movie-bytes"
    assert "content-disposition" not in response.headers


def test_movie_download_sets_a_filename(client):
    response = client.get("/scripts/simple.yaml/movie?download=1")
    assert "attachment" in response.headers["content-disposition"]
    assert "simple.mp4" in response.headers["content-disposition"]


def test_movie_supports_range_requests(client):
    """iOS cannot scrub a video without this."""
    response = client.get("/scripts/simple.yaml/movie", headers={"Range": "bytes=0-3"})
    assert response.status_code == 206
    assert response.content == b"movi"
    assert response.headers["content-range"] == "bytes 0-3/11"


def test_absent_movie_is_404(client):
    assert client.get("/scripts/h3/beach.yaml/movie").status_code == 404


def test_scene_clip(client):
    response = client.get("/scripts/simple.yaml/clip/001-opening.webm")
    assert response.status_code == 200
    assert response.content == b"clip-bytes"


def test_log_is_plain_text(client):
    response = client.get("/scripts/simple.yaml/log/001-opening.attempt1.log")
    assert response.status_code == 200
    assert "rendering" in response.text


# --- path traversal --------------------------------------------------------


@pytest.mark.parametrize("url", [
    "/scripts/simple.yaml/clip/..%2F..%2F..%2Fetc%2Fpasswd",
    "/scripts/simple.yaml/log/..%2F..%2Fstate.json",
    "/assets/file/..%2F..%2Fetc%2Fpasswd",
    "/assets/thumb/..%2F..%2Fetc%2Fpasswd",
    "/drafts/..%2F..%2Fetc%2Fpasswd",
])
def test_traversal_is_refused(client, url):
    assert client.get(url).status_code in (400, 404)


def test_traversal_never_returns_file_content(client):
    response = client.get("/assets/file/..%2F..%2F..%2Fetc%2Fpasswd")
    assert b"root:" not in response.content


# --- drafts ----------------------------------------------------------------


def test_drafts_index(client):
    body = client.get("/drafts").text
    assert "Beach picnic" in body
    assert "h3-prompt-writing" in body      # the explanation of what a draft is


def test_draft_detail_shows_the_expand_command(client):
    body = client.get("/drafts/picnic").text
    assert "sandwich" in body
    assert "drafts/picnic.md" in body
    assert "h3-prompt-writing" in body


def test_missing_draft_is_404(client):
    assert client.get("/drafts/nope").status_code == 404


def test_create_draft(client, web_workspace):
    response = client.post("/drafts", data={"title": "Josy At The Beach"},
                           follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/drafts/josy-at-the-beach"
    path = web_workspace.drafts_dir / "josy-at-the-beach.md"
    assert path.read_text() == "# Josy At The Beach\n\n"


def test_creating_an_existing_draft_opens_it_without_clobbering(client, web_workspace):
    before = (web_workspace.drafts_dir / "picnic.md").read_text()
    client.post("/drafts", data={"title": "picnic"}, follow_redirects=False)
    assert (web_workspace.drafts_dir / "picnic.md").read_text() == before


def test_save_draft_normalises_crlf(client, web_workspace):
    response = client.post("/drafts/picnic", data={"text": "line one\r\nline two\r\n"},
                           follow_redirects=False)
    assert response.status_code == 303
    assert (web_workspace.drafts_dir / "picnic.md").read_text() == "line one\nline two\n"


def test_delete_draft(client, web_workspace):
    client.post("/drafts/picnic/delete", follow_redirects=False)
    assert not (web_workspace.drafts_dir / "picnic.md").exists()


def test_draft_save_cannot_escape_the_drafts_dir(client, web_workspace):
    response = client.post("/drafts/..%2F..%2Fpwned", data={"text": "x"},
                           follow_redirects=False)
    assert response.status_code in (400, 404)
    assert not (web_workspace.root.parent / "pwned.md").exists()


# --- assets ----------------------------------------------------------------


def test_assets_index(client):
    body = client.get("/assets").text
    assert "josy-reference.jpg" in body
    assert "used by 1" in body
    assert "unused" in body


def test_asset_download(client):
    response = client.get("/assets/file/josy-reference.jpg")
    assert response.status_code == 200
    assert response.content == b"anchor-bytes"


def test_upload(client, web_workspace, monkeypatch):
    from moviemakr.web import assets as A

    monkeypatch.setattr(A, "image_size", lambda path: (800, 600))
    response = client.post(
        "/assets",
        files=[("files", ("New Photo.jpg", b"jpeg-bytes", "image/jpeg"))],
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert (web_workspace.assets_dir / "new-photo.jpg").read_bytes() == b"jpeg-bytes"


def test_upload_rejection_is_reported_not_raised(client, web_workspace):
    response = client.post(
        "/assets",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    assert not (web_workspace.assets_dir / "notes.txt").exists()


def test_upload_location_header_is_encoded(client, monkeypatch):
    """Filenames land in a Location header; a raw space there is invalid."""
    from moviemakr.web import assets as A

    monkeypatch.setattr(A, "image_size", lambda path: (800, 600))
    response = client.post(
        "/assets",
        files=[("files", ("a.jpg", b"x", "image/jpeg"))],
        follow_redirects=False,
    )
    assert " " not in response.headers["location"]


# --- script upload ---------------------------------------------------------


@pytest.fixture
def script_bytes(base_script):
    import yaml
    return yaml.safe_dump(base_script, sort_keys=False).encode()


def test_index_offers_the_upload_form(client):
    body = client.get("/").text
    assert 'action="/scripts"' in body
    assert 'name="folder"' in body
    assert 'value="h3"' in body          # the datalist of existing folders


def test_script_upload(client, web_workspace, script_bytes):
    response = client.post(
        "/scripts",
        files=[("files", ("Josy Beach Drive.yaml", script_bytes, "text/yaml"))],
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert (web_workspace.scripts_dir / "josy-beach-drive.yaml").is_file()


def test_script_upload_into_a_folder(client, web_workspace, script_bytes):
    client.post(
        "/scripts",
        data={"folder": "h3"},
        files=[("files", ("kitchen.yaml", script_bytes, "text/yaml"))],
        follow_redirects=False,
    )
    assert (web_workspace.scripts_dir / "h3" / "kitchen.yaml").is_file()


def test_uploaded_script_appears_on_the_index(client, web_workspace, script_bytes):
    client.post("/scripts", files=[("files", ("kitchen.yaml", script_bytes, "text/yaml"))],
                follow_redirects=False)
    assert client.get("/scripts/kitchen.yaml").status_code == 200


def test_script_upload_rejection_is_reported_not_raised(client, web_workspace):
    response = client.post(
        "/scripts",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    assert not (web_workspace.scripts_dir / "notes.txt").exists()


def test_script_upload_will_not_clobber_without_replace(client, web_workspace, script_bytes):
    before = (web_workspace.scripts_dir / "simple.yaml").read_text()
    response = client.post(
        "/scripts",
        files=[("files", ("simple.yaml", script_bytes, "text/yaml"))],
        follow_redirects=False,
    )
    assert "error=" in response.headers["location"]
    assert (web_workspace.scripts_dir / "simple.yaml").read_text() == before


def test_script_upload_replaces_when_asked(client, web_workspace, script_bytes):
    response = client.post(
        "/scripts",
        data={"replace": "true"},
        files=[("files", ("simple.yaml", script_bytes, "text/yaml"))],
        follow_redirects=False,
    )
    assert "uploaded=" in response.headers["location"]
    assert "test-movie" in (web_workspace.scripts_dir / "simple.yaml").read_text()


def test_a_script_that_will_not_load_is_stored_with_a_warning(client, web_workspace,
                                                              base_script):
    """The refs are uploaded separately, so the script often arrives first."""
    import yaml

    base_script["continuity"] = {"anchors": ["not-here-yet.jpg"]}
    data = yaml.safe_dump(base_script, sort_keys=False).encode()
    response = client.post(
        "/scripts",
        files=[("files", ("kitchen.yaml", data, "text/yaml"))],
        follow_redirects=False,
    )
    location = response.headers["location"]
    assert "warning=" in location and "error=" not in location
    assert (web_workspace.scripts_dir / "kitchen.yaml").is_file()


def test_script_upload_cannot_escape_the_scripts_dir(client, web_workspace, script_bytes):
    response = client.post(
        "/scripts",
        data={"folder": "../../pwned"},
        files=[("files", ("evil.yaml", script_bytes, "text/yaml"))],
        follow_redirects=False,
    )
    assert "error=" in response.headers["location"]
    assert not (web_workspace.root.parent / "pwned").exists()


def test_upload_location_header_is_encoded_for_scripts(client, script_bytes):
    response = client.post(
        "/scripts",
        files=[("files", ("a.yaml", script_bytes, "text/yaml"))],
        follow_redirects=False,
    )
    assert " " not in response.headers["location"]


def test_raw_yaml_download_sets_a_filename(client):
    response = client.get("/scripts/h3/beach.yaml/raw?download=1")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert "beach.yaml" in response.headers["content-disposition"]
    assert "name: beach drive" in response.text
