"""`safe_path` is the security boundary: the server hands out workspace files.

Stdlib-only, so these run whether or not the `web` extra is installed.
"""

from __future__ import annotations

import pytest

from moviemakr.web.paths import UnsafePath, rel_key, safe_path, safe_stem


@pytest.fixture
def base(tmp_path):
    root = tmp_path / "scripts"
    (root / "h3").mkdir(parents=True)
    (root / "h3" / "beach.yaml").write_text("name: beach\n")
    return root


def test_resolves_a_nested_key(base):
    assert safe_path(base, "h3/beach.yaml") == base / "h3" / "beach.yaml"


def test_allows_a_path_that_does_not_exist_yet(base):
    """Drafts are written through this: the file appears after validation."""
    assert safe_path(base, "new.md") == base / "new.md"


@pytest.mark.parametrize("bad", [
    "",
    ".",
    "/",
    "..",
    "../secrets",
    "../../../etc/passwd",
    "h3/../../escape",
    "/etc/passwd",
    "h3/../../../etc/passwd",
])
def test_rejects_traversal(base, bad):
    with pytest.raises(UnsafePath):
        safe_path(base, bad)


def test_rejects_a_symlink_pointing_out_of_the_tree(base, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("nope")
    (base / "link").symlink_to(secret)
    with pytest.raises(UnsafePath):
        safe_path(base, "link")


def test_a_symlink_inside_the_tree_is_fine(base):
    (base / "alias.yaml").symlink_to(base / "h3" / "beach.yaml")
    assert safe_path(base, "alias.yaml") == base / "h3" / "beach.yaml"


def test_rel_key_round_trips(base):
    path = base / "h3" / "beach.yaml"
    assert safe_path(base, rel_key(base, path)) == path


@pytest.mark.parametrize("raw,expected", [
    ("Beach Drive.JPG", "beach-drive"),
    ("IMG_1234.jpeg", "img-1234"),
    ("PXL_20220304_081603820.jpg", "pxl-20220304-081603820"),
    ("../../etc/passwd", "passwd"),
    ("a/b/c.png", "c"),
    ("photo.tar.gz", "photo-tar"),
    ("...", "untitled"),
    ("   ", "untitled"),
    ("", "untitled"),
])
def test_safe_stem(raw, expected):
    assert safe_stem(raw) == expected


def test_safe_stem_never_returns_a_path():
    assert "/" not in safe_stem("../../etc/passwd")
