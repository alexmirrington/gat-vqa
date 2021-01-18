"""Tests for custom (de)serialisation functions."""
import os.path
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

import pytest

from gat_vqa.utilities.serialisation import path_deserializer, path_serializer


@pytest.fixture(scope="module", name="jsons")
def fixture_jsons_path():
    """Import jsons and register `pathlib.Path` (de)serialisers."""
    import jsons  # pylint: disable=import-outside-toplevel

    jsons.set_deserializer(path_deserializer, PurePath)
    jsons.set_serializer(path_serializer, PurePath)

    return jsons


def test_dump_singlepart_relative_path(jsons):
    """Test `jsons.dump` on a single part relative path."""
    assert jsons.dump(Path("abc")) == "abc"


def test_dump_singlepart_absolute_path(jsons):
    """Test `jsons.dump` on a single part absolute path."""
    assert jsons.dump(Path("abc").resolve()) == os.path.abspath("abc")


def test_dump_singlepart_pure_windows_path(jsons):
    """Test `jsons.dump` on a single part Windows path."""
    assert jsons.dump(PureWindowsPath("abc")) == "abc"


def test_dump_singlepart_pure_posix_path(jsons):
    """Test `jsons.dump` on a single part Posix path."""
    assert jsons.dump(PurePosixPath("abc")) == "abc"


def test_dump_multipart_relative_path(jsons):
    """Test `jsons.dump` on a multi-part relative path."""
    assert jsons.dump(Path("abc", "def", "ghi")) == "abc/def/ghi"
    assert jsons.dump(Path("abc/def/ghi")) == "abc/def/ghi"


def test_dump_multipart_absolute_path(jsons):
    """Test `jsons.dump` on a multi-part absolute path."""
    assert jsons.dump(Path("abc", "def", "ghi").resolve()) == os.path.abspath(
        "abc/def/ghi"
    )
    assert jsons.dump(Path("abc/def/ghi").resolve()) == os.path.abspath("abc/def/ghi")


def test_dump_multipart_pure_windows_path(jsons):
    """Test `jsons.dump` on a multi-part Windows path."""
    assert jsons.dump(PureWindowsPath("abc", "def", "ghi")) == "abc/def/ghi"
    assert jsons.dump(PureWindowsPath("abc/def/ghi")) == "abc/def/ghi"
    assert jsons.dump(PureWindowsPath("abc\\def\\ghi")) == "abc/def/ghi"


def test_dump_multipart_pure_posix_path(jsons):
    """Test `jsons.dump` on a multi-part Posix path."""
    assert jsons.dump(PurePosixPath("abc", "def", "ghi")) == "abc/def/ghi"
    assert jsons.dump(PurePosixPath("abc/def/ghi")) == "abc/def/ghi"
    assert jsons.dump(PurePosixPath("abc\\def\\ghi")) == "abc\\def\\ghi"


def test_dump_multipart_drived_pure_windows_path(jsons):
    """Test `jsons.dump` on a drived multi-part Windows path."""
    assert jsons.dump(PureWindowsPath("Z:\\", "abc", "def", "ghi")) == "Z:/abc/def/ghi"
    assert jsons.dump(PureWindowsPath("Z:/abc/def/ghi")) == "Z:/abc/def/ghi"
    assert jsons.dump(PureWindowsPath("Z:\\abc\\def\\ghi")) == "Z:/abc/def/ghi"


def test_dump_multipart_drived_pure_posix_path(jsons):
    """Test `jsons.dump` on a drived multi-part Posix path."""
    assert jsons.dump(PurePosixPath("Z:", "abc", "def", "ghi")) == "Z:/abc/def/ghi"
    assert jsons.dump(PurePosixPath("Z:/abc/def/ghi")) == "Z:/abc/def/ghi"
    assert jsons.dump(PurePosixPath("Z:\\abc\\def\\ghi")) == "Z:\\abc\\def\\ghi"


def test_load_singlepart_relative_path(jsons):
    """Test `jsons.load` on a single part relative path."""
    assert jsons.load("abc", Path) == Path("abc")


def test_load_singlepart_pure_windows_path(jsons):
    """Test `jsons.load` on a single part Windows path."""
    assert jsons.load("abc", PureWindowsPath) == PureWindowsPath("abc")


def test_load_singlepart_pure_posix_path(jsons):
    """Test `jsons.load` on a single part Posix path."""
    assert jsons.load("abc", PurePosixPath) == PurePosixPath("abc")


def test_load_multipart_relative_path(jsons):
    """Test `jsons.load` on a multi-part relative path."""
    assert jsons.load("abc/def/ghi", Path) == Path("abc", "def", "ghi")
    assert jsons.load("abc/def/ghi", Path) == Path("abc/def/ghi")


def test_load_multipart_pure_windows_path(jsons):
    """Test `jsons.load` on a multi-part Windows path."""
    # We should be able to load Posix-style paths on Windows.
    assert jsons.load("abc/def/ghi", PureWindowsPath) == PureWindowsPath(
        "abc", "def", "ghi"
    )
    assert jsons.load("abc/def/ghi", PureWindowsPath) == PureWindowsPath("abc/def/ghi")
    assert jsons.load("abc/def/ghi", PureWindowsPath) == PureWindowsPath(
        "abc\\def\\ghi"
    )
    # We should be able to load Windows-style paths on Windows.
    assert jsons.load("abc\\def\\ghi", PureWindowsPath) == PureWindowsPath(
        "abc", "def", "ghi"
    )
    assert jsons.load("abc\\def\\ghi", PureWindowsPath) == PureWindowsPath(
        "abc/def/ghi"
    )
    assert jsons.load("abc\\def\\ghi", PureWindowsPath) == PureWindowsPath(
        "abc\\def\\ghi"
    )


def test_load_multipart_pure_posix_path(jsons):
    """Test `jsons.load` on a multi-part Posix path."""
    # We should be able to load Posix-style paths on Posix systems.
    assert jsons.load("abc/def/ghi", PurePosixPath) == PurePosixPath(
        "abc", "def", "ghi"
    )
    assert jsons.load("abc/def/ghi", PurePosixPath) == PurePosixPath("abc/def/ghi")
    assert jsons.load("abc/def/ghi", PurePosixPath) != PurePosixPath("abc\\def\\ghi")

    # Backslashes on Posix systems should be interpreted as escapes.
    assert jsons.load("abc\\def\\ghi", PurePosixPath) != PurePosixPath(
        "abc", "def", "ghi"
    )
    assert jsons.load("abc\\def\\ghi", PurePosixPath) != PurePosixPath("abc/def/ghi")
    assert jsons.load("abc\\def\\ghi", PurePosixPath) == PurePosixPath("abc\\def\\ghi")


def test_load_multipart_drived_pure_windows_path(jsons):
    """Test `jsons.load` on a drived multi-part Windows path."""
    # We should be able to load Posix-style paths on Windows.
    assert jsons.load("Z:/abc/def/ghi", PureWindowsPath) == PureWindowsPath(
        "Z:\\", "abc", "def", "ghi"
    )
    assert jsons.load("Z:/abc/def/ghi", PureWindowsPath) == PureWindowsPath(
        "Z:/abc/def/ghi"
    )
    assert jsons.load("Z:/abc/def/ghi", PureWindowsPath) == PureWindowsPath(
        "Z:\\abc\\def\\ghi"
    )

    # We should be able to load Windows-style paths on Windows.
    assert jsons.load("Z:\\abc\\def\\ghi", PureWindowsPath) == PureWindowsPath(
        "Z:\\", "abc", "def", "ghi"
    )
    assert jsons.load("Z:\\abc\\def\\ghi", PureWindowsPath) == PureWindowsPath(
        "Z:/abc/def/ghi"
    )
    assert jsons.load("Z:\\abc\\def\\ghi", PureWindowsPath) == PureWindowsPath(
        "Z:\\abc\\def\\ghi"
    )


def test_load_multipart_drived_pure_posix_path(jsons):
    """Test `jsons.load` on a drived multi-part Posix path."""
    # We should be able to load Posix-style paths on Windows.
    assert jsons.load("Z:/abc/def/ghi", PurePosixPath) == PurePosixPath(
        "Z:", "abc", "def", "ghi"
    )
    assert jsons.load("Z:/abc/def/ghi", PurePosixPath) == PurePosixPath(
        "Z:/abc/def/ghi"
    )
    assert jsons.load("Z:/abc/def/ghi", PurePosixPath) != PurePosixPath(
        "Z:\\abc\\def\\ghi"
    )

    # Backslashes on Posix systems should be interpreted as escapes.
    assert jsons.load("Z:\\abc\\def\\ghi", PurePosixPath) != PurePosixPath(
        "Z:", "abc", "def", "ghi"
    )
    assert jsons.load("Z:\\abc\\def\\ghi", PurePosixPath) != PurePosixPath(
        "Z:/abc/def/ghi"
    )
    assert jsons.load("Z:\\abc\\def\\ghi", PurePosixPath) == PurePosixPath(
        "Z:\\abc\\def\\ghi"
    )


def test_dump_posix_load_windows(jsons):
    """Test `jsons.dump` on a Posix system and `jsons.load` on a Windows system."""
    dump_result = jsons.dump(PurePosixPath("abc", "def", "ghi"))
    assert dump_result == "abc/def/ghi"
    load_result = jsons.load(dump_result, PureWindowsPath)
    assert load_result == PureWindowsPath("abc", "def", "ghi")


def test_dump_windows_load_posix(jsons):
    """Test `jsons.dump` on a Windows system and `jsons.load` on a Posix system."""
    dump_result = jsons.dump(PureWindowsPath("abc", "def", "ghi"))
    assert dump_result == "abc/def/ghi"
    load_result = jsons.load(dump_result, PurePosixPath)
    assert load_result == PurePosixPath("abc", "def", "ghi")


def test_dump_posix_load_posix(jsons):
    """Test `jsons.dump` and `jsons.load` on a Posix system."""
    dump_result = jsons.dump(PurePosixPath("abc", "def", "ghi"))
    assert dump_result == "abc/def/ghi"
    load_result = jsons.load(dump_result, PurePosixPath)
    assert load_result == PurePosixPath("abc", "def", "ghi")


def test_dump_windows_load_windows(jsons):
    """Test `jsons.dump` and `jsons.load` on a Windows system."""
    dump_result = jsons.dump(PureWindowsPath("abc", "def", "ghi"))
    assert dump_result == "abc/def/ghi"
    load_result = jsons.load(dump_result, PureWindowsPath)
    assert load_result == PureWindowsPath("abc", "def", "ghi")
