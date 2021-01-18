"""Custom serialisation and deserialisation functions."""
from pathlib import PurePath
from typing import Any


def path_deserializer(obj: str, cls: type = PurePath, **_: Any) -> PurePath:
    """Deserialize a string to a `pathlib.PurePath` object.

    Since `pathlib` implements `PurePath`, no filename or existence checks are
    performed.

    Params:
    ------
    `obj`: the string to deserialize.
    `cls`: the class to convert to, a subclass of `PurePath`.

    Returns:
    --------
    A path of type `cls`.
    """
    path: PurePath = cls(obj)
    return path


def path_serializer(obj: PurePath, **_: Any) -> str:
    """Serialize a pathlib.PurePath` object to a `str`, Posix-style.

    Posix-style strings are used as they can be used to create `pathlib.Path`
    objects on both Posix and Windows systems, but Windows-style strings can
    only be used to create valid `pathlib.Path` objects on Windows.

    Params:
    ------
    `obj`: the string to deserialize.

    Returns:
    --------
    A string, the serialised path.
    """
    return obj.as_posix()
