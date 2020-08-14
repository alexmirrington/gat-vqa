"""Common schema definitions."""

from schema import And, Schema

SLICE_SCHEMA = Schema(
    And(lambda s: slice(*s.split(":")), lambda s: 1 <= len(s.split(":")) <= 2)
)
