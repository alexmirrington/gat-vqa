"""Common schema definitions."""

from schema import And, Schema

SLICE = Schema(
    And(lambda s: slice(*s.split(":")), lambda s: 1 <= len(s.split(":")) <= 2)
)
