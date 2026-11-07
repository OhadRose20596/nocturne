"""Grader: the last number in the output equals the expected number."""
from ._common import last_number


def grade(output: str, expected) -> float:
    got = last_number(output)
    if got is None:
        return 0.0
    return 1.0 if abs(got - float(expected)) < 1e-6 else 0.0
