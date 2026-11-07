"""Grader: any accepted answer appears in the normalized output."""
from ._common import normalize


def grade(output: str, expected) -> float:
    got = normalize(output)
    answers = expected if isinstance(expected, list) else [expected]
    return 1.0 if any(normalize(str(a)) in got for a in answers) else 0.0
