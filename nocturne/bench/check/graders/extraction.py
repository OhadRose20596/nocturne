"""Grader: normalized output equals the expected span, or contains it as a short answer."""
from ._common import normalize


def grade(output: str, expected) -> float:
    got, want = normalize(output), normalize(str(expected))
    if got == want:
        return 1.0
    # Accept a short answer that wraps the span, e.g. "The answer is Dana Levi."
    if want in got and len(got) <= 3 * len(want) + 20:
        return 1.0
    return 0.0
