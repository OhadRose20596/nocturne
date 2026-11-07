"""Grader: output parses as JSON and matches the expected object key by key."""
from ._common import first_json_object


def grade(output: str, expected) -> float:
    got = first_json_object(output)
    if not isinstance(got, dict) or not isinstance(expected, dict):
        return 0.0
    if not expected:
        return 1.0 if got == expected else 0.0
    hits = sum(1 for k, v in expected.items() if k in got and got[k] == v)
    extra = len(set(got) - set(expected))
    return max(0.0, (hits - extra) / len(expected))
