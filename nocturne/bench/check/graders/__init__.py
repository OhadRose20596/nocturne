"""One grader per prompt category. grade(output, expected) -> float in [0, 1], or None when the grader could not run
safely (code without a sandbox). Deterministic, offline."""
from . import arithmetic, code, extraction, factual, json_format, long_structured

GRADERS = {
    "arithmetic": arithmetic.grade,
    "extraction": extraction.grade,
    "json_format": json_format.grade,
    "code": code.grade,
    "factual": factual.grade,
    "long_structured": long_structured.grade,
}


def grade(category: str, output: str, expected) -> float | None:
    if category not in GRADERS:
        raise KeyError(f"no grader for category {category!r}")
    v = GRADERS[category](output or "", expected)
    return None if v is None else float(v)
