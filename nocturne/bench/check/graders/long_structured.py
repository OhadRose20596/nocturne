"""Grader: structure checks on long output. Score = fraction of checks passed.
expected = {"format": numbered|bullets|table, "min_items", "max_items", "must_contain": [...]}"""
import re

NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
BULLET = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}", re.MULTILINE)


def count_items(output: str, fmt: str) -> int:
    if fmt == "numbered":
        return len(NUMBERED.findall(output))
    if fmt == "bullets":
        return len(BULLET.findall(output))
    if fmt == "table":
        rows = [r for r in TABLE_ROW.findall(output) if not TABLE_SEP.match(r)]
        return max(0, len(rows) - 1)  # minus the header row
    return 0


def grade(output: str, expected) -> float:
    checks = []
    n = count_items(output, expected.get("format", ""))
    checks.append(n >= int(expected.get("min_items", 1)))
    if "max_items" in expected:
        checks.append(n <= int(expected["max_items"]))
    low = output.lower()
    for word in expected.get("must_contain", []):
        checks.append(word.lower() in low)
    return sum(checks) / len(checks) if checks else 0.0
