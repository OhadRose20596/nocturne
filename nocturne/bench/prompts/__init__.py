"""Eval set loader. The set version comes from the file: a first line {"set": ..., "version": ...} without an "id"
is a header; otherwise a _v<N> suffix in the file name; otherwise "1" (eval_set.jsonl is the frozen v1)."""
import json
import re
from pathlib import Path

PATH = Path(__file__).with_name("eval_set.jsonl")
SETS = {p.stem: p for p in Path(__file__).parent.glob("*.jsonl")}


def path_for(set_name: str | None) -> Path:
    """prompts/<set>.jsonl; the default set when None."""
    if not set_name:
        return PATH
    if set_name not in SETS:
        raise KeyError(f"unknown prompt set {set_name!r}; known: {', '.join(sorted(SETS))}")
    return SETS[set_name]


def _rows_and_meta(path: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    meta = {"set": path.stem.split("_v")[0], "version": None}
    if rows and "id" not in rows[0] and "version" in rows[0]:
        meta.update({k: str(v) for k, v in rows[0].items()})
        rows = rows[1:]
    if meta["version"] is None:
        m = re.search(r"_v(\d+)$", path.stem)
        meta["version"] = m.group(1) if m else "1"
    return rows, meta


SET_NAME, SET_VERSION = _rows_and_meta(PATH)[1]["set"], _rows_and_meta(PATH)[1]["version"]


def load(subset: int | None = None, categories: list[str] | None = None, path: Path = PATH) -> list[dict]:
    rows, _ = _rows_and_meta(path)
    if categories:
        rows = [r for r in rows if r["category"] in categories]
    if subset:
        rows = rows[:subset]
    return rows


def meta(path: Path = PATH) -> dict:
    return _rows_and_meta(path)[1]
