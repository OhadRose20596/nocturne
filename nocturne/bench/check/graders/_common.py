"""Shared helpers for graders. Pure string functions, no I/O."""
import json
import re

CODE_BLOCK = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s]+", " ", s)
    return s.strip(" .!?:;\"'`*")


def strip_code_fences(text: str) -> str:
    m = CODE_BLOCK.search(text)
    return m.group(1) if m else text


def last_number(text: str):
    nums = NUMBER.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def first_json_object(text: str):
    """Return the first parseable JSON object or array in text, else None."""
    text = strip_code_fences(text)
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None
