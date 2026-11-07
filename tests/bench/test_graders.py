import json

from nocturne.bench.check import correctness
from nocturne.bench.check.graders import GRADERS, grade
from nocturne.bench import prompts


def test_eval_set_shape():
    rows = prompts.load()
    assert len(rows) == 33
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    for r in rows:
        assert set(r) == {"id", "category", "prompt", "expected", "max_tokens"}
        assert r["category"] in GRADERS
        assert r["max_tokens"] > 0
    by_cat = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    assert by_cat["long_structured"] == 3
    assert all(v == 6 for k, v in by_cat.items() if k != "long_structured")


def test_subset_and_categories():
    assert len(prompts.load(subset=10)) == 10
    assert all(r["category"] == "code" for r in prompts.load(categories=["code"]))


def test_arithmetic():
    assert grade("arithmetic", "936", 936) == 1.0
    assert grade("arithmetic", "The answer is 1,250.", 1250) == 1.0
    assert grade("arithmetic", "937", 936) == 0.0
    assert grade("arithmetic", "no idea", 936) == 0.0


def test_extraction():
    assert grade("extraction", "Dana Levi", "Dana Levi") == 1.0
    assert grade("extraction", "The order went to Dana Levi.", "Dana Levi") == 1.0
    assert grade("extraction", "Dana", "Dana Levi") == 0.0
    long = "Dana Levi " + "x" * 200
    assert grade("extraction", long, "Dana Levi") == 0.0


def test_json_format():
    exp = {"name": "Alice", "age": 30}
    assert grade("json_format", '{"name": "Alice", "age": 30}', exp) == 1.0
    assert grade("json_format", 'Here:\n```json\n{"name": "Alice", "age": 30}\n```', exp) == 1.0
    assert grade("json_format", '{"name": "Alice", "age": 31}', exp) == 0.5
    assert grade("json_format", '{"name": "Alice", "age": 30, "x": 1}', exp) == 0.5
    assert grade("json_format", "not json", exp) == 0.0
    assert grade("json_format", "[1, 2]", exp) == 0.0


def _needs_sandbox():
    from nocturne.bench.check.graders import code as code_grader
    if code_grader.sandbox_prefix() is None:
        import pytest
        pytest.skip("no sandbox-exec or unshare on this host; code grader returns None here")


def test_code_pass_fail_timeout():
    _needs_sandbox()
    tests = {"tests": ["assert is_even(4) is True", "assert is_even(7) is False", "assert is_even(0) is True"]}
    good = "```python\ndef is_even(n):\n    return n % 2 == 0\n```"
    assert grade("code", good, tests) == 1.0
    partial = "def is_even(n):\n    return True\n"
    assert abs(grade("code", partial, tests) - 2 / 3) < 1e-9
    assert grade("code", "def is_even(n): raise ValueError", tests) == 0.0
    assert grade("code", "this is not python", tests) == 0.0
    from nocturne.bench.check.graders import code as code_grader
    loop = "import time\ndef is_even(n):\n    time.sleep(5)\n"
    assert code_grader.grade(loop, tests, timeout_s=0.5) == 0.0


def test_code_cannot_fake_pass_and_caps_at_one():
    _needs_sandbox()
    tests = {"tests": ["assert False"]}
    # I024: a candidate that prints PASS used to score 1 although the only test is `assert False`
    assert grade("code", 'print("PASS")\nprint("PASS")\n', tests) == 0.0
    assert grade("code", 'import sys\nsys.stderr.write("PASS\\n")\n', tests) == 0.0
    two = {"tests": ["assert f() == 1", "assert f() == 1"]}
    assert grade("code", "def f():\n    return 1\n", two) == 1.0
    # a test that exits the interpreter counts as failed, not as a crash of the grader
    assert grade("code", "def f():\n    return 1\n", {"tests": ["raise SystemExit(0)", "assert f() == 1"]}) == 0.5


def test_code_is_isolated_from_cwd(tmp_path, monkeypatch):
    _needs_sandbox()
    # A planted json.py in cwd must not be importable by the candidate (python -I).
    (tmp_path / "json.py").write_text("raise RuntimeError('planted')")
    monkeypatch.chdir(tmp_path)
    tests = {"tests": ["assert f() == 1"]}
    assert grade("code", "import json\ndef f():\n    return 1\n", tests) == 1.0


def test_factual():
    assert grade("factual", "Tokyo", ["Tokyo"]) == 1.0
    assert grade("factual", "There are 8 planets.", ["8", "eight"]) == 1.0
    assert grade("factual", "Eight", ["8", "eight"]) == 1.0
    assert grade("factual", "Kyoto", ["Tokyo"]) == 0.0


def test_long_structured():
    exp = {"format": "numbered", "min_items": 3, "max_items": 3, "must_contain": ["password"]}
    out = "1. Plug in.\n2. Set the password.\n3. Done."
    assert grade("long_structured", out, exp) == 1.0
    assert grade("long_structured", "1. Plug in.\n2. Set the password.", exp) == 2 / 3
    bullets = {"format": "bullets", "min_items": 2, "must_contain": []}
    assert grade("long_structured", "- a\n- b\n", bullets) == 1.0
    table = {"format": "table", "min_items": 2, "max_items": 2, "must_contain": ["Rust"]}
    md = "| Lang | Typing |\n|---|---|\n| Rust | static |\n| Go | static |"
    assert grade("long_structured", md, table) == 1.0


def test_graders_are_deterministic():
    for r in prompts.load():
        a = grade(r["category"], "some output 42", r["expected"])
        b = grade(r["category"], "some output 42", r["expected"])
        assert a == b and 0.0 <= a <= 1.0


def test_unknown_category():
    import pytest
    with pytest.raises(KeyError):
        grade("poetry", "x", None)


def test_first_divergence():
    assert correctness.first_divergence("a b c", "a b c") == 3
    assert correctness.first_divergence("a b c", "a x c") == 1
    assert correctness.first_divergence("a b", "a b c") == 2
    assert correctness.first_divergence("", "") == 0


def test_compare_and_passes():
    rows = [
        {"id": "p1", "category": "arithmetic", "prompt": "", "expected": 2, "max_tokens": 4},
        {"id": "p2", "category": "factual", "prompt": "", "expected": ["Tokyo"], "max_tokens": 4},
    ]
    out = {"p1": "2", "p2": "Kyoto"}
    base = {"p1": "2", "p2": "Tokyo"}
    c = correctness.compare(out, base, rows, baseline_task_accuracy=1.0)
    assert c["exact_match_ratio"] == 0.5
    assert c["first_divergence_median"] == 0.5
    assert c["task_accuracy"] == 0.5
    assert not correctness.passes(c)
    c2 = correctness.compare(base, base, rows, baseline_task_accuracy=1.0)
    assert c2["exact_match_ratio"] == 1.0 and correctness.passes(c2)
    assert not correctness.passes({"task_accuracy": 1.0, "baseline_task_accuracy": None})
