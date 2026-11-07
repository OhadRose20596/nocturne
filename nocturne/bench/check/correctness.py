"""Correctness of a cell against the baseline record of the same model on the requester host.
exact_match_ratio: share of prompts whose text equals the baseline text.
first_divergence_median: median index of the first differing word (prompt length when identical).
task_accuracy: mean grader score over the prompts that ran and could be graded; prompts whose grader returned None
(code without a sandbox) are listed in `skipped` and excluded from the mean.
quality_pass: task_accuracy >= baseline_task_accuracy - 0.05, both computed on the same prompt ids."""
import statistics

from .graders import grade


def first_divergence(a: str, b: str) -> int:
    wa, wb = (a or "").split(), (b or "").split()
    for i, (x, y) in enumerate(zip(wa, wb)):
        if x != y:
            return i
    return min(len(wa), len(wb)) if len(wa) != len(wb) else len(wa)


def grade_all(outputs: dict, prompts: list) -> tuple[dict, list[str]]:
    """{prompt_id: score or None}, [prompt ids skipped]. Missing output scores 0."""
    scores = {p["id"]: grade(p["category"], outputs.get(p["id"], ""), p["expected"]) for p in prompts}
    return scores, [i for i, v in scores.items() if v is None]


def task_accuracy(outputs: dict, prompts: list) -> float | None:
    scores, _ = grade_all(outputs, prompts)
    vals = [v for v in scores.values() if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def compare(outputs: dict, baseline_outputs: dict, prompts: list, baseline_task_accuracy=None, graded: list | None = None) -> dict:
    """baseline_task_accuracy is recomputed from baseline_outputs on these prompts when they are given, so a subset
    run (A4 on the long prompts) is compared with the baseline on the same ids. `graded` restricts grading to a subset
    of the prompts (fixed-token runs grade nothing and keep only the identity check); identity is over all prompts."""
    if graded is None:
        graded = prompts
    ids = [p["id"] for p in prompts]
    common = [i for i in ids if i in outputs and i in baseline_outputs]
    exact = (sum(1 for i in common if outputs[i] == baseline_outputs[i]) / len(common)) if common else None
    div = statistics.median([first_divergence(outputs[i], baseline_outputs[i]) for i in common]) if common else None
    if baseline_outputs and graded:
        baseline_task_accuracy = task_accuracy(baseline_outputs, graded)
    return {
        "exact_match_ratio": exact,
        "first_divergence_median": div,
        "task_accuracy": task_accuracy(outputs, graded) if graded else None,
        "baseline_task_accuracy": baseline_task_accuracy,
    }


def passes(correctness: dict, tolerance: float = 0.05) -> bool | None:
    """None when there is no baseline or no graded prompt: unknown, not failed."""
    base = correctness.get("baseline_task_accuracy")
    acc = correctness.get("task_accuracy")
    if base is None or acc is None:
        return None
    return acc >= base - tolerance
