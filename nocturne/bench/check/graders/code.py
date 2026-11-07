"""Grader: run the candidate code plus each assert in one isolated subprocess.
expected = {"tests": [assert statements]}. Score = passed / total. Timeout (5 s) or crash = 0.
Isolation: `python -I`, no network. macOS: `sandbox-exec` with a deny-network profile. Linux: `unshare -rn`.
When neither works the category is not run: grade() returns None and the run notes `skipped_unsandboxed`."""
import platform
import secrets
import shutil
import subprocess
import sys

from ._common import strip_code_fences

# Verdicts go to stderr as "<nonce> <index> PASS|FAIL". The candidate owns stdout and cannot know the nonce, so a
# solution that prints "PASS" scores nothing (I024). An assert that fails raises AssertionError and lands in FAIL.
HARNESS = r'''
import sys
src = sys.stdin.read()
ns = {}
def verdict(i, ok):
    sys.stderr.write(NONCE + " " + str(i) + (" PASS" if ok else " FAIL") + "\n")
try:
    exec(compile(src, "<candidate>", "exec"), ns)
except BaseException:
    for i in range(len(TESTS)):
        verdict(i, False)
    sys.exit(0)
for i, t in enumerate(TESTS):
    try:
        exec(compile(t, "<test>", "exec"), ns)
        verdict(i, True)
    except BaseException:
        verdict(i, False)
'''
SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)"
SKIPPED = "skipped_unsandboxed"
_prefix: list[str] | None | bool = False        # False = not probed yet


def sandbox_prefix(system: str | None = None, run=subprocess.run) -> list[str] | None:
    """Command prefix that denies network, or None when no sandbox is available on this host."""
    system = system or platform.system()
    cands = []
    if system == "Darwin" and shutil.which("sandbox-exec"):
        cands.append(["sandbox-exec", "-p", SANDBOX_PROFILE])
    if system == "Linux" and shutil.which("unshare"):
        cands.append(["unshare", "-rn"])
    for c in cands:
        try:
            r = run(c + [sys.executable, "-I", "-c", "pass"], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            return c
    return None


def grade(output: str, expected, timeout_s: float = 5.0) -> float | None:
    global _prefix
    tests = list(expected.get("tests", []))
    if not tests:
        return 0.0
    if _prefix is False:
        _prefix = sandbox_prefix()
    if _prefix is None:
        return None
    code = strip_code_fences(output)
    nonce = secrets.token_hex(8)
    prog = "TESTS = " + repr(tests) + "\nNONCE = " + repr(nonce) + "\n" + HARNESS
    try:
        r = subprocess.run(_prefix + [sys.executable, "-I", "-c", prog],
                           input=code, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return 0.0
    passed = set()
    for line in (r.stderr or "").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == nonce and parts[2] == "PASS" and parts[1].isdigit():
            passed.add(int(parts[1]))
    return min(1.0, len(passed & set(range(len(tests)))) / len(tests))
