"""
Tiny test runner shared by all milestone test files, so they run with or
without pytest:

    python -m sw.tests.test_mX_...          (this runner)
    python -m pytest sw/tests               (pytest)

Missing prerequisites (dataset not downloaded, no GPU, ...) are reported as
SKIP, never as a silent PASS.
"""
import os
import traceback


class Skip(Exception):
    pass


def skip(reason: str):
    """Skip the current test under either runner."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        import pytest
        pytest.skip(reason)
    raise Skip(reason)


def need(path):
    """Skip the current test if a required file is missing."""
    if not path.exists():
        skip(f"{path} missing")


def run_all(namespace: dict) -> int:
    tests = [(n, f) for n, f in namespace.items() if n.startswith("test_") and callable(f)]
    n_pass = n_fail = n_skip = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            n_pass += 1
        except Skip as e:
            print(f"SKIP  {name}: {e}")
            n_skip += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0
