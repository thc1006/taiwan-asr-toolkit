# -*- coding: utf-8 -*-
"""CI guard: README / docs / notebook claims of "N tests" must match the
actual collected pytest count.

Pre-v0.5.5 the docs drifted: README badge said "72 passed", line 287 said
56, CONTRIBUTING said 56, llms.txt said 56, CHANGELOG never logged the
69 → 72 transition. This test re-runs `pytest --collect-only -q`, parses
the count, and asserts the docs all reference that exact number.
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

# Files allowed to mention a test count claim. New mentions go here so the
# guard catches them.
DOCS_TO_CHECK = [
    ROOT / "README.md",
    ROOT / "llms.txt",
    ROOT / "docs/ARCHITECTURE.md",
    ROOT / "docs/PROMOTION.md",
    ROOT / "examples/quickstart.ipynb",
]

# Patterns that look like a test count claim. We extract the leading int
# and assert it equals the live collected count.
COUNT_PATTERNS = [
    re.compile(r"(\d{2,4})\s+TDD\s+tests"),
    re.compile(r"All\s+(\d{2,4})\s+tests"),
    re.compile(r"MIT\s+·\s+(\d{2,4})\s+tests"),
    re.compile(r"tests-(\d{2,4})%20passed"),
    # ARCHITECTURE-style: "56 tests · 4 priority tiers"
    re.compile(r"(\d{2,4})\s+tests\s+·"),
    # llms.txt parens form: "(104 TDD tests including 5 invariants...)"
    re.compile(r"\((\d{2,4})\s+TDD\s+tests"),
]


def _live_collected_count() -> int:
    """Run pytest --collect-only and parse the 'N tests collected' line."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    output = proc.stdout + proc.stderr
    m = re.search(r"(\d+)\s+tests?\s+collected", output)
    assert m, f"Could not parse pytest collection count from output:\n{output[-1000:]}"
    return int(m.group(1))


@pytest.mark.fast
def test_doc_test_count_claims_match_live_count():
    """All N-tests claims across docs must reference the same number that
    pytest currently collects."""
    live = _live_collected_count()
    drift = []
    for path in DOCS_TO_CHECK:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for pat in COUNT_PATTERNS:
            for match in pat.finditer(text):
                claimed = int(match.group(1))
                if claimed != live:
                    # Find which line this match is on for a useful error.
                    line_no = text.count("\n", 0, match.start()) + 1
                    drift.append(
                        f"  {path.relative_to(ROOT)}:{line_no} claims "
                        f"{claimed} tests; live collection has {live}"
                    )
    assert not drift, (
        f"Test-count claim drift detected (live={live}). Update these:\n"
        + "\n".join(drift)
    )
