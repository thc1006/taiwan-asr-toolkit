# -*- coding: utf-8 -*-
"""Regression test for v0.5.5 S4: sdist must ship tests/conftest.py.

Pre-v0.5.5: setuptools auto-shipped test_*.py via package discovery but
NOT conftest.py or tests/__init__.py. A downstream developer pulling the
sdist and running `pytest tests/` got ~15 'fixture not found' ERRORs even
though wheel install + clone-from-git both worked.

This test rebuilds the sdist locally and asserts conftest.py is present.
Marked @slow because building the sdist takes a few seconds.
"""
from __future__ import annotations
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


@pytest.mark.slow
def test_sdist_includes_conftest(tmp_path):
    """The built sdist tarball must contain tests/conftest.py."""
    repo_root = Path(__file__).parent.parent
    if not (repo_root / "pyproject.toml").is_file():
        pytest.skip("Not in a source checkout")
    if shutil.which("python") is None:
        pytest.skip("python not on PATH")

    proc = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(tmp_path)],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"sdist build failed (likely missing 'build' module): {proc.stderr[-500:]}")

    tarballs = list(tmp_path.glob("*.tar.gz"))
    assert tarballs, f"no sdist produced in {tmp_path}"
    [tarball] = tarballs

    with tarfile.open(tarball) as tf:
        names = tf.getnames()

    has_conftest = any(n.endswith("/tests/conftest.py") for n in names)
    assert has_conftest, (
        f"sdist {tarball.name} must include tests/conftest.py; "
        f"existing tests/ entries: "
        f"{sorted(n for n in names if '/tests/' in n and n.endswith('.py'))[:5]}"
    )


@pytest.mark.fast
def test_manifest_in_present():
    """Cheap proxy for the same invariant: MANIFEST.in must explicitly include
    tests/. Catches the regression instantly without rebuilding sdist."""
    repo_root = Path(__file__).parent.parent
    manifest = repo_root / "MANIFEST.in"
    if not manifest.is_file():
        pytest.fail(
            "MANIFEST.in is missing. Required to ensure tests/conftest.py ships "
            "in the sdist. Add e.g.: 'recursive-include tests *.py'."
        )
    content = manifest.read_text(encoding="utf-8")
    assert "tests" in content, (
        f"MANIFEST.in must reference tests/. Current content:\n{content}"
    )
