# -*- coding: utf-8 -*-
"""Regression test for v0.5.5 M7: examples/quickstart.ipynb must NOT use raw
f-string interpolation of audio_path inside shell escapes.

Pre-v0.5.5 cells 4 and 6 had:

    !{sys.executable} -m taiwan_asr.breeze "{audio_path}" --glossary-file builtin

IPython substitutes `str(audio_path)` literally into the shell command line.
Filenames with `"`, `$`, backtick (or other shell-special chars) break parsing
or trigger command substitution. A Colab user uploading e.g. `weird"name.mp3`
hits a confusing shell-side error.

Fix: route the path through an env var:

    import os
    os.environ["TAIWAN_ASR_INPUT"] = str(audio_path)
    !{sys.executable} -m taiwan_asr.breeze "$TAIWAN_ASR_INPUT" --glossary-file builtin

The shell expands $TAIWAN_ASR_INPUT after parsing, so quoting hazards stay inside
Python and never reach the shell parser.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest


NB_PATH = Path(__file__).parent.parent / "examples" / "quickstart.ipynb"


@pytest.fixture(scope="module")
def notebook():
    if not NB_PATH.is_file():
        pytest.skip(f"notebook not found: {NB_PATH}")
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _shell_lines_in_cell(cell) -> str:
    """Concatenate all source lines of a code cell."""
    if cell.get("cell_type") != "code":
        return ""
    return "".join(cell["source"])


@pytest.mark.fast
def test_no_raw_audio_path_fstring_in_shell_escape(notebook):
    """No code cell may pass `"{audio_path}"` directly to a shell escape."""
    offenders = []
    for i, cell in enumerate(notebook["cells"]):
        src = _shell_lines_in_cell(cell)
        # the dangerous pattern: a !-line containing literal {audio_path}
        for line in src.splitlines():
            stripped = line.lstrip()
            if not stripped.startswith("!"):
                continue
            if "{audio_path}" in line:
                offenders.append((i, line))
    assert not offenders, (
        f"Cells with shell-escape f-string of audio_path "
        f"(vulnerable to filenames with $, \", backtick): {offenders}"
    )


@pytest.mark.fast
def test_audio_path_routed_via_env_var(notebook):
    """The notebook should set an env var (e.g. os.environ[...]=str(audio_path))
    before any shell call referencing the audio file."""
    has_env_route = False
    for cell in notebook["cells"]:
        src = _shell_lines_in_cell(cell)
        if "os.environ[" in src and "audio_path" in src:
            has_env_route = True
            break
    assert has_env_route, (
        "Notebook should set os.environ['...'] = str(audio_path) before any "
        "shell escape that needs the path. This decouples Python-side filenames "
        "from shell parsing."
    )
