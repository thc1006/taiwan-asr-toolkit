# -*- coding: utf-8 -*-
"""Regression test for v0.5.5 M6: breeze._ensure_ct2 must survive missing
optional files in --copy_files.

Pre-v0.5.5 (breeze.py:163-172): the converter command hard-coded a list of
9 filenames including added_tokens.json. If MediaTek changes the Breeze
checkpoint to drop one (newer Whisper variants don't have it), the
subprocess fails with non-zero exit and we raise:

    RuntimeError("CT2 轉換失敗,請改用 --backend transformers: ...")

Fix: on first failure, retry with a minimal --copy_files list
(preprocessor_config.json + tokenizer.json). Only raise if THAT also fails.
"""
from __future__ import annotations
from unittest.mock import patch, MagicMock

import pytest
import subprocess

torch = pytest.importorskip("torch")
pytest.importorskip("faster_whisper")

from taiwan_asr.breeze import FasterWhisperBackend, HwConfig
from taiwan_asr.common import S2TW


def _backend(tmp_path):
    cfg = HwConfig(
        device="cpu", cuda_idx=0, dtype=torch.float32, ct2_compute="float32",
        batch_size=1, cpu_threads=1, num_workers=1,
        chunk_sec=4.0, pad_sec=0.4, desc="test",
    )
    b = FasterWhisperBackend(cfg=cfg, beam=1, s2tw=S2TW(False))
    b.ct2_dir = str(tmp_path / "ct2")
    return b


@pytest.mark.fast
def test_ensure_ct2_retries_with_minimal_copy_files_on_failure(tmp_path):
    """First subprocess call fails (full --copy_files list). Backend retries
    with a minimal list. Second call succeeds — no RuntimeError."""
    backend = _backend(tmp_path)
    call_args = []

    def fake_run(cmd, **kwargs):
        call_args.append(list(cmd))
        if len(call_args) == 1:
            # First attempt with full --copy_files list -> fail
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        # Second attempt -> succeed; create marker so subsequent _ensure_ct2 skips
        import os
        os.makedirs(backend.ct2_dir, exist_ok=True)
        open(f"{backend.ct2_dir}/model.bin", "w").close()
        return MagicMock(returncode=0)

    with patch("taiwan_asr.breeze.subprocess.run", side_effect=fake_run):
        backend._ensure_ct2()

    assert len(call_args) >= 2, (
        f"Backend should retry with minimal --copy_files on failure; "
        f"got {len(call_args)} subprocess calls."
    )
    # First call had the full list; second should be smaller
    first = call_args[0]
    second = call_args[1]
    first_copy = first.count("--copy_files") > 0 and len(first)
    second_copy = second.count("--copy_files") > 0 and len(second)
    assert second_copy < first_copy, (
        f"Retry must use a SMALLER copy_files list; "
        f"first={first_copy} args, second={second_copy} args."
    )


@pytest.mark.fast
def test_ensure_ct2_propagates_when_minimal_also_fails(tmp_path):
    """If even the minimal retry fails, raise — don't infinite-loop."""
    backend = _backend(tmp_path)

    def always_fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

    with patch("taiwan_asr.breeze.subprocess.run", side_effect=always_fail):
        with pytest.raises(RuntimeError, match="CT2 轉換失敗"):
            backend._ensure_ct2()


@pytest.mark.fast
def test_ensure_ct2_skipped_when_already_built(tmp_path):
    """If model.bin already exists, no subprocess call should happen."""
    backend = _backend(tmp_path)
    import os
    os.makedirs(backend.ct2_dir, exist_ok=True)
    open(f"{backend.ct2_dir}/model.bin", "w").close()

    with patch("taiwan_asr.breeze.subprocess.run") as run_mock:
        backend._ensure_ct2()
        run_mock.assert_not_called()
