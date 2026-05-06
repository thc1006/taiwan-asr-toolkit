---
name: Bug report
about: Report a reproducible problem
title: "[BUG] "
labels: bug
---

## What happened
<!-- Brief description of the bug -->

## How to reproduce
1. ...
2. ...
3. ...

## Expected vs actual

**Expected**: ...

**Actual**: ...

## Environment

- OS:
- Python version: `python --version`
- GPU + driver: `nvidia-smi`
- PyTorch version: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
- Toolkit commit/version:
- Audio source (file format, sample rate, length):

## Test suite status

```
$ pytest -m fast
# paste output
```

Are the Breeze invariant tests passing? `pytest -m breeze_invariant`

## Logs / error output

```
# paste full traceback or relevant log
```
