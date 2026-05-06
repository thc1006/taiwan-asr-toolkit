---
name: Feature request
about: Suggest a feature
title: "[FEAT] "
labels: enhancement
---

## What problem does this solve
<!-- e.g. "ASR can't handle code-switching between Mandarin and Hokkien" -->

## Proposed solution

## Alternatives considered

## Will this break the Breeze invariant?
The toolkit's contract is that `MediaTek-Research/Breeze-ASR-25` is the Taiwan-Mandarin
specialist. Any change must keep the 5 `@pytest.mark.breeze_invariant` tests green.

- [ ] My proposal preserves the Breeze model ID
- [ ] My proposal preserves Traditional Chinese (Taiwan) output
- [ ] My proposal does not weaken the `condition_on_previous_text=False` Chinese decoding default
