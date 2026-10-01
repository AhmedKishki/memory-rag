# Upstream format fixtures

This file is **not written by this package**. It is the exact bytes UltraRAG's own memory server produced, captured on 2026-09-22 by running `servers/memory/src/memory.py` from the pinned revision `3a709a2aea3fbe46acca59c422621c94b6e86857` (declared version `0.3.0.2`) with `ULTRARAG_UI_STORAGE_ROOT` pointing at a temporary directory, then calling

```python
get_global_memory("fixture")
```

| File | What upstream wrote it with |
| --- | --- |
| `standing_memory.md` | the template `get_global_memory` creates on the first read of a user |

`tests/test_fidelity.py` compares what this package writes against this file, so a change to the format fails a test instead of drifting quietly.

The same revision also wrote a `daily_round.md` beside it, which this package does not produce. A memory here is a set of statements, and anything worth keeping is recorded as one, so the store neither writes nor reads a dated exchange log. The divergence is recorded in the README's choice table. That fixture is kept in this directory's history rather than deleted from it, so the round format upstream uses is still recoverable from the revision that produced it.

To recapture the standing document after an upstream revision change, run the module from the new checkout the same way and replace the file; the revision recorded in `src/memory_ultra_rag_mcp/reference.py` moves with it.
