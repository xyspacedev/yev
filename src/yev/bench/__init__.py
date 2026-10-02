"""External benchmarks: `yev bench build` (laptop, network) writes chat rows; `yev bench score` (GPU, offline)
runs the model and the benchmark's own scorer; `yev bench overlap` measures 8-gram overlap with training."""
from __future__ import annotations

from yev.bench import decidebench, dynabench_test, jevbench, rjudge, wildguardtest

BENCHES = {m.NAME: m for m in (decidebench, jevbench, wildguardtest, dynabench_test, rjudge)}
