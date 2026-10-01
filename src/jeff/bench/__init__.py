"""External benchmarks: `jeff bench build` (laptop, network) writes chat rows; `jeff bench score` (GPU, offline)
runs the model and the benchmark's own scorer; `jeff bench overlap` measures 8-gram overlap with training."""
from __future__ import annotations

from jeff.bench import decidebench, dynabench_test, jevbench, rjudge, wildguardtest

BENCHES = {m.NAME: m for m in (decidebench, jevbench, wildguardtest, dynabench_test, rjudge)}
