# bench

Reference-engine benchmarks compared against sinter's own decode harness
(`llm/qwen35/zz_paths_test.go`, `TestScratchDecodePaths`).

## mlx-lm

`mlxlm_tokenexact.py` is the comparison that matters. It feeds mlx-lm the
exact token ids sinter reported for the same prompts, so both engines decode
the same sequence and do the same amount of work. Decode time runs from the
first generated token to the last, excluding prefill, on both sides.

`mlxlm_bench.py` is the earlier string-prompt version. It is kept only to
show why token-exact matters: mlx-lm's chat template emits ` thinking` where
sinter's does not, so the two generate different content and the comparison
is confounded.

### Setup

The Homebrew `mlx` install (0.32.3) is broken — it ships no `mlx.utils`, so
`import mlx_lm` fails on every Python on the machine. Use a venv:

```bash
python3.12 -m venv /tmp/sprout/mlxenv
/tmp/sprout/mlxenv/bin/pip install mlx mlx-lm
```

### Run

```bash
/tmp/sprout/mlxenv/bin/python bench/mlxlm_tokenexact.py \
  /Users/alanp/dev/llm-models/qwen3.5-4b-4bit 3
```

Benchmark on a quiet machine. `sysctl -n vm.loadavg` above ~10 makes the
numbers unusable; a concurrent VM or Spotlight indexing moved them by more
than the effects being measured.

### Results

See "mlx-lm is the reference" in `PROGRESS-vision.md` for the current table
and what it implies.
