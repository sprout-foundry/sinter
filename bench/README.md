# bench

Reference-engine benchmarks compared against sinter's own decode harness
(`llm/qwen35/zz_paths_test.go`, `TestScratchDecodePaths`).

## mlx-lm

`mlx_vs_sinter.py` is the comparison that matters. It encodes the SAME
prompts through mlx-lm's own tokenizer (15 and 2541 tokens), runs best of 3,
and measures decode first-token-to-last. Pair it with sinter's
`TestScratchMlxPaired` (llm/qwen35/zz_ablate_test.go, gitignored), which
uses the identical prompts and accounting.

`mlxlm_tokenexact.py` is kept only as a warning: it fed mlx-lm the token ids
sinter *generated* (62-63 tokens), so its "long" rows measured a short
prompt and made mlx-lm look flat across context. That false flatness
mis-attributed sinter's context cost for a whole session. Do not use it.

`mlxlm_bench.py` is the earliest string-prompt version. It is confounded
differently (mlx-lm's template emits ` thinking` where sinter's does not).

### Setup

The Homebrew `mlx` install (0.32.3) is broken — it ships no `mlx.utils`, so
`import mlx_lm` fails on every Python on the machine. Use a venv:

```bash
python3.12 -m venv /tmp/sprout/mlxenv
/tmp/sprout/mlxenv/bin/pip install mlx mlx-lm
```

### Run

```bash
/tmp/sprout/mlxenv/bin/python bench/mlx_vs_sinter.py \
  /Users/alanp/dev/llm-models/Qwen3.5-2B-4bit

SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/Qwen3.5-2B-4bit \
  go test -count=1 -run TestScratchMlxPaired -v ./llm/qwen35/
```

Benchmark on a quiet machine. `sysctl -n vm.loadavg` above ~10 makes the
numbers unusable; a concurrent VM or Spotlight indexing moved them by more
than the effects being measured.

### Results

Corrected paired numbers (2B, load ~5-12): mlx-lm 185.6 short / 177.5 long
(−4.4% with context); sinter 158.6 / 141.6 (−10.7%). Gap: −14.5% short,
−20.2% long. See "mlx-lm is the reference" in `PROGRESS-vision.md` for what
this implies — the constant short-context term is the larger one.
