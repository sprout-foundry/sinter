# sinter — handoff

Last updated: 2026-11-17 17:05

Vision input works end to end on MLX and GGML, and the bugs that were
blocking it are fixed and released. This document is the forward plan:
what to do next, in order, with the measurements and dead ends already
established so they are not repeated.

## Where things stand

- **sinter v0.8.2** is tagged and pushed. `main` carries the decode-path
  default change below (unreleased).
- **sprout** is on `sinter v0.8.2`, committed and pushed to `main`; publishing
  the sprout release is the owner's step.
- MLX is the product path (M1 Max, 64GB, 400 GB/s). GGML is secondary and has
  a known large-image instability (see Gotchas).

Shipped in this stretch: the MLX header/ABI fix, three bugs it unmasked, the
vision merger's missing biases, sharded-checkpoint `TensorShape`, the
per-image feature cache, and the image token cap (1024). Then the decode-path
defaults: pipelined decode is now opt-in and prompt-lookup is off, which
unlocks compiled decode as the default and is worth 5-10% on every model
measured.

## Do this next, in order

### 1. Close the remaining decode gap — it is not launch overhead

**Status: the diagnosis is done; the fix is not.** Compiled decode used to be
dead code: `useCompiled` requires `!usePipelined`, and pipelined decode was
default-ON, so `SINTER_COMPILED_DECODE=0` changed nothing (which is exactly
what the earlier 47.3 vs 47.2 tok/s measured). Pipelined is now opt-in, so
compiled runs by default and wins:

| model | eager | compiled | gain |
|---|---|---|---|
| qwen3.6-35b-a3b-4bit, short ctx | 54.4/54.6 | 58.8/59.1 | +8% |
| qwen3.6-35b-a3b-4bit, 2541 tok | 43.3/45.7 | 50.5/48.8 | +10% |
| Qwen3.5-2B-4bit | 123-130 | 140-142 | +10% |
| qwen3.5-4b-4bit | 73.7-74.5 | 77.8-79.2 | +5% |

The DeltaNet block's four input projections (`in_proj_qkv`, `_z`, `_b`, `_a`)
now load as one merged wide matmul (`llm.MergeLinears`, split back with
constant-bounded slices). Output is bit-identical, and it is worth another
+3%: 35B 58.8/59.1 → 60.1/61.0/61.2 tok/s, 4B 77.8/79.2 → 80.5/81.4, 2B a
wash. Prefill is unchanged (8261ms for the 2541-token prompt).

**Launch overhead is ruled out as the dominant cost.** MLX's kernel fusion on
top of compile (`SINTER_COMPILED_DECODE_MODE=fuse`) measured 54.8/50.3 at short
context and 45.2/45.6 at 2541 tokens — no better than eager. Replay alone is
what pays: 160 chained elementwise ops cost 2.0ms eager and 0.37ms compiled.
And the merge experiment shows kernel COUNT is not the lever either: a
standalone benchmark made 4 qmm look 5x slower than 1 merged qmm, but in the
real single-command-buffer graph the same change is worth 3%. Standalone
per-call timings on this machine overstate by 3-4x because each AsyncEval
commits its own command buffer — do not trust them for absolute cost.

**Where the ~16ms/token actually goes** (35B, short context, compiled):

- weight reads ~1669MB/token (1383MB across 40 layers + 286MB `lm_head`)
- MoE (40 layers): 3.1ms for 660MB — 213 GB/s, already dense-equivalent, so
  the expert dispatch is NOT the problem. The router chain is inside the noise
- linear attention (30 layers): ~4.5ms, of which ~2.9ms is the projections
  (now merged) and the rest is the recurrent kernel, conv and norms
- vocabulary head: 2.0ms (286MB, 143 GB/s); argmax over 248k vocab: 0.5ms
- full attention (10 layers): ~1ms
- elementwise norms and residual adds: ~0.4ms under compile (2.2ms eager)
- compiled-closure staging: ~1.5ms (30 recurrent states, 2MB fp32 each,
  crossing the closure boundary in and out every step)

**The 35B is not bandwidth-starved — it is per-layer-overhead-bound.** Per
layer, on the same machine, compiled decode costs:

| model | layers | MB/layer | µs/layer | effective GB/s |
|---|---|---|---|---|
| Qwen3.5-2B-4bit | 24 | 60 | 296 | 202 |
| qwen3.5-4b-4bit | 32 | 84 | 394 | 213 |
| qwen3.6-35b-a3b-4bit | 40 | 42 | 408 | 102 |

The 35B reads *half* the bytes per layer of the 4B in the *same* time. Its
decode is set by a ~400µs per-layer cost that barely moves with bytes, times
40 layers. That is why merging kernels gave 3% rather than the 5x a
micro-benchmark suggested, and why chasing bandwidth is the wrong frame:
something per-layer is paying a fixed cost. Two candidates worth a real GPU
profile (Metal capture, read in Instruments): kernel-to-kernel serialization
inside each layer, and the MoE's 8-expert `gather_qmm` being internally
serialized even though its aggregate bandwidth looks fine.

**Micro-benchmark caveat, learned the hard way.** Standalone per-call timings
on this machine overstate by 3-4x because each `AsyncEval` commits its own
command buffer, and they also carry a fixed per-run cost that only amortizes
at high iteration counts: the recurrent kernel measured 0.238ms/layer at 10
chained iterations, 0.145 at 50, and 0.084 at 200. Always report the iteration
count, and prefer full-model A/B ablations over isolated timings.

**Target.** >100 tok/s on the 35B. Summing the parts gives ~11.5ms of real
work, i.e. a ~87 tok/s ceiling with the current kernels, and the per-layer
table above says the remaining distance is per-layer overhead rather than
bytes. Reaching 100 needs a GPU profile to find what that ~400µs is, not
another scheduling change.

**Consumer follow-up.** sprout's `pkg/localmodel/local_provider_chat.go` sets
`PromptLookupMaxDrafts = 6` in two places, so production still takes the
prompt-lookup path even though it now measures slower than compiled decode.
Removing those two assignments (and bumping sprout onto the sinter release
that carries the new defaults) is what actually delivers the 5-10% in the
product.

**Measure with** `llm/qwen35/zz_perf_test.go` (gitignored scratch):
`TestScratchPerfProfile` prints prefill/decode splits for a short turn, a
2541-token system prompt, a growing conversation, and an image request.
`TestScratchDecodePaths` (also scratch) reports decode-only tok/s for a short
and a long prompt, which is the cleaner A/B harness for path changes.

### 2. Persist the system-prompt prefix across process restarts

**Why.** The 2541-token system prompt costs **8489ms** of prefill on the 35B
(299 tok/s). Prefix reuse already makes later turns cheap (~604ms for a
~177-token delta), and the timestamp is injected on the *user* message
(`pkg/agent/seed_query.go`), not the system prompt, so the prefix is stable.
But a fresh process starts cold and pays the 8.5s again.

**Existing pieces.** `KVCache` already carries `lastHidden`,
`SnapshotPrefixCopy`, `RestorePrefix`, and a spill file at format version 2
that includes the hidden state. `CachedHiddenArchitecture` exposes
`LastHidden` / `LogitsFromHidden`. So the question is whether a restarted
process can already load a slot, and if not, what is missing — not whether
the mechanism exists.

**Blockers to respect.** A DeltaNet layer's recurrent state is a single
accumulated value; the earlier attempt to rewind it produced a plausible but
wrong answer. The working design keeps the last position's hidden state
instead of rewinding, so any persistence must store the same thing. The key
must include model identity, prompt hash and positions, so a stale slot is
never served to a different model or prompt.

**Target.** Cold-start first turn on the 35B drops from ~8.5s to ~0.1s when
the system prompt is unchanged.

### 3. Consider lowering the image cap to 512

Only after 1 and 2. `DefaultImageMaxTokens` is 1024
(`llm/openaisserver/multimodal.go`). Dropping to 512 halves image prefill:

| image tokens | 35B prefill | 2B prefill |
|---|---|---|
| 987 (cap 1024) | 4055ms | 1369ms |
| 495 (cap 512) | ~2100ms | ~650ms |

Quality: 495 tokens still transcribed the 1080x2340 screenshot correctly on
the 2B, and the 4B is fine at every size tested. The risk is dense text and
small UI labels. This is a quality/latency tradeoff, so it needs a decision,
not a default flip.

## Measured baseline — do not re-measure

qwen3.6-35b-a3b-4bit on M1 Max, compiled decode (the current default):

| scenario | prefill | decode |
|---|---|---|
| short turn | 94ms | 61 tok/s |
| system prompt, 2541 tok | 8489ms (299 tok/s) | — |
| conv delta, ~177 tok | 604ms | 49 tok/s |
| image, 987 tok | 4055ms | 21.7 tok/s |

Qwen3.5-2B-4bit, same machine: prefill ~1736 tok/s, decode ~141 tok/s
(compiled), image 987 tok → 1369ms. qwen3.5-4b-4bit: 81 tok/s (compiled).

**Already ruled out by measurement:**

- **Prefill chunk size** — swept 128→4096 in the code comments; throughput is
  flat. It is a memory knob, not a speed knob.
- **Compiled decode** — it was never running (see step 1). Now default-ON and
  worth 5-10%.
- **Kernel fusion** — `SINTER_COMPILED_DECODE_MODE=fuse` is no better than
  eager at either context length. Launch count is not the bottleneck.
- **Pipelined decode** — slower than compiled on all three models
  (35B 52.6/53.0, 2B 112.3/116.3, 4B 75.3/75.3). Now opt-in via
  `SINTER_PIPELINE_DECODE=1`.
- **Prompt-lookup** — slower than compiled even on verbatim-echo prompts
  (35B 51.6-53.8 at k=4 and k=6, 2B 106-114). Instrumented: 362 of 414 decode
  steps found no n-gram candidate, so the search and verify batches cost more
  than the accepted drafts save. `openaisserver` no longer enables it.
- **MoE expert dispatch** — 213 GB/s, dense-equivalent. The router chain is
  inside the noise.
- **Kernel count** — merging the four DeltaNet input projections into one
  matmul (bit-identical) is worth 3%, not the 5x a standalone benchmark
  suggests. Fewer kernels is not the lever.
- **Sampling path** — greedy/no-penalty vs the defaults is identical, so CPU
  argmax over the 248k vocab is not the cost (argmax itself is 0.5ms).
- **MTP** — none of the three checkpoints (`Qwen3.5-2B-4bit`,
  `qwen3.5-4b-bf16`, `qwen3.6-35b-a3b-4bit`) ships MTP tensors.

## Don't redo

- **MLX ABI skew is fixed.** Vendored headers are mlx-c v0.7.0 and match the
  installed `libmlxc` 0.7.0. `mlx_fast_scaled_dot_product_attention` takes
  `force_fused` before the stream. If a `no fused kernel` or
  `non-empty mlx_stream` error reappears, check the header version against
  the installed library *first* — mlx-c passes structs by value, so a stale
  header misreads arguments instead of failing to link. A pinned consumer on
  an older sinter is the other cause: sprout was on v0.7.2 (mlx-c 0.6.0
  headers) and failed exactly this way.
- **The vision merger's biases are applied** (`mergerFC1B`, `mergerFC2B`).
  Guarded by `TestMergerAppliesBias`.
- **Sharded checkpoints work** (`TensorShape` routes through the owning
  shard). Guarded by `TestShardedTensorShape`.
- **The reference harness in `/tmp/sprout` is only trustworthy for the vision
  tower.** It produces degenerate text at every image scale (144–629 tokens)
  while sinter transcribes the same screenshot correctly at 495. Its
  `lm_head` is correctly tied, so the text path is broken elsewhere. Do not
  read its generated text or logits as ground truth; its tower dumps are
  usable.

## Gotchas that cost time

- **The commit tool is scoped to the session's workspace root** (`sinter`).
  It refuses `repo_dir` outside it, so commits in `sprout` need either a
  session rooted there or a plain `git commit`.
- **`zz_*_test.go` is gitignored.** Scratch diagnostics live on disk but
  never get committed. `llm/qwen35/zz_blocks_test.go` is the per-block tower
  comparison that found the merger bug; keep it.
  `llm/qwen35/zz_towerparity_test.go` fails against a stale
  `/tmp/sprout/refdump` (2048 vs 2560 hidden) — it is a dump-version mismatch,
  not a regression.
- **`TestNoLegacyHomeDotSprout`** (sprout `pkg/configuration`) flakes when a
  live sprout session creates/removes `.sprout/revisions` dirs while the
  guard walks the tree. Passes on re-run; unrelated to dependency bumps.
- **Large image prefills abort on GGML.** A 987-token image dies in
  `gatedDeltaNet.forward`'s `z` reshape
  (`GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b))`), and with
  `SINTER_GGML_DEBUG=1` it instead fails earlier in the prefix snapshot
  (`pin of 16777216 bytes exceeds tensor nbytes 73728`). Two different
  failures on one input means instability at this length, not one wrong
  reshape. Small images are fine and the same request is correct on MLX.
- **`TestProbeQwen35GGML`** crashes the process during pipelined decode on
  clean HEAD. Because it takes the process down, a multi-package
  `-tags ggml` run reports unrelated packages as FAIL; run packages
  individually.
- **Benchmark on a quiet machine.** Every decode number in this document is
  decode-only tok/s from a greedy run. A concurrent `tsc --noEmit` or a
  transcription job moved the same measurement by 2x, and load average above
  ~10 makes the numbers unusable. Check `sysctl -n vm.loadavg` first.
- **`ThinkingTokens` defaults to false**, so a prompt that makes the model
  think emits zero tokens through `onToken` and a scratch harness reports
  "0 tok". Set `cfg.ThinkingTokens = true` in harnesses that count tokens.

## Verified commands

```bash
go build ./... && go build -tags ggml ./...

# MLX
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/Qwen3.5-2B-4bit \
  go test -count=1 ./llm/qwen35/ ./llm/ ./mlx/ ./llm/openaisserver/

# Decode-path A/B (scratch, decode-only tok/s, short and long prompt)
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.6-35b-a3b-4bit \
  go test -count=1 -run TestScratchDecodePaths -v ./llm/qwen35/

# Performance harness (scratch, prints prefill/decode splits)
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.6-35b-a3b-4bit \
  go test -count=1 -run TestScratchPerfProfile -v ./llm/qwen35/

# Image cap end to end over HTTP (scratch)
SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/Qwen3.5-2B-4bit \
  go test -count=1 -run TestScratchImageCapOnScreenshot -v ./llm/openaisserver/

# GGML (run serially, packages individually — see Gotchas)
SINTER_BACKEND=ggml SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.5-2b-4bit \
  go test -count=1 -tags ggml ./llm/vision/ ./llm/mropeparity/ ./llm/openaisserver/
```
