# sinter — handoff

Last updated: 2026-11-18 08:57

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
measured. Then the cross-restart prefix store (see "Just shipped" below),
which takes a cold first turn on the 35B from 8.7s to 0.4s.

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

The routed MoE `gate_proj` and `up_proj` merged the same way, along the
intermediate axis (`llm.MergeLinearsAxis`, one `gather_qmm` instead of two).
Also bit-identical (same token ids), worth +2.2% on an interleaved A/B:
baseline 58.9/59.1/59.6 vs merged 60.3/60.4/60.8, merged winning all three
pairs. Two merges of the same class, each ~2-3% — consistent with the
per-layer-overhead picture below, not with launch count being the lever.

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

- weight reads ~1590MB/token (505MB DeltaNet projections, 660MB MoE, 136MB
  full attention, 286MB `lm_head`)
- MoE (40 layers): 3.1ms for 660MB — 213 GB/s, already dense-equivalent, so
  the expert dispatch is NOT the problem. The router chain is inside the noise
- linear attention (30 layers): ~4.5ms, of which ~2.9ms is the projections
  (now merged) and the rest is the recurrent kernel, conv and norms
- vocabulary head: 2.0ms (286MB, 143 GB/s); argmax over 248k vocab: 0.5ms
- full attention (10 layers): ~1ms
- elementwise norms and residual adds: ~0.4ms under compile (2.2ms eager)

**Closure staging is not a cost — that line was wrong.** The earlier estimate
of ~1.5ms for staging 30 recurrent states in and out does not survive
measurement. `Closure.Apply` passes array handles, not data: a 65-input apply
dispatches in **73µs** (`TestSpikeApply65Inputs`), and the whole per-token
apply is 1ms of CPU against 12ms of GPU, so it is entirely hidden. The
recurrent state's 120MB/token of read+write is real, but it is the kernel's
own traffic, counted below, not a boundary copy.

**Nothing here is bandwidth-bound.** Against each block's own floor at
400 GB/s, every one runs 2-5x slow:

| block | bytes/token | floor | measured | effective |
|---|---|---|---|---|
| DeltaNet projections (30) | 505MB | 1.26ms | 2.9ms | 174 GB/s |
| MoE (40) | 660MB | 1.65ms | 3.1ms | 213 GB/s |
| full attention (10) | 136MB | 0.34ms | 1.0ms | 136 GB/s |
| vocabulary head | 286MB | 0.72ms | 2.0ms | 143 GB/s |
| DeltaNet recurrent state (30) | 120MB | 0.30ms | 1.6ms | **75 GB/s** |

The recurrent kernel is the worst and the simplest: it loads a 2MB fp32
state, does one step of ~8 flops per element, and stores it back — a
memcpy-shaped kernel at 75 GB/s. It is fp32 because the checkpoint declares
`mamba_ssm_dtype: float32`, so halving it to bf16 is a numerics change, not
a free win.

**Two knobs tested, both dead ends.** The recurrent kernel's threadgroup
count is not the problem: `SetThreadGroup(32, 4/8/16, 1)` — 1024, 512 and
256 threadgroups — measured 54.7 / 54.6 / 53.8 tok/s interleaved at 2541
tokens. No effect. And the compile transform is not a lever either:
`SINTER_COMPILED_DECODE_MODE=fuse` measured 64.5/63.7 against 62.6/62.2
default at short context, and 59.8/60.3 against 60.0/60.2 at 2541 tokens —
within noise at long context, +2-3% at short, not worth trading the
bit-exact parity guarantee for.

**The fused gated-delta kernel is worth +14%, and it is already the
default.** Interleaved and bit-identical: 63.6/63.6 vs 55.7/55.3 tok/s at
short context, 58.6/57.2 vs 50.9/50.2 at 2541 tokens (`SINTER_DELTA_OPS=1`
forces the ops loop). That is the single largest measured win in the decode
path, and it points the way: the ops loop materializes ~20 intermediates per
layer over a 2MB state array, so what costs is bytes written to intermediate
buffers, not launches.

**Op count on small tensors is free — fusing the norm chains would buy
nothing.** The obvious next step after the delta result was to fuse the
per-layer norm and elementwise chains (two `scaleRMSNorm` plus
`rmsNormGated` per DeltaNet layer, plus the MoE router's ~17 ops). Measured
instead of assumed: chaining 16 extra exact no-op elementwise ops on each
layer's output — 640 extra ops per token — measured 63.0 against 64.4 tok/s,
i.e. nothing. (The first generation with a new op count is slow because MLX
builds a compile plan for the new graph; ignore it.) A 2048-element bf16
multiply is ~20ns of memory traffic, so 640 of them is 13µs. The "~2.3µs per
op" figure in circulation came from a chain over a 2MB tensor and does not
transfer to the small ones.

So op count is not the lever in general — bytes on the large tensors are.
That rules out the whole fusion-of-small-chains direction, which is where
the remaining ops are.

**Steady-state decode, interleaved on a quiet machine** (the earlier
single-shot numbers were contaminated by the first generation after load,
which pays Metal kernel compilation):

| model | short | 2541-token prompt |
|---|---|---|
| Qwen3.5-2B-4bit | 145 | 126 |
| qwen3.5-4b-4bit | 82 | 70 |
| qwen3.6-35b-a3b-4bit | 63 | 60 |

The 35B's long-context figure is much better than the 48.5 a single cold run
reports: once the first generation is excluded it is ~60, not ~49.

Every block being uniformly 2-5x off its floor is the signature of a
latency/occupancy problem, not a bytes problem — which is exactly what a
Metal capture would resolve, and why no host-side restructuring has moved
the number.

**Per-turn setup is ~8-10ms, and it is not worth caching.** The compiled
closure is traced and compiled on every `Generate` and released on return,
which looked like an easy win. Measured: the Go-side trace is 3-6µs,
`Compile` is 2-4µs, and the first apply is 8-10ms. Caching the closure
across turns would save that 8-10ms and nothing more, so it is not worth the
shape-keying it would need. `SINTER_LOCAL_DEBUG=1` now prints the
breakdown.

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

A slot-count sweep of `gather_qmm` supports the second: k=1, 2, 4, 8, 16 cost
0.152, 0.154, 0.155, 0.173, 0.191ms, and k=8 pointing at the *same* expert
costs the same as k=8 distinct (0.161 vs 0.170). Reading 8x the weight bytes
is nearly free, so the call is fixed-cost dominated, not bandwidth bound. The
absolute numbers are unusable (the identical full-MoE measurement swung 1.02ms
→ 14.77ms between runs on a contended machine) — only the within-run shape is
informative.

**Micro-benchmark caveat, learned the hard way.** Standalone per-call timings
on this machine overstate by 3-4x because each `AsyncEval` commits its own
command buffer, and they also carry a fixed per-run cost that only amortizes
at high iteration counts: the recurrent kernel measured 0.238ms/layer at 10
chained iterations, 0.145 at 50, and 0.084 at 200. Always report the iteration
count, and prefer full-model A/B ablations over isolated timings.

**Profiled: the GPU is 95% busy during decode, so stalls are not the
problem.** The handoff had two candidates for the ~400µs per layer: kernel
serialization inside a layer, and the MoE gather being internally
serialized. A Metal System Trace of a marked decode window (64 tokens,
0.986s = 15.4ms/token) settles the first: **GPU busy 0.938s of 0.986s =
95.1% coverage**, 23 gaps totalling 47ms, mean 2.0ms, max 20.3ms. There is
no idle time to reclaim — even eliminating every gap would buy 5%.

So the time is inside the kernels, which is what the floor table above
already said. Candidate #1 is dead; only kernel speed is left.

Method, since the trace tooling was not obvious: `.gputrace` bundles are
Xcode's format and are not parseable (Apple's private "MTSP" container, and
21GB because they dump every buffer). `xctrace record --template 'Metal
System Trace'` produces a 158MB `.trace` whose tables `xctrace export`
writes as XML, and a Go test that writes wall-clock markers around the
decode lets the window be cut exactly. See Verified commands.

**Target.** >100 tok/s on the 35B. Summing the measured parts gives ~11.5ms
of real work, i.e. a ~87 tok/s ceiling with the current kernels. The floor
table above says where that ceiling comes from: the large-tensor blocks all
run 2-5x off their own bandwidth floor. Small-op count is not a factor
(measured above) and neither is dependency serialization (profiled above),
so the remaining work is making those kernels saturate memory. The two
furthest from their floor are the DeltaNet recurrent kernel (75 GB/s — a
memcpy-shaped kernel over a 2MB fp32 state) and the vocabulary head
(143 GB/s). What is left to determine is occupancy versus access pattern,
which needs per-kernel GPU counters rather than the timeline: the Metal
System Trace used above has the Shader Timeline disabled, so it gives
coverage but not per-kernel attribution.

The capture path is verified working — see the command in Verified
commands. It needs `MTL_CAPTURE_ENABLED=1`, which the earlier attempt was
missing.

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

### 2. Consider lowering the image cap to 512

Only after 1. `DefaultImageMaxTokens` is 1024
(`llm/openaisserver/multimodal.go`). Dropping to 512 halves image prefill:

| image tokens | 35B prefill | 2B prefill |
|---|---|---|
| 987 (cap 1024) | 4055ms | 1369ms |
| 495 (cap 512) | ~2100ms | ~650ms |

Quality: 495 tokens still transcribed the 1080x2340 screenshot correctly on
the 2B, and the 4B is fine at every size tested. The risk is dense text and
small UI labels. This is a quality/latency tradeoff, so it needs a decision,
not a default flip.

## Just shipped — cross-restart prefix persistence

The 2541-token system prompt used to cost a full prefill on every cold start.
It now loads from disk instead. `llm/prefix_store.go` persists the warm
system-prefix KV slot under
`${SINTER_CACHE_DIR:-${XDG_CACHE_HOME:-~/.cache}/sinter}/prefix/<model
fingerprint>/`, and both `WarmSystemPrefix` and `Generate` restore it.

35B, `TestScratchColdStart`, same prompt, greedy, separate processes:

| | warm step | first-turn total | tokens |
|---|---|---|---|
| cold (empty store) | 8315ms | 8742ms | `[271 17 10 17 283 220 19]` |
| warm (restored) | 36ms | 395ms | `[271 17 10 17 283 220 19]` |

**Identical token ids** are the correctness proof: a restored DeltaNet state
that differed from the prefilled one would change them. The 116MB slot file
reads back in ~40ms (unified memory, so the "upload" is a memcpy).

Design constraints worth keeping:

- **Only the warm prefix is persisted.** Persisting each conversation's own
  slot would rewrite hundreds of megabytes per turn for a hit the in-memory
  slots already serve.
- **Matching is stricter than a hash.** The stored token sequence is compared
  exactly and must be a true prefix; `imageKey` and `posDelta` must match
  too. A partial overlap is never served, for the same reason
  `bestPrefixSlot` refuses one.
- **The fingerprint covers the code, not just the weights.** It hashes
  `config.json` and `tokenizer.json`, plus each safetensors shard's name,
  size and mtime, plus `prefixCacheEpoch`. Bump that constant on any change
  to rope, norms or the recurrent kernel — the weights would fingerprint
  identically while the stored K/V went stale.
- **A bad file is a cache miss, never an error.** Fingerprint mismatch,
  wrong layer count, truncated payload and an absurd declared token count all
  fall through to a normal prefill.
- **`SINTER_PREFIX_DISK=0`** disables it; `SINTER_CACHE_DIR` relocates the
  root (the live tests use the former to stay hermetic).

Not done, and not needed yet: persisting a conversation's own prefix, and
compressing the slot (fp16 would halve the 116MB).

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
- **Kernel fusion** — `SINTER_COMPILED_DECODE_MODE=fuse` is +2-3% at short
  context (64.5/63.7 vs 62.6/62.2) and within noise at 2541 tokens
  (59.8/60.3 vs 60.0/60.2). Not worth trading the bit-exact parity
  guarantee for. Launch count is not the bottleneck.
- **Delta kernel threadgroup count** — `SetThreadGroup(32, 4/8/16, 1)`, i.e.
  1024/512/256 threadgroups, measured 54.7/54.6/53.8 tok/s interleaved at
  2541 tokens. The kernel is not threadgroup-scheduling-bound.
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
  suggests. But do not conclude that kernels never matter: the fused
  gated-delta kernel is worth **+14%** over the ops loop (63.6/63.6 vs
  55.7/55.3 short, 58.6/57.2 vs 50.9/50.2 at 2541 tokens). The difference is
  what the ops loop writes: ~20 intermediates per layer over a 2MB state
  array. Bytes to intermediate buffers is the lever, launch count is not.
- **Op count on small tensors** — 640 extra no-op elementwise ops per token
  (16 per layer) measured 63.0 vs 64.4 tok/s. Free. Fusing the per-layer
  norm chains or the MoE router would therefore buy nothing; the remaining
  ops are all on small tensors.
- **Sampling path** — greedy/no-penalty vs the defaults is identical, so CPU
  argmax over the 248k vocab is not the cost (argmax itself is 0.5ms).
- **Closure staging** — a 65-input `Closure.Apply` dispatches in 73µs
  (`TestSpikeApply65Inputs`), and the per-token apply is 1ms of CPU against
  12ms of GPU, so it is fully hidden. Handles are passed, not data. There is
  no 1.5ms boundary copy to reclaim.
- **Caching the compiled closure across turns** — it is traced and compiled
  on every `Generate`, but both are under a millisecond; the ~40ms first
  apply is the only per-turn cost. Not worth shape-keying.
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
- **Under `-tags ggml`, two large-image tests also crash on clean HEAD**
  with `fault 0x187d3c5e8`: `TestImageCacheHitSkipsPrefill`
  (`llm/mropeparity`) and `TestImageRequestLiveModel`
  (`llm/openaisserver`). Same instability as the large-image abort below,
  not a regression — verified by stashing to clean HEAD and reproducing.
- **The `llm` package's own tests import only the MLX backend**, so
  `go test -tags ggml ./llm/` fails four tests with "no GPU backend
  available" (`TestGeneration`, `TestCacheDecodeParity`,
  `TestPrefixCacheParity`, `TestPrefixCacheSpeedsUp`). Test the `llm`
  package on MLX; use the ggml tags for `llm/vision`, `llm/mropeparity` and
  `llm/openaisserver`.
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

# Cold-start A/B for the prefix store (scratch). Run twice against the same
# SINTER_CACHE_DIR: the first run populates the store, the second restores.
# Add SINTER_COLDSTART_NO_WARM=1 to exercise Generate's own adoption path.
SINTER_CACHE_DIR=/tmp/sprout/prefix-ab SINTER_BACKEND=metal \
SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.6-35b-a3b-4bit \
  go test -count=1 -run TestScratchColdStart -v ./llm/qwen35/

# Closure-boundary cost for 65 inputs (prints dispatch vs dispatch+eval)
SINTER_BACKEND=metal go test -count=1 -run TestSpikeApply65Inputs -v ./mlx/

# Metal capture of steady-state decode, for per-kernel GPU times in
# Instruments. MTL_CAPTURE_ENABLED=1 is REQUIRED — without it the Metal debug
# layer is absent and start_capture fails with "Capture layer is not
# inserted". The bundle is large (21GB for the 35B, because it dumps every
# buffer) and only Xcode can open it, not Instruments; it is Apple's private
# format and not parseable. Prefer the xctrace route below.
mkdir -p /tmp/sprout/capture
MTL_CAPTURE_ENABLED=1 SINTER_BACKEND=metal \
SINTER_MTP_PARITY_MODEL=/Users/alanp/dev/llm-models/qwen3.6-35b-a3b-4bit \
SINTER_CAPTURE_DIR=/tmp/sprout/capture \
  go test -count=1 -run TestScratchMetalCapture -v ./llm/

# GPU timeline as parseable XML, which is how the 95% decode coverage above
# was measured. TestScratchTracePhases writes wall-clock markers around one
# decode to /tmp/sprout/phases.txt so the window can be cut exactly.
cd /Users/alanp/dev/sprout-foundry/sinter && go test -c -o /tmp/sprout/qwen35.test ./llm/qwen35/
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.6-35b-a3b-4bit \
  xcrun xctrace record --template 'Metal System Trace' --output /tmp/sprout/ph.trace \
  --launch -- /tmp/sprout/qwen35.test -test.run TestScratchTracePhases -test.v
xcrun xctrace export --input /tmp/sprout/ph.trace --toc   # gives start-date + table list
xcrun xctrace export --input /tmp/sprout/ph.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output /tmp/sprout/ph_gpu.xml
# Then: filter rows to gpu-channel-name=Compute and metal-nesting-level=0,
# resolve the XML's id/ref deduplication, merge overlapping intervals, and
# compare merged busy time against the window span.

# Image cap end to end over HTTP (scratch)
SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/Qwen3.5-2B-4bit \
  go test -count=1 -run TestScratchImageCapOnScreenshot -v ./llm/openaisserver/

# GGML (run serially, packages individually — see Gotchas)
SINTER_BACKEND=ggml SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.5-2b-4bit \
  go test -count=1 -tags ggml ./llm/vision/ ./llm/mropeparity/ ./llm/openaisserver/
```
