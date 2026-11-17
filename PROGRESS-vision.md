# sinter — handoff

Last updated: 2026-11-17 15:40

Vision input works end to end on MLX and GGML, and the bugs that were
blocking it are fixed and released. This document is the forward plan:
what to do next, in order, with the measurements and dead ends already
established so they are not repeated.

## Where things stand

- **sinter v0.8.2** is tagged and pushed. `main` is in sync with origin.
- **sprout** is on `sinter v0.8.2`, committed and pushed to `main`; publishing
  the sprout release is the owner's step.
- MLX is the product path (M1 Max, 64GB, 400 GB/s). GGML is secondary and has
  a known large-image instability (see Gotchas).

Shipped in this stretch: the MLX header/ABI fix, three bugs it unmasked, the
vision merger's missing biases, sharded-checkpoint `TensorShape`, the
per-image feature cache, and the image token cap (1024).

## Do this next, in order

### 1. Fuse the decode layer — the biggest win

**Why.** Decode is overhead-bound, not bandwidth-bound. On the 35B MoE
(3B active), decode runs at ~47 tok/s while pulling only ~70 GB/s of the
400 GB/s available — roughly 5x theoretical headroom. The 2B sits at
~100 tok/s, also far under its bandwidth ceiling. With 40 layers and roughly
15 kernel launches each, per-token launch overhead is the plausible
dominant cost.

**Do not start by writing kernels.** First establish whether the existing
compiled-decode path engages at all, because the measurements say it is
currently doing nothing:

- `SINTER_COMPILED_DECODE=0` vs default changed decode by 0.1 tok/s
  (47.3 vs 47.2) on the 35B.
- No `compiled decode unavailable (...)` line was ever logged, which is what
  `llm/model.go:940` prints when `PrepareCompiledDecode` fails.
- So either `useCompiled` is false (the conditions at `llm/model.go:936-938`
  are not met) or the compiled closure is being built but not helping.

Check, in this order: does `*Qwen35` satisfy `CompiledGreedyArchitecture`;
does `PrepareCompiledDecode` actually run (`llm/qwen35/compiled_decode_mlx.go`);
and does `q.cd` end up non-nil. Note `useGPUArgmax` requires
`Temperature <= 0` **and** `RepetitionPenalty == 0`, and
`DefaultGenerateConfig` sets `RepetitionPenalty: 1.1` — so a default-config
call takes none of the fast paths. That was a measurement trap once already.

**Then.** Existing scaffolding to build on, not to duplicate:

- `llm/qwen35/compiled_decode_mlx.go` — the `mlx_compile` closure over a
  fixed-capacity KV window.
- `llm/qwen35/compiled_decode_body_mlx.go` — the traced body.
- `llm/qwen35/compiled_delta_layer_mlx.go` — mirrors `gatedDeltaNet.forward`
  at S=1 with the recurrent state as explicit I/O.
- `llm/qwen35/fused_kernels.go` — `fusedComputeG`, `fusedGatedDeltaUpdate`.
- `mlx/` — the `mlx_fast_*` wrappers and the generated dispatch shim.

The direction that matches the diagnosis is covering a whole layer per
launch rather than one op per launch, starting with the DeltaNet layers
(30 of 40) where `compiled_delta_layer_mlx.go` already has the shape.

**Target.** >100 tok/s on the 35B, and a measurable gain on the 2B. Anything
smaller means the launches were not the cost, and the next suspect is the
MoE expert dispatch (`gather_qmm` in `llm/qwen35/moe.go`).

**Measure with** `llm/qwen35/zz_perf_test.go` (gitignored scratch):
`TestScratchPerfProfile` prints prefill/decode splits for a short turn, a
2541-token system prompt, a growing conversation, and an image request.

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

qwen3.6-35b-a3b-4bit on M1 Max:

| scenario | prefill | decode |
|---|---|---|
| short turn | 94ms | 37 tok/s |
| system prompt, 2541 tok | 8489ms (299 tok/s) | — |
| conv delta, ~177 tok | 604ms | 47 tok/s |
| image, 987 tok | 4055ms | 21.7 tok/s |

Qwen3.5-2B-4bit, same machine: prefill ~1736 tok/s, decode ~100 tok/s,
image 987 tok → 1369ms.

Of the 35B image prefill, roughly 3.3s is text and ~0.75s is the vision
tower, so the token count dominates the tower.

**Already ruled out by measurement:**

- **Prefill chunk size** — swept 128→4096 in the code comments; throughput is
  flat. It is a memory knob, not a speed knob.
- **Compiled decode** — on/off identical (see step 1).
- **Sampling path** — greedy/no-penalty vs the defaults is identical, so CPU
  argmax over the 248k vocab is not the cost.
- **MTP** — none of the three checkpoints (`Qwen3.5-2B-4bit`,
  `qwen3.5-4b-bf16`, `qwen3.6-35b-a3b-4bit`) ships MTP tensors.
- **MoE routing** — already batched through `gather_qmm`.
- **Prompt lookup** — attempted, but the harness generated zero tokens, so
  the result is inconclusive rather than negative. Worth a better harness if
  copy-heavy agent output is the target.

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

## Verified commands

```bash
go build ./... && go build -tags ggml ./...

# MLX
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/Qwen3.5-2B-4bit \
  go test -count=1 ./llm/qwen35/ ./llm/ ./mlx/

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
