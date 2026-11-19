# sinter — handoff

Last updated: 2026-11-18 09:12

Vision input works end to end on MLX and GGML, and the bugs that were
blocking it are fixed and released. This document is the forward plan:
what to do next, in order, with the measurements and dead ends already
established so they are not repeated.

## Where things stand

- **sinter v0.8.3** is tagged and pushed. It carries everything below:
  compiled decode as the default, the three projection merges, and the
  cross-restart prefix store.
- **sprout** is on `sinter v0.8.3` (`c8c6cea954`), committed and pushed to
  `main`; publishing the sprout release is the owner's step.
- MLX is the product path (M1 Max, 64GB, 400 GB/s). GGML is secondary and has
  a known large-image instability (see Gotchas).

Shipped in this stretch: the MLX header/ABI fix, three bugs it unmasked, the
vision merger's missing biases, sharded-checkpoint `TensorShape`, the
per-image feature cache, and the image token cap (1024). Then the decode-path
defaults: pipelined decode is now opt-in and prompt-lookup is off, which
unlocks compiled decode as the default and is worth 5-10% on every model
measured. Then three bit-identical projection merges (DeltaNet inputs, routed
MoE gate/up, shared-expert inputs), and the cross-restart prefix store (see
"Just shipped" below), which takes a cold first turn on the 35B from 8.7s to
0.4s.

## Engine comparison — llama.cpp is the only verified cross-engine number

| engine | 4B | 35B-A3B | notes |
|---|---|---|---|
| sinter (compiled) | ~81 | ~63 | current default |
| sinter (eager) | — | 56.1 | `SINTER_COMPILED_DECODE=0` |
| llama.cpp (Q4_K_M) | 62.2 ± 0.8 | n/a | `llama-bench -p 0 -n 64 -r 3`, 4B GGUF |

llama.cpp is ~23% slower than sinter on the 4B. Its quant is Q4_K_M rather
than MLX 4-bit, so treat it as an order-of-magnitude check, not a
like-for-like one. It is not the tool to chase.

### mlx-lm is the reference, and sinter is 15-32% behind it

Measured on a quiet machine (load 1.7-2.6) with **token-exact prompts**: the
same token ids sinter used are fed to mlx-lm, so both engines decode the same
sequence and the same amount of work. Decode is first-generated-token to
last, excluding prefill, on both sides. mlx-lm is a venv at
`/tmp/sprout/mlxenv` (python3.12, mlx 0.32.4, mlx-lm); the Homebrew `mlx`
0.32.3 install is broken (`mlx.utils` missing) and must not be used.

| model | prompt | mlx-lm | sinter | gap |
|---|---|---|---|---|
| qwen3.5-4b-4bit | short (~63 tok) | 98.2 | 81.4 | **−17%** |
| qwen3.5-4b-4bit | long (~2500 tok) | 98.2 | 69.5 | **−29%** |
| qwen3.6-35b-a3b-4bit | short (~63 tok) | 73.4 | 62.6 | **−15%** |
| qwen3.6-35b-a3b-4bit | long (~2500 tok) | 74.9 | 51.2 | **−32%** |

Two things follow, and they are the most useful facts in this document:

1. **The gap is real** — 15-17% at short context, and it reproduces across
   both models within a point or two. An earlier revision of this document
   recorded 98.8/76.0 from a run that could not be reproduced; the numbers
   above are the reproducible ones and happen to be close, but they are
   measured, not asserted.
2. **The gap widens sharply with context** — 17% → 29% on the 4B and 15% →
   32% on the 35B, going from a ~63-token prompt to a ~2500-token one (the
   same short prompt behind a repeated system prefix). mlx-lm is flat across
   both (98.2/98.2 and 73.4/74.9) while sinter loses 12-19%.
   **Something in sinter's decode scales with sequence length that does not
   scale in mlx-lm.** That is the target, and it is a different target from
   every per-step-overhead hypothesis chased so far.

**The capacity-sized buffer is ruled out as the cause.** The obvious
candidate was compiled decode's over-allocated K/V buffer: attention runs
over `capacity` positions rather than `used`, so the mask, the scatter and
SDPA all grow with the buffer, and `kvGrowStep` is 256 — at a 63-token prompt
the buffer is already 4x the sequence. Measured, it does not explain the
gap: sinter's **eager** path uses an exact-length window and no array mask,
and it degrades by the same ratio (76.4 → 68.0 tok/s, −11%, against compiled's
81.2 → 69.3, −15%). Both paths lose ~1.5ms/token going from 63 to ~2500
tokens, so the cost is in something both share — attention over a real
sequence, the DeltaNet recurrence, or the MoE — not in the buffer design.
mlx-lm loses nothing over the same range, so whatever it is, mlx-lm is not
paying it.

That narrowed the target to a per-token cost that grows with sequence length
and is common to both of sinter's decode paths. **The ablation has now
pinned it to full attention** — see "Attention is what makes the step grow
with context" below. Attention accounts for 107% of the growth; the MoE and
DeltaNet are flat.

What survives from the earlier investigation is the **source reading**, which
does not depend on mlx-lm being runnable:

- **mlx-lm does not compile.** `mx.compile` appears only in
  `sample_utils.py` (the sampling functions), never around the model forward.
  Its loop is eager plus `mx.async_eval` pipelining.
- **Its loop is the same algorithm as sinter's `usePipelined`.** Both build
  the next step chained onto the current token array, `async_eval` it, then
  read back the current one. So sinter's pipelined path measuring the eager
  number is not a bug — see "Pipelining is ruled out" below.
- **Its `KVCache` mutates in place and returns an exact-length slice**
  (`self.keys[..., prev:offset, :] = keys`), so at decode it passes **no
  attention mask**. sinter's compiled path over-allocates to a fixed capacity
  and must pass an additive array mask in every full-attention layer. Dropping
  the mask in the chain harness is worth 64µs/block (309 -> 245µs,
  0.64ms/token), and the full-capacity `Where` scatter adds ~0.29ms.
- **Its MoE and DeltaNet are less optimized than sinter's**: four separate
  DeltaNet input projections where sinter merges them into one, and two
  `gather_qmm` calls where sinter merges gate/up into one.

So the mask and scatter are real, measured differences worth ~6% of the step
— but with the mlx-lm baseline withdrawn, there is no verified evidence that
sinter is behind anything on this hardware. The honest position is that
sinter's compiled path is the fastest decode measured here.


## Do this next, in order

### 1. Decode efficiency — the cost model is measured, the ceiling is not

**Status: the cost model is measured; the ceiling is not.** Compiled decode
used to be dead code: `useCompiled` requires `!usePipelined`, and pipelined
decode was default-ON, so `SINTER_COMPILED_DECODE=0` changed nothing (which
is exactly what the earlier 47.3 vs 47.2 tok/s measured). Pipelined is now
opt-in, so compiled runs by default and wins:

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

**Where the ~15.4ms/token actually goes** (35B, short context, compiled).
Measured by switching each block off in the traced graph and timing the
decode loop (`SINTER_ABLATE=attn|moe|delta`, 3 rounds, best of each — the
numbers reproduce to ±0.5%):

| component | share | ms/token | tok/s with it removed |
|---|---|---|---|
| MoE (40 layers) | **50%** | 7.61 | 129.6 |
| full attention (10 layers) | **35%** | 5.34 | 100.1 |
| DeltaNet block (30 layers) | 7% | 1.06 | 70.1 |
| vocabulary head | 6% | ~0.9 | — |
| **total** | | **15.4** | 65.1 |

The four add up to the whole step, which is the check that makes this
trustworthy. Weight reads are ~1590MB/token (505MB DeltaNet projections,
660MB MoE, 136MB full attention, 286MB `lm_head`) — a 4ms floor at 400 GB/s,
so the model runs at roughly a quarter of peak bandwidth and **no single
block is close to its own floor**: the MoE moves 660MB in 7.6ms (87 GB/s),
full attention 136MB in 5.3ms (26 GB/s).

**Attention is what makes the step grow with context.** The same ablation on
the 4B at both prompt lengths (`TestScratchAblateSteps`, which forces a fixed
step count via `GenerateConfig.IgnoreEOS` so an ablation that destroys the
model still costs the same number of steps — `TestScratchAblate` cannot price
`attn` or `moe` because both stop on the first token):

| component | short (~63 tok) | long (~2500 tok) | change |
|---|---|---|---|
| full attention (8 layers) | 4.77 ms (40%) | 5.91 ms (45%) | **+1.14** |
| MoE (24 layers) | 4.75 ms (40%) | 4.60 ms (35%) | −0.15 |
| DeltaNet (24 layers) | 0.80 ms (7%) | 0.88 ms (7%) | +0.08 |
| **whole step** | **11.94** | **13.19** | **+1.25** |

Attention accounts for **107%** of the growth (the MoE slightly shrinks,
inside noise) and the other two are flat. That is the whole context-dependent
gap: sinter loses ~1.25ms/token going to a 2500-token prompt and attention is
all of it. The per-token growth is far larger than the ~1ms the earlier
estimate gave attention, and it is the term to attack — mlx-lm is flat over
the same range, so it is not paying this.

The MoE block is dominated by the two routed-expert gathers. Chained the
same way (`TestScratchMoeChain`, `SINTER_MOE_VARIANT` for the sub-groups),
the block is 270-300µs, and the variants separate it:

| group | µs/block | share |
|---|---|---|
| routed experts (gate_up + down gathers) | ~140-155 | ~52% |
| shared expert (3 matvecs) | 25-40 | ~12% |
| router chain | 13-28 | ~7% |
| residual (weighting, sum, reshapes) | ~60 | ~20% |

The routed gathers read 12.6MB of the block's 14.4MB at **90 GB/s**, 4.4x off
the floor, and the doc's earlier k=1..16 sweep showed the call is
fixed-cost-dominated rather than bandwidth-bound. That makes them the single
largest op group in the model (~5.6ms/token isolated across 40 layers) and
the biggest remaining target — but the cost is inside MLX's `gather_qmm`, so
a Go-level change cannot reach it. `sorted_indices` is the documented
performance hint and is currently passed false; testing it needs an argsort
binding sinter does not have, and the flat k-sweep suggests index handling is
not what costs the time.

The router is confirmed negligible (13-28µs), and ablating the score
weighting and the sum over the expert axis changes nothing (292 vs 299µs), so
the residual is run-to-run variance and non-additivity rather than a real op
group.

**The shared expert's three input projections are now one matvec.** Its
`gate_proj`, `up_proj` and `shared_expert_gate` all read the same x and write
disjoint output rows, so `MergeLinears` folds them into a single launch — two
fewer per layer, 40 layers deep. Bit-identical (same token ids), and worth
+0.5% at short context and +3.8% at 2541 tokens (interleaved, 3 pairs:
short 62.4/62.8/62.3 vs 61.8/62.5/62.2, long 50.1/51.6/50.5 vs
50.3/48.1/48.2). The long-context gain is the larger one because the saved
launches are a fixed cost per layer, so they weigh more against a smaller
per-token budget.

**Block-level ablations do not decompose — do not bisect inside a block.**
The full-attention block measures 5.3ms, but every part of it measures near
zero: merging q/k/v into one matvec is +0.4%, ablating `o_proj` is -0.3%
(noise), ablating SDPA is 0.46ms, and the K/V scatter is 65µs. That sums to
~0.5ms against a 5.3ms block. Removing a block shortens the dependency chain
feeding the next block's MoE router, so what the block-level number captures
is partly critical-path latency, not work. Use these numbers to rank blocks,
not to locate a kernel; per-kernel GPU counters are the only thing that
resolves inside a block.

**The attention block's real GPU work is 266µs, not 530µs.** Chaining the
actual `compiledFullAttentionLayer` 100 times and evaluating once
(`TestScratchAttnChain`) — which defeats CSE and amortizes one command-buffer
commit across the whole chain — gives **266µs per block, 2.66ms per token
across 10 blocks, 51 GB/s**. So about half of the 5.3ms in-model figure is
latency, and half is real GPU work that is still 8x off the 400 GB/s floor.

**Per-op isolation does not work at this scale.** Timing each sub-op over 100
distinct inputs and evaluating each output separately reports 1338µs for
`q_proj` alone (6 GB/s) — every `Eval` is its own sync point, so the numbers
are meaningless. The chained-block method above is the only trustworthy
isolation available without per-kernel counters, and bisecting *within* the
block therefore needs env-gated variants of the block itself, measured as
chains.

**This replaces the earlier floor-table estimates, which were wrong by 2-5x**
and pointed at the wrong blocks. Two in particular:

- The **vocabulary head was never the problem.** An ablation that replaces
  its output with zeros appeared to save 13.3ms (86% of the step) — but that
  variant runs at 2.21ms/token, *below* the 3.65ms weight-read floor, which
  is physically impossible. MLX skips the layer computation once the logits
  stop depending on the hidden state, so the measurement was an artifact. The
  head's real cost is ~0.9ms. **Any ablation that breaks the graph's
  dependency chain is invalid; check the result against the bandwidth floor
  before believing it.**
- The **DeltaNet block is cheap** (1.1ms), not the 4.5ms previously
  attributed to it. The projections were already merged and the recurrent
  kernel is not the bottleneck it looked like.

**Full attention is the anomaly worth chasing next**: 10 layers, 136MB of
weights, 5.3ms — 26 GB/s, the worst effective bandwidth in the model, and
534µs per layer at a context short enough that SDPA itself is trivial. Two
candidate explanations have been measured and eliminated:

- **Not the K/V `Where`-scatter.** Inflating the fixed capacity from 256 to
  16384 positions (64x) costs 4.08ms, i.e. **0.25µs per position**: ~65µs at
  natural short-context capacity and ~0.7ms at 2541 tokens. It is a real
  context-length tax (~3% at long context) but not the 5.3ms. Measured with
  `SINTER_COMPILED_CAPACITY`.
- **Not per-call overhead on the projections.** Merging q/k/v into one wide
  matmul (`MergeLinears`, bit-identical — same token ids) removes 20
  quantized matvecs per token and measured **+0.4%**, i.e. nothing. Reverted
  rather than kept for the kernel count.

That leaves the rest of the block: two RoPE calls, q/k norms, the transposes,
the output gate and `o_proj`. Ablating SDPA alone is not a usable probe — it
made decode *slower* (43.6 vs 65.1 tok/s), because feeding the query straight
through changes both the layout `o_proj` sees and the downstream MoE routing,
so the comparison is not like-for-like.

**Closure staging is not a cost — that line was wrong.** The earlier estimate
of ~1.5ms for staging 30 recurrent states in and out does not survive
measurement. `Closure.Apply` passes array handles, not data: a 65-input apply
dispatches in **73µs** (`TestSpikeApply65Inputs`), and the whole per-token
apply is 1ms of CPU against 12ms of GPU, so it is entirely hidden. The
recurrent state's 120MB/token of read+write is real, but it is the kernel's
own traffic, not a boundary copy.

The recurrent kernel looked like the worst offender on paper — it loads a 2MB
fp32 state, does one step of ~8 flops per element, and stores it back, a
memcpy-shaped kernel — but the ablation shows the whole DeltaNet block is
only 1.1ms of the step, so it is not where the time is. It is fp32 because
the checkpoint declares `mamba_ssm_dtype: float32`, so halving it to bf16 is
a numerics change, not a free win, and not worth spending on a 7% block.

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
something per-layer is paying a fixed cost. The component ablation above
narrows it to two blocks — MoE (7.6ms) and full attention (5.3ms) — and
rules out the DeltaNet block (1.1ms) that this table's averaging hid.

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

So the time is inside the kernels, which is what the component ablation
already said. Candidate #1 is dead; only kernel speed is left.

Method, since the trace tooling was not obvious: `.gputrace` bundles are
Xcode's format and are not parseable (Apple's private "MTSP" container, and
21GB because they dump every buffer). Opening one for the 35B made Xcode try
to address 355GB on a 64GB machine, so that route is a dead end at this
model size. `xctrace record --template 'Metal System Trace'` produces a
158MB `.trace` whose tables `xctrace export` writes as XML, and a Go test
that writes wall-clock markers around the decode lets the window be cut
exactly. See Verified commands.

**Target.** There is no verified external reference to aim at. The old
">100 tok/s" figure was never supported by a measurement, and the
"~65 tok/s ceiling with today's kernels" claim that replaced it rested on the
mlx-lm numbers that are now withdrawn (see "Engine comparison" above).
sinter's compiled path at ~63 tok/s on the 35B is the fastest decode measured
on this hardware. What is defensible is a floor, not a ceiling: the step is
15.9ms and the weight reads alone are ~4.2ms at the machine's 400 GB/s, so
there is real headroom, but nothing here quantifies how much.

Two blocks dominate the step, in this order:

1. **MoE — 7.6ms in-model; ~140-155µs/block isolated, of which the two
   routed-expert gathers are ~52%.** The largest single op group in the model
   (~5.6ms/token isolated across 40 layers), reading 12.6MB at 90 GB/s. The
   `gather_qmm` slot sweep says the call is fixed-cost dominated (k=1 and k=8
   cost the same), so the lever is per-call overhead, not bytes — and that
   overhead is inside MLX, not reachable from Go. `sorted_indices` is the one
   documented hint still untested; it needs an argsort binding. Note mlx-lm
   uses *two* `gather_qmm` calls here where sinter merges to one, so this
   block is not where sinter loses to mlx-lm.
2. **Full attention — 2.66ms of real GPU work per token (266µs/block, 51
   GB/s), plus ~2.6ms of critical-path latency.** The isolated block is 8x
   off its 400 GB/s floor. Two measured differences from mlx-lm: the additive
   mask mlx-lm avoids by keeping its cache exact-length (0.64ms/token), and
   the full-capacity `Where` scatter (~0.29ms). Bisecting further needs
   env-gated variants of the block measured as 100-iteration chains
   (`TestScratchAttnChain`); per-op timing and in-model sub-ablations both
   fail here, for the reasons above.

**Pipelining is not a lever here.** mlx-lm's `generate_step` and
sinter's `usePipelined` loop are the *same algorithm*: build the next step
chained onto the current token array, `async_eval` it, then read back the
current one — so the read-back of step i overlaps the dispatch of step i+1.
Read both side by side (`mlx_lm/generate.py` lines 454-467 vs the
`usePipelined` branch in `llm/model.go`). sinter's pipelined path measuring
the eager number is therefore not a bug to fix: with a 95%-busy GPU and a
hard data dependency between consecutive tokens, there is almost nothing to
overlap. Do not spend more time here.

**What compiled decode costs, and what it buys.** Measured per token:

| cost | per token | why sinter pays it |
|---|---|---|
| closure staging | ~1.5ms | 30 DeltaNet recurrent states cross the closure boundary in and out every step |
| additive SDPA mask | 0.64ms | the K/V buffer is capacity-sized, so attention needs a mask |
| in-graph `Where` scatter | 0.29ms | the new K/V row is written by a full-capacity select |

All three follow from needing static shapes. The eager path avoids all of
them — `AppendWindow` returns a slice to `used` and `forward.go` passes no
array mask — and still measures 56 tok/s against compiled's 63. So compiling
the elementwise chain is worth more than the 2.4ms these three cost (2.2ms
eager vs 0.4ms compiled for norms and residual adds alone). Compiled decode
is therefore the right default, and these three are its price, not a bug to
remove.

**The headroom, now identified.** The mlx-lm comparison above is the verified
reference: 15-17% at short context and 29-32% at ~60 tokens, on both models.
The step is 15.9ms and the weight reads alone are ~4.2ms at 400 GB/s, so
there is room; the context-dependent part of the gap is the part worth
chasing, because it is the part that grows.


Per-kernel GPU counters would settle the rest. The Metal System Trace has the
Shader Timeline disabled and xctrace exposes no option to enable it, so it
gives coverage but not per-kernel attribution. Xcode's Metal debugger is the
only route, and the 35B's `.gputrace` is not openable on a 64GB machine — use
a smaller model there.

The capture path is verified working — see the command in Verified
commands. It needs `MTL_CAPTURE_ENABLED=1`, which the earlier attempt was
missing.

**Consumer follow-up — done.** sinter v0.8.3 is tagged and pushed, and sprout
is on it (`c8c6cea954`): both `PromptLookupMaxDrafts = 6` assignments are
gone from `pkg/localmodel/local_provider_chat.go`, so the compiled path is
what production now runs. Verified by the gate chain rather than by
inspection alone: `DefaultGenerateConfig` leaves `PromptLookupMaxDrafts` at
0, sprout sets `Temperature=0`/`RepetitionPenalty=0`, so `useGPUArgmax` is
true, `usePromptLookup` is false, and `useCompiled` engages.

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
- **Pipelined decode as a source of headroom** — sinter's loop is the same
  algorithm as mlx-lm's `generate_step` (build next chained on the current
  token, `async_eval`, then read back the current). It measures the eager
  number because with a 95%-busy GPU and a hard token-to-token data
  dependency there is nothing to overlap. Also slower than compiled on all
  three models (35B 52.6/53.0, 2B 112.3/116.3, 4B 75.3/75.3). Opt-in via
  `SINTER_PIPELINE_DECODE=1`; do not re-chase it.
- **Prompt-lookup** — slower than compiled even on verbatim-echo prompts
  (35B 51.6-53.8 at k=4 and k=6, 2B 106-114). Instrumented: 362 of 414 decode
  steps found no n-gram candidate, so the search and verify batches cost more
  than the accepted drafts save. `openaisserver` no longer enables it, and
  sprout no longer sets it.
- **MoE expert dispatch being the *dispatch*'s fault** — the block is 50% of
  the step (7.6ms for 660MB, 87 GB/s), but the `gather_qmm` slot sweep shows
  k=1 and k=8 cost the same, so the cost is per-call fixed overhead, not the
  expert selection or the bytes gathered.
- **Kernel count** — merging the four DeltaNet input projections into one
  matmul (bit-identical) is worth 3%, not the 5x a standalone benchmark
  suggests. But do not conclude that kernels never matter: the fused
  gated-delta kernel is worth **+14%** over the ops loop (63.6/63.6 vs
  55.7/55.3 short, 58.6/57.2 vs 50.9/50.2 at 2541 tokens). The difference is
  what the ops loop writes: ~20 intermediates per layer over a 2MB state
  array. Bytes to intermediate buffers is the lever, launch count is not.
  Merging full attention's q/k/v the same way (bit-identical, 20 fewer
  matvecs per token) measured **+0.4%** and was reverted — the lever stops
  paying outside the DeltaNet block.
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
- **An ablation that breaks the graph's dependency chain is invalid.** Replacing
  the vocabulary head's output with a constant made decode appear to run at
  2.21ms/token — below the 3.65ms floor implied by the model's own weight
  bytes, which is impossible. MLX stopped computing the layers once nothing
  downstream depended on them. Always check an ablation against the
  bandwidth floor before believing it; ablations that keep the head depending
  on the real hidden state (`attn`, `moe`, `delta`) are the ones that hold up.
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

# mlx-lm reference on the same token ids (token-exact, decode-only tok/s).
# Needs a venv with mlx + mlx-lm; the Homebrew mlx install is broken (no
# mlx.utils) and cannot import mlx_lm. See bench/README.md.
python3.12 -m venv /tmp/sprout/mlxenv && /tmp/sprout/mlxenv/bin/pip install mlx mlx-lm
/tmp/sprout/mlxenv/bin/python bench/mlxlm_tokenexact.py \
  /Users/alanp/dev/llm-models/qwen3.5-4b-4bit 3

# Component cost attribution (scratch, SINTER_ABLATE=attn|moe|delta)
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.6-35b-a3b-4bit \
  go test -count=1 -run TestScratchAblate -v ./llm/qwen35/

# Chained-block timing (scratch; the only trustworthy isolation without
# per-kernel counters — 100 chained iterations, one eval)
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.6-35b-a3b-4bit \
  go test -count=1 -run TestScratchAttnChain -v ./llm/qwen35/

# MoE block chain (scratch; SINTER_MOE_VARIANT=norouted|norouter|noshared|nosum
# bisects its op groups)
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.6-35b-a3b-4bit \
  go test -count=1 -run TestScratchMoeChain -v ./llm/qwen35/

# K/V buffer capacity sweep (scratch; SINTER_COMPILED_CAPACITY isolates the
# O(capacity) scatter cost)
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.6-35b-a3b-4bit \
  go test -count=1 -run TestScratchCapacity -v ./llm/qwen35/

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
# inserted".
#
# This route is effectively unusable for a large model. The bundle is 21GB for
# the 35B (it dumps every buffer), only Xcode can open it (not Instruments),
# it is Apple's private format and not parseable — and opening it made Xcode
# try to address 355GB of memory on a 64GB machine, so it had to be killed.
# Use the xctrace route below instead.
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
