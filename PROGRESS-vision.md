# Vision input for sinter — progress

Last updated: 2026-11-16 13:50

## Status

Vision works end to end on **both backends**, and an exact prompt repeat —
image or text — now skips the prefill (309ms → 2ms and 44ms → 1ms). GGML was
finished first; MLX was blocked by an ABI mismatch that has now been found
and fixed, and the three bugs it had been masking are fixed too.

Verified on MLX (`Qwen3.5-2B-4bit`) over HTTP: a base64 PNG of vertical
stripes returns "this image displays a vibrant, abstract pattern composed
of vertical stripes that transition smoothly from deep blue on the".

## The MLX blocker: header/library ABI skew

The installed `libmlxc` is 0.7.0; the vendored headers were 0.6.0. mlx-c
passes structs and enums by value, so a stale header misreads arguments
instead of failing to link. v0.7.0 inserted a `bool force_fused` before the
stream in `mlx_fast_scaled_dot_product_attention`, so under the old header
the stream handle landed in the `force_fused` slot and the real stream was
read from whatever followed — producing

> force_fused=True but no fused kernel is available: the fused kernels
> require a GPU (Metal) stream

on every MLX model path. That message is why the earlier diagnosis went
looking at the device and the stream, both of which were correct all along
(`mlx_get_default_device` → gpu, both stream constructors → gpu).

Found by calling `libmlxc` from Python `ctypes` with the 0.7.0 signature:
the same call succeeded immediately. Confirmed by diffing every prototype
between the two header sets — `mlx_fast_scaled_dot_product_attention` is the
only symbol this package uses whose signature changed (145 used, 1 changed).

Fix: refresh `mlx/mlxc_headers` to v0.7.0, regenerate the dispatch shim,
add the `forceFused` argument. The `tensor.Backend` interface is unchanged;
the Metal backend passes `false` so unsupported shapes take the unfused
fallback instead of erroring.

## Bugs the ABI fix unmasked

1. **`ApplyMRoPE` rotated the full head dim** — it was handed a
   `rotaryDims`-wide table but rotated all 256 columns, so the first real
   image failed on a broadcast mismatch. Now takes `rotaryDims` and stitches
   the untouched tail back on. Covered by `TestApplyMRoPEPartialRotation`
   (verified to fail against the old behaviour).
2. **The vision tower kept a stale stream** — it captured the model-load
   stream but runs on the inference thread, and MLX streams only evaluate on
   the thread that created them. `Qwen35.SetStream` now propagates to the
   tower.
3. **`loadFlattenedTransposed` returned a lazy graph** — unlike
   `loadLinearT`, it never evaluated its transpose/reshape, so the patch-embed
   weight carried the load-time stream into the inference thread. Same
   failure mode, different array. Now evaluates before returning.
4. **Double-free in the tower's patch embed** — `defer h.Free()` was bound
   before `h` was reassigned, leaking the matmul result and freeing the bias
   result twice.
5. **Vision attention passed `"none"`** — mlx-c rejects it; full attention is
   an empty `mask_mode` with no mask array. GGML treats anything that is not
   `"causal"`/mask as full attention, so `""` is correct for both.

## What works

End to end over HTTP against `qwen3.5-2b-4bit` on GGML and `Qwen3.5-2B-4bit`
on MLX: a request carrying a base64 PNG of vertical stripes returns a
description of vertical stripes.

Parity against the reference (`Qwen3VLVisionModel` with the checkpoint's own
weights):

- patch-embed permutation and position interpolation: bit-exact
- rotary tables: agree to 3e-7
- LayerNorm and both GELUs: exact
- SDPA and Linear: within 4e-4 of a CPU reference
- mRoPE text tables: match to 2.4e-7
- text-only output: byte-identical to the pre-change build

## Corrections to the original estimate

1. **"27-layer" was wrong for this checkpoint** — depth 24, hidden 1024,
   16 heads, head_dim 64.
2. **`deepstack_visual_indexes` is `[]`** — no per-layer merge into the
   language model. One merger MLP at the end.
3. **Patch embed is Conv3D, not Conv2D** — reshape to `[N, 1536]` and
   matmul. No new backend op.
4. **`<|image_pad|>` expansion is not a template job** — the run length
   depends on the image grid, known only after preprocessing.
5. **`head_dim` is 256, not 64**, and `partial_rotary_factor` is 0.25. So 64
   dims rotate → 32 frequencies, which is what `mrope_section=[11,11,10]`
   sums to.
6. **`rope_theta` is 10000000**, not 10000.

## Prefix caching: landed, by keeping the hidden state

The plan called for content-keyed prefix caching that skips work. An early
attempt measured **3.32s → 77ms (43×)** and **produced a wrong answer**,
which was put down to a DeltaNet layer's recurrent state being one
accumulated value with no way back.

That was true but framed the problem wrong. Nothing needs rewinding: the
logits come from the last position's hidden state, which is `[1,1,H]` — a
few KB — not from the recurrent state. So the cache now keeps it.

`KVCache` carries `lastHidden`, set during prefill, copied by
`SnapshotPrefixCopy`, retained by `RestorePrefix`, released by `Free`, and
written to the spill file (format version 1 → 2, so an old file is rejected
rather than misread). `CachedHiddenArchitecture` exposes `LastHidden` and
`LogitsFromHidden`; qwen35 implements it, and an architecture that does not
keeps the old full-prefill behaviour.

On an exact prompt hit — image or text — the code restores the slot's cache
and runs only the final norm plus the vocabulary head. **Prefill 309ms → 2ms**
for images and **44ms → 1ms** for text, with the answer identical across
runs, including after the slot has been spilled to disk and reloaded.

Image prompts now store a prefix slot; the keying (`imageKeyFor`, hashed
over preprocessed pixels + grid + order) was already correct and is what
makes the match safe.

Two tests guard it, because the earlier attempt produced a plausible but
wrong answer:

- `TestImageCacheHitSkipsPrefill` and `TestTextExactHitSkipsPrefill` assert
  the work actually went away, by comparing warm prefill against a cold run.
  The correctness tests next door would pass even if the hit silently fell
  back to a full prefill. Both verified to fail with the optimization
  disabled (126ms → 105ms, and 34ms → 35ms).
- `TestTextMultiTurnStillDeltas` checks the other direction: a growing
  conversation must keep taking the delta path, or every turn would return
  the first turn's answer.
- `TestKVCacheDiskRoundTrip` covers the hidden state, so a spill that
  dropped it surfaces here rather than as a silent full prefill on reload.

Getting this far also exposed two GGML ownership bugs, both fixed:

1. **`Array.Free` freed the graph allocator unconditionally**, before the
   refcount check. A retained array shares the tensor and refcount but has
   no allocator of its own, so freeing the original left every retained
   wrapper pointing into a freed graph buffer.
2. **`SnapshotPrefix` retained arrays into recyclable storage.** The KV
   cache's buffers go back to the pool on the next forward, so a snapshot
   taken before a prefill and read after it saw recycled memory. Prefix
   slots now use `SnapshotPrefixCopy`, which reads back and rebuilds;
   `SnapshotPrefix` remains for the MTP and prompt-lookup rollback paths,
   which free their snapshot before anything else runs.

Still open: nothing on the caching path. A text prompt whose whole prompt
matches a slot now takes the same route as an image prompt (44ms → 1ms),
and a growing conversation still uses the delta path.

## Verified commands

```bash
go build ./... && go build -tags ggml ./...          # both OK

# MLX
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/Qwen3.5-2B-4bit \
  go test -count=1 ./llm/qwen35/ ./llm/ ./mlx/

# GGML (run serially — see below)
SINTER_BACKEND=ggml SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.5-2b-4bit \
  go test -count=1 -tags ggml ./llm/vision/ ./llm/mropeparity/ ./llm/openaisserver/
```

## Pre-existing failures (not caused by this work)

- `TestPrefixCacheParity`, `TestPrefixCacheSpeedsUp` — need a registered
  backend; fail identically on HEAD.
- `TestProbeQwen35GGML` — crashes in `ggml_backend_graph_compute` during
  pipelined decode; reproduces on HEAD. It takes the process down, so a
  multi-package `-tags ggml` run reports unrelated packages as FAIL when it
  fires; run packages individually to see real results.
- `TestImageCacheKeyLiveModel` on GGML — intermittently faults in
  `ggml_reshape` during a repeated image prefill. Clean HEAD crashes at the
  same rate, and it reproduces with prefix-slot storage disabled entirely,
  so it is not the cache. Re-run to get a verdict.
- `TestDeltaKernelByteIdentity`, `TestDeltaKernelNilState`,
  `TestCPUQuantFastPaths` in `tensor/ggml` — fail identically on HEAD.
