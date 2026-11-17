# Vision input for sinter — progress

Last updated: 2026-11-17 13:20

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

## Vision tower output is cached per image

The prefix cache skips the *text* prefill for a repeated prompt, but a new
question about an image already seen still re-ran the vision tower. The
tower's output depends only on the prepared pixels, so `imageFeatureCache`
in `llm/qwen35/vision_cache.go` keys it on `vision.Pixels.Hash` — patch grid
plus every pixel value — and an 8-entry LRU serves repeat images.

Measured on MLX (`Qwen3.5-2B-4bit`), a 1024x768 image (3072 patches, 768
tokens), distinct prompt text each turn so the prefix slot cannot help:

| | request | prefill |
|---|---|---|
| tower re-run | 1.26s | ~0.98s |
| tower cached | 0.70s | ~0.42s |

The tower alone is 510ms at that size (24 blocks × ~48ms; GGML 2.9s).

**Entries are host-side float32, not backend arrays.** Holding a retained
GPU array across calls is not safe on GGML: `registerArray` keys the
tensor→Array registry on the C tensor pointer, so `RetainArray` overwrites
the original wrapper's entry and freeing the original deletes it, leaving
the retained wrapper pointing at a tensor with no registered logical shape.
A later `Reshape` then computes element counts from a stale shape and
`ggml_reshape_1d` aborts. Keeping the values on the host costs one upload
(~6MB) per hit, which is negligible next to the encoder it replaces.

`TestImageFeatureCacheSkipsTower` asserts both halves — the features match a
fresh encode bit-for-bit, and the warm call is at least 4x faster — because
a cache that returned wrong features would still look fast and one that
silently missed would still return correct features. Verified to fail with
`SINTER_VISION_CACHE_SIZE=0` (525ms vs 561ms cold).

## Fixed: batched GGML concatenation returned garbage

Writing the multi-image cache test surfaced a pre-existing backend bug.
`GGMLBackend.ConcatenateAxis` chained `ggml_concat` nodes in the recyclable
ops context, then queued the result under batched evaluation. A flush
triggered by a later op recycles that context, so the queued graph read freed
nodes — and a multi-image vision encode returned raw patch data instead of
merged features. Single-image encodes never concatenate, which is why this
went unnoticed; it reproduces on HEAD (2 images, batched GGML, every value
wrong) and disappears with `SINTER_GGML_BATCH=0`.

`ConcatenateAxis` now flushes pending ops and evaluates through
`evalOpEager`, so the chain runs before anything can recycle its nodes.
Concatenation is rare and off the decode path, so the synchronous eval costs
nothing measurable. `TestImageFeatureCacheMultiImage` is the guard: it checks
cold, all-cached and mixed multi-image encodes agree bit-for-bit, and it
fails without the fix.

## Fixed: the merger's biases were never applied

The tower's last step is a 2x2 patch merge: LayerNorm over the tower width,
concatenate each block's four patches, then two linear layers down to the
text hidden size. `loadVisionTower` loaded both weights and neither bias, so
every image reached the language model with a constant offset missing from
its features.

It degraded rather than broke, which is why it survived. All 24 blocks
matched the reference at cosine 1.000000; only the merged output was off
(cosine 0.969, mean absolute 0.012). The committed parity test gates on mean
absolute difference at 0.03, so the bug sat under the gate and passed.

Found by dumping the reference's per-block hidden states and comparing block
by block — everything matched until the merger. With both biases applied the
merged output matches the reference to 2e-4.

The visible effect was image degeneration at larger token counts. On a
1080x2340 screenshot, 987 image tokens used to collapse into a repeated
character loop (`C D E F A B C D E F...`); it now transcribes the screen
correctly. 2482 tokens still degenerates, which looks like the model's own
limit rather than a missing weight.

`TestMergerAppliesBias` pins the merger with hidden size 1, where LayerNorm
collapses to zero and the expected output is exact; it fails with either
bias detached. The parity gate is now 0.015, below the 0.020 this bug
produced.

## The reference harness is only trustworthy for the tower

Worth knowing before the next parity attempt: the transformers harness in
`/tmp/sprout` produces degenerate text at *every* image scale — 144, 220,
406, 495 and 629 tokens — while sinter transcribes the same screenshot
correctly at 495. Its `lm_head` is properly tied to `embed_tokens`, so the
output head is not the cause; the text path is broken somewhere else (the
`linear_attn.conv1d` layout guess in `ref_load.py` is the likely suspect).

So the harness's *vision tower* dumps are usable and its *generated text* and
*logits* are not. The logit-parity and layer-parity scratch tests compare
against those unusable values and should be read as "the reference is wrong
here", not as sinter diverging.

## Image token budget

Image cost is capped per request. `DefaultImageMaxTokens` (1024) lowers the
pixel budget `smart_resize` already enforces, so a large screenshot is
resized down before patchifying rather than truncated. `New` installs it, and
a caller with a bigger model can raise `Server.ImageMaxTokens`.

The measurements behind 1024, on a 1080x2340 screenshot:

| image tokens | 2B prefill | 2B answer | 4B prefill | 4B answer |
|---|---|---|---|---|
| 2482 | 4517ms | degenerate | 7152ms | correct |
| 987 | 1369ms | correct | 2371ms | correct |
| 495 | 652ms | correct | 1357ms | correct |
| 230 | 335ms | correct | 714ms | correct |
| 112 | 181ms | correct | 461ms | correct |

The cap sits where the weakest supported model stays coherent, not where the
largest is fastest: the 2B collapses into a repeated character loop above
roughly a thousand tokens, while the 4B reads 2482 correctly.

## Verified commands

```bash
go build ./... && go build -tags ggml ./...          # both OK

# MLX
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/Qwen3.5-2B-4bit \
  go test -count=1 ./llm/qwen35/ ./llm/ ./mlx/

# GGML (run serially — see below)
SINTER_BACKEND=ggml SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/qwen3.5-2b-4bit \
  go test -count=1 -tags ggml ./llm/vision/ ./llm/mropeparity/ ./llm/openaisserver/

# Vision tower feature cache: the warm call must skip the encoder
SINTER_BACKEND=metal SINTER_VISION_MODEL=/Users/alanp/dev/llm-models/Qwen3.5-2B-4bit \
  go test -count=1 -run TestImageFeatureCacheSkipsTower -v ./llm/qwen35/
```

## Pre-existing failures (not caused by this work)

- `TestPrefixCacheParity`, `TestPrefixCacheSpeedsUp` — need a registered
  backend; fail identically on HEAD.
- `TestProbeQwen35GGML` — crashes in `ggml_backend_graph_compute` during
  pipelined decode; reproduces on HEAD. It takes the process down, so a
  multi-package `-tags ggml` run reports unrelated packages as FAIL when it
  fires; run packages individually to see real results.
- `TestImageCacheKeyLiveModel` on GGML — intermittently faults in
  `ggml_reshape_1d` during a repeated image prefill. Clean HEAD crashes at
  the same rate (9/14 pass vs 10/14 with the feature cache), and it
  reproduces with the feature cache disabled entirely, so it is not the
  cache. Re-run to get a verdict.
- Large image prefills on GGML abort. A 987-token image through the HTTP
  path dies in `gatedDeltaNet.forward`'s `z` reshape
  (`GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b))`), and with
  `SINTER_GGML_DEBUG=1` set it instead fails earlier in the prefix snapshot
  (`pin of 16777216 bytes exceeds tensor nbytes 73728`). Two different
  failure modes on the same input says the GGML path is unstable at this
  length rather than wrong in one place. Small images are fine — the 96px
  fixture passes — and the same 987-token request is correct on MLX. Not
  caused by the token cap: it changes the image size, not the text path.
- `TestDeltaKernelByteIdentity`, `TestDeltaKernelNilState`,
  `TestCPUQuantFastPaths` in `tensor/ggml` — fail identically on HEAD.

## Fixed along the way

- `GGMLBackend.ConcatenateAxis` returned garbage for inputs that were still
  pending under batched evaluation (see above). Found by the multi-image
  feature-cache test; it affected every multi-image vision prefill on GGML.
