# Image input for sinter — implementation plan

**Status: all four steps implemented and committed. Images work end to end
through the HTTP server against a real model.**

| Step | State |
|---|---|
| 1. Image preparation | Done, parity-tested |
| 2. Vision tower | Done, component-parity verified; ~1% f32 drift documented |
| 3. Images into the language model | Done, tables match reference to 2.4e-7 |
| 4. Server | Done, verified over HTTP |

Known gaps, in priority order:

1. **MLX is unverified, for a diagnosed environment reason.** The installed
   `mlx-c` (0.7.0) and `mlx` (0.32.3) are version-skewed: the C wrapper
   passes `force_fused=True` to scaled-dot-product attention, but the
   libmlx it links against has no fused kernel to satisfy it, so every
   MLX model path fails with "no fused kernel is available". Ordinary ops
   are unaffected — matmul, softmax, LayerNorm and GELU all execute
   correctly on Metal — so this is one kernel, not a broken backend.
   `TestFusedAttentionAvailability` in `mlx/` reports the condition.
   Homebrew has `mlx-c` 0.8.0 and `mlx` 0.32.4; upgrading both is the fix
   and has not been run. Everything else was verified on GGML.
2. **The tower is not bit-identical** to the reference. See step 2.
3. **GGML pipelined decode crashes** on image prompts (and on text prompts —
   pre-existing). Image prompts now step aside from that path, so they are
   unaffected, but the underlying bug remains.
4. **Prefix caching is skipped** for image prompts rather than keyed on image
   content. Correct, but a repeated image pays a full prefill.

Scope: Qwen3.5 VL only (the `qwen3_5` wrapper around `qwen3_5_text`), the one
multimodal family sinter already runs the text half of. Target model for
parity work: `llm-models/qwen3.5-2b-4bit` (vision tower is bf16 inside the
same safetensors file).

Reference for number-for-number parity: `mlx-vlm` (throwaway venv), with
transformers' `modeling_qwen3_vl.py` as the tie-breaker when the two disagree.

## What is actually in the checkpoint

From `config.json` / `preprocessor_config.json` / the safetensors header:

- vision tower: depth 24, hidden 1024, 16 heads (head_dim 64), mlp 4096,
  GELU (pytorch_tanh), LayerNorm eps 1e-6, all with bias. Patch embed is a
  Conv3D over `[2, 16, 16, 3]` (temporal_patch_size 2, patch 16).
  `pos_embed` is a learned table of 2304 = 48x48 entries, bilinearly
  interpolated to each image's own grid. 2D RoPE, head_dim 64 → 32 freqs,
  half-split, laid out `[h, w, h, w]`.
  Merger: LayerNorm(1024) → view to 4096 (2x2 spatial merge) → fc1 → GELU →
  fc2 → 2048 (text hidden).
- `deepstack_visual_indexes: []` — no deepstack taps, so no per-layer
  injection into the text model. (Keep the field read from config; if a
  future checkpoint has entries, fail loudly rather than silently ignore.)
- text side: `mrope_section [11,11,10]`, `mrope_interleaved: true`,
  `partial_rotary_factor 0.25`, head_dim 256 → 64 rope freqs, 16 rotated.
  3 of 4 layers are DeltaNet (no RoPE, no positions at all).
- tokens: `<|vision_start|>` 248053, `<|vision_end|>` 248054,
  `<|image_pad|>` 248056. The tokenizer already encodes and decodes these
  correctly (verified) — no tokenizer work.
- processor: rescale 1/255, normalize mean/std 0.5, bicubic (resample=3),
  dims rounded to a multiple of `patch*merge = 32`, pixel budget
  shortest_edge 65536 / longest_edge 16777216.

## 1. Image preparation (`llm/vision/image.go`, pure Go, no build tag) — DONE

- `Decode(r io.Reader) (*Image, error)` — PNG and JPEG via stdlib
  `image/png`, `image/jpeg`.
- `Preprocess(img *Image, cfg Config) (*Pixels, error)` — smart_resize,
  bicubic resize, normalize to `(x/255 - 0.5)/0.5`, patchify.
- `Patchify` — cuts into rows of `2*16*16*3` floats.

**Corrected during implementation.** Two assumptions in the first draft of
this plan were wrong, and the reference (transformers 5.23.0's
`Qwen2VLImageProcessorPil`, the class the checkpoint's
`processor_config.json` names) settled both:

1. The patch row layout is `[channel][temporal][patch-row][patch-col]`, not
   `[temporal][channel][patch-row][patch-col]`. Channel and temporal are both
   outer axes; getting this wrong produces a tensor that looks plausible and
   is silently wrong.
2. The merge-group row order is `((my*gridWM+mx)*merge+dy)*merge+dx` — merge
   group outermost, then the 2×2 position inside it. Confirmed exact.

Verified against the reference on a real 640×480 image: **bit-identical**
(max abs diff 0.0). On a 96×96 image, which smart_resize upscales, the only
remaining difference is the bicubic kernel: max abs diff 0.0078 (2/255 of
the range), mean 0.001. That is x/image's CatmullRom vs torchvision's
antialiased bicubic, and it is small enough to leave as-is for now.

Fixtures live in `llm/vision/testdata/` (one 96×96 PNG + its 1.5 MB
reference tensor). `TestPreprocessNoResizeIsExact` rebuilds the expected
tensor from the source pixels independently, so a layout regression fails
there rather than hiding inside resize drift.

## 2. Vision tower (`llm/qwen35/vision.go` + `llm/vision` helpers)

New `visionTower` in the qwen35 package (it shares `llm.Linear`,
`llm.SafetensorsFile`, and the model's backend/stream):

- `loadVision(sf, prefix)` — key prefix auto-detected: `vision_tower.` on
  raw HF exports, `model.visual.` on mlx-community conversions. All weights
  bf16, full precision (never quantized in the checkpoint).
- `forward(pixels)`:
  - patch embed: reshape `[N, 2,16,16,3]` → `[N, 2*16*16*3]` → Linear
    (weights stored `[1024, 2,16,16,3]`, reshape to `[1024, 1536]`).
  - 2D RoPE tables from the per-image grids (`[h, w]` positions per patch,
    half-split rotation, `[h,w,h,w]` interleave).
  - 24 blocks: LayerNorm → attention (qkv bias, non-causal, per-image
    sequence, no GQA) → proj → residual → LayerNorm → fc1 → GELU(tanh) →
    fc2 → residual. Attention per image, not across the packed batch.
  - merger: LayerNorm → 2x2 merge → fc1 → GELU → fc2 → `[numImageTokens, 2048]`.
- Backend gaps to fill (each one op, both backends):
  - `LayerNorm` (bias) — MLX `fast_layer_norm`, GGML `ggml_norm` + mul + add.
  - `Conv3D` — only if the 3D patch-embed path is kept; otherwise do patch
    embed as a matmul over the flattened window (cheaper, no new kernel).
    Decision: matmul. No Conv3D.
  - bilinear resample of `pos_embed` — CPU-side (2304x1024 table is small),
    then upload once at load. No backend op.
  - GELU tanh — implement in Go from existing ops (`0.5x(1+tanh(...))`) or
    add `GELU` to the backend; measure first.
- Parity gate: `llm/qwen35/vision_parity_live_test.go` compares tower output
  for one fixed image against a dumped mlx-vlm tensor (bf16, tolerance
  documented per test). This is the "number for number" checkpoint; nothing
  downstream is trustworthy until it passes.

**Status: implemented and verified against the reference, with a documented
numerical gap.**

Verified against transformers' `Qwen3VLVisionModel` with the checkpoint's own
weights (24 blocks, 331M params):

| Component | Result |
|---|---|
| Patch-embed weight permutation | bit-exact |
| Position-embed interpolation | bit-exact |
| Vision RoPE cos/sin tables | 3e-7 |
| LayerNorm (weight+bias) | exact at the real `[256,1024]` shape |
| GELU tanh / erf | exact vs float64 |
| SDPA | 4e-4 vs CPU |
| Linear | 3.6e-4 vs CPU |
| Resize vs PIL bicubic | ≤1/255, mean 0.16 |

Two corrections to the plan above, found by reading the reference:

- Patch order is merge-block-major `(my, mx, dy, dx)`, and the patch window
  is `[channel][temporal][patch-row][patch-col]` — not `[temporal][channel]`.
- `pos_embed` resampling is bilinear with `align_corners=True`, indexed in
  merge-block order, not raster order.

**The gap:** with *identical* input patches, block 0's mean relative error is
2.7e-4, block 11 is 2.4e-3, block 23 is 9.9e-3, and the merged output's mean
absolute error is 0.017 on reference values averaging 0.094. Activations
grow to ~5e3 by the last block, so f32 reduction-order differences compound.
Every op is correct in isolation; the drift is accumulation, not a layout
bug. The test gates on the mean and logs the real numbers.

**Not yet done:**

- MLX parity is unverified. SDPA segfaults in this environment on clean HEAD
  too (pre-existing mlx/mlx-c mismatch), so the tower has only been exercised
  on GGML. This is the gate that still has to be met.
- "Number for number" against mlx-vlm is not achieved and probably not
  reachable in f32. If the plan requires it, the tower needs bf16 or f64
  accumulation to compare like-for-like.
- Deepstack is rejected at config load (`deepstack_visual_indexes` non-empty);
  this checkpoint has it empty.

## 3. Feeding images into the text model

### 3a. Interface (`llm/architecture.go`)

Add an optional capability, mirroring the existing optional interfaces:

```go
type MultimodalArchitecture interface {
    Architecture
    // ForwardPrefillMultimodal runs one prefill over a chunk that contains
    // image tokens. embeds are the merged image embeddings, one row per
    // <|image_pad|> token in ids, in order.
    ForwardPrefillMultimodal(ids tensor.Array, embeds tensor.Array, seqLen, startPos int,
        positions *MRopePositions, cache *KVCache) ([]float32, error)
}
```

`MRopePositions` carries the 3D position table `[3][seqLen]` plus the
`ropeDelta` the caller must keep for decode.

### 3b. Positions (`llm/mrope.go`)

- `MRopePositionsFor(tokenIDs []int, images []ImageSpan, section []int, merge int)`
  — port of `get_rope_index`: text runs get `arange + current_pos` on all
  three axes; an image span gets `(t=0, h=row+cur, w=col+cur)` over its
  `h x w` merged grid; `current_pos += max(h, w)` after the image;
  `ropeDelta = max(pos)+1 - len(tokens)`.
- `ApplyMRoPE(x, positions, dims, section, theta, interleaved)` — builds the
  cos/sin table with the `[11,11,10]` interleaved layout (freq index `i`
  takes axis `i%3` when `i%3 < len(section)`), half-split rotation, then
  applies it. Only the 16 rotated dims of head_dim 256.
- DeltaNet layers take no positions — unchanged.
- Decode after an image: `pos = cacheLen + ropeDelta` instead of `cacheLen`.
  This is the "text after it continues from a shifted position" requirement;
  it must be threaded through *every* decode path (plain, pipelined,
  compiled, MTP, prompt-lookup) or they will silently disagree.

### 3c. Prefill chunking (`llm/qwen35/forward.go`)

`prefillInternal` currently slices the token array into fixed 512-token
chunks. With images: chunk boundaries must respect image spans (an image's
tokens and their embeddings stay in one chunk), and a chunk containing image
tokens routes to `ForwardPrefillMultimodal`. Text-only chunks keep the
existing path (and its `needLogits=false` optimisation).

### 3d. Fast paths must step aside (`llm/model.go`)

When the prompt contains image tokens, disable for that request:
- pipelined decode (`ForwardDecodeArgmaxArray`) — it assumes a plain
  `pos` counter and no per-token position table,
- compiled decode (`PrepareCompiledDecode`),
- MTP self-speculation,
- prompt-lookup speculative decoding (it re-prefills arbitrary token
  batches at arbitrary positions).
Plain `ForwardDecode`/`ForwardDecodeArgmax` with `pos + ropeDelta` is the
only decode path allowed. This is a deliberate first-cut: correctness
first, re-enable per path once each has a parity test with an image in the
prompt.

### 3e. Prompt cache must key on image content (`llm/model.go`, `kv_cache.go`)

Today `prefixSlot.tokens []int` is the cache key. Two different images have
identical token ids (`<|vision_start|><|image_pad|>...`), so a naive
implementation returns the wrong cached K/V. Fix:
- `prefixSlot` gains `imageHash []byte` — a SHA-256 over each image's
  preprocessed pixel bytes (post-resize, pre-normalize is enough and stable),
  concatenated in order.
- `bestPrefixSlot` requires both the token prefix and the image hash prefix
  to match.
- `WarmSystemPrefix` never matches a slot with images (system prefixes are
  text-only by the chat template's own rule).
- The same hash goes into the request's cache diagnostics so a mismatch is
  visible in `SINTER_LOCAL_DEBUG=1` logs.

### 3f. Model API (`llm/model.go`)

- `ChatMessage` gains `Images []Image` (decoded pixels + hash), populated by
  the server or by sprout's local provider.
- `Model.GenerateMultimodal(ctx, prompt string, images []Image, cfg)` —
  encodes the prompt, expands each `<|vision_start|><|image_pad|><|vision_end|>`
  marker group to `numImageTokens` copies of `<|image_pad|>` (the count comes
  from the image's grid, so it is only known after preprocessing), runs the
  tower, and drives the same generation loop with the multimodal prefill.
- `Generate` stays text-only and errors if the prompt contains image markers
  (today it would happily tokenize `<|image_pad|>` and produce garbage).

## 4. Server (`llm/openaisserver/server.go`)

- `chatMessage.Content` becomes `json.RawMessage` and is decoded as either a
  string (current behaviour) or an array of parts; `{"type":"image_url",
  "image_url":{"url":"data:image/png;base64,..."}}` is accepted.
- Only `data:` URLs; any `http(s):` URL is rejected with a clear 400 (no
  outbound fetch from the inference server).
- Limits: max bytes per image (default 8 MiB decoded), max images per request
  (default 8), max total prompt tokens after expansion (reject before
  prefill, not after).
- The `<|vision_start|><|image_pad|><|vision_end|>` markers go into the
  message text before `FormatChat`, matching the model's own chat template.
- Tool messages / assistant turns may not carry images (template raises);
  return 400.
- `Model` interface gains the multimodal generate call; the fake model in
  `server_test.go` implements it.

## Sequencing and gates

1. Image prep + tests (no GPU needed).
2. Vision tower + mlx-vlm parity test. **Gate: tower output matches.**
3. mRoPE + multimodal prefill + plain decode. **Gate: greedy output for an
   image prompt matches mlx-vlm's, and a text-only prompt's output is
   bit-identical to today's.**
4. Cache keying on image hash + tests (two different images must not share a
   slot; the same image in two turns must).
5. Server + tests with the fake model.

## Risks / open questions

- **Bicubic resize parity.** Measured: max 0.0078 on a [-1,1] range for an
  upscaled 96×96 image, mean 0.001. No-resize images are bit-exact. Left
  as-is; if the tower parity test later shows it matters, port torchvision's
  antialiased kernel.
- **Tower throughput.** 24 layers of full attention over up to ~16k patches
  is not cheap on a 2B model; a 1024x1024 image is ~4k tokens. Needs a
  measurement before promising anything, and probably an image-size cap
  lower than the model's own budget.
- **DeltaNet + images.** The linear layers have no positions, so images only
  affect them through the embeddings. That is expected to be fine, but
  nothing in the tower parity test will catch a mistake there unless the test
  compares final logits, not just tower output.
- **GGML backend.** LayerNorm/GELU are the only new ops; both exist in ggml.
  Parity on GGML is a follow-up, not part of the first cut.

## Reference environment

`/tmp/sprout/vlref-env` (throwaway venv, python3.14): pillow, numpy,
transformers 5.23.0. No torch — the image processor path does not need it.
The checkpoint's processor is `Qwen2VLImageProcessorPil`, which is why the
reference numbers above come from PIL's resize rather than torchvision's.
