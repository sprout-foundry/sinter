//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package qwen35

import (
	"fmt"
	"math"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/tensor"
)

// visionTower is the Qwen3.5 ViT that turns prepared image pixels into the
// merged embeddings the language model consumes in place of <|image_pad|>.
//
// The tower runs as one non-causal pass over the packed patches of all
// images in the request; attention never crosses an image boundary (see
// forward, which splits per image).
type visionTower struct {
	cfg     visionConfig
	backend tensor.Backend
	stream  tensor.Stream

	patchW    tensor.Array // [hidden, temporal*patch*patch*in_ch]
	patchBias tensor.Array // [hidden]
	posEmbed  tensor.Array // [numPositionEmbeddings, hidden]

	blocks []visionBlock

	mergerNormW tensor.Array // [hidden]
	mergerNormB tensor.Array // [hidden]
	mergerFC1   *llm.Linear  // hidden*merge^2 -> hidden*merge^2
	mergerFC2   *llm.Linear  // hidden*merge^2 -> out_hidden
}

// visionBlock is one ViT block. Both LayerNorms carry bias, unlike the
// text model's RMSNorm.
type visionBlock struct {
	norm1W, norm1B tensor.Array
	norm2W, norm2B tensor.Array
	qkvW           *llm.Linear // hidden -> 3*hidden
	qkvB           tensor.Array
	projW          *llm.Linear // hidden -> hidden
	projB          tensor.Array
	fc1W           *llm.Linear // hidden -> intermediate
	fc1B           tensor.Array
	fc2W           *llm.Linear // intermediate -> hidden
	fc2B           tensor.Array
}

// loadVisionTower reads the tower's weights. prefix is the safetensors key
// prefix, which differs between exports: raw HF uses "vision_tower.",
// mlx-community conversions use "model.visual.".
func loadVisionTower(sf *llm.SafetensorsFile, prefix string, cfg visionConfig, b tensor.Backend, s tensor.Stream) (*visionTower, error) {
	get := func(name string) (tensor.Array, error) {
		return sf.Get(prefix+name, b, s)
	}

	// Patch embed is a Conv3d with kernel == stride == (temporal, patch,
	// patch), so it is exactly a matmul against the flattened window. The
	// stored weight is [hidden, temporal, patch, patch, in_ch]; flattening
	// its trailing axes and pre-transposing gives [in, out] for MatMul.
	//
	// The flatten happens on the raw bytes rather than through a reshape op:
	// ggml tensors cap at 4 dimensions, so a 5-D source cannot be
	// materialized there at all.
	patchWT, err := loadFlattenedTransposed(sf, prefix+"patch_embed.proj.weight", b, s)
	if err != nil {
		return nil, fmt.Errorf("vision patch_embed weight: %w", err)
	}

	patchBias, err := get("patch_embed.proj.bias")
	if err != nil {
		patchWT.Free()
		return nil, fmt.Errorf("vision patch_embed bias: %w", err)
	}
	posEmbed, err := get("pos_embed.weight")
	if err != nil {
		patchWT.Free()
		patchBias.Free()
		return nil, fmt.Errorf("vision pos_embed: %w", err)
	}

	t := &visionTower{
		cfg:       cfg,
		backend:   b,
		stream:    s,
		patchW:    patchWT,
		patchBias: patchBias,
		posEmbed:  posEmbed,
	}

	loadLinear := func(name string) (*llm.Linear, error) {
		// The tower is always full precision in the checkpoint; passing nil
		// quantization keeps it that way even when the text model is 4-bit.
		return llm.LoadLinear(sf, prefix+name+".weight", b, s, nil)
	}
	loadBias := func(name string) (tensor.Array, error) {
		return get(name + ".bias")
	}

	for i := 0; i < cfg.Depth; i++ {
		p := fmt.Sprintf("blocks.%d.", i)
		var blk visionBlock
		if blk.norm1W, err = get(p + "norm1.weight"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d norm1: %w", i, err)
		}
		if blk.norm1B, err = get(p + "norm1.bias"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d norm1 bias: %w", i, err)
		}
		if blk.norm2W, err = get(p + "norm2.weight"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d norm2: %w", i, err)
		}
		if blk.norm2B, err = get(p + "norm2.bias"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d norm2 bias: %w", i, err)
		}
		if blk.qkvW, err = loadLinear(p + "attn.qkv"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d qkv: %w", i, err)
		}
		if blk.qkvB, err = loadBias(p + "attn.qkv"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d qkv bias: %w", i, err)
		}
		if blk.projW, err = loadLinear(p + "attn.proj"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d proj: %w", i, err)
		}
		if blk.projB, err = loadBias(p + "attn.proj"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d proj bias: %w", i, err)
		}
		if blk.fc1W, err = loadLinear(p + "mlp.linear_fc1"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d fc1: %w", i, err)
		}
		if blk.fc1B, err = loadBias(p + "mlp.linear_fc1"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d fc1 bias: %w", i, err)
		}
		if blk.fc2W, err = loadLinear(p + "mlp.linear_fc2"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d fc2: %w", i, err)
		}
		if blk.fc2B, err = loadBias(p + "mlp.linear_fc2"); err != nil {
			t.free()
			return nil, fmt.Errorf("vision block %d fc2 bias: %w", i, err)
		}
		t.blocks = append(t.blocks, blk)
	}

	if t.mergerNormW, err = get("merger.norm.weight"); err != nil {
		t.free()
		return nil, fmt.Errorf("vision merger norm: %w", err)
	}
	if t.mergerNormB, err = get("merger.norm.bias"); err != nil {
		t.free()
		return nil, fmt.Errorf("vision merger norm bias: %w", err)
	}
	if t.mergerFC1, err = loadLinear("merger.linear_fc1"); err != nil {
		t.free()
		return nil, fmt.Errorf("vision merger fc1: %w", err)
	}
	if t.mergerFC2, err = loadLinear("merger.linear_fc2"); err != nil {
		t.free()
		return nil, fmt.Errorf("vision merger fc2: %w", err)
	}
	return t, nil
}

func (t *visionTower) free() {
	freeArr(t.patchW)
	freeArr(t.patchBias)
	freeArr(t.posEmbed)
	for i := range t.blocks {
		blk := &t.blocks[i]
		freeArr(blk.norm1W)
		freeArr(blk.norm1B)
		freeArr(blk.norm2W)
		freeArr(blk.norm2B)
		freeLinear(blk.qkvW)
		freeArr(blk.qkvB)
		freeLinear(blk.projW)
		freeArr(blk.projB)
		freeLinear(blk.fc1W)
		freeArr(blk.fc1B)
		freeLinear(blk.fc2W)
		freeArr(blk.fc2B)
	}
	freeArr(t.mergerNormW)
	freeArr(t.mergerNormB)
	freeLinear(t.mergerFC1)
	freeLinear(t.mergerFC2)
}

func freeLinear(l *llm.Linear) {
	if l != nil {
		l.Free()
	}
}

// visionImage is one image's patch grid, as prepared by the vision package.
type visionImage struct {
	GridTHW [3]int // temporal, height, width in patches
}

// forward runs the tower over one image's patches and returns the merged
// embeddings, [numTokens, outHidden], where numTokens is the patch count
// divided by merge^2.
//
// patches is [numPatches, temporal*patch*patch*in_ch] in the layout the
// vision package produces.
func (t *visionTower) forward(patches tensor.Array, img visionImage) (tensor.Array, error) {
	s := t.stream
	b := t.backend
	cfg := t.cfg

	gridT, gridH, gridW := img.GridTHW[0], img.GridTHW[1], img.GridTHW[2]
	if gridT != 1 {
		return nil, fmt.Errorf("vision: temporal grid %d, only single-frame images are supported", gridT)
	}
	numPatches := gridH * gridW

	// Patch embed: [numPatches, window] @ [window, hidden] + bias.
	proj, err := b.MatMul(patches, t.patchW, s)
	if err != nil {
		return nil, fmt.Errorf("vision patch embed matmul: %w", err)
	}
	defer proj.Free()
	h, err := addVectorBias(proj, t.patchBias, numPatches, cfg.HiddenSize, b, s)
	if err != nil {
		return nil, fmt.Errorf("vision patch embed bias: %w", err)
	}
	defer h.Free()

	// Learned position embeddings, bilinearly resampled to this grid.
	pos, err := t.interpolatePosEmbed(gridH, gridW, numPatches)
	if err != nil {
		return nil, err
	}
	defer pos.Free()
	hPos, err := b.Add(h, pos, s)
	if err != nil {
		return nil, fmt.Errorf("vision pos embed add: %w", err)
	}
	defer hPos.Free()

	// 2D rotary tables for q and k, per patch.
	cos, sin, err := t.rotaryTables(gridH, gridW)
	if err != nil {
		return nil, err
	}
	defer cos.Free()
	defer sin.Free()

	hidden := hPos
	owned := false
	for i := range t.blocks {
		out, err := t.block(&t.blocks[i], hidden, numPatches, cos, sin)
		if owned {
			hidden.Free()
		}
		if err != nil {
			return nil, fmt.Errorf("vision block %d: %w", i, err)
		}
		hidden = out
		owned = true
	}
	if !owned {
		// Depth 0 is not a real configuration, but keep ownership explicit.
		hidden = b.RetainArray(hidden)
	}
	defer hidden.Free()

	return t.merge(hidden, numPatches)
}

// block runs one ViT block: pre-norm attention with residual, then a
// pre-norm GELU MLP with residual.
func (t *visionTower) block(blk *visionBlock, x tensor.Array, seqLen int, cos, sin tensor.Array) (tensor.Array, error) {
	s := t.stream
	b := t.backend
	cfg := t.cfg
	headDim := cfg.HeadDim()

	normed, err := b.LayerNorm(x, blk.norm1W, blk.norm1B, float32(cfg.LayerNormEps), s)
	if err != nil {
		return nil, fmt.Errorf("norm1: %w", err)
	}
	defer normed.Free()

	qkv, err := blk.qkvW.Forward(normed, b, s)
	if err != nil {
		return nil, fmt.Errorf("qkv: %w", err)
	}
	defer qkv.Free()
	qkv, err = addVectorBias(qkv, blk.qkvB, seqLen, 3*cfg.HiddenSize, b, s)
	if err != nil {
		return nil, fmt.Errorf("qkv bias: %w", err)
	}
	defer qkv.Free()

	// [seqLen, 3, heads, headDim] -> q, k, v each [1, heads, seqLen, headDim]
	qkv4, err := b.Reshape(qkv, []int{seqLen, 3, cfg.NumHeads, headDim}, s)
	if err != nil {
		return nil, fmt.Errorf("qkv reshape: %w", err)
	}
	defer qkv4.Free()

	var parts [3]tensor.Array
	for i := 0; i < 3; i++ {
		sl, err := b.Slice(qkv4, []int{0, i, 0, 0}, []int{seqLen, i + 1, cfg.NumHeads, headDim}, []int{1, 1, 1, 1}, s)
		if err != nil {
			for j := 0; j < i; j++ {
				parts[j].Free()
			}
			return nil, fmt.Errorf("qkv slice %d: %w", i, err)
		}
		// [seqLen, 1, heads, headDim] -> [1, heads, seqLen, headDim]
		tr, err := b.TransposeAxes(sl, []int{1, 2, 0, 3}, s)
		if err != nil {
			sl.Free()
			for j := 0; j < i; j++ {
				parts[j].Free()
			}
			return nil, fmt.Errorf("qkv transpose %d: %w", i, err)
		}
		sl.Free()
		resh, err := b.Reshape(tr, []int{1, cfg.NumHeads, seqLen, headDim}, s)
		if err != nil {
			tr.Free()
			for j := 0; j < i; j++ {
				parts[j].Free()
			}
			return nil, fmt.Errorf("qkv reshape %d: %w", i, err)
		}
		tr.Free()
		parts[i] = resh
	}
	q, k, v := parts[0], parts[1], parts[2]
	defer q.Free()
	defer k.Free()
	defer v.Free()

	qRot, err := applyVisionRoPE(q, cos, sin, b, s)
	if err != nil {
		return nil, fmt.Errorf("q rope: %w", err)
	}
	defer qRot.Free()
	kRot, err := applyVisionRoPE(k, cos, sin, b, s)
	if err != nil {
		return nil, fmt.Errorf("k rope: %w", err)
	}
	defer kRot.Free()

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	// Vision attention is non-causal across the whole image: an empty
	// mask_mode with no mask array is full attention.
	ctx, err := b.FastScaledDotProductAttention(qRot, kRot, v, scale, "", nil, nil, s)
	if err != nil {
		return nil, fmt.Errorf("attention: %w", err)
	}
	defer ctx.Free()

	// [1, heads, seqLen, headDim] -> [seqLen, hidden]
	ctxT, err := b.TransposeAxes(ctx, []int{0, 2, 1, 3}, s)
	if err != nil {
		return nil, fmt.Errorf("ctx transpose: %w", err)
	}
	defer ctxT.Free()
	ctxFlat, err := b.Reshape(ctxT, []int{seqLen, cfg.HiddenSize}, s)
	if err != nil {
		return nil, fmt.Errorf("ctx reshape: %w", err)
	}
	defer ctxFlat.Free()

	attnOut, err := blk.projW.Forward(ctxFlat, b, s)
	if err != nil {
		return nil, fmt.Errorf("proj: %w", err)
	}
	defer attnOut.Free()
	attnOut, err = addVectorBias(attnOut, blk.projB, seqLen, cfg.HiddenSize, b, s)
	if err != nil {
		return nil, fmt.Errorf("proj bias: %w", err)
	}

	residual, err := b.Add(x, attnOut, s)
	if err != nil {
		attnOut.Free()
		return nil, fmt.Errorf("attn residual: %w", err)
	}
	attnOut.Free()
	defer residual.Free()

	normed2, err := b.LayerNorm(residual, blk.norm2W, blk.norm2B, float32(cfg.LayerNormEps), s)
	if err != nil {
		return nil, fmt.Errorf("norm2: %w", err)
	}
	defer normed2.Free()

	fc1, err := blk.fc1W.Forward(normed2, b, s)
	if err != nil {
		return nil, fmt.Errorf("fc1: %w", err)
	}
	defer fc1.Free()
	fc1, err = addVectorBias(fc1, blk.fc1B, seqLen, cfg.IntermediateSize, b, s)
	if err != nil {
		return nil, fmt.Errorf("fc1 bias: %w", err)
	}
	defer fc1.Free()

	// The block MLP uses the config's hidden_act, which is
	// gelu_pytorch_tanh for these checkpoints; the merger uses exact GELU.
	exact := cfg.HiddenAct == "gelu"
	act, err := b.GELU(fc1, exact, s)
	if err != nil {
		return nil, fmt.Errorf("gelu: %w", err)
	}
	defer act.Free()

	fc2, err := blk.fc2W.Forward(act, b, s)
	if err != nil {
		return nil, fmt.Errorf("fc2: %w", err)
	}
	defer fc2.Free()
	fc2, err = addVectorBias(fc2, blk.fc2B, seqLen, cfg.HiddenSize, b, s)
	if err != nil {
		return nil, fmt.Errorf("fc2 bias: %w", err)
	}
	defer fc2.Free()

	return b.Add(residual, fc2, s)
}

// merge applies the merger: LayerNorm over the tower width, then a 2x2
// spatial merge into one token, then an MLP down to the text hidden size.
func (t *visionTower) merge(x tensor.Array, numPatches int) (tensor.Array, error) {
	s := t.stream
	b := t.backend
	cfg := t.cfg
	merge := cfg.SpatialMergeSize
	merged := cfg.HiddenSize * merge * merge

	normed, err := b.LayerNorm(x, t.mergerNormW, t.mergerNormB, float32(cfg.LayerNormEps), s)
	if err != nil {
		return nil, fmt.Errorf("merger norm: %w", err)
	}
	defer normed.Free()

	// Patch rows are already grouped in merge blocks by the vision package's
	// patchify, so a plain reshape to [numTokens, hidden*merge^2] lines each
	// group's patches up in the order the reference's permute produces.
	flat, err := b.Reshape(normed, []int{numPatches / (merge * merge), merged}, s)
	if err != nil {
		return nil, fmt.Errorf("merger reshape: %w", err)
	}
	defer flat.Free()

	fc1, err := t.mergerFC1.Forward(flat, b, s)
	if err != nil {
		return nil, fmt.Errorf("merger fc1: %w", err)
	}
	defer fc1.Free()

	// nn.GELU() in the reference merger is the exact erf form.
	act, err := b.GELU(fc1, true, s)
	if err != nil {
		return nil, fmt.Errorf("merger gelu: %w", err)
	}
	defer act.Free()

	return t.mergerFC2.Forward(act, b, s)
}

// readF32 returns an array's values as float32, casting when the stored
// dtype is not already float32 (the tower's weights are bf16).
func (t *visionTower) readF32(a tensor.Array) ([]float32, error) {
	if a.Dtype() == tensor.Float32 {
		return a.Float32Data()
	}
	f32, err := t.backend.AsType(a, tensor.Float32, t.stream)
	if err != nil {
		return nil, err
	}
	defer f32.Free()
	return f32.Float32Data()
}

// loadFlattenedTransposed reads a convolution kernel and returns it as an
// [window, out] matrix ready for MatMul against flattened patch windows.
//
// The checkpoint stores the Conv3d kernel as [out, temporal, patch, patch,
// in_ch] (MLX layout). Flattening must produce the same element order the
// reference's convolution reduces over, which is [in_ch, temporal, patch,
// patch] — channel first.
//
// The permutation runs over the raw bytes rather than a backend op: ggml
// caps tensors at 4 dimensions, so the stored 5-D form cannot be
// materialized there at all.
func loadFlattenedTransposed(sf *llm.SafetensorsFile, name string, b tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	stored := sf.TensorShape(name)
	if len(stored) != 5 {
		return nil, fmt.Errorf("%s: shape %v, want [out, temporal, patch, patch, in_ch]", name, stored)
	}
	outDim := stored[0]
	temporal, ph, pw, inCh := stored[1], stored[2], stored[3], stored[4]
	window := inCh * temporal * ph * pw

	// Read the source at its stored rank, then permute on the host. The
	// backend may reject the 5-D shape (ggml), so read the bytes directly
	// through a 2-D view of the same data.
	src, err := sf.GetReshaped(name, []int{outDim, temporal * ph * pw * inCh}, b, s)
	if err != nil {
		return nil, err
	}
	defer src.Free()
	raw, err := src.RawBytes()
	if err != nil {
		return nil, fmt.Errorf("%s: read bytes: %w", name, err)
	}
	elem := len(raw) / (outDim * temporal * ph * pw * inCh)
	if elem == 0 {
		return nil, fmt.Errorf("%s: empty weight", name)
	}

	// Index into the stored [out][t][h][w][c] order, write [out][c][t][h][w].
	permuted := make([]byte, len(raw))
	plane := ph * pw
	for o := 0; o < outDim; o++ {
		for t := 0; t < temporal; t++ {
			for hw := 0; hw < plane; hw++ {
				for c := 0; c < inCh; c++ {
					srcOff := (((o*temporal+t)*plane+hw)*inCh + c) * elem
					dstIdx := ((o*inCh+c)*temporal+t)*plane + hw
					copy(permuted[dstIdx*elem:(dstIdx+1)*elem], raw[srcOff:srcOff+elem])
				}
			}
		}
	}

	arr, err := b.NewArrayFromBytes(permuted, []int{outDim, window}, src.Dtype())
	if err != nil {
		return nil, fmt.Errorf("%s: build flattened weight: %w", name, err)
	}
	defer arr.Free()
	tr, err := b.Transpose(arr, s)
	if err != nil {
		return nil, fmt.Errorf("%s: transpose: %w", name, err)
	}
	defer tr.Free()
	// Some backends keep the transposed rank (ggml returns [1,1,window,out]),
	// so normalize back to the 2-D shape MatMul expects.
	weight, err := b.Reshape(tr, []int{window, outDim}, s)
	if err != nil {
		return nil, fmt.Errorf("%s: reshape: %w", name, err)
	}
	// Materialize before returning: the tower runs on the inference thread,
	// and a lazy graph built here would keep this load-time stream baked in,
	// failing to evaluate later with "no Stream(gpu, 0) in current thread".
	if err := weight.Eval(); err != nil {
		weight.Free()
		return nil, fmt.Errorf("%s: eval: %w", name, err)
	}
	return weight, nil
}

// addVectorBias adds a [width] bias to a [rows, width] tensor.
func addVectorBias(x, bias tensor.Array, rows, width int, b tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	if bias == nil {
		return b.RetainArray(x), nil
	}
	bias2, err := b.Reshape(bias, []int{1, width}, s)
	if err != nil {
		return nil, fmt.Errorf("bias reshape: %w", err)
	}
	defer bias2.Free()
	return b.Add(x, bias2, s)
}

// interpolatePosEmbed bilinearly resamples the learned square position grid
// to this image's grid, matching the reference's
// get_vision_interpolation_indices_and_weights with align_corners=True:
// each patch takes the four surrounding corners of the source table,
// weighted by the fractional distance between them.
//
// Patch rows arrive in merge-block order, so the (row, col) of patch i is
// derived from i the same way the reference decodes it.
func (t *visionTower) interpolatePosEmbed(gridH, gridW, numPatches int) (tensor.Array, error) {
	side := int(math.Round(math.Sqrt(float64(t.cfg.NumPositionEmbeddings))))
	if side*side != t.cfg.NumPositionEmbeddings {
		return nil, fmt.Errorf("vision: num_position_embeddings %d is not a perfect square", t.cfg.NumPositionEmbeddings)
	}

	table, err := t.readF32(t.posEmbed)
	if err != nil {
		return nil, fmt.Errorf("vision pos_embed read: %w", err)
	}
	hidden := t.cfg.HiddenSize
	merge := t.cfg.SpatialMergeSize
	out := make([]float32, numPatches*hidden)

	blocksW := gridW / merge
	last := float64(side - 1)

	for i := 0; i < numPatches; i++ {
		// Decode the merge-block-major index into (row, col).
		inCol := i % merge
		inRow := (i / merge) % merge
		blockCol := (i / (merge * merge)) % blocksW
		blockRow := i / (merge * merge * blocksW)
		row := blockRow*merge + inRow
		col := blockCol*merge + inCol

		// align_corners=True: endpoints map to 0 and side-1.
		srcH := float64(row) * last / math.Max(float64(gridH-1), 1)
		srcW := float64(col) * last / math.Max(float64(gridW-1), 1)

		hFloor := int(math.Floor(srcH))
		wFloor := int(math.Floor(srcW))
		dh := srcH - float64(hFloor)
		dw := srcW - float64(wFloor)
		hCeil := min(hFloor+1, side-1)
		wCeil := min(wFloor+1, side-1)

		base := i * hidden
		i00 := (hFloor*side + wFloor) * hidden
		i01 := (hFloor*side + wCeil) * hidden
		i10 := (hCeil*side + wFloor) * hidden
		i11 := (hCeil*side + wCeil) * hidden
		w00 := float32((1 - dh) * (1 - dw))
		w01 := float32((1 - dh) * dw)
		w10 := float32(dh * (1 - dw))
		w11 := float32(dh * dw)
		for k := 0; k < hidden; k++ {
			out[base+k] = table[i00+k]*w00 + table[i01+k]*w01 + table[i10+k]*w10 + table[i11+k]*w11
		}
	}

	return t.backend.NewArrayFromFloat32(out, []int{numPatches, hidden})
}

// rotaryTables builds the 2D rotary cos/sin tables for the patch grid.
//
// Each patch gets a (row, col) coordinate; frequencies are
// 1/theta^(2i/dim) over dim = headDim/2, and the table is laid out
// [h, w, h, w] so the half-split rotation covers both axes.
func (t *visionTower) rotaryTables(gridH, gridW int) (cos, sin tensor.Array, err error) {
	cfg := t.cfg
	headDim := cfg.HeadDim()
	ropeDim := headDim / 2 // the reference builds dim = head_dim // 2
	halfRope := ropeDim / 2
	const theta = 10000.0

	invFreq := make([]float64, halfRope)
	for i := 0; i < halfRope; i++ {
		invFreq[i] = 1.0 / math.Pow(theta, float64(2*i)/float64(ropeDim))
	}

	// The reference builds a ropeDim-wide table (row freq block then column
	// freq block), then duplicates it to fill headDim: emb = cat(freqs, freqs).
	merge := cfg.SpatialMergeSize
	blocksW := gridW / merge
	numPatches := gridH * gridW
	cosData := make([]float32, numPatches*headDim)
	sinData := make([]float32, numPatches*headDim)

	for i := 0; i < numPatches; i++ {
		// Same merge-block-major decode as the position embedding.
		inCol := i % merge
		inRow := (i / merge) % merge
		blockCol := (i / (merge * merge)) % blocksW
		blockRow := i / (merge * merge * blocksW)
		r := blockRow*merge + inRow
		c := blockCol*merge + inCol

		base := i * headDim
		for k := 0; k < halfRope; k++ {
			rowAngle := float64(r) * invFreq[k]
			colAngle := float64(c) * invFreq[k]
			cosData[base+k] = float32(math.Cos(rowAngle))
			sinData[base+k] = float32(math.Sin(rowAngle))
			cosData[base+halfRope+k] = float32(math.Cos(colAngle))
			sinData[base+halfRope+k] = float32(math.Sin(colAngle))
		}
		copy(cosData[base+ropeDim:base+headDim], cosData[base:base+ropeDim])
		copy(sinData[base+ropeDim:base+headDim], sinData[base:base+ropeDim])
	}

	cosArr, err := t.backend.NewArrayFromFloat32(cosData, []int{1, 1, numPatches, headDim})
	if err != nil {
		return nil, nil, fmt.Errorf("vision rope cos: %w", err)
	}
	sinArr, err := t.backend.NewArrayFromFloat32(sinData, []int{1, 1, numPatches, headDim})
	if err != nil {
		cosArr.Free()
		return nil, nil, fmt.Errorf("vision rope sin: %w", err)
	}
	return cosArr, sinArr, nil
}

// applyVisionRoPE applies the half-split rotation to q or k:
// out = x*cos + rotateHalf(x)*sin.
func applyVisionRoPE(x, cos, sin tensor.Array, b tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	shape := x.Shape()
	if len(shape) != 4 {
		return nil, fmt.Errorf("vision rope: shape %v, want 4-D", shape)
	}
	headDim := shape[3]
	half := headDim / 2

	first, err := b.Slice(x, []int{0, 0, 0, 0}, []int{shape[0], shape[1], shape[2], half}, []int{1, 1, 1, 1}, s)
	if err != nil {
		return nil, fmt.Errorf("vision rope slice first: %w", err)
	}
	defer first.Free()
	second, err := b.Slice(x, []int{0, 0, 0, half}, []int{shape[0], shape[1], shape[2], headDim}, []int{1, 1, 1, 1}, s)
	if err != nil {
		return nil, fmt.Errorf("vision rope slice second: %w", err)
	}
	defer second.Free()

	negSecond, err := b.Negative(second, s)
	if err != nil {
		return nil, fmt.Errorf("vision rope negate: %w", err)
	}
	defer negSecond.Free()

	rotated, err := b.ConcatenateAxis([]tensor.Array{negSecond, first}, 3, s)
	if err != nil {
		return nil, fmt.Errorf("vision rope concat: %w", err)
	}
	defer rotated.Free()

	scaled, err := b.Multiply(x, cos, s)
	if err != nil {
		return nil, fmt.Errorf("vision rope x*cos: %w", err)
	}
	defer scaled.Free()
	rotScaled, err := b.Multiply(rotated, sin, s)
	if err != nil {
		return nil, fmt.Errorf("vision rope rot*sin: %w", err)
	}
	defer rotScaled.Free()

	return b.Add(scaled, rotScaled, s)
}
