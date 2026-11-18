//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package llm

import (
	"fmt"
	"strings"

	"github.com/sprout-foundry/sinter/tensor"
)

// Linear is a projection weight with two representations:
//
//   - Full precision: wT is the pre-transposed [in, out] weight; Forward does
//     a plain MatMul (the fast path, no dequant).
//   - Quantized: qW is the packed int32 weight [out, in*bits/32] (PyTorch
//     layout, NOT pre-transposed), qScales/qBiases are the per-group scale
//     and bias. Forward calls mlx_quantized_matmul which dequantizes on the
//     fly in the Metal kernel.
//
// Quantized weights stay in [out, in] layout because packing is bit-interleaved
// along the in axis; a transpose would corrupt the layout. mlx_quantized_matmul
// takes transpose=true to compensate.
type Linear struct {
	wT tensor.Array // [in, out], nil when quantized

	qW         tensor.Array // packed int32 [out, in*bits/32]
	qScales    tensor.Array // [out, in/group_size]
	qBiases    tensor.Array // [out, in/group_size], optional
	qGroupSize int
	qBits      int
	qMode      string

	// bias is an optional additive bias [out] (Qwen2 QKV projections have
	// them; Qwen3 dropped them). Applied after the matmul in Forward.
	bias tensor.Array
}

// IsQuantized reports whether this projection holds packed quantized weights.
func (l *Linear) IsQuantized() bool { return l.qW != nil }

// Accessors for MoE gather_qmm (need raw quantized weight tensors).
func (l *Linear) QW() tensor.Array      { return l.qW }
func (l *Linear) QScales() tensor.Array { return l.qScales }
func (l *Linear) QBiases() tensor.Array { return l.qBiases }
func (l *Linear) QGroupSize() int       { return l.qGroupSize }
func (l *Linear) QBits() int            { return l.qBits }
func (l *Linear) QMode() string         { return l.qMode }
func (l *Linear) WT() tensor.Array      { return l.wT }

// NumExperts returns the number of experts from the weight's first dimension.
// For non-MoE (2D) weights, returns 0.
func (l *Linear) NumExperts() int {
	if l.qW != nil {
		shape := l.qW.Shape()
		if len(shape) == 3 {
			return shape[0]
		}
	} else if l.wT != nil {
		shape := l.wT.Shape()
		if len(shape) == 3 {
			return shape[0]
		}
	}
	return 0
}

// Forward computes x @ W (+ bias when present; x is [..., in], result [..., out]).
// Calls through the backend interface.
func (l *Linear) Forward(x tensor.Array, b tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	var out tensor.Array
	var err error
	if l.qW != nil {
		out, err = b.QuantizedMatMul(x, l.qW, l.qScales, l.qBiases, true, l.qGroupSize, l.qBits, l.qMode, s)
	} else {
		out, err = b.MatMul(x, l.wT, s)
	}
	if err != nil {
		return nil, err
	}
	if l.bias == nil {
		return out, nil
	}
	// Broadcast-add the [out] bias over the leading dims: reshape to
	// [1, 1, out] so it broadcasts against [..., seq, out].
	bias3, err := b.Reshape(l.bias, []int{1, 1, l.bias.Shape()[0]}, s)
	if err != nil {
		out.Free()
		return nil, fmt.Errorf("reshape bias: %w", err)
	}
	defer bias3.Free()
	biased, err := b.Add(out, bias3, s)
	if err != nil {
		out.Free()
		return nil, fmt.Errorf("add bias: %w", err)
	}
	out.Free()
	return biased, nil
}

func (l *Linear) Free() {
	freeArr(l.wT)
	freeArr(l.qW)
	freeArr(l.qScales)
	freeArr(l.qBiases)
	freeArr(l.bias)
}

// MergeLinears concatenates projections that read the same input into one
// wider projection, so a single matmul replaces several that each stream the
// same activations. At decode shape every projection is a separate kernel
// launch, and the launch dominates the (small) memory transfer, so this is
// the difference between four kernels and one per block.
//
// Output rows appear in the order the parts are given: the caller splits the
// result along the last axis at the same boundaries. The merged projection is
// bit-identical to the separate ones — each output row's accumulation is
// independent of the others.
//
// Returns an error, leaving the parts untouched, when they cannot share a
// kernel: mixed quantization, differing quant parameters, or a bias on only
// some of them.
func MergeLinears(b tensor.Backend, s tensor.Stream, parts ...*Linear) (*Linear, error) {
	return MergeLinearsAxis(b, s, 0, parts...)
}

// MergeLinearsAxis is MergeLinears with an explicit concatenation axis for
// the packed quantized weights. MoE expert tensors are [experts, inter,
// packed_hidden], so gate and up — which read the same input and differ only
// along the intermediate axis — merge along axis 1. Full-precision weights
// are stored pre-transposed, so their axis is shifted by one.
func MergeLinearsAxis(b tensor.Backend, s tensor.Stream, axis int, parts ...*Linear) (*Linear, error) {
	if len(parts) < 2 {
		return nil, fmt.Errorf("merge linears: need at least two projections")
	}
	quantized, full := 0, 0
	for _, p := range parts {
		if p == nil {
			return nil, fmt.Errorf("merge linears: nil projection")
		}
		if p.qW != nil {
			quantized++
		}
		if p.wT != nil {
			full++
		}
	}
	if quantized != 0 && full != 0 {
		return nil, fmt.Errorf("merge linears: mixed quantized and full-precision projections")
	}

	out := &Linear{}
	switch {
	case quantized == len(parts):
		first := parts[0]
		for _, p := range parts[1:] {
			if p.qGroupSize != first.qGroupSize || p.qBits != first.qBits || p.qMode != first.qMode {
				return nil, fmt.Errorf("merge linears: quant parameters differ")
			}
		}
		ws := make([]tensor.Array, len(parts))
		scales := make([]tensor.Array, len(parts))
		biases := make([]tensor.Array, len(parts))
		for i, p := range parts {
			ws[i], scales[i], biases[i] = p.qW, p.qScales, p.qBiases
		}
		var err error
		if out.qW, err = b.ConcatenateAxis(ws, axis, s); err != nil {
			return nil, fmt.Errorf("merge linears: concat weights: %w", err)
		}
		if out.qScales, err = b.ConcatenateAxis(scales, axis, s); err != nil {
			out.Free()
			return nil, fmt.Errorf("merge linears: concat scales: %w", err)
		}
		if out.qBiases, err = b.ConcatenateAxis(biases, axis, s); err != nil {
			out.Free()
			return nil, fmt.Errorf("merge linears: concat biases: %w", err)
		}
		out.qGroupSize, out.qBits, out.qMode = first.qGroupSize, first.qBits, first.qMode
	case full == len(parts):
		// Full-precision weights are stored pre-transposed [..., in, out], so
		// the parts concatenate one axis further along than the packed form.
		ws := make([]tensor.Array, len(parts))
		for i, p := range parts {
			ws[i] = p.wT
		}
		var err error
		if out.wT, err = b.ConcatenateAxis(ws, axis+1, s); err != nil {
			return nil, fmt.Errorf("merge linears: concat weights: %w", err)
		}
	default:
		return nil, fmt.Errorf("merge linears: projections have no weights")
	}

	// Optional bias: all or none.
	withBias := 0
	for _, p := range parts {
		if p.bias != nil {
			withBias++
		}
	}
	if withBias != 0 {
		if withBias != len(parts) {
			out.Free()
			return nil, fmt.Errorf("merge linears: bias on only some projections")
		}
		bs := make([]tensor.Array, len(parts))
		for i, p := range parts {
			bs[i] = p.bias
		}
		var err error
		if out.bias, err = b.ConcatenateAxis(bs, 0, s); err != nil {
			out.Free()
			return nil, fmt.Errorf("merge linears: concat bias: %w", err)
		}
	}

	// Materialize once at load time rather than on the first decode step.
	for _, a := range []tensor.Array{out.qW, out.qScales, out.qBiases, out.wT, out.bias} {
		if a == nil {
			continue
		}
		if err := a.Eval(); err != nil {
			out.Free()
			return nil, fmt.Errorf("merge linears: eval: %w", err)
		}
	}
	return out, nil
}

// NewLinearFull creates a full-precision linear from a pre-transposed weight.
func NewLinearFull(wT tensor.Array) *Linear { return &Linear{wT: wT} }

// LoadLinearBias loads the optional additive bias for a projection
// ("{base}.bias"). Returns a nil-array Linear wrapper (bias only) when the
// key is absent — callers merge it via SetBias. Kept separate so full-
// precision and quantized loads share one bias path.
func (l *Linear) SetBias(sf *SafetensorsFile, base string, b tensor.Backend, s tensor.Stream) error {
	if !sf.Has(base + ".bias") {
		return nil // no bias (Qwen3-style)
	}
	bias, err := sf.Get(base+".bias", b, s)
	if err != nil {
		return fmt.Errorf("load %s.bias: %w", base, err)
	}
	l.bias = bias
	return nil
}

// LoadLinear loads a projection weight. When quant is nil it loads and
// pre-transposes the float weight (the existing full-precision path). When
// quant is non-nil it either loads a pre-quantized triplet from the file
// (keys `{base}.weight`/`.scales`/`.biases`, as mlx-community models store)
// or quantizes the loaded float weight at load time.
//
// `name` may be the full weight key ("...q_proj.weight") or the base
// ("...q_proj") — the pre-quantized triplet check strips a trailing
// ".weight" so `{base}.scales` resolves.
func LoadLinear(sf *SafetensorsFile, name string, b tensor.Backend, s tensor.Stream, quant *QuantConfig) (*Linear, error) {
	base := strings.TrimSuffix(name, ".weight")

	if quant == nil {
		wT, err := loadLinearT(sf, name, b, s)
		if err != nil {
			return nil, err
		}
		return &Linear{wT: wT}, nil
	}

	if sf.Has(base + ".scales") {
		// Pre-quantized triplet in the safetensors file.
		return loadQuantizedTriplet(sf, base, b, s, quant)
	}

	// Quantize a full-precision weight at load time.
	w, err := sf.Get(name, b, s)
	if err != nil {
		return nil, err
	}
	defer w.Free()
	return quantizeLinear(w, b, s, quant)
}

// loadLinearT loads a float weight and pre-transposes it from [out, in]
// (PyTorch layout) to [in, out] so decode-time matmuls skip the transpose.
func loadLinearT(sf *SafetensorsFile, name string, b tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	w, err := sf.Get(name, b, s)
	if err != nil {
		return nil, err
	}
	defer w.Free()

	wT, err := b.Transpose(w, s)
	if err != nil {
		return nil, fmt.Errorf("transpose %s: %w", name, err)
	}
	if err := wT.Eval(); err != nil {
		wT.Free()
		return nil, fmt.Errorf("eval transpose %s: %w", name, err)
	}
	return wT, nil
}

// loadQuantizedTriplet loads the packed weight + scales + biases tensors that
// mlx-community stores for a pre-quantized model. `name` is the base key
// ("...q_proj"); the triplet keys are `{name}.weight`/`.scales`/`.biases`.
func loadQuantizedTriplet(sf *SafetensorsFile, name string, b tensor.Backend, s tensor.Stream, quant *QuantConfig) (*Linear, error) {
	w, err := sf.Get(name+".weight", b, s)
	if err != nil {
		return nil, err
	}
	scales, err := sf.Get(name+".scales", b, s)
	if err != nil {
		w.Free()
		return nil, err
	}
	var biases tensor.Array
	if sf.Has(name + ".biases") {
		biases, err = sf.Get(name+".biases", b, s)
		if err != nil {
			w.Free()
			scales.Free()
			return nil, err
		}
	}

	// Infer actual packed bits from weight/scales shape ratio. MLX models
	// may mix packing widths (e.g. 5-bit config but embedding stored as
	// 6-bit). MLX ops validate bits against the shape, so we must pass the
	// correct value, not the config's nominal bits.
	bits := inferQuantBits(w.Shape(), scales.Shape(), quant.GroupSize, quant.Bits)

	// If the backend can't handle native quantization, dequantize now.
	// On GGML, this re-quantizes to Q4_0 for ARM-optimized kernels.
	if !b.NativeQuantization() {
		fullW, err := dequantizeToFull(b, s, w, scales, biases, bits, quant.GroupSize)
		if err != nil {
			w.Free()
			scales.Free()
			if biases != nil {
				biases.Free()
			}
			return nil, fmt.Errorf("dequantize %s: %w", name, err)
		}
		// For Q4_0 tensors, skip the transpose — ggml_mul_mat handles [out, in]
		// layout directly with quantized kernels. Transposing a Q4_0 tensor
		// would require dequantization, defeating the purpose.
		// For F32 tensors (non-Q4_0 backends), pre-transpose as before.
		if _, ok := b.(GGMLQuantizer); ok {
			// Q4_0 path: store as-is in [out, in] layout. MatMul must use
			// transpose=true semantics (handled by ggml_mul_mat natively).
			return &Linear{wT: fullW}, nil
		}
		wT, err := b.Transpose(fullW, s)
		if err != nil {
			fullW.Free()
			return nil, fmt.Errorf("transpose %s: %w", name, err)
		}
		if err := wT.Eval(); err != nil {
			fullW.Free()
			wT.Free()
			return nil, fmt.Errorf("eval transpose %s: %w", name, err)
		}
		return &Linear{wT: wT}, nil
	}

	return &Linear{
		qW:         w,
		qScales:    scales,
		qBiases:    biases,
		qGroupSize: quant.GroupSize,
		qBits:      bits,
		qMode:      quant.Mode,
	}, nil
}

// inferQuantBits derives the actual packed bits from the weight and scales
// shapes. MLX stores quantized weights as uint32 with packed_dim =
// in_dim * bits / 32. When the config's nominal bits doesn't match the
// actual packing (common with 5-bit models where the embedding uses 6-bit
// packing), use the shape-derived value.
func inferQuantBits(weightShape, scalesShape []int, groupSize, configBits int) int {
	if len(weightShape) < 2 || len(scalesShape) < 2 || groupSize == 0 {
		return configBits
	}
	packedDim := weightShape[1]
	numGroups := scalesShape[1]
	inDim := numGroups * groupSize
	if inDim == 0 {
		return configBits
	}
	shapeBits := packedDim * 32 / inDim
	if shapeBits <= 0 {
		return configBits
	}
	return shapeBits
}

// quantizeLinear runs MLX quantization on a full-precision weight and
// materializes the triplet on the loading thread (lazy arrays bind to the
// thread-local stream, so evals must happen here — see loadLinearT).
func quantizeLinear(w tensor.Array, b tensor.Backend, s tensor.Stream, quant *QuantConfig) (*Linear, error) {
	parts, err := b.Quantize(w, quant.GroupSize, quant.Bits, quant.Mode, s)
	if err != nil {
		return nil, fmt.Errorf("quantize %s: %w", "weight", err)
	}
	if len(parts) < 2 {
		for _, p := range parts {
			p.Free()
		}
		return nil, fmt.Errorf("quantize: expected [weight, scales] got %d arrays", len(parts))
	}
	for _, p := range parts {
		if err := p.Eval(); err != nil {
			for _, q := range parts {
				q.Free()
			}
			return nil, fmt.Errorf("eval quantized weight: %w", err)
		}
	}
	l := &Linear{
		qW:         parts[0],
		qScales:    parts[1],
		qGroupSize: quant.GroupSize,
		qBits:      quant.Bits,
		qMode:      quant.Mode,
	}
	if len(parts) > 2 {
		l.qBiases = parts[2]
	}
	return l, nil
}
