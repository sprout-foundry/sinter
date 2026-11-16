//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package llm

import (
	"fmt"
	"math"

	"github.com/sprout-foundry/sinter/tensor"
)

// MRoPE (multimodal RoPE) gives each token three position coordinates —
// time, row, column — and interleaves their rotary frequencies across the
// head dimension. Text tokens use the same value for all three, which
// reproduces ordinary 1-D RoPE exactly; image tokens carry a (t, h, w)
// triple derived from their patch grid.
//
// The reference builds cos/sin per position rather than rotating with a
// scalar offset, so this path produces explicit tables and applies the
// rotation directly. That keeps it independent of the backend's RoPE
// kernels, which take a single offset and cannot express per-position
// frequencies.
//
// Recompose order for mrope_section = [T, H, W]: start from T's frequencies,
// overwrite every third entry starting at 1 with H's, and every third entry
// starting at 2 with W's. The result is duplicated to fill the head dim,
// matching the reference's cat(freqs, freqs).
type MRoPE struct {
	Sections []int // [T, H, W] frequency counts
	Theta    float64
	HeadDim  int
	invFreq  []float64
}

// NewMRoPE builds the rotary frequency table. sections is the model's
// mrope_section; headDim is the attention head width.
func NewMRoPE(sections []int, theta float64, headDim int) (*MRoPE, error) {
	if len(sections) != 3 {
		return nil, fmt.Errorf("mrope: sections must have 3 entries (T,H,W), got %v", sections)
	}
	half := headDim / 2
	total := sections[0] + sections[1] + sections[2]
	if total != half {
		return nil, fmt.Errorf("mrope: sections %v sum to %d, want head_dim/2 = %d", sections, total, half)
	}
	inv := make([]float64, half)
	for i := 0; i < half; i++ {
		inv[i] = 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
	}
	return &MRoPE{Sections: sections, Theta: theta, HeadDim: headDim, invFreq: inv}, nil
}

// freqIndexAxis reports which coordinate (0=T, 1=H, 2=W) drives frequency
// slot i after recomposition.
func (m *MRoPE) freqIndexAxis() []int {
	half := m.HeadDim / 2
	axis := make([]int, half)
	for i := range axis {
		axis[i] = 0 // T fills every slot first
	}
	for dim, offset := range []int{1, 2} {
		length := m.Sections[dim] * 3
		for idx := offset; idx < length && idx < half; idx += 3 {
			axis[idx] = dim + 1
		}
	}
	return axis
}

// Tables builds cos and sin for the given per-token (t, h, w) positions,
// each of length n. The result is laid out [1, 1, n, headDim] for direct
// broadcasting against [1, heads, n, headDim] attention tensors.
func (m *MRoPE) Tables(t, h, w []int, b tensor.Backend, s tensor.Stream) (cosArr, sinArr tensor.Array, err error) {
	n := len(t)
	if len(h) != n || len(w) != n {
		return nil, nil, fmt.Errorf("mrope: position slices differ in length (%d, %d, %d)", len(t), len(h), len(w))
	}
	half := m.HeadDim / 2
	axis := m.freqIndexAxis()

	cosData := make([]float32, n*m.HeadDim)
	sinData := make([]float32, n*m.HeadDim)
	pos := [3][]int{t, h, w}
	for i := 0; i < n; i++ {
		base := i * m.HeadDim
		for f := 0; f < half; f++ {
			angle := float64(pos[axis[f]][i]) * m.invFreq[f]
			c := float32(math.Cos(angle))
			sn := float32(math.Sin(angle))
			cosData[base+f] = c
			sinData[base+f] = sn
			// Duplicate to fill the head dim (cat(freqs, freqs)).
			cosData[base+half+f] = c
			sinData[base+half+f] = sn
		}
	}

	shape := []int{1, 1, n, m.HeadDim}
	cosArr, err = b.NewArrayFromFloat32(cosData, shape)
	if err != nil {
		return nil, nil, fmt.Errorf("mrope cos: %w", err)
	}
	sinArr, err = b.NewArrayFromFloat32(sinData, shape)
	if err != nil {
		cosArr.Free()
		return nil, nil, fmt.Errorf("mrope sin: %w", err)
	}
	return cosArr, sinArr, nil
}

// ApplyMRoPE rotates x (shape [1, heads, n, headDim]) by the given cos/sin
// tables. The rotation pairs element i with element i+headDim/2, which is
// the non-traditional (GPT-NeoX) convention FastRoPE uses with
// traditional=false.
func ApplyMRoPE(x, cos, sin tensor.Array, b tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	shape := x.Shape()
	if len(shape) != 4 {
		return nil, fmt.Errorf("mrope: expected 4-D input, got %v", shape)
	}
	half := shape[3] / 2

	// rot(x) = cat(-x2, x1) over the head dim.
	x1, err := b.Slice(x, []int{0, 0, 0, 0}, []int{shape[0], shape[1], shape[2], half}, []int{1, 1, 1, 1}, s)
	if err != nil {
		return nil, fmt.Errorf("mrope slice x1: %w", err)
	}
	defer x1.Free()
	x2, err := b.Slice(x, []int{0, 0, 0, half}, []int{shape[0], shape[1], shape[2], shape[3]}, []int{1, 1, 1, 1}, s)
	if err != nil {
		return nil, fmt.Errorf("mrope slice x2: %w", err)
	}
	defer x2.Free()

	negX2, err := b.Negative(x2, s)
	if err != nil {
		return nil, fmt.Errorf("mrope negate: %w", err)
	}
	defer negX2.Free()
	rotated, err := b.ConcatenateAxis([]tensor.Array{negX2, x1}, 3, s)
	if err != nil {
		return nil, fmt.Errorf("mrope concat: %w", err)
	}
	defer rotated.Free()

	scaled, err := b.Multiply(x, cos, s)
	if err != nil {
		return nil, fmt.Errorf("mrope x*cos: %w", err)
	}
	defer scaled.Free()
	rotScaled, err := b.Multiply(rotated, sin, s)
	if err != nil {
		return nil, fmt.Errorf("mrope rot*sin: %w", err)
	}
	defer rotScaled.Free()

	return b.Add(scaled, rotScaled, s)
}
