//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package llm

import (
	"math"
	"testing"

	"github.com/sprout-foundry/sinter/tensor"
)

// TestMRoPEPositions pins the 3-D position layout against the reference's
// get_rope_index. The expected values come from running the reference
// algorithm on: 5 text tokens, one 4x4-patch image (2x2 merged grid, 4
// tokens), then 3 text tokens.
func TestMRoPEPositions(t *testing.T) {
	const merge = 2
	images := []ImagePlaceholder{{Start: 5, GridTHW: [3]int{1, 4, 4}}}
	gotT, gotH, gotW, advance, err := MRoPEPositions(12, images, merge)
	if err != nil {
		t.Fatalf("MRoPEPositions: %v", err)
	}

	wantT := []int{0, 1, 2, 3, 4, 5, 5, 5, 5, 7, 8, 9}
	wantH := []int{0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9}
	wantW := []int{0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 9}
	for i := range wantT {
		if gotT[i] != wantT[i] || gotH[i] != wantH[i] || gotW[i] != wantW[i] {
			t.Fatalf("token %d: got (%d,%d,%d), want (%d,%d,%d)",
				i, gotT[i], gotH[i], gotW[i], wantT[i], wantH[i], wantW[i])
		}
	}
	if advance != 10 {
		t.Errorf("advance = %d, want 10", advance)
	}
	// Text after the image continues from the 3-D position, not the token
	// count: 12 tokens but only 10 positions were consumed.
	if delta := MRoPEDelta(advance, 12); delta != -2 {
		t.Errorf("delta = %d, want -2", delta)
	}
}

// TestMRoPEPositionsImageFirst covers an image at the very start of the
// prompt, where the text counter has not advanced yet.
func TestMRoPEPositionsImageFirst(t *testing.T) {
	images := []ImagePlaceholder{{Start: 0, GridTHW: [3]int{1, 4, 4}}}
	gotT, gotH, gotW, advance, err := MRoPEPositions(7, images, 2)
	if err != nil {
		t.Fatalf("MRoPEPositions: %v", err)
	}
	wantT := []int{0, 0, 0, 0, 2, 3, 4}
	wantH := []int{0, 0, 1, 1, 2, 3, 4}
	wantW := []int{0, 1, 0, 1, 2, 3, 4}
	for i := range wantT {
		if gotT[i] != wantT[i] || gotH[i] != wantH[i] || gotW[i] != wantW[i] {
			t.Fatalf("token %d: got (%d,%d,%d), want (%d,%d,%d)",
				i, gotT[i], gotH[i], gotW[i], wantT[i], wantH[i], wantW[i])
		}
	}
	if advance != 5 {
		t.Errorf("advance = %d, want 5", advance)
	}
}

// TestMRoPEPositionsTextOnly checks that a prompt with no images reduces to
// plain 1-D positions, which is what keeps text-only output unchanged.
func TestMRoPEPositionsTextOnly(t *testing.T) {
	gotT, gotH, gotW, advance, err := MRoPEPositions(5, nil, 2)
	if err != nil {
		t.Fatalf("MRoPEPositions: %v", err)
	}
	for i := 0; i < 5; i++ {
		if gotT[i] != i || gotH[i] != i || gotW[i] != i {
			t.Fatalf("token %d: got (%d,%d,%d), want (%d,%d,%d)", i, gotT[i], gotH[i], gotW[i], i, i, i)
		}
	}
	if advance != 5 {
		t.Errorf("advance = %d, want 5", advance)
	}
	if delta := MRoPEDelta(advance, 5); delta != 0 {
		t.Errorf("text-only delta = %d, want 0", delta)
	}
}

// TestMRoPEPositionsRejectsOverrun ensures a placeholder whose span exceeds
// the prompt is an error rather than a silent truncation.
func TestMRoPEPositionsRejectsOverrun(t *testing.T) {
	images := []ImagePlaceholder{{Start: 0, GridTHW: [3]int{1, 8, 8}}}
	if _, _, _, _, err := MRoPEPositions(4, images, 2); err == nil {
		t.Fatal("expected an error for an image span longer than the prompt")
	}
}

// TestMRoPETablesLayout checks the frequency recomposition: with
// mrope_section [11,11,10] and head_dim 64, slots 0,3,6..30 come from T,
// 1,4,7..31 from H, and 2,5,8..29 from W.
func TestMRoPETablesLayout(t *testing.T) {
	m, err := NewMRoPE([]int{11, 11, 10}, 10000, 64)
	if err != nil {
		t.Fatalf("NewMRoPE: %v", err)
	}
	axis := m.freqIndexAxis()
	if len(axis) != 32 {
		t.Fatalf("axis length = %d, want 32", len(axis))
	}
	counts := [3]int{}
	for _, a := range axis {
		counts[a]++
	}
	if counts != [3]int{11, 11, 10} {
		t.Errorf("axis counts = %v, want [11 11 10]", counts)
	}
	// Spot-check the interleave pattern.
	for _, tc := range []struct{ idx, want int }{
		{0, 0}, {1, 1}, {2, 2}, {3, 0}, {4, 1}, {5, 2}, {31, 1}, {30, 0}, {29, 2},
	} {
		if axis[tc.idx] != tc.want {
			t.Errorf("axis[%d] = %d, want %d", tc.idx, axis[tc.idx], tc.want)
		}
	}
}

// TestMRoPETablesTextMatchesPlainRoPE is the invariant that keeps text-only
// generation bit-compatible: when t == h == w, every slot uses the same
// position, so the table must equal ordinary 1-D RoPE's cos/sin.
func TestMRoPETablesTextMatchesPlainRoPE(t *testing.T) {
	const headDim = 64
	m, err := NewMRoPE([]int{11, 11, 10}, 10000, headDim)
	if err != nil {
		t.Fatalf("NewMRoPE: %v", err)
	}
	half := headDim / 2
	inv := make([]float64, half)
	for i := 0; i < half; i++ {
		inv[i] = 1.0 / math.Pow(10000, float64(2*i)/float64(headDim))
	}
	axis := m.freqIndexAxis()
	for _, p := range []int{0, 1, 7, 100} {
		tbl := make([]float64, half)
		for f := 0; f < half; f++ {
			tbl[f] = math.Cos(float64(p) * inv[f])
		}
		for f := 0; f < half; f++ {
			// Every axis carries the same position, so whichever one drives
			// slot f, the angle is identical.
			if axis[f] != 0 && axis[f] != 1 && axis[f] != 2 {
				t.Fatalf("slot %d has invalid axis %d", f, axis[f])
			}
			want := tbl[f]
			got := math.Cos(float64(p) * inv[f])
			if math.Abs(got-want) > 1e-12 {
				t.Fatalf("slot %d: %v != %v", f, got, want)
			}
		}
	}
}

// TestApplyMRoPEPartialRotation checks that only the first rotaryDims columns
// of the head rotate. A full-head rotation against a rotaryDims-wide table
// does not fail loudly — it either errors on the shape mismatch or, when the
// shapes happen to line up, silently rotates dims the model expects to pass
// through unchanged.
func TestApplyMRoPEPartialRotation(t *testing.T) {
	backend := tensor.DetectBackend()
	if backend == nil || !backend.Available() {
		t.Skip("no tensor backend available on this platform")
	}
	stream, err := backend.DefaultGPUStream()
	if err != nil {
		t.Fatalf("get stream: %v", err)
	}
	defer stream.Free()

	const heads, n, headDim, rotaryDims = 1, 1, 8, 4
	// Distinct values so a wrongly-rotated tail is detectable: the tail
	// (columns 4..7) must come back bit-identical.
	vals := []float32{1, 2, 3, 4, 100, 200, 300, 400}
	x, err := backend.NewArrayFromFloat32(vals, []int{1, heads, n, headDim})
	if err != nil {
		t.Fatalf("new array: %v", err)
	}
	defer x.Free()

	// cos=1, sin=0 for every position: rotation is the identity, so the
	// whole tensor must round-trip unchanged.
	cos, err := backend.NewArrayFromFloat32([]float32{1, 1, 1, 1}, []int{1, 1, n, rotaryDims})
	if err != nil {
		t.Fatalf("new cos: %v", err)
	}
	defer cos.Free()
	sin, err := backend.NewArrayFromFloat32([]float32{0, 0, 0, 0}, []int{1, 1, n, rotaryDims})
	if err != nil {
		t.Fatalf("new sin: %v", err)
	}
	defer sin.Free()

	out, err := ApplyMRoPE(x, cos, sin, rotaryDims, backend, stream)
	if err != nil {
		t.Fatalf("ApplyMRoPE: %v", err)
	}
	defer out.Free()
	if err := out.Eval(); err != nil {
		t.Fatalf("eval: %v", err)
	}
	got, err := out.Float32Data()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != len(vals) {
		t.Fatalf("got %d values, want %d", len(got), len(vals))
	}
	for i := range vals {
		if math.Abs(float64(got[i]-vals[i])) > 1e-5 {
			t.Fatalf("element %d: got %v, want %v (identity rotation must be a no-op)", i, got[i], vals[i])
		}
	}
}
