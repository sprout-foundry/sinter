//go:build darwin && arm64 && cgo

package mlx

import (
	"math"
	"runtime"
	"testing"
)

// TestSpikeLayerNormAndGELU checks the two ops the vision tower added
// against straightforward float64 Go implementations. Both are new
// primitives composed from existing kernels (LayerNorm wraps
// mlx_fast_layer_norm; GELU is built from erf/tanh), so a mistake here
// would be invisible until the tower's parity test, where it would be
// indistinguishable from a layout bug.
func TestSpikeLayerNormAndGELU(t *testing.T) {
	if !Available() {
		t.Skip("Metal not available")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	s, err := NewGPUStream()
	if err != nil {
		t.Skipf("no stream: %v", err)
	}
	defer s.Free()

	const rows, dim = 4, 64
	const eps = 1e-6

	x := make([]float32, rows*dim)
	for i := range x {
		// Spread of magnitudes so mean/variance are non-trivial.
		x[i] = float32(math.Sin(float64(i)*0.37)) * float32(1+(i%7))
	}
	w := make([]float32, dim)
	b := make([]float32, dim)
	for i := range w {
		w[i] = 0.5 + float32(i%5)*0.1
		b[i] = float32(i%3) * 0.25
	}

	xArr, err := NewArrayFromFloat32(x, []int{rows, dim})
	if err != nil {
		t.Fatalf("x array: %v", err)
	}
	defer xArr.Free()
	wArr, err := NewArrayFromFloat32(w, []int{dim})
	if err != nil {
		t.Fatalf("w array: %v", err)
	}
	defer wArr.Free()
	bArr, err := NewArrayFromFloat32(b, []int{dim})
	if err != nil {
		t.Fatalf("b array: %v", err)
	}
	defer bArr.Free()

	ln, err := LayerNorm(xArr, wArr, bArr, eps, s)
	if err != nil {
		t.Fatalf("LayerNorm: %v", err)
	}
	defer ln.Free()
	gotLN, err := ln.Float32Data()
	if err != nil {
		t.Fatalf("LayerNorm data: %v", err)
	}

	for r := 0; r < rows; r++ {
		row := x[r*dim : (r+1)*dim]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= dim
		var variance float64
		for _, v := range row {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= dim
		for c := 0; c < dim; c++ {
			want := (float64(row[c])-mean)/math.Sqrt(variance+eps)*float64(w[c]) + float64(b[c])
			got := float64(gotLN[r*dim+c])
			if math.Abs(got-want) > 2e-3 {
				t.Fatalf("LayerNorm[%d,%d] = %v, want %v", r, c, got, want)
			}
		}
	}

	// GELU, both variants, over a range that exercises the tails.
	xs := make([]float32, 41)
	for i := range xs {
		xs[i] = float32(-6 + float64(i)*0.3)
	}
	xsArr, err := NewArrayFromFloat32(xs, []int{1, len(xs)})
	if err != nil {
		t.Fatalf("gelu input: %v", err)
	}
	defer xsArr.Free()

	for _, exact := range []bool{true, false} {
		out, err := GELU(xsArr, exact, s)
		if err != nil {
			t.Fatalf("GELU(exact=%v): %v", exact, err)
		}
		got, err := out.Float32Data()
		out.Free()
		if err != nil {
			t.Fatalf("GELU data: %v", err)
		}
		for i, v := range xs {
			fv := float64(v)
			var want float64
			if exact {
				want = 0.5 * fv * (1 + math.Erf(fv/math.Sqrt2))
			} else {
				want = 0.5 * fv * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(fv+0.044715*fv*fv*fv)))
			}
			if math.Abs(float64(got[i])-want) > 2e-3 {
				t.Fatalf("GELU(exact=%v)[%d] x=%v = %v, want %v", exact, i, v, got[i], want)
			}
		}
	}
}
