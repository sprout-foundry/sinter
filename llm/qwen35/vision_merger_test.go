//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package qwen35

import (
	"math"
	"testing"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/tensor"
)

// TestMergerAppliesBias guards the merger's two biases. The tower's merged
// output is the only thing the language model sees of an image, and a
// missing bias here is silent: the features stay plausible and only the
// text degrades. The checkpoint carries both, so they must be applied.
//
// Hidden size 1 makes LayerNorm collapse to zero (one element has no
// variance), which leaves the two linear layers and their biases as the
// only thing under test — and makes the expected value exact.
func TestMergerAppliesBias(t *testing.T) {
	b := tensor.DetectBackend()
	if b == nil || !b.Available() {
		t.Skip("no backend")
	}
	s, err := b.DefaultGPUStream()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Free()

	ones, _ := b.NewArrayFromFloat32([]float32{1}, []int{1})
	defer ones.Free()
	zeros, _ := b.NewArrayFromFloat32([]float32{0}, []int{1})
	defer zeros.Free()

	// fc1: weight [[2]], bias [0.5]; fc2: weight [[3]], bias [1.5].
	w1, _ := b.NewArrayFromFloat32([]float32{2}, []int{1, 1})
	defer w1.Free()
	b1, _ := b.NewArrayFromFloat32([]float32{0.5}, []int{1})
	defer b1.Free()
	w2, _ := b.NewArrayFromFloat32([]float32{3}, []int{1, 1})
	defer w2.Free()
	b2, _ := b.NewArrayFromFloat32([]float32{1.5}, []int{1})
	defer b2.Free()

	tower := &visionTower{
		cfg: visionConfig{
			HiddenSize:       1,
			SpatialMergeSize: 1,
			OutHiddenSize:    1,
			LayerNormEps:     1e-6,
		},
		backend:     b,
		stream:      s,
		mergerNormW: ones,
		mergerNormB: zeros,
		mergerFC1:   llm.NewLinearFull(w1),
		mergerFC1B:  b1,
		mergerFC2:   llm.NewLinearFull(w2),
		mergerFC2B:  b2,
	}

	// Two patches, merge size 1, so two merged tokens.
	x, _ := b.NewArrayFromFloat32([]float32{7, -3}, []int{2, 1})
	defer x.Free()

	out, err := tower.merge(x, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()

	got, err := out.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("output has %d values, want 2", len(got))
	}

	// LayerNorm gives 0, so fc1 yields its bias, GELU is exact, and fc2 adds
	// its own bias.
	act := 0.5 * 0.5 * (1 + math.Erf(0.5/math.Sqrt2)) // GELU(0.5) = 0.5*Phi(0.5)
	want := float32(act*3 + 1.5)
	for i, v := range got {
		if math.Abs(float64(v-want)) > 1e-4 {
			t.Errorf("token %d: got %v, want %v (missing merger bias?)", i, v, want)
		}
	}
}
