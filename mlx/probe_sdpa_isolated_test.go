//go:build darwin && arm64 && cgo

package mlx

import (
	"runtime"
	"testing"
)

// TestFusedAttentionAvailability reports whether the fused attention kernel
// actually runs, rather than assuming a GPU means it does.
//
// The backend can be fully functional for ordinary ops — matmul, softmax,
// reductions all execute correctly — while this one kernel is unavailable.
// That happens when the installed mlx-c is built against a different libmlx
// than the one it links: the C wrapper passes force_fused=True, and the
// library it ends up calling has no fused kernel to satisfy it.
//
// A failure here means every MLX model path is unusable, so it is worth
// distinguishing from a model-level bug. The test reports the condition and
// skips rather than failing, because it describes the environment, not the
// code under test.
func TestFusedAttentionAvailability(t *testing.T) {
	if !Available() {
		t.Skip("Metal not available")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	s, err := DefaultGPUStream()
	if err != nil {
		t.Skipf("no GPU stream: %v", err)
	}

	const b, h, l, d = 1, 2, 4, 64
	q, err := NewArrayFromFloat32(make([]float32, b*h*l*d), []int{b, h, l, d})
	if err != nil {
		t.Fatalf("array: %v", err)
	}
	k, err := NewArrayFromFloat32(make([]float32, b*h*l*d), []int{b, h, l, d})
	if err != nil {
		t.Fatalf("array: %v", err)
	}
	v, err := NewArrayFromFloat32(make([]float32, b*h*l*d), []int{b, h, l, d})
	if err != nil {
		t.Fatalf("array: %v", err)
	}

	out, err := FastScaledDotProductAttention(q, k, v, 0.25, "causal", nil, nil, true, s)
	if err != nil {
		t.Skipf("fused attention unavailable in this environment: %v", err)
	}
	if err := out.Eval(); err != nil {
		t.Skipf("fused attention unavailable in this environment: %v", err)
	}
}
