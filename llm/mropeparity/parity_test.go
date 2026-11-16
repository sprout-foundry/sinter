//go:build ggml && cgo && ((darwin && arm64) || (linux && (arm64 || amd64)))

// Package mropeparity holds the mRoPE table parity test.
//
// It lives in its own package because it blank-imports the ggml backend to
// register it. Doing that inside package llm would change which backend
// llm's own tests detect, and several of them exercise capabilities (such as
// on-the-fly quantization) that the ggml backend stubs out.
package mropeparity

import (
	"math"
	"runtime"
	"testing"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/llm/vision"
	"github.com/sprout-foundry/sinter/tensor"
	_ "github.com/sprout-foundry/sinter/tensor/ggml"
)

// TestMRoPETablesParityReference compares the rotary tables against the
// reference implementation (transformers' Qwen3_5TextRotaryEmbedding with
// this checkpoint's config: head_dim 256, partial_rotary_factor 0.25,
// rope_theta 1e7, mrope_section [11,11,10]).
//
// The fixture covers 5 text tokens, one 4x4-patch image (2x2 merged grid),
// then 3 text tokens.
func TestMRoPETablesParityReference(t *testing.T) {
	backend := tensor.DetectBackend()
	if backend == nil || !backend.Available() {
		t.Skip("no GPU backend available")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	stream, err := backend.DefaultGPUStream()
	if err != nil {
		t.Fatalf("gpu stream: %v", err)
	}
	defer stream.Free()

	// The tables cover the rotated slice: head_dim 256 * 0.25 = 64.
	const rotatedDims = 64
	m, err := llm.NewMRoPE([]int{11, 11, 10}, 10000000, rotatedDims)
	if err != nil {
		t.Fatalf("NewMRoPE: %v", err)
	}

	tPos := []int{0, 1, 2, 3, 4, 5, 5, 5, 5, 7, 8, 9}
	hPos := []int{0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9}
	wPos := []int{0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 9}

	cos, sin, err := m.Tables(tPos, hPos, wPos, backend, stream)
	if err != nil {
		t.Fatalf("Tables: %v", err)
	}
	defer cos.Free()
	defer sin.Free()

	gotCos, err := cos.Float32Data()
	if err != nil {
		t.Fatalf("read cos: %v", err)
	}
	gotSin, err := sin.Float32Data()
	if err != nil {
		t.Fatalf("read sin: %v", err)
	}

	wantCos, cosShape, err := vision.LoadNpyFloat32("../testdata/ref_mrope_cos.npy")
	if err != nil {
		t.Fatalf("load reference cos: %v", err)
	}
	wantSin, _, err := vision.LoadNpyFloat32("../testdata/ref_mrope_sin.npy")
	if err != nil {
		t.Fatalf("load reference sin: %v", err)
	}
	if len(wantCos) != len(gotCos) {
		t.Fatalf("cos has %d values, reference has %d (shape %v)", len(gotCos), len(wantCos), cosShape)
	}

	// The reference computes in float32 too, so the only difference should
	// be the order of the cos/sin evaluation.
	const tol = 1e-6
	maxCos, maxSin := 0.0, 0.0
	for i := range wantCos {
		if d := math.Abs(float64(gotCos[i] - wantCos[i])); d > maxCos {
			maxCos = d
		}
		if d := math.Abs(float64(gotSin[i] - wantSin[i])); d > maxSin {
			maxSin = d
		}
	}
	t.Logf("mrope tables: max cos diff %.3e, max sin diff %.3e", maxCos, maxSin)
	if maxCos > tol || maxSin > tol {
		t.Fatalf("mrope tables differ from the reference: cos %.3e, sin %.3e (tolerance %v)", maxCos, maxSin, tol)
	}
}
