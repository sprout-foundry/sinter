//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package qwen35

import (
	"math/rand"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/llm/vision"
	"github.com/sprout-foundry/sinter/tensor"
)

// TestImageFeatureCacheSkipsTower checks the cache actually removes the
// encoder from a repeat image, and that a hit returns the same features a
// fresh encode would.
//
// Both halves matter. A cache that returned wrong features would still look
// fast, and one that silently missed would still return correct features —
// so neither a timing check nor a correctness check alone catches a
// regression. This asserts both.
func TestImageFeatureCacheSkipsTower(t *testing.T) {
	dir := os.Getenv("SINTER_VISION_MODEL")
	if dir == "" {
		dir = os.Getenv("SINTER_MTP_PARITY_MODEL")
	}
	if dir == "" {
		t.Skip("set SINTER_VISION_MODEL to a qwen3.5 VL model directory")
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	backend := tensor.DetectBackend()
	if backend == nil || !backend.Available() {
		t.Skip("no GPU backend available")
	}

	cfg, err := loadVisionConfig(dir + "/config.json")
	if err != nil || cfg == nil {
		t.Skipf("no vision config in %s: %v", dir, err)
	}
	sf, err := llm.OpenSafetensors(dir + "/model.safetensors")
	if err != nil {
		t.Fatalf("open safetensors: %v", err)
	}
	defer sf.Release()
	prefix := "vision_tower."
	if !sf.Has(prefix + "patch_embed.proj.weight") {
		prefix = "model.visual."
	}
	stream, err := backend.DefaultGPUStream()
	if err != nil {
		t.Fatalf("gpu stream: %v", err)
	}
	defer stream.Free()

	tower, err := loadVisionTower(sf, prefix, *cfg, backend, stream)
	if err != nil {
		t.Fatalf("load tower: %v", err)
	}
	defer tower.free()

	// The cache needs only the hidden size (to validate tower output) and
	// the merge factor (ImageTokens); the rest of the text model is unused.
	q := &Qwen35{
		cfg: llm.ModelConfig{
			HiddenSize:         cfg.OutHiddenSize,
			VisionSpatialMerge: cfg.SpatialMergeSize,
		},
		backend:      backend,
		stream:       stream,
		vision:       tower,
		featureCache: newImageFeatureCache(),
	}
	defer q.featureCache.free()

	// A 1024x768 image is 3072 patches — ~0.5s through the tower on a 2B
	// model, which is what makes the skip measurable.
	img := &vision.Image{Width: 1024, Height: 768, Pixels: make([]uint8, 1024*768*3)}
	r := rand.New(rand.NewSource(7))
	for i := range img.Pixels {
		img.Pixels[i] = uint8(r.Intn(256))
	}
	px, err := vision.Preprocess(img, vision.DefaultConfig())
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}

	encode := func() ([]float32, time.Duration) {
		t0 := time.Now()
		f, err := q.encodeImages([]*vision.Pixels{px}, nil)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		// Eval inside the timing: the tower's graph is lazy, so the work
		// otherwise lands on whichever later op first synchronises.
		if err := f.Eval(); err != nil {
			f.Free()
			t.Fatalf("eval: %v", err)
		}
		d := time.Since(t0)
		data, err := f.Float32Data()
		f.Free()
		if err != nil {
			t.Fatalf("read features: %v", err)
		}
		return data, d
	}

	cold, coldDur := encode()
	warm, warmDur := encode()

	if len(cold) != len(warm) {
		t.Fatalf("cached features have %d values, fresh encode has %d", len(warm), len(cold))
	}
	for i := range cold {
		if cold[i] != warm[i] {
			t.Fatalf("cached feature %d = %v, fresh encode = %v", i, warm[i], cold[i])
		}
	}

	t.Logf("cold %v, warm %v", coldDur, warmDur)
	// The tower dominates a cold encode, so a working cache has to be
	// several times faster. A generous bound: it only has to prove the
	// encoder is gone, not pin down how much faster it is.
	if warmDur*4 > coldDur {
		t.Fatalf("cached encode took %v vs %v cold — the tower is still running", warmDur, coldDur)
	}
}
