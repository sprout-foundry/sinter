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

// visionCacheFixture is a loaded vision tower plus a Qwen35 that holds only
// the fields encodeImages reads, so these tests exercise the cache without
// paying for the text model.
type visionCacheFixture struct {
	q      *Qwen35
	tower  *visionTower
	stream tensor.Stream
}

// newVisionCacheFixture loads the tower from SINTER_VISION_MODEL. It skips
// when no weights are on disk.
func newVisionCacheFixture(t *testing.T) *visionCacheFixture {
	t.Helper()
	dir := os.Getenv("SINTER_VISION_MODEL")
	if dir == "" {
		dir = os.Getenv("SINTER_MTP_PARITY_MODEL")
	}
	if dir == "" {
		t.Skip("set SINTER_VISION_MODEL to a qwen3.5 VL model directory")
	}

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
	t.Cleanup(sf.Release)

	prefix := "vision_tower."
	if !sf.Has(prefix + "patch_embed.proj.weight") {
		prefix = "model.visual."
	}
	stream, err := backend.DefaultGPUStream()
	if err != nil {
		t.Fatalf("gpu stream: %v", err)
	}
	t.Cleanup(stream.Free)

	tower, err := loadVisionTower(sf, prefix, *cfg, backend, stream)
	if err != nil {
		t.Fatalf("load tower: %v", err)
	}
	t.Cleanup(tower.free)

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
	t.Cleanup(q.featureCache.free)

	return &visionCacheFixture{q: q, tower: tower, stream: stream}
}

// encode runs one encodeImages call and reads the result back. Evaluation
// happens inside the returned duration: the tower's graph is lazy, so the
// work otherwise lands on whichever later op first synchronises.
func (f *visionCacheFixture) encode(t *testing.T, pixels []*vision.Pixels, spans []llm.ImageInput) ([]float32, time.Duration) {
	t.Helper()
	t0 := time.Now()
	a, err := f.q.encodeImages(pixels, spans)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if err := a.Eval(); err != nil {
		a.Free()
		t.Fatalf("eval: %v", err)
	}
	d := time.Since(t0)
	data, err := a.Float32Data()
	a.Free()
	if err != nil {
		t.Fatalf("read features: %v", err)
	}
	return data, d
}

// noiseImage builds a deterministic random image at the given size. Random
// pixels keep the tower's activations from being degenerate, so the timing
// reflects real work.
func noiseImage(seed int64, width, height int) *vision.Image {
	img := &vision.Image{Width: width, Height: height, Pixels: make([]uint8, width*height*3)}
	r := rand.New(rand.NewSource(seed))
	for i := range img.Pixels {
		img.Pixels[i] = uint8(r.Intn(256))
	}
	return img
}

// TestImageFeatureCacheSkipsTower checks the cache actually removes the
// encoder from a repeat image, and that a hit returns the same features a
// fresh encode would.
//
// Both halves matter. A cache that returned wrong features would still look
// fast, and one that silently missed would still return correct features —
// so neither a timing check nor a correctness check alone catches a
// regression. This asserts both.
func TestImageFeatureCacheSkipsTower(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	f := newVisionCacheFixture(t)

	// A 1024x768 image is 3072 patches — ~0.5s through the tower on a 2B
	// model, which is what makes the skip measurable.
	px, err := vision.Preprocess(noiseImage(7, 1024, 768), vision.DefaultConfig())
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}

	cold, coldDur := f.encode(t, []*vision.Pixels{px}, nil)
	warm, warmDur := f.encode(t, []*vision.Pixels{px}, nil)

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

// TestImageFeatureCacheMultiImage covers the paths a single image cannot:
// several images in one prompt, with some cached and some not. The mixed case
// is where ownership goes wrong — cached uploads and fresh encodes share one
// pieces slice, and the concatenation must see all of them exactly once.
//
// It also guards a GGML backend bug this test found: concatenating arrays that
// are still pending under batched evaluation returned garbage (raw patch data
// instead of features), because the concat chain lived in the recyclable ops
// context and a later flush freed those nodes before the graph ran. See
// GGMLBackend.ConcatenateAxis. A single-image encode never concatenates, so
// only this test reaches it.
func TestImageFeatureCacheMultiImage(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	f := newVisionCacheFixture(t)

	var imgs []*vision.Pixels
	var spans []llm.ImageInput
	start := 0
	for seed := int64(1); seed <= 3; seed++ {
		px, err := vision.Preprocess(noiseImage(seed, 128, 128), vision.DefaultConfig())
		if err != nil {
			t.Fatalf("preprocess: %v", err)
		}
		imgs = append(imgs, px)
		spans = append(spans, llm.ImageInput{Start: start, GridTHW: px.GridTHW})
		start += px.NumTokens()
	}

	// All cold, then all cached, in the same cache.
	cold, _ := f.encode(t, imgs, spans)
	warm, _ := f.encode(t, imgs, spans)

	// Mixed: only the middle image is resident, so the other two go through
	// the tower alongside a cached one.
	f.q.featureCache.free()
	middle := []*vision.Pixels{imgs[1]}
	f.encode(t, middle, []llm.ImageInput{{Start: 0, GridTHW: imgs[1].GridTHW}})
	mixed, _ := f.encode(t, imgs, spans)

	if len(cold) != len(warm) || len(cold) != len(mixed) {
		t.Fatalf("length mismatch: cold=%d warm=%d mixed=%d", len(cold), len(warm), len(mixed))
	}
	for i := range cold {
		if cold[i] != warm[i] {
			t.Logf("lens cold=%d warm=%d; first diff at %d: cold=%v warm=%v", len(cold), len(warm), i, cold[i], warm[i])
			t.Fatalf("all-cached differs at %d: %v vs %v", i, warm[i], cold[i])
		}
		if cold[i] != mixed[i] {
			t.Fatalf("mixed differs at %d: %v vs %v", i, mixed[i], cold[i])
		}
	}
}
