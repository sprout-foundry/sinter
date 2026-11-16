//go:build ggml && cgo && ((darwin && arm64) || (linux && (arm64 || amd64)))

package mropeparity

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/sprout-foundry/sinter/llm"
	_ "github.com/sprout-foundry/sinter/llm/qwen35"
	"github.com/sprout-foundry/sinter/llm/vision"
	_ "github.com/sprout-foundry/sinter/tensor/ggml"
)

// TestImageCacheHitSkipsPrefill checks the property the cached hidden state
// exists for: a repeated image prompt must not re-run the prefill.
//
// The correctness test next door would still pass if the hit silently fell
// back to a full prefill, so this one asserts the work actually went away.
// It compares prefill wall time against a first, cold run rather than an
// absolute number, because the absolute cost depends on the machine.
func TestImageCacheHitSkipsPrefill(t *testing.T) {
	dir := os.Getenv("SINTER_VISION_MODEL")
	if dir == "" {
		t.Skip("set SINTER_VISION_MODEL to a qwen3.5 VL model directory")
	}
	m, err := llm.NewModel(dir)
	if err != nil {
		t.Skipf("load model: %v", err)
	}
	defer m.Close()

	f, err := os.Open("../vision/testdata/src_96.png")
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()
	img, err := vision.Decode(f)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	px, err := vision.Preprocess(img, vision.DefaultConfig())
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}

	prompt := "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe.<|im_end|>\n<|im_start|>assistant\n"
	run := func() (string, int64) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		gen := llm.DefaultGenerateConfig()
		gen.MaxTokens = 16
		gen.Temperature = 0
		var ids []int
		if err := m.GenerateMultimodal(ctx, prompt, []llm.ImagePart{{Pixels: px}}, gen, func(id int) { ids = append(ids, id) }); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return m.DecodeTokenIDs(ids), m.LastPromptMs()
	}

	first, coldMs := run()
	second, warmMs := run()
	t.Logf("prefill: cold=%dms warm=%dms", coldMs, warmMs)

	if first != second {
		t.Fatalf("cache hit changed the answer:\n first: %q\n again: %q", first, second)
	}
	if warmMs == 0 {
		t.Fatalf("warm prefill reported 0ms (cold %dms); the timer did not run", coldMs)
	}
	// A skipped prefill is one final norm plus the vocabulary head, which is
	// orders of magnitude below re-encoding 64 image tokens through 24
	// layers. A generous factor still separates the two cleanly.
	if warmMs*4 > coldMs {
		t.Fatalf("warm prefill %dms is not meaningfully below cold %dms: the cache hit did not skip the prefill", warmMs, coldMs)
	}
}
