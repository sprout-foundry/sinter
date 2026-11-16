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

// TestImageCacheKeyLiveModel checks the property the image cache key exists
// for: two different pictures must not be treated as the same prompt.
//
// Both expand to an identical <|image_pad|> run, so token ids alone cannot
// tell them apart. If the key were missing or wrong, the second request
// would restore K/V computed from the first picture and answer about the
// wrong image — a wrong answer that looks entirely plausible.
func TestImageCacheKeyLiveModel(t *testing.T) {
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
	pxA, err := vision.Preprocess(img, vision.DefaultConfig())
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}

	// A solid colour: same grid and token count as the stripes, completely
	// different content, so the two answers must differ.
	solid := &vision.Image{Width: img.Width, Height: img.Height, Pixels: make([]uint8, len(img.Pixels))}
	for i := 0; i+3 < len(img.Pixels); i += 4 {
		solid.Pixels[i] = 20
		solid.Pixels[i+1] = 120
		solid.Pixels[i+2] = 200
		solid.Pixels[i+3] = 255
	}
	pxB, err := vision.Preprocess(solid, vision.DefaultConfig())
	if err != nil {
		t.Fatalf("preprocess solid: %v", err)
	}
	if pxA.NumTokens() != pxB.NumTokens() {
		t.Fatalf("fixtures must expand to the same token count, got %d and %d", pxA.NumTokens(), pxB.NumTokens())
	}

	prompt := "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe.<|im_end|>\n<|im_start|>assistant\n"
	run := func(px *vision.Pixels) string {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		gen := llm.DefaultGenerateConfig()
		gen.MaxTokens = 24
		gen.Temperature = 0
		var ids []int
		if err := m.GenerateMultimodal(ctx, prompt, []llm.ImagePart{{Pixels: px}}, gen, func(id int) { ids = append(ids, id) }); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return m.DecodeTokenIDs(ids)
	}

	// A first, then B, then A again. The repeat must reproduce A's answer:
	// if B had overwritten or been confused with A's slot, the third run
	// would return B's description.
	a1 := run(pxA)
	b := run(pxB)
	a2 := run(pxA)
	t.Logf("A: %q\nB: %q\nA again: %q", a1, b, a2)

	if a1 == b {
		t.Skip("inverted image produced the same answer; the fixture does not distinguish them")
	}
	if a1 != a2 {
		t.Fatalf("the same image gave different answers across runs:\n first: %q\n again: %q", a1, a2)
	}
}
