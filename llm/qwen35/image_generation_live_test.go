//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package qwen35

import (
	"context"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/llm/vision"
	"github.com/sprout-foundry/sinter/tensor"
)

// TestImageGenerationLiveModel runs a prompt with a real image through the
// full path — vision tower, feature splice, 3-D positions, decode — and
// checks the answer actually describes the picture.
//
// The fixture is vertical stripes of red/green gradient with a blue
// stripe pattern, so a working pipeline says something about stripes or
// colour rather than producing unrelated text.
func TestImageGenerationLiveModel(t *testing.T) {
	dir := os.Getenv("SINTER_VISION_MODEL")
	if dir == "" {
		dir = os.Getenv("SINTER_MTP_PARITY_MODEL")
	}
	if dir == "" {
		t.Skip("set SINTER_VISION_MODEL to a qwen3.5 VL model directory")
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if b := tensor.DetectBackend(); b == nil || !b.Available() {
		t.Skip("no GPU backend available")
	}

	m, err := llm.NewModel(dir)
	if err != nil {
		t.Skipf("load model: %v", err)
	}
	defer m.Close()

	f, err := os.Open("../vision/testdata/src_96.png")
	if err != nil {
		t.Fatalf("open image: %v", err)
	}
	defer f.Close()
	img, err := vision.Decode(f)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	pixels, err := vision.Preprocess(img, vision.DefaultConfig())
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}

	prompt := "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" +
		"Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n"

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = 20
	cfg.Temperature = 0
	cfg.PromptLookupMaxDrafts = 0

	var tokens []int
	err = m.GenerateMultimodal(context.Background(), prompt,
		[]llm.ImagePart{{Pixels: pixels}}, cfg,
		func(id int) { tokens = append(tokens, id) })
	if err != nil {
		t.Fatalf("GenerateMultimodal: %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("model produced no tokens")
	}

	text := strings.ToLower(m.DecodeTokenIDs(tokens))
	t.Logf("answer: %q", text)

	// The image is stripes and colour gradients. A broken image path (wrong
	// features, misaligned positions, unrelated K/V) produces generic or
	// off-topic text instead.
	keywords := []string{"stripe", "stripe", "line", "vertical", "colour", "color", "gradient", "abstract", "pattern"}
	for _, k := range keywords {
		if strings.Contains(text, k) {
			return
		}
	}
	t.Fatalf("answer %q does not describe the striped test image", text)
}
