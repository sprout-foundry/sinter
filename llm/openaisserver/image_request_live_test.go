//go:build ggml && cgo && ((darwin && arm64) || (linux && (arm64 || amd64)))

package openaisserver

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/sprout-foundry/sinter/llm"
	_ "github.com/sprout-foundry/sinter/llm/qwen35"
	_ "github.com/sprout-foundry/sinter/tensor/ggml"
)

// TestImageRequestLiveModel sends a real OpenAI-style request carrying a
// data URL through the HTTP handler and checks the answer describes the
// picture. This exercises the whole chain: JSON part decoding, base64
// decode, preprocessing, the vision tower, 3-D positions, and decode.
func TestImageRequestLiveModel(t *testing.T) {
	dir := os.Getenv("SINTER_VISION_MODEL")
	if dir == "" {
		dir = os.Getenv("SINTER_MTP_PARITY_MODEL")
	}
	if dir == "" {
		t.Skip("set SINTER_VISION_MODEL to a qwen3.5 VL model directory")
	}
	m, err := llm.NewModel(dir)
	if err != nil {
		t.Skipf("load model: %v", err)
	}
	defer m.Close()

	// The fixture is the striped test image from the vision package.
	raw, err := os.ReadFile("../vision/testdata/src_96.png")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw)

	body, _ := json.Marshal(map[string]any{
		"model": "test",
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": "Describe this image in one sentence."},
				{"type": "image_url", "image_url": map[string]string{"url": dataURL}},
			},
		}},
		"max_tokens":  20,
		"temperature": 0,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	New(m, "test", 0).Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var resp chatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("no choices in response")
	}
	text := strings.ToLower(resp.Choices[0].Message.Content)
	t.Logf("answer: %q", text)

	for _, k := range []string{"stripe", "line", "vertical", "colour", "color", "gradient", "abstract", "pattern"} {
		if strings.Contains(text, k) {
			return
		}
	}
	t.Fatalf("answer %q does not describe the striped test image", text)
}

// TestImageRequestRejectsBadPayload checks the handler returns 400 rather
// than 500 for a malformed image, so a client error is not logged as a
// server fault.
func TestImageRequestRejectsBadPayload(t *testing.T) {
	body, _ := json.Marshal(map[string]any{
		"model": "test",
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64,bm90YW5pbWFnZQ=="}},
			},
		}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	New(&fakeModel{}, "test", 0).Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body.String())
	}
}

// TestPNGFixtureIsDecodable guards the fixture itself, so a corrupted test
// image fails here instead of looking like a vision bug.
func TestPNGFixtureIsDecodable(t *testing.T) {
	f, err := os.Open("../vision/testdata/src_96.png")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if img.Bounds().Dx() != 96 || img.Bounds().Dy() != 96 {
		t.Errorf("fixture is %v, want 96x96", img.Bounds())
	}
	_ = image.Rect
}
