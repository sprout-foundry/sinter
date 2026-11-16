package openaisserver

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"github.com/sprout-foundry/sinter/llm/vision"
	"image"
	"image/png"
	"strings"
	"testing"
)

// pngDataURL builds a data URL for a small solid-colour PNG.
func pngDataURL(t *testing.T, w, h int) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = 0x80
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

// TestMessageContentStringAndParts checks both content encodings decode, and
// that the array form keeps the text/image interleaving.
func TestMessageContentStringAndParts(t *testing.T) {
	var plain chatMessage
	if err := json.Unmarshal([]byte(`{"role":"user","content":"hello"}`), &plain); err != nil {
		t.Fatalf("plain string: %v", err)
	}
	if plain.Content.Text != "hello" || len(plain.Content.Images) != 0 {
		t.Fatalf("plain string decoded to %+v", plain.Content)
	}

	var multi chatMessage
	body := `{"role":"user","content":[
		{"type":"text","text":"before "},
		{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}},
		{"type":"text","text":" after"}
	]}`
	if err := json.Unmarshal([]byte(body), &multi); err != nil {
		t.Fatalf("parts: %v", err)
	}
	if len(multi.Content.Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(multi.Content.Images))
	}
	got := multi.Content.renderText()
	want := "before " + imageMarker + " after"
	if got != want {
		t.Errorf("rendered %q, want %q", got, want)
	}
}

// TestMessageContentRejectsUnknownPart ensures an unrecognised part type is
// an error rather than being silently dropped (which would shift the marker
// count out of step with the images).
func TestMessageContentRejectsUnknownPart(t *testing.T) {
	var m chatMessage
	body := `{"role":"user","content":[{"type":"audio","audio":{}}]}`
	if err := json.Unmarshal([]byte(body), &m); err == nil {
		t.Fatal("expected an error for an unsupported part type")
	}
}

// TestDecodeDataURLRejectsNonDataURL checks that remote URLs are refused: the
// server must not become a proxy for arbitrary network fetches.
func TestDecodeDataURLRejectsNonDataURL(t *testing.T) {
	if _, err := decodeDataURL("https://example.com/cat.png", vision.DefaultConfig()); err == nil {
		t.Fatal("expected an error for an http URL")
	}
}

// TestDecodeDataURLEnforcesSizeLimit checks the byte cap is applied before
// the image is decoded, and that the error mentions the limit.
func TestDecodeDataURLEnforcesSizeLimit(t *testing.T) {
	oversized := "data:image/png;base64," + strings.Repeat("A", MaxImageBytes*4/3+16)
	_, err := decodeDataURL(oversized, vision.DefaultConfig())
	if err == nil {
		t.Fatal("expected an error for an oversized image")
	}
	if !strings.Contains(err.Error(), "limit") {
		t.Errorf("error %q should mention the size limit", err)
	}
}

// TestDecodeDataURLRoundTrip decodes a real PNG and checks it comes back
// preprocessed with a sane patch grid.
func TestDecodeDataURLRoundTrip(t *testing.T) {
	url := pngDataURL(t, 96, 96)
	px, err := decodeDataURL(url, vision.DefaultConfig())
	if err != nil {
		t.Fatalf("decodeDataURL: %v", err)
	}
	if px.NumPatches() == 0 {
		t.Fatal("no patches produced")
	}
	if px.GridTHW[1]%16 != 0 || px.GridTHW[2]%16 != 0 {
		t.Errorf("grid %v not a multiple of the patch size", px.GridTHW)
	}
	if len(px.Data) != px.NumPatches()*16*16*3*2 {
		t.Errorf("data has %d values for %d patches", len(px.Data), px.NumPatches())
	}
}

// TestDecodeRequestImagesEnforcesCount checks the per-request image cap.
func TestDecodeRequestImagesEnforcesCount(t *testing.T) {
	s := &Server{}
	url := pngDataURL(t, 32, 32)
	msgs := make([]chatMessage, 1)
	for i := 0; i < MaxImagesPerRequest+1; i++ {
		msgs[0].Content.Images = append(msgs[0].Content.Images, url)
	}
	if _, err := s.decodeRequestImages(msgs); err == nil {
		t.Fatal("expected an error for too many images")
	}
}

// TestDecodeRequestImagesNoneIsNil keeps the text-only path allocation-free.
func TestDecodeRequestImagesNoneIsNil(t *testing.T) {
	s := &Server{}
	msgs := []chatMessage{{Content: messageContent{Text: "hi"}}}
	got, err := s.decodeRequestImages(msgs)
	if err != nil {
		t.Fatalf("decodeRequestImages: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil images, got %d", len(got))
	}
}
