package openaisserver

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/llm/vision"
)

// MaxImageBytes caps the decoded size of one image. A 4 MB PNG is roughly a
// 2048x2048 screenshot, well past what the model's patch budget accepts, so
// anything larger is a client mistake rather than a legitimate request.
const MaxImageBytes = 4 << 20

// MaxImagesPerRequest caps how many images one request may carry. Each image
// costs a vision-tower pass plus its patch tokens, so a flood of them is a
// memory risk on a machine sized for one model.
const MaxImagesPerRequest = 8

// DefaultImageMaxTokens caps how many <|image_pad|> tokens one image may
// expand to. Both the vision tower and the text prefill scale with it, so it
// is the main lever on image-request latency: on a 1080x2340 screenshot the
// 2B model needs 4.5s of prefill at 2482 tokens and 1.4s at 987.
//
// It is also a quality ceiling for the small models. Past roughly a thousand
// image tokens the 2B starts returning a repeated character loop instead of
// the screen's text, while the 4B still reads it correctly — so the cap is
// set where the weakest supported model stays coherent, and a caller with a
// larger model can raise it.
//
// The cap resizes the image down before patchifying, which is what the
// reference processor does with a smaller budget: the model only ever sees a
// smaller image, never a truncated one.
const DefaultImageMaxTokens = 1024

// contentPart is one entry of an OpenAI-style multimodal content array.
// A plain-string content decodes into a single text part.
type contentPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text"`
	ImageURL *imageURLObject `json:"image_url,omitempty"`
}

type imageURLObject struct {
	URL string `json:"url"`
	// Detail is accepted and ignored: the model has one fixed patch budget.
	Detail string `json:"detail,omitempty"`
}

// messageContent decodes a `content` field that may be either a JSON string
// or an array of parts. OpenAI sends the array form when images are present.
type messageContent struct {
	Text string
	// Images are the data URLs, in order.
	Images []string
	// Segments preserves the original interleaving of text and images so
	// the prompt places each marker where the client put it.
	Segments []contentSegment
}

// contentSegment is one piece of a multimodal message: either literal text
// or an image (identified by its index into Images).
type contentSegment struct {
	Text    string
	ImageIx int // -1 for a text segment
}

// imageMarker is what the Qwen3.5 chat template emits for one image. The
// model expands the single <|image_pad|> to the image's real token count at
// request time.
const imageMarker = "<|vision_start|><|image_pad|><|vision_end|>"

// renderText rebuilds the message text with an image marker at each image's
// position, so the template sees the images where the client put them.
func (c *messageContent) renderText() string {
	if len(c.Images) == 0 {
		return c.Text
	}
	var sb strings.Builder
	for _, seg := range c.Segments {
		if seg.ImageIx >= 0 {
			sb.WriteString(imageMarker)
			continue
		}
		sb.WriteString(seg.Text)
	}
	return sb.String()
}

func (c *messageContent) UnmarshalJSON(data []byte) error {
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "null" {
		return nil
	}
	if strings.HasPrefix(trimmed, "\"") {
		return json.Unmarshal(data, &c.Text)
	}
	var parts []contentPart
	if err := json.Unmarshal(data, &parts); err != nil {
		return err
	}
	var texts []string
	for _, p := range parts {
		switch p.Type {
		case "text", "input_text":
			texts = append(texts, p.Text)
			c.Segments = append(c.Segments, contentSegment{Text: p.Text, ImageIx: -1})
		case "image_url", "input_image":
			if p.ImageURL == nil || p.ImageURL.URL == "" {
				return fmt.Errorf("image_url part has no url")
			}
			c.Segments = append(c.Segments, contentSegment{ImageIx: len(c.Images)})
			c.Images = append(c.Images, p.ImageURL.URL)
		default:
			return fmt.Errorf("unsupported content part type %q", p.Type)
		}
	}
	c.Text = strings.Join(texts, "")
	return nil
}

// decodeDataURL turns a `data:image/png;base64,...` URL into pixels ready for
// the vision tower. Plain http(s) URLs are rejected: fetching them would make
// the server a proxy for arbitrary network requests.
func decodeDataURL(url string, cfg vision.Config) (*vision.Pixels, error) {
	if !strings.HasPrefix(url, "data:") {
		return nil, fmt.Errorf("only data: URLs are supported, got %q", truncateURL(url))
	}
	comma := strings.IndexByte(url, ',')
	if comma < 0 {
		return nil, fmt.Errorf("malformed data URL")
	}
	header, payload := url[len("data:"):comma], url[comma+1:]
	if !strings.Contains(header, ";base64") {
		return nil, fmt.Errorf("data URL must be base64-encoded")
	}
	// Reject before decoding: base64 inflates by 4/3, so bound the payload
	// rather than allocating a decoded buffer we then discard.
	if len(payload) > MaxImageBytes*4/3+4 {
		return nil, fmt.Errorf("image exceeds the %d byte limit", MaxImageBytes)
	}
	raw, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return nil, fmt.Errorf("decode base64 image: %w", err)
	}
	if len(raw) > MaxImageBytes {
		return nil, fmt.Errorf("image exceeds the %d byte limit", MaxImageBytes)
	}
	img, err := vision.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return vision.Preprocess(img, cfg)
}

// truncateURL shortens a URL for error messages so a multi-megabyte data URL
// does not end up in a log line or an HTTP error body.
func truncateURL(u string) string {
	if len(u) <= 64 {
		return u
	}
	return u[:64] + "..."
}

// buildImageParts pairs each decoded image with the marker the chat template
// emitted for it. The template writes one <|vision_start|><|image_pad|>
// <|vision_end|> group per image, in message order.
func buildImageParts(images []*vision.Pixels) []llm.ImagePart {
	parts := make([]llm.ImagePart, len(images))
	for i, px := range images {
		parts[i] = llm.ImagePart{Pixels: px}
	}
	return parts
}

// decodeRequestImages decodes every image in the request, in message order,
// enforcing the count and per-image size limits.
func (s *Server) decodeRequestImages(msgs []chatMessage) ([]*vision.Pixels, error) {
	total := 0
	for _, m := range msgs {
		total += len(m.Content.Images)
	}
	if total == 0 {
		return nil, nil
	}
	if total > MaxImagesPerRequest {
		return nil, fmt.Errorf("request has %d images, the limit is %d", total, MaxImagesPerRequest)
	}
	out := make([]*vision.Pixels, 0, total)
	cfg := vision.DefaultConfig().WithMaxTokens(s.ImageMaxTokens)
	for _, m := range msgs {
		for _, url := range m.Content.Images {
			px, err := decodeDataURL(url, cfg)
			if err != nil {
				return nil, err
			}
			out = append(out, px)
		}
	}
	return out, nil
}
