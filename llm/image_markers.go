//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package llm

import "fmt"

// expandImageMarkers replaces each <|image_pad|> token with the run of
// tokens the vision tower produces for the corresponding image, and returns
// the expanded token ids plus each image's span.
//
// The chat template emits one <|image_pad|> per image between
// <|vision_start|> and <|vision_end|>. The real count depends on the image's
// patch grid, which is only known after preprocessing, so the expansion
// cannot happen in the template.
func (m *Model) expandImageMarkers(tokenIDs []int, parts []ImagePart) ([]int, []ImageInput, error) {
	if m.tokenizer == nil {
		return nil, nil, fmt.Errorf("llm: no tokenizer")
	}
	padID := m.tokenizer.IDOf("<|image_pad|>")
	if padID == 0 {
		return nil, nil, fmt.Errorf("llm: tokenizer has no <|image_pad|> token; model does not accept images")
	}
	mm, ok := m.arch.(MultimodalArchitecture)
	if !ok {
		return nil, nil, fmt.Errorf("llm: model %q does not support image input", m.cfg.Arch)
	}

	// Count the markers first: the number of images must match exactly, or
	// the position assignment would silently shift.
	markers := 0
	for _, id := range tokenIDs {
		if id == padID {
			markers++
		}
	}
	if markers != len(parts) {
		return nil, nil, fmt.Errorf("llm: prompt has %d image markers but %d images were supplied", markers, len(parts))
	}

	out := make([]int, 0, len(tokenIDs))
	spans := make([]ImageInput, 0, len(parts))
	next := 0
	for _, id := range tokenIDs {
		if id != padID {
			out = append(out, id)
			continue
		}
		part := parts[next]
		if part.Pixels == nil {
			return nil, nil, fmt.Errorf("llm: image %d has no pixels", next)
		}
		n := mm.ImageTokens(part.Pixels.GridTHW)
		if n <= 0 {
			return nil, nil, fmt.Errorf("llm: image %d expands to %d tokens", next, n)
		}
		spans = append(spans, ImageInput{Start: len(out), GridTHW: part.Pixels.GridTHW})
		for i := 0; i < n; i++ {
			out = append(out, padID)
		}
		next++
	}
	return out, spans, nil
}
