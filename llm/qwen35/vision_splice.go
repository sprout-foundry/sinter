//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package qwen35

import (
	"fmt"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/llm/vision"
	"github.com/sprout-foundry/sinter/tensor"
)

// spliceImageFeatures returns h with each image span's rows replaced by the
// vision tower's output for that image.
//
// h is [1, seqLen, hidden]. The features are built as a single
// [1, totalImageTokens, hidden] tensor in prompt order, so the splice walks
// the spans in order, copying the rows before each span from h and the span
// itself from the features.
func (q *Qwen35) spliceImageFeatures(h tensor.Array, seqLen int) (tensor.Array, error) {
	s := q.stream
	b := q.backend
	hidden := q.cfg.HiddenSize

	if len(q.imgSpans) == 0 {
		return nil, fmt.Errorf("qwen35: image features present but no spans")
	}

	// Verify the spans tile the features and fit the prompt before copying
	// anything — a mismatch here would silently shift every later position.
	featureRows := 0
	for i, span := range q.imgSpans {
		want := q.ImageTokens(span.GridTHW)
		if want <= 0 {
			return nil, fmt.Errorf("qwen35: image %d has no tokens", i)
		}
		if span.Start < 0 || span.Start+want > seqLen {
			return nil, fmt.Errorf("qwen35: image %d span [%d,%d) outside prompt of %d tokens", i, span.Start, span.Start+want, seqLen)
		}
		featureRows += want
	}
	if shape := q.imgFeatures.Shape(); len(shape) != 3 || shape[1] != featureRows || shape[2] != hidden {
		return nil, fmt.Errorf("qwen35: image features shape %v does not match %d rows of %d", shape, featureRows, hidden)
	}

	// Walk the prompt, alternating text runs (taken from h) and image spans
	// (taken from the features).
	pieces := make([]tensor.Array, 0, 2*len(q.imgSpans)+1)
	freed := make([]tensor.Array, 0, len(pieces))
	defer func() {
		for _, p := range freed {
			p.Free()
		}
	}()

	cursor := 0
	featCursor := 0
	for _, span := range q.imgSpans {
		if span.Start > cursor {
			text, err := b.Slice(h, []int{0, cursor, 0}, []int{1, span.Start, hidden}, []int{1, 1, 1}, s)
			if err != nil {
				return nil, fmt.Errorf("qwen35: slice text before image: %w", err)
			}
			pieces = append(pieces, text)
			freed = append(freed, text)
		}
		n := q.ImageTokens(span.GridTHW)
		img, err := b.Slice(q.imgFeatures, []int{0, featCursor, 0}, []int{1, featCursor + n, hidden}, []int{1, 1, 1}, s)
		if err != nil {
			return nil, fmt.Errorf("qwen35: slice image features: %w", err)
		}
		pieces = append(pieces, img)
		freed = append(freed, img)
		featCursor += n
		cursor = span.Start + n
	}
	if cursor < seqLen {
		text, err := b.Slice(h, []int{0, cursor, 0}, []int{1, seqLen, hidden}, []int{1, 1, 1}, s)
		if err != nil {
			return nil, fmt.Errorf("qwen35: slice trailing text: %w", err)
		}
		pieces = append(pieces, text)
		freed = append(freed, text)
	}

	out, err := b.ConcatenateAxis(pieces, 1, s)
	if err != nil {
		return nil, fmt.Errorf("qwen35: concatenate image features: %w", err)
	}
	return out, nil
}

// ImageTokens reports how many <|image_pad|> tokens one image expands to.
func (q *Qwen35) ImageTokens(gridTHW [3]int) int {
	merge := q.cfg.VisionSpatialMergeSize()
	rows := gridTHW[1] / merge
	cols := gridTHW[2] / merge
	t := gridTHW[0]
	if t == 0 {
		t = 1
	}
	return rows * cols * t
}

// encodeImages runs the vision tower over each image and concatenates the
// merged features into one [1, totalTokens, hidden] tensor.
func (q *Qwen35) encodeImages(pixels []*vision.Pixels, spans []llm.ImageInput) (tensor.Array, error) {
	s := q.stream
	b := q.backend
	hidden := q.cfg.HiddenSize

	var pieces []tensor.Array
	defer func() {
		for _, p := range pieces {
			p.Free()
		}
	}()

	for i, px := range pixels {
		if px == nil {
			return nil, fmt.Errorf("qwen35: image %d has no pixels", i)
		}
		rows := px.NumPatches()
		cols := len(px.Data) / rows
		patches, err := b.NewArrayFromFloat32(px.Data, []int{rows, cols})
		if err != nil {
			return nil, fmt.Errorf("qwen35: image %d patch array: %w", i, err)
		}
		merged, err := q.vision.forward(patches, visionImage{GridTHW: px.GridTHW})
		patches.Free()
		if err != nil {
			return nil, fmt.Errorf("qwen35: image %d tower: %w", i, err)
		}
		// The tower returns [tokens, outHidden]; the text model wants
		// [1, tokens, hidden]. outHidden must equal hidden for the splice.
		shape := merged.Shape()
		if len(shape) != 2 || shape[1] != hidden {
			merged.Free()
			return nil, fmt.Errorf("qwen35: image %d tower output %v, want [tokens,%d]", i, shape, hidden)
		}
		reshaped, err := b.Reshape(merged, []int{1, shape[0], shape[1]}, s)
		merged.Free()
		if err != nil {
			return nil, fmt.Errorf("qwen35: image %d reshape: %w", i, err)
		}
		pieces = append(pieces, reshaped)
	}

	out, err := b.ConcatenateAxis(pieces, 1, s)
	if err != nil {
		return nil, fmt.Errorf("qwen35: concatenate image features: %w", err)
	}
	return out, nil
}
