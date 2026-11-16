//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package llm

import "fmt"

// ImagePlaceholder describes one image's tokens inside a prompt: the token
// span they occupy and the patch grid they came from. The span's length must
// equal gridH*merge * gridW*merge, i.e. the number of tokens the vision
// tower produces for that image.
type ImagePlaceholder struct {
	Start   int // first token index of the image span
	GridTHW [3]int
}

// MRoPEPositions computes the (t, h, w) position of every token in a prompt
// containing images, following the reference's get_rope_index:
//
//   - Text runs advance a single counter used for all three coordinates.
//   - An image's tokens get (t=base, h=base+row, w=base+col) over its merged
//     grid, and the counter then advances by max(rows, cols) — the image
//     occupies a square as far as the text stream is concerned.
//
// The returned slices have one entry per token. totalAdvance is the value
// the counter reached, which callers use to shift positions for text that
// follows the prompt (decode steps continue from there, not from the token
// count).
//
// images must be sorted by Start and must not overlap.
func MRoPEPositions(numTokens int, images []ImagePlaceholder, mergeSize int) (t, h, w []int, totalAdvance int, err error) {
	if mergeSize <= 0 {
		return nil, nil, nil, 0, fmt.Errorf("mrope: merge size must be positive, got %d", mergeSize)
	}
	t = make([]int, numTokens)
	h = make([]int, numTokens)
	w = make([]int, numTokens)

	pos := 0
	next := 0
	for i := 0; i < numTokens; {
		if next < len(images) && images[next].Start == i {
			img := images[next]
			gridT, gridH, gridW := img.GridTHW[0], img.GridTHW[1], img.GridTHW[2]
			rows := gridH / mergeSize
			cols := gridW / mergeSize
			n := rows * cols * gridT
			if i+n > numTokens {
				return nil, nil, nil, 0, fmt.Errorf("mrope: image at %d spans %d tokens, past the prompt end (%d)", i, n, numTokens)
			}
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					for tt := 0; tt < gridT; tt++ {
						idx := i + (r*cols+c)*gridT + tt
						t[idx] = pos + tt
						h[idx] = pos + r
						w[idx] = pos + c
					}
				}
			}
			advance := rows
			if cols > advance {
				advance = cols
			}
			pos += advance
			i += n
			next++
			continue
		}

		// Text run up to the next image (or the end).
		end := numTokens
		if next < len(images) {
			end = images[next].Start
		}
		for ; i < end; i++ {
			t[i], h[i], w[i] = pos, pos, pos
			pos++
		}
	}
	return t, h, w, pos, nil
}

// MRoPEDelta is the shift between the last 3-D position and the token count.
// Text generated after a prompt containing images continues from
// lastPosition + 1, not from numTokens, so decode steps offset by this
// delta. Mirrors the reference's mrope_position_deltas.
func MRoPEDelta(lastPosition, numTokens int) int {
	return lastPosition - numTokens
}
