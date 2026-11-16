package llm

import (
	"testing"

	"github.com/sprout-foundry/sinter/llm/vision"
)

// TestImageKeyDistinguishesImages pins the cache key's job: two different
// images must not produce the same key, because they expand to the same
// <|image_pad|> run and a token-only match would reuse K/V computed from
// the wrong picture.
func TestImageKeyDistinguishesImages(t *testing.T) {
	mk := func(vals ...float32) []ImagePart {
		return []ImagePart{{
			Pixels: &vision.Pixels{Data: vals, GridTHW: [3]int{1, 4, 4}},
		}}
	}

	a := imageKeyFor(mk(1, 2, 3, 4))
	b := imageKeyFor(mk(1, 2, 3, 4))
	c := imageKeyFor(mk(1, 2, 3, 5))

	if a != b {
		t.Error("identical images must produce identical keys")
	}
	if a == c {
		t.Error("different images must produce different keys")
	}
	if a == "" || c == "" {
		t.Error("an image prompt must not produce the empty (text-only) key")
	}
	if got := imageKeyFor(nil); got != "" {
		t.Errorf("text-only key = %q, want empty", got)
	}
}

// TestImageKeyCoversGridAndOrder checks the two non-pixel inputs that change
// the tower's output: the patch grid, and which image comes first.
func TestImageKeyCoversGridAndOrder(t *testing.T) {
	same := []float32{1, 2, 3, 4}

	gridA := imageKeyFor([]ImagePart{{Pixels: &vision.Pixels{Data: same, GridTHW: [3]int{1, 4, 4}}}})
	gridB := imageKeyFor([]ImagePart{{Pixels: &vision.Pixels{Data: same, GridTHW: [3]int{1, 2, 8}}}})
	if gridA == gridB {
		t.Error("the same pixels on different grids must not share a key")
	}

	first := &vision.Pixels{Data: []float32{1, 1}, GridTHW: [3]int{1, 2, 2}}
	second := &vision.Pixels{Data: []float32{2, 2}, GridTHW: [3]int{1, 2, 2}}
	ab := imageKeyFor([]ImagePart{{Pixels: first}, {Pixels: second}})
	ba := imageKeyFor([]ImagePart{{Pixels: second}, {Pixels: first}})
	if ab == ba {
		t.Error("image order must be part of the key")
	}
}

// TestBestPrefixSlotRespectsImageKey checks that matching is gated on the
// key even when the token ids line up exactly — the case that would
// otherwise silently reuse another picture's cache.
func TestBestPrefixSlotRespectsImageKey(t *testing.T) {
	m := &Model{}
	tokens := []int{1, 2, 3, 4, 5}
	m.prefixSlots = append(m.prefixSlots, &prefixSlot{tokens: tokens, imageKey: "picture-a"})

	shared, idx := m.bestPrefixSlot(tokens, "picture-a")
	if idx != 0 || shared != len(tokens) {
		t.Fatalf("same key: got shared=%d idx=%d, want full match", shared, idx)
	}

	shared, idx = m.bestPrefixSlot(tokens, "picture-b")
	if idx != -1 {
		t.Errorf("different key matched slot %d (shared=%d); it must not match", idx, shared)
	}
}
