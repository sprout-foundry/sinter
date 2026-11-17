//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package qwen35

import (
	"fmt"
	"os"
	"strconv"
	"sync"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/llm/vision"
	"github.com/sprout-foundry/sinter/tensor"
)

// visionFeatureCacheSize is how many images' merged features stay resident.
// Each entry is [imageTokens, hidden] float32: 768 tokens of a 2B model is
// ~6 MB, so the default costs ~50 MB at worst and usually far less. The
// working set is the handful of images one conversation keeps referring to.
const visionFeatureCacheSize = 8

// imageFeatureCache retains the vision tower's merged output keyed on the
// prepared pixels. The tower is the dominant cost of an image request — about
// half a second for a 1024x768 image on a 2B model — and its output depends
// only on the pixels, so a repeat image (the same screenshot read on every
// turn) can skip it entirely.
//
// Entries are host-side float32 rather than backend arrays. A retained GPU
// array outliving the call that produced it is not safe on GGML: the
// backend's tensor→Array registry is keyed on the C tensor pointer, and
// freeing the original wrapper deletes that entry while the retained one
// still points at the tensor, so a later op reads a stale logical shape and
// ggml aborts on an element-count mismatch. Keeping the values on the host
// sidesteps the wrapper lifetime question; a hit costs one upload, which is
// negligible next to the encoder it replaces.
//
// Keyed on vision.Pixels.Hash, which covers the patch grid and every pixel
// value, so a hit is bit-identical to a fresh encode rather than an
// approximation.
type imageFeatureCache struct {
	mu      sync.Mutex
	entries []*imageFeatureEntry // most recently used last
	max     int
}

type imageFeatureEntry struct {
	key    [32]byte
	shape  []int
	values []float32
}

func newImageFeatureCache() *imageFeatureCache {
	max := visionFeatureCacheSize
	if v := os.Getenv("SINTER_VISION_CACHE_SIZE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			max = n
		}
	}
	return &imageFeatureCache{max: max}
}

// get returns a copy of the cached features for key, moving the entry to
// most-recently used. A copy keeps the caller from aliasing storage the cache
// may evict.
func (c *imageFeatureCache) get(key [32]byte) ([]float32, []int, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i, e := range c.entries {
		if e.key != key {
			continue
		}
		c.entries = append(c.entries[:i], c.entries[i+1:]...)
		c.entries = append(c.entries, e)
		return append([]float32(nil), e.values...), append([]int(nil), e.shape...), true
	}
	return nil, nil, false
}

// put stores values under key, evicting the least recently used entry when
// full. The cache copies values, so the caller keeps ownership of its slice.
func (c *imageFeatureCache) put(key [32]byte, values []float32, shape []int) {
	if c.max <= 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for len(c.entries) >= c.max {
		c.entries = c.entries[1:]
	}
	c.entries = append(c.entries, &imageFeatureEntry{
		key:    key,
		shape:  append([]int(nil), shape...),
		values: append([]float32(nil), values...),
	})
}

// free releases every retained entry.
func (c *imageFeatureCache) free() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = nil
}

// encodeImages runs the vision tower over each image and concatenates the
// merged features into one [1, totalTokens, hidden] tensor.
//
// An image whose features are already cached skips the tower. spans, when
// non-nil, is the prompt's image spans; when every image is cached the
// concatenated result is reused whole, which is the common case of one image
// per prompt.
func (q *Qwen35) encodeImages(pixels []*vision.Pixels, spans []llm.ImageInput) (tensor.Array, error) {
	s := q.stream
	b := q.backend
	hidden := q.cfg.HiddenSize

	if len(pixels) == 0 {
		return nil, fmt.Errorf("qwen35: no images to encode")
	}
	if q.featureCache == nil {
		q.featureCache = newImageFeatureCache()
	}

	// Whole-prompt hit: every image is cached, so the concatenation is
	// exactly what the prompt needs and the tower can be skipped for all of
	// them.
	if len(spans) == len(pixels) {
		if whole, ok := q.cachedPromptFeatures(pixels); ok {
			return whole, nil
		}
	}

	// pieces holds every array this call owns; the deferred loop is the only
	// place they are released. The returned array is a separate retained
	// reference, so callers own exactly one array regardless of path.
	pieces := make([]tensor.Array, 0, len(pixels)+1)
	defer func() {
		for _, p := range pieces {
			p.Free()
		}
	}()

	for i, px := range pixels {
		if px == nil {
			return nil, fmt.Errorf("qwen35: image %d has no pixels", i)
		}
		key := px.Hash()
		if values, shape, ok := q.featureCache.get(key); ok {
			a, err := b.NewArrayFromFloat32(values, shape)
			if err == nil {
				pieces = append(pieces, a)
				continue
			}
			// A failed upload is not worth failing the request over: fall
			// through and re-encode.
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
		// Read the values back for the cache before this call's reference
		// goes away; the array itself is not retained.
		values, readErr := reshaped.Float32Data()
		if readErr == nil {
			q.featureCache.put(key, values, []int{1, shape[0], shape[1]})
		}
		pieces = append(pieces, reshaped)
	}

	if len(pieces) == 1 {
		return b.RetainArray(pieces[0]), nil
	}
	out, err := b.ConcatenateAxis(pieces, 1, s)
	if err != nil {
		return nil, fmt.Errorf("qwen35: concatenate image features: %w", err)
	}
	pieces = append(pieces, out)
	return b.RetainArray(out), nil
}

// cachedPromptFeatures returns the concatenated features for a prompt whose
// images are all cached, so the result is the exact tensor a fresh encode
// would produce. The caller owns the returned array; the intermediates this
// builds are released here.
func (q *Qwen35) cachedPromptFeatures(pixels []*vision.Pixels) (tensor.Array, bool) {
	pieces := make([]tensor.Array, 0, len(pixels))
	defer func() {
		for _, p := range pieces {
			p.Free()
		}
	}()
	for _, px := range pixels {
		if px == nil {
			return nil, false
		}
		values, shape, ok := q.featureCache.get(px.Hash())
		if !ok {
			return nil, false
		}
		// A cached entry is only reusable for this image if its token count
		// matches the grid the prompt expects. The hash covers the grid, so
		// this holds by construction; the check keeps a corrupt entry from
		// silently shifting every later position.
		if len(shape) != 3 || shape[1] != q.ImageTokens(px.GridTHW) {
			return nil, false
		}
		a, err := q.backend.NewArrayFromFloat32(values, shape)
		if err != nil {
			return nil, false
		}
		pieces = append(pieces, a)
	}
	if len(pieces) == 1 {
		return q.backend.RetainArray(pieces[0]), true
	}
	out, err := q.backend.ConcatenateAxis(pieces, 1, q.stream)
	if err != nil {
		return nil, false
	}
	// Hand the caller its own reference; out is released by the defer.
	retained := q.backend.RetainArray(out)
	out.Free()
	return retained, true
}
