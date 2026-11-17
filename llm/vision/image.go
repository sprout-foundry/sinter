// Package vision holds the image-side pieces of sinter's multimodal path:
// decoding, the reference preprocessing (smart resize, normalize, patchify),
// and the shared types that cross into the llm package.
//
// Nothing here touches a GPU backend, so it builds and tests on every
// platform.
package vision

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"

	xdraw "golang.org/x/image/draw"
)

// Config mirrors the model's preprocessor_config.json. The zero value is not
// usable; use DefaultConfig.
type Config struct {
	PatchSize         int // 16
	TemporalPatchSize int // 2
	MergeSize         int // 2 — the 2x2 spatial merge in the vision tower
	ShortestEdge      int // min pixel budget (shortest_edge)
	LongestEdge       int // max pixel budget (longest_edge)
	Mean              [3]float32
	Std               [3]float32

	// MaxTokens caps how many <|image_pad|> tokens one image may expand to.
	// Zero means no cap beyond LongestEdge.
	//
	// The encoder and the text prefill both scale with this, so it is the
	// main lever on image-request latency: a 1024x768 image is 768 tokens
	// uncapped and ~0.5s in the tower alone. Lowering it resizes the image
	// down before patchifying, which is what the reference processor does
	// when it is given a smaller budget — the model has no notion of a
	// "capped" image, only of a smaller one.
	MaxTokens int
}

// DefaultConfig returns the Qwen3.5 VL preprocessing parameters
// (Qwen2VLImageProcessorFast with Qwen3.5's patch geometry).
func DefaultConfig() Config {
	return Config{
		PatchSize:         16,
		TemporalPatchSize: 2,
		MergeSize:         2,
		ShortestEdge:      65536,
		LongestEdge:       16777216,
		Mean:              [3]float32{0.5, 0.5, 0.5},
		Std:               [3]float32{0.5, 0.5, 0.5},
	}
}

// WithMaxTokens returns a copy of the config that caps one image at n
// <|image_pad|> tokens (0 removes the cap). It lowers LongestEdge, which is
// the budget smart_resize already enforces, so no new resize path is needed.
func (c Config) WithMaxTokens(n int) Config {
	c.MaxTokens = n
	if n <= 0 {
		return c
	}
	// One token is one merge group of patch*merge pixels on a side.
	side := c.PatchSize * c.MergeSize
	c.LongestEdge = n * side * side
	return c
}

// Factor is the divisor the reference rounds both dimensions to: one patch
// times one merge step.
func (c Config) Factor() int { return c.PatchSize * c.MergeSize }

// Image is a decoded RGB image. Pixels are 8-bit RGB, row-major, 3 bytes per
// pixel — the reference resizes in uint8 before converting to float.
type Image struct {
	Width, Height int
	Pixels        []uint8 // len = Width*Height*3
}

// Decode reads a PNG or JPEG image. The result is always RGB (alpha is
// dropped, matching do_convert_rgb=true).
func Decode(r io.Reader) (*Image, error) {
	src, format, err := image.Decode(r)
	if err != nil {
		return nil, fmt.Errorf("vision: decode image: %w", err)
	}
	switch format {
	case "png", "jpeg":
	default:
		return nil, fmt.Errorf("vision: unsupported image format %q (png and jpeg only)", format)
	}
	return FromImage(src), nil
}

// FromImage converts any image.Image to 8-bit RGB.
func FromImage(src image.Image) *Image {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	out := &Image{Width: w, Height: h, Pixels: make([]uint8, w*h*3)}
	for y := 0; y < h; y++ {
		row := out.Pixels[y*w*3 : (y+1)*w*3]
		for x := 0; x < w; x++ {
			// RGBA() returns 16-bit alpha-premultiplied values; shifting by 8
			// is what image/color's own conversions do. For opaque images
			// this is exact.
			r, g, bl, _ := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
			row[x*3+0] = uint8(r >> 8)
			row[x*3+1] = uint8(g >> 8)
			row[x*3+2] = uint8(bl >> 8)
		}
	}
	return out
}

// SmartResize applies the reference's smart_resize: both dimensions become
// multiples of factor, the aspect ratio is preserved as closely as the
// rounding allows, and the pixel count stays within [minPixels, maxPixels].
func SmartResize(height, width, factor, minPixels, maxPixels int) (int, int, error) {
	if height <= 0 || width <= 0 {
		return 0, 0, fmt.Errorf("vision: invalid image size %dx%d", width, height)
	}
	long, short := max(height, width), min(height, width)
	if float64(long)/float64(short) > 200 {
		return 0, 0, fmt.Errorf("vision: aspect ratio %d:%d exceeds the reference limit of 200", long, short)
	}

	hBar := int(math.Round(float64(height)/float64(factor))) * factor
	wBar := int(math.Round(float64(width)/float64(factor))) * factor
	if hBar*wBar > maxPixels {
		beta := math.Sqrt(float64(height*width) / float64(maxPixels))
		hBar = max(factor, int(math.Floor(float64(height)/beta/float64(factor)))*factor)
		wBar = max(factor, int(math.Floor(float64(width)/beta/float64(factor)))*factor)
	} else if hBar*wBar < minPixels {
		beta := math.Sqrt(float64(minPixels) / float64(height*width))
		hBar = int(math.Ceil(float64(height)*beta/float64(factor))) * factor
		wBar = int(math.Ceil(float64(width)*beta/float64(factor))) * factor
	}
	return hBar, wBar, nil
}

// Resize returns a copy of img scaled to exactly width x height using
// bicubic interpolation. The reference uses torchvision's antialiased
// bicubic; x/image's CatmullRom is the same 4-lobe cubic kernel, so values
// agree closely but not bit-exactly (see the parity notes in
// vision_parity_test.go).
func Resize(img *Image, width, height int) *Image {
	if width == img.Width && height == img.Height {
		cp := &Image{Width: width, Height: height, Pixels: make([]uint8, len(img.Pixels))}
		copy(cp.Pixels, img.Pixels)
		return cp
	}
	src := &image.RGBA{
		Pix:    make([]uint8, img.Width*img.Height*4),
		Stride: img.Width * 4,
		Rect:   image.Rect(0, 0, img.Width, img.Height),
	}
	for i := 0; i < img.Width*img.Height; i++ {
		src.Pix[i*4+0] = img.Pixels[i*3+0]
		src.Pix[i*4+1] = img.Pixels[i*3+1]
		src.Pix[i*4+2] = img.Pixels[i*3+2]
		src.Pix[i*4+3] = 0xff
	}
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	xdraw.CatmullRom.Scale(dst, dst.Bounds(), src, src.Bounds(), xdraw.Src, nil)

	out := &Image{Width: width, Height: height, Pixels: make([]uint8, width*height*3)}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			s := y*dst.Stride + x*4
			d := (y*width + x) * 3
			out.Pixels[d+0] = dst.Pix[s+0]
			out.Pixels[d+1] = dst.Pix[s+1]
			out.Pixels[d+2] = dst.Pix[s+2]
		}
	}
	return out
}

// Pixels is one image prepared for the vision tower.
type Pixels struct {
	// Data is the normalized pixel tensor in the reference's patch layout:
	// numPatches rows of TemporalPatchSize*PatchSize*PatchSize*3 floats, in
	// merge-grouped patch order (see Patchify).
	Data []float32
	// GridTHW is (temporal, height, width) in patches. Temporal is always 1;
	// a single image is duplicated across the temporal patch, which is a
	// no-op for the values but required for the shape.
	GridTHW [3]int
	// ResizedHeight and ResizedWidth are the post-smart-resize dimensions.
	ResizedHeight, ResizedWidth int
}

// NumPatches is the number of patch rows in Data, i.e. the vision tower's
// sequence length for this image.
func (p *Pixels) NumPatches() int { return p.GridTHW[1] * p.GridTHW[2] }

// NumTokens is the number of <|image_pad|> tokens this image expands to —
// one per 2x2 merged patch group.
func (p *Pixels) NumTokens() int {
	m := p.GridTHW[1] * p.GridTHW[2]
	return m / 4
}

// Preprocess runs the full reference pipeline on img: smart resize, then
// rescale and normalize, then patchify.
func Preprocess(img *Image, cfg Config) (*Pixels, error) {
	h, w, err := SmartResize(img.Height, img.Width, cfg.Factor(), cfg.ShortestEdge, cfg.LongestEdge)
	if err != nil {
		return nil, err
	}
	resized := Resize(img, w, h)
	return Patchify(resized, cfg), nil
}

// Patchify converts a resized RGB image into the vision tower's input
// layout without resizing it further. The image must be at least one merge
// group (patch*merge) on each side; callers that accept arbitrary sizes go
// through Preprocess, which resizes first.
//
// The reference reshapes [C, H, W] to
// [C, H/merge, merge, patch, W/merge, merge, patch], permutes to
// [H/merge, W/merge, merge, merge, C, patch, patch], and flattens each
// (merge, merge) group into one row — so patch rows are grouped in 2x2
// blocks, and within a row the order is merge-row, merge-col, channel,
// patch-row, patch-col.
func Patchify(img *Image, cfg Config) *Pixels {
	p := cfg.PatchSize
	m := cfg.MergeSize
	gridH := img.Height / p
	gridW := img.Width / p
	gridHM := gridH / m
	gridWM := gridW / m

	rowLen := cfg.TemporalPatchSize * p * p * 3
	numPatches := gridH * gridW
	data := make([]float32, numPatches*rowLen)

	if gridHM == 0 || gridWM == 0 {
		// Too small to form a merge group. Returning zeros here would look
		// like a legitimate all-black image to the tower, so leave the data
		// empty and let the caller's size check catch it.
		return &Pixels{
			GridTHW:       [3]int{1, gridH, gridW},
			ResizedHeight: img.Height,
			ResizedWidth:  img.Width,
		}
	}

	// Row index for a patch at merge-group (my, mx) and position (dy, dx)
	// inside it: groups are laid out row-major, then the 2x2 block.
	rowOf := func(my, mx, dy, dx int) int {
		return ((my*gridWM+mx)*m+dy)*m + dx
	}

	// Row layout, from the reference's reshape/permute:
	//   [C, temporal, p, p]  flattened per patch row.
	// So both the channel and the temporal slot are outer axes relative to
	// the patch grid: all of channel 0's first frame, then its second frame,
	// then channel 1's, and so on. A single image repeats the same frame
	// across the temporal slots.
	for my := 0; my < gridHM; my++ {
		for mx := 0; mx < gridWM; mx++ {
			for dy := 0; dy < m; dy++ {
				for dx := 0; dx < m; dx++ {
					off := rowOf(my, mx, dy, dx) * rowLen
					for c := 0; c < 3; c++ {
						for t := 0; t < cfg.TemporalPatchSize; t++ {
							for py := 0; py < p; py++ {
								pyAbs := (my*m+dy)*p + py
								for px := 0; px < p; px++ {
									pxAbs := (mx*m+dx)*p + px
									src := (pyAbs*img.Width + pxAbs) * 3
									v := float32(img.Pixels[src+c]) / 255
									v = (v - cfg.Mean[c]) / cfg.Std[c]
									data[off] = v
									off++
								}
							}
						}
					}
				}
			}
		}
	}

	return &Pixels{
		Data:          data,
		GridTHW:       [3]int{1, gridH, gridW},
		ResizedHeight: img.Height,
		ResizedWidth:  img.Width,
	}
}

// Hash identifies the prepared pixels. Two turns that reference the same
// image produce the same hash, so the prompt cache can tell them apart from
// a different image that yields identical token ids.
func (p *Pixels) Hash() [32]byte {
	h := sha256.New()
	var hdr [12]byte
	for i, v := range p.GridTHW {
		hdr[i*4+0] = byte(v)
		hdr[i*4+1] = byte(v >> 8)
		hdr[i*4+2] = byte(v >> 16)
		hdr[i*4+3] = byte(v >> 24)
	}
	h.Write(hdr[:])
	buf := make([]byte, 0, 4096)
	for _, f := range p.Data {
		var b [4]byte
		bits := math.Float32bits(f)
		b[0] = byte(bits)
		b[1] = byte(bits >> 8)
		b[2] = byte(bits >> 16)
		b[3] = byte(bits >> 24)
		buf = append(buf, b[:]...)
		if len(buf) >= 4096 {
			h.Write(buf)
			buf = buf[:0]
		}
	}
	h.Write(buf)
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

// ErrNoImage reports an empty or unreadable image payload.
var ErrNoImage = errors.New("vision: no image data")
