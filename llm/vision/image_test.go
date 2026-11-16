package vision

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"
)

// solidPNG encodes a w x h PNG filled with one RGB colour.
func solidPNG(t *testing.T, w, h int, c color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

func TestDecodePNG(t *testing.T) {
	data := solidPNG(t, 8, 4, color.RGBA{R: 10, G: 20, B: 30, A: 255})
	img, err := Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if img.Width != 8 || img.Height != 4 {
		t.Fatalf("size = %dx%d, want 8x4", img.Width, img.Height)
	}
	if len(img.Pixels) != 8*4*3 {
		t.Fatalf("pixel len = %d, want %d", len(img.Pixels), 8*4*3)
	}
	want := []uint8{10, 20, 30}
	for i := 0; i < len(img.Pixels); i += 3 {
		if img.Pixels[i] != want[0] || img.Pixels[i+1] != want[1] || img.Pixels[i+2] != want[2] {
			t.Fatalf("pixel %d = %v, want %v", i/3, img.Pixels[i:i+3], want)
		}
	}
}

func TestDecodeRejectsGarbage(t *testing.T) {
	if _, err := Decode(bytes.NewReader([]byte("not an image"))); err == nil {
		t.Fatal("expected an error for non-image input")
	}
}

// TestSmartResizeMatchesReference pins the values the reference's
// smart_resize produces for the Qwen3.5 geometry (factor 32). The expected
// values were generated with the reference implementation itself, not by
// hand — the pixel budget (not the rounding) dominates most cases.
func TestSmartResizeMatchesReference(t *testing.T) {
	const factor = 32
	cases := []struct {
		name           string
		h, w           int
		minPix, maxPix int
		wantH, wantW   int
	}{
		// 64x96 = 6144 pixels, below the 65536 floor, so it scales up.
		{"below floor, wide", 64, 96, 65536, 16777216, 224, 320},
		// 100x100 is below the floor too: ceil(100*sqrt(65536/10000)/32)*32 = 256.
		{"below floor, square", 100, 100, 65536, 16777216, 256, 256},
		{"below floor, near", 110, 110, 65536, 16777216, 256, 256},
		// 4096x4096 = 16.7M pixels; with a 1M ceiling, floor(4096/4/32)*32 = 992.
		{"above ceiling", 4096, 4096, 65536, 1000000, 992, 992},
		// Inside the budget and already aligned: untouched.
		{"aligned, in budget", 256, 512, 65536, 16777216, 256, 512},
		// Inside the budget, rounds to the nearest multiple.
		{"rounds to nearest", 300, 460, 65536, 16777216, 288, 448},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h, w, err := SmartResize(tc.h, tc.w, factor, tc.minPix, tc.maxPix)
			if err != nil {
				t.Fatalf("SmartResize: %v", err)
			}
			if h != tc.wantH || w != tc.wantW {
				t.Fatalf("SmartResize(%d,%d) = %dx%d, want %dx%d", tc.h, tc.w, h, w, tc.wantH, tc.wantW)
			}
			if h%factor != 0 || w%factor != 0 {
				t.Fatalf("result %dx%d not a multiple of %d", h, w, factor)
			}
		})
	}
}

func TestSmartResizeRejectsExtremeAspect(t *testing.T) {
	if _, _, err := SmartResize(10, 3000, 32, 1, 1<<30); err == nil {
		t.Fatal("expected an error for a 300:1 aspect ratio")
	}
}

// TestPatchifyLayout checks the merge-grouped patch order on a small image
// where every pixel is distinct. A 2-patch-wide, 2-patch-high image (32x32)
// is exactly one merge group of four patch rows.
func TestPatchifyLayout(t *testing.T) {
	cfg := DefaultConfig()
	const w, h = 32, 32
	img := &Image{Width: w, Height: h, Pixels: make([]uint8, w*h*3)}
	// Encode (x, y) into the pixel so we can identify each source pixel.
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := (y*w + x) * 3
			img.Pixels[i+0] = uint8(x)
			img.Pixels[i+1] = uint8(y)
			img.Pixels[i+2] = 0
		}
	}

	p := Patchify(img, cfg)
	if p.GridTHW != [3]int{1, 2, 2} {
		t.Fatalf("grid = %v, want [1 2 2]", p.GridTHW)
	}
	if p.NumPatches() != 4 {
		t.Fatalf("NumPatches = %d, want 4", p.NumPatches())
	}
	if p.NumTokens() != 1 {
		t.Fatalf("NumTokens = %d, want 1", p.NumTokens())
	}

	rowLen := cfg.TemporalPatchSize * cfg.PatchSize * cfg.PatchSize * 3
	if len(p.Data) != 4*rowLen {
		t.Fatalf("data len = %d, want %d", len(p.Data), 4*rowLen)
	}

	// Row order is (my, mx, dy, dx); with one merge group that is
	// dy,dx = (0,0), (0,1), (1,0), (1,1) — i.e. patch (0,0), (0,1), (1,0), (1,1).
	// Within a row the layout is [channel][temporal][patch-row][patch-col],
	// so the first float of each channel's temporal-0 block is that patch's
	// top-left pixel.
	wantFirst := []struct{ x, y int }{{0, 0}, {16, 0}, {0, 16}, {16, 16}}
	const chanStride = 2 * 16 * 16 // temporal * patch area
	for row, want := range wantFirst {
		base := row * rowLen
		// channel 0 holds x, channel 1 holds y in this synthetic image.
		gotX := p.Data[base+0*chanStride]
		gotY := p.Data[base+1*chanStride]
		wantX := float32(want.x)/255*2 - 1
		wantY := float32(want.y)/255*2 - 1
		if math.Abs(float64(gotX-wantX)) > 1e-6 || math.Abs(float64(gotY-wantY)) > 1e-6 {
			t.Fatalf("row %d first pixel = (%v,%v), want (%v,%v)", row, gotX, gotY, wantX, wantY)
		}
	}

	// The temporal slots of a single-frame image must be identical.
	frameLen := chanStride
	for c := 0; c < 3; c++ {
		for i := 0; i < frameLen/2; i++ {
			a := p.Data[c*chanStride+i]
			b := p.Data[c*chanStride+frameLen/2+i]
			if a != b {
				t.Fatalf("channel %d: temporal slots differ at %d: %v != %v", c, i, a, b)
			}
		}
	}
}

// TestPatchifyNormalization checks the (x/255 - mean)/std transform.
func TestPatchifyNormalization(t *testing.T) {
	cfg := DefaultConfig()
	// 32x32 is the smallest image that forms a full merge group.
	img := &Image{Width: 32, Height: 32, Pixels: make([]uint8, 32*32*3)}
	for i := range img.Pixels {
		img.Pixels[i] = 255
	}
	p := Patchify(img, cfg)
	if len(p.Data) == 0 {
		t.Fatal("no patch data")
	}
	for i, v := range p.Data {
		if math.Abs(float64(v)-1.0) > 1e-6 {
			t.Fatalf("all-white pixel %d = %v, want 1.0", i, v)
		}
	}

	for i := range img.Pixels {
		img.Pixels[i] = 0
	}
	p = Patchify(img, cfg)
	for i, v := range p.Data {
		if math.Abs(float64(v)+1.0) > 1e-6 {
			t.Fatalf("all-black pixel %d = %v, want -1.0", i, v)
		}
	}
}

// TestPatchifyTooSmall guards the silent-zeros case: an image smaller than
// one merge group must produce no data rather than a plausible-looking
// all-black tensor.
func TestPatchifyTooSmall(t *testing.T) {
	cfg := DefaultConfig()
	img := &Image{Width: 16, Height: 16, Pixels: make([]uint8, 16*16*3)}
	for i := range img.Pixels {
		img.Pixels[i] = 255
	}
	if got := Patchify(img, cfg); len(got.Data) != 0 {
		t.Fatalf("expected no patch data for a sub-merge-group image, got %d floats", len(got.Data))
	}
}

func TestPreprocessGrid(t *testing.T) {
	cfg := DefaultConfig()
	img := &Image{Width: 100, Height: 100, Pixels: make([]uint8, 100*100*3)}
	p, err := Preprocess(img, cfg)
	if err != nil {
		t.Fatalf("Preprocess: %v", err)
	}
	// 100x100 is below the 65536-pixel floor, so smart_resize scales it to
	// 256x256; at patch 16 that is a 16x16 grid.
	if p.GridTHW != [3]int{1, 16, 16} {
		t.Fatalf("grid = %v, want [1 16 16]", p.GridTHW)
	}
	if p.ResizedWidth != 256 || p.ResizedHeight != 256 {
		t.Fatalf("resized = %dx%d, want 256x256", p.ResizedWidth, p.ResizedHeight)
	}
	if p.NumTokens() != 64 {
		t.Fatalf("NumTokens = %d, want 64", p.NumTokens())
	}
}

// TestHashDistinguishesImages is the property the prompt cache depends on:
// two different images must not hash alike, and the same image must hash
// identically across calls.
func TestHashDistinguishesImages(t *testing.T) {
	cfg := DefaultConfig()
	a := &Image{Width: 32, Height: 32, Pixels: make([]uint8, 32*32*3)}
	b := &Image{Width: 32, Height: 32, Pixels: make([]uint8, 32*32*3)}
	for i := range a.Pixels {
		a.Pixels[i] = 0
		b.Pixels[i] = 255
	}

	pa := Patchify(a, cfg)
	pb := Patchify(b, cfg)
	ha, ha2, hb := pa.Hash(), Patchify(a, cfg).Hash(), pb.Hash()

	if ha != ha2 {
		t.Fatal("same image produced different hashes")
	}
	if ha == hb {
		t.Fatal("different images produced the same hash")
	}
}

func TestResizePreservesSizeWhenUnchanged(t *testing.T) {
	img := &Image{Width: 4, Height: 4, Pixels: make([]uint8, 4*4*3)}
	for i := range img.Pixels {
		img.Pixels[i] = uint8(i)
	}
	out := Resize(img, 4, 4)
	if out.Width != 4 || out.Height != 4 {
		t.Fatalf("size = %dx%d", out.Width, out.Height)
	}
	for i := range img.Pixels {
		if out.Pixels[i] != img.Pixels[i] {
			t.Fatalf("same-size resize changed pixel %d: %d != %d", i, out.Pixels[i], img.Pixels[i])
		}
	}
}
