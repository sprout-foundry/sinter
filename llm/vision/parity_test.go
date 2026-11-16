package vision

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"
)

// loadNpyFloat32 reads a C-order little-endian float32 .npy file (the format
// numpy.save writes) and returns its flat data and shape. Only the subset the
// test fixtures need is supported.
func loadNpyFloat32(t *testing.T, path string) ([]float32, []int) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(raw) < 10 || !bytes.HasPrefix(raw, []byte("\x93NUMPY")) {
		t.Fatalf("%s: not a .npy file", path)
	}
	major := raw[6]
	var headerLen int
	var headerStart int
	if major == 1 {
		headerLen = int(binary.LittleEndian.Uint16(raw[8:10]))
		headerStart = 10
	} else {
		headerLen = int(binary.LittleEndian.Uint32(raw[8:12]))
		headerStart = 12
	}
	header := string(raw[headerStart : headerStart+headerLen])
	dataStart := headerStart + headerLen

	// The header is a Python dict literal: {'descr': '<f4', 'fortran_order': False, 'shape': (1200, 1536), }
	// Take the value after "'descr':" up to the next quote.
	descrStart := bytes.Index([]byte(header), []byte("'descr':"))
	if descrStart < 0 {
		t.Fatalf("%s: no descr in header %q", path, header)
	}
	rest := header[descrStart+len("'descr':"):]
	q1 := bytes.IndexByte([]byte(rest), '\'')
	if q1 < 0 {
		t.Fatalf("%s: unterminated descr in header %q", path, header)
	}
	q2 := bytes.IndexByte([]byte(rest[q1+1:]), '\'')
	if q2 < 0 {
		t.Fatalf("%s: unterminated descr in header %q", path, header)
	}
	descr := rest[q1+1 : q1+1+q2]
	if descr != "<f4" {
		t.Fatalf("%s: unsupported dtype %q, want <f4", path, descr)
	}

	var shape []int

	// Shape lives between '(' and ')'.
	open := bytes.IndexByte([]byte(header), '(')
	close := bytes.IndexByte([]byte(header), ')')
	if open < 0 || close < 0 || close < open {
		t.Fatalf("%s: no shape in header %q", path, header)
	}
	for _, f := range bytes.Split([]byte(header[open+1:close]), []byte(",")) {
		f = bytes.TrimSpace(f)
		if len(f) == 0 {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(string(f), "%d", &n); err != nil {
			t.Fatalf("%s: parse shape field %q: %v", path, f, err)
		}
		shape = append(shape, n)
	}

	total := 1
	for _, d := range shape {
		total *= d
	}
	payload := raw[dataStart:]
	if len(payload) != total*4 {
		t.Fatalf("%s: payload %d bytes, want %d", path, len(payload), total*4)
	}
	out := make([]float32, total)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
	}
	return out, shape
}

func trimQuotes(s string) string {
	for len(s) > 0 && (s[len(s)-1] == '\'' || s[len(s)-1] == '"') {
		s = s[:len(s)-1]
	}
	return s
}

// TestPreprocessMatchesReference compares the full preprocessing pipeline
// against a fixture produced by the model's own image processor
// (transformers' Qwen2VLImageProcessorPil, the class the checkpoint's
// processor_config.json names). The fixture is a 96x96 image, which
// smart_resize upscales to 256x256, so this covers resize, normalization,
// patch ordering, and the merge-group layout together.
//
// The 96x96 case resizes, so the reference's antialiased bicubic and
// x/image's CatmullRom differ slightly. The tolerance below is the measured
// drift (max 0.0079 on a [-1,1] range). A no-resize image matches bit-exactly;
// see TestPreprocessNoResizeIsExact.
func TestPreprocessMatchesReference(t *testing.T) {
	want, shape := loadNpyFloat32(t, "testdata/ref_96.npy")
	if len(shape) != 2 || shape[1] != 1536 {
		t.Fatalf("fixture shape %v, want [N 1536]", shape)
	}

	f, err := os.Open("testdata/src_96.png")
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer f.Close()
	img, err := Decode(f)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	p, err := Preprocess(img, DefaultConfig())
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}
	if p.GridTHW != [3]int{1, 16, 16} {
		t.Fatalf("grid = %v, want [1 16 16]", p.GridTHW)
	}
	if len(p.Data) != len(want) {
		t.Fatalf("got %d values, want %d", len(p.Data), len(want))
	}

	const tol = 0.01
	worst, worstIdx := 0.0, -1
	for i := range want {
		d := math.Abs(float64(p.Data[i] - want[i]))
		if d > worst {
			worst, worstIdx = d, i
		}
	}
	if worst > tol {
		row, col := worstIdx/1536, worstIdx%1536
		t.Fatalf("max abs diff %.6f at row %d col %d (got %v want %v), tolerance %v",
			worst, row, col, p.Data[worstIdx], want[worstIdx], tol)
	}
	t.Logf("max abs diff %.6f (tolerance %v)", worst, tol)
}

// TestPreprocessNoResizeIsExact pins the stronger property: when smart_resize
// picks the image's own dimensions, preprocessing is bit-identical to the
// reference. This is the case that isolates layout from interpolation, so a
// layout regression shows up here rather than hiding inside resize drift.
func TestPreprocessNoResizeIsExact(t *testing.T) {
	cfg := DefaultConfig()
	// 256x256 is exactly the shortest_edge floor, so smart_resize leaves it
	// alone.
	const n = 256
	img := &Image{Width: n, Height: n, Pixels: make([]uint8, n*n*3)}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			i := (y*n + x) * 3
			img.Pixels[i+0] = uint8(255 * x / n)
			img.Pixels[i+1] = uint8(255 * y / n)
			if (x/8)%2 == 0 {
				img.Pixels[i+2] = 255
			}
		}
	}

	p, err := Preprocess(img, cfg)
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}
	if p.ResizedWidth != n || p.ResizedHeight != n {
		t.Fatalf("resized to %dx%d, expected no resize", p.ResizedWidth, p.ResizedHeight)
	}

	// Independently build the expected tensor straight from the source
	// pixels, using the reference's [C, temporal, patch-row, patch-col]
	// row layout. This duplicates Patchify's intent on purpose: it is the
	// specification the implementation has to match.
	const patch, merge = 16, 2
	gridH, gridW := n/patch, n/patch
	gridHM, gridWM := gridH/merge, gridW/merge
	rowLen := cfg.TemporalPatchSize * patch * patch * 3
	want := make([]float32, gridH*gridW*rowLen)
	for my := 0; my < gridHM; my++ {
		for mx := 0; mx < gridWM; mx++ {
			for dy := 0; dy < merge; dy++ {
				for dx := 0; dx < merge; dx++ {
					row := ((my*gridWM+mx)*merge+dy)*merge + dx
					off := row * rowLen
					for c := 0; c < 3; c++ {
						for tt := 0; tt < cfg.TemporalPatchSize; tt++ {
							for py := 0; py < patch; py++ {
								for px := 0; px < patch; px++ {
									src := (((my*merge+dy)*patch+py)*n + (mx*merge+dx)*patch + px) * 3
									v := float32(img.Pixels[src+c]) / 255
									want[off] = (v - cfg.Mean[c]) / cfg.Std[c]
									off++
								}
							}
						}
					}
				}
			}
		}
	}

	if len(p.Data) != len(want) {
		t.Fatalf("got %d values, want %d", len(p.Data), len(want))
	}
	for i := range want {
		if p.Data[i] != want[i] {
			row, col := i/rowLen, i%rowLen
			t.Fatalf("mismatch at row %d offset %d: got %v want %v", row, col, p.Data[i], want[i])
		}
	}
}
