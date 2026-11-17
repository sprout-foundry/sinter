//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package qwen35

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/llm/vision"
	"github.com/sprout-foundry/sinter/tensor"
)

// TestVisionTowerParityLiveModel compares the tower's merged output against
// a dump from the reference implementation (transformers' Qwen3VLVisionModel
// with the checkpoint's own weights).
//
// The dump is testdata/ref_tower_96.npy, produced from testdata/src_96.png.
// Regenerate with:
//
//	/tmp/sprout/vlref-env/bin/python (see the vision package's parity notes)
//
// The model directory comes from SINTER_VISION_MODEL, falling back to the
// usual parity-model variable, so this skips when no weights are on disk.
func TestVisionTowerParityLiveModel(t *testing.T) {
	dir := os.Getenv("SINTER_VISION_MODEL")
	if dir == "" {
		dir = os.Getenv("SINTER_MTP_PARITY_MODEL")
	}
	if dir == "" {
		t.Skip("set SINTER_VISION_MODEL to a qwen3.5 VL model directory")
	}

	// MLX streams are thread-local: the stream, the weights loaded onto it,
	// and every op chained off them must all run on one OS thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	backend := tensor.DetectBackend()
	if backend == nil || !backend.Available() {
		t.Skip("no GPU backend available")
	}

	cfg, err := loadVisionConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatalf("load vision config: %v", err)
	}
	if cfg == nil {
		t.Skipf("%s has no vision_config", dir)
	}

	sf, err := llm.OpenSafetensors(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatalf("open safetensors: %v", err)
	}
	defer sf.Release()

	prefix := "vision_tower."
	if !sf.Has(prefix + "patch_embed.proj.weight") {
		prefix = "model.visual."
	}
	if !sf.Has(prefix + "patch_embed.proj.weight") {
		t.Skipf("%s has no vision tower weights", dir)
	}

	stream, err := backend.DefaultGPUStream()
	if err != nil {
		t.Fatalf("gpu stream: %v", err)
	}
	defer stream.Free()

	tower, err := loadVisionTower(sf, prefix, *cfg, backend, stream)
	if err != nil {
		t.Fatalf("load tower: %v", err)
	}
	defer tower.free()

	// Prepare the same image the reference dump used.
	f, err := os.Open("../vision/testdata/src_96.png")
	if err != nil {
		t.Fatalf("open source image: %v", err)
	}
	defer f.Close()
	img, err := vision.Decode(f)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	pixels, err := vision.Preprocess(img, vision.DefaultConfig())
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}

	patches, err := backend.NewArrayFromFloat32(pixels.Data, []int{pixels.NumPatches(), len(pixels.Data) / pixels.NumPatches()})
	if err != nil {
		t.Fatalf("patch array: %v", err)
	}
	defer patches.Free()

	merged, err := tower.forward(patches, visionImage{GridTHW: pixels.GridTHW})
	if err != nil {
		t.Fatalf("tower forward: %v", err)
	}
	defer merged.Free()

	got, err := merged.Float32Data()
	if err != nil {
		t.Fatalf("read merged: %v", err)
	}

	want, shape, err := vision.LoadNpyFloat32("../vision/testdata/ref_tower_96.npy")
	if err != nil {
		t.Fatalf("load reference tower output: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("output has %d values, reference has %d (shape %v)", len(got), len(want), shape)
	}

	// The two implementations agree structurally but not bit-exactly: the
	// tower runs in bf16 here while the dump is f32, so accumulation order
	// compounds through 24 blocks. Measured mean absolute 0.0085 on both MLX
	// and GGML, on reference values averaging 0.094.
	//
	// The gate sits below what a missing weight or bias produces — the
	// merger's biases were once skipped, which pushed this to 0.020 — and
	// above the drift. TestMergerAppliesBias pins the merger directly, so
	// this only has to catch structural breakage: a layout, position or
	// ordering bug shows up as most of the output being wrong at once.
	var maxAbs, sumAbs float64
	maxRel := 0.0
	var refAbsMax float64
	for i := range want {
		d := math.Abs(float64(got[i] - want[i]))
		if d > maxAbs {
			maxAbs = d
		}
		sumAbs += d
		if a := math.Abs(float64(want[i])); a > refAbsMax {
			refAbsMax = a
		}
		if math.Abs(float64(want[i])) > 1e-3 {
			if r := d / math.Abs(float64(want[i])); r > maxRel {
				maxRel = r
			}
		}
	}
	meanAbs := sumAbs / float64(len(want))
	t.Logf("merged output: max abs %.6f, mean abs %.6f, max rel %.4f, reference |max| %.4f",
		maxAbs, meanAbs, maxRel, refAbsMax)

	const maxMeanAbs = 0.015
	if meanAbs > maxMeanAbs {
		firstBad := -1
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 0.1 {
				firstBad = i
				break
			}
		}
		row, col := -1, -1
		if firstBad >= 0 && len(shape) == 2 {
			row, col = firstBad/shape[1], firstBad%shape[1]
		}
		t.Fatalf("mean abs diff %.6f exceeds %v (first large diff at row %d col %d: got %v want %v)",
			meanAbs, maxMeanAbs, row, col, got[firstBad], want[firstBad])
	}
}
