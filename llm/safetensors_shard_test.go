//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package llm

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeShard writes a one-tensor safetensors file. The tensor holds
// len(shape) multiplied-out float32 zeros; the values do not matter, only the
// header metadata the readers consult.
func writeShard(t *testing.T, path, name string, shape []int) {
	t.Helper()
	n := 1
	for _, d := range shape {
		n *= d
	}
	header := map[string]safetensorEntry{
		name: {DType: "F32", Shape: shape, DataOffsets: []int{0, n * 4}},
	}
	hb, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 8+len(hb)+n*4)
	binary.LittleEndian.PutUint64(buf[:8], uint64(len(hb)))
	copy(buf[8:], hb)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestShardedTensorShape covers a sharded checkpoint's metadata accessors.
// TensorShape used to return nil for every tensor in a sharded model, which
// made the vision tower reject its patch-embed weight ("shape [], want [out,
// temporal, patch, patch, in_ch]") and stopped the model from loading at all.
func TestShardedTensorShape(t *testing.T) {
	dir := t.TempDir()
	shardA := "model-00001-of-00002.safetensors"
	shardB := "model-00002-of-00002.safetensors"
	writeShard(t, filepath.Join(dir, shardA), "vision_tower.patch_embed.proj.weight", []int{1152, 2, 16, 16, 3})
	writeShard(t, filepath.Join(dir, shardB), "language_model.model.embed_tokens.weight", []int{248320, 2048})

	idx := safetensorsIndex{WeightMap: map[string]string{
		"vision_tower.patch_embed.proj.weight":     shardA,
		"language_model.model.embed_tokens.weight": shardB,
	}}
	ib, err := json.Marshal(idx)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o644); err != nil {
		t.Fatal(err)
	}

	sf, err := OpenSafetensors(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatalf("open sharded: %v", err)
	}
	defer sf.Release()
	if sf.weightMap == nil {
		t.Fatal("expected a sharded reader")
	}

	for name, want := range map[string][]int{
		"vision_tower.patch_embed.proj.weight":     {1152, 2, 16, 16, 3},
		"language_model.model.embed_tokens.weight": {248320, 2048},
	} {
		got := sf.TensorShape(name)
		if len(got) != len(want) {
			t.Fatalf("%s: shape %v, want %v", name, got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: shape %v, want %v", name, got, want)
			}
		}
		if !sf.Has(name) {
			t.Errorf("%s: Has = false", name)
		}
	}

	if got := sf.TensorShape("not.a.tensor"); got != nil {
		t.Errorf("missing tensor: shape %v, want nil", got)
	}
	if sf.Has("not.a.tensor") {
		t.Error("missing tensor: Has = true")
	}
}
