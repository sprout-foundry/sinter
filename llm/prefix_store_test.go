//go:build darwin && arm64 && cgo

package llm

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	_ "github.com/sprout-foundry/sinter/mlx" // registers a real tensor.Backend for this test
	"github.com/sprout-foundry/sinter/tensor"
)

// fakeModelDir builds the files newPrefixStore fingerprints, so a test model
// looks like a real checkpoint without loading one.
func fakeModelDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range map[string]string{
		"config.json":       `{"arch":"test"}`,
		"tokenizer.json":    `{"model":{}}`,
		"model.safetensors": "weights",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	return dir
}

// testCache builds a small cache with one full-attention layer and one
// DeltaNet layer, which is the shape a real slot has.
func testCache(t *testing.T, backend tensor.Backend, stream tensor.Stream) *KVCache {
	t.Helper()
	c := NewKVCache(2, stream, backend)
	k, err := backend.NewArrayFromFloat32([]float32{1, 2, 3, 4}, []int{1, 1, 2, 2})
	if err != nil {
		t.Fatalf("new K: %v", err)
	}
	v, err := backend.NewArrayFromFloat32([]float32{5, 6, 7, 8}, []int{1, 1, 2, 2})
	if err != nil {
		t.Fatalf("new V: %v", err)
	}
	if err := c.Store(0, k, v); err != nil {
		t.Fatalf("store layer 0: %v", err)
	}
	state, err := backend.NewArrayFromFloat32([]float32{0.5, -0.5}, []int{1, 1, 2, 1})
	if err != nil {
		t.Fatalf("new State: %v", err)
	}
	conv, err := backend.NewArrayFromFloat32([]float32{1.25}, []int{1, 1, 1})
	if err != nil {
		t.Fatalf("new ConvState: %v", err)
	}
	if err := c.StoreState(1, state, conv); err != nil {
		t.Fatalf("store layer 1 state: %v", err)
	}
	hidden, err := backend.NewArrayFromFloat32([]float32{9, 10}, []int{1, 1, 2})
	if err != nil {
		t.Fatalf("new hidden: %v", err)
	}
	c.SetLastHidden(hidden)
	return c
}

// TestPrefixStoreRoundTrip guards the mechanism behind cross-restart prefix
// reuse: a slot written by one process must load into another with its K/V,
// recurrent state and last hidden state intact, and must only be served to a
// prompt it is a true prefix of.
func TestPrefixStoreRoundTrip(t *testing.T) {
	backend := tensor.DetectBackend()
	if backend == nil || !backend.Available() {
		t.Skip("no tensor backend available on this platform")
	}
	stream, err := backend.DefaultGPUStream()
	if err != nil {
		t.Fatalf("get stream: %v", err)
	}
	t.Setenv("SINTER_CACHE_DIR", t.TempDir())

	store := newPrefixStore(fakeModelDir(t))
	if store == nil {
		t.Fatal("newPrefixStore returned nil with a valid model dir")
	}

	cache := testCache(t, backend, stream)
	defer cache.Free()
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	if err := store.Save(prefixStoreEntry{tokens: tokens}, cache); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// A prompt that extends the stored prefix must find it, and must report
	// the stored length as the shared prefix.
	longer := append(append([]int(nil), tokens...), 11, 12, 13)
	entry, path, ok := store.Find(longer, "")
	if !ok {
		t.Fatal("Find did not match a prompt that extends the stored prefix")
	}
	if len(entry.tokens) != len(tokens) {
		t.Fatalf("matched %d tokens, want %d", len(entry.tokens), len(tokens))
	}

	// A prompt that diverges must not match: DeltaNet state cannot be
	// rewound, so a partial overlap would restore a wrong recurrent state.
	diverged := append([]int(nil), tokens...)
	diverged[4] = 99
	if _, _, ok := store.Find(diverged, ""); ok {
		t.Fatal("Find matched a prompt that diverges mid-prefix")
	}
	// A shorter prompt cannot contain the stored prefix.
	if _, _, ok := store.Find(tokens[:5], ""); ok {
		t.Fatal("Find matched a prompt shorter than the stored prefix")
	}

	loaded, err := store.Load(path, 2, stream, backend)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer loaded.Free()

	l0, err := loaded.Get(0)
	if err != nil {
		t.Fatalf("get layer 0: %v", err)
	}
	gotK, err := l0.K.Float32Data()
	if err != nil {
		t.Fatalf("K.Float32Data: %v", err)
	}
	assertFloat32Equal(t, "K", gotK, []float32{1, 2, 3, 4})

	l1, err := loaded.Get(1)
	if err != nil {
		t.Fatalf("get layer 1: %v", err)
	}
	gotState, err := l1.State.Float32Data()
	if err != nil {
		t.Fatalf("State.Float32Data: %v", err)
	}
	assertFloat32Equal(t, "State", gotState, []float32{0.5, -0.5})

	if loaded.LastHidden() == nil {
		t.Fatal("last hidden state did not survive the store round-trip")
	}
	gotHidden, err := loaded.LastHidden().Float32Data()
	if err != nil {
		t.Fatalf("hidden Float32Data: %v", err)
	}
	assertFloat32Equal(t, "hidden", gotHidden, []float32{9, 10})
}

// TestPrefixStoreRejectsOtherModel guards the fingerprint gate: a slot file
// copied into another model's directory must be refused rather than restored,
// because its K/V came from different weights.
func TestPrefixStoreRejectsOtherModel(t *testing.T) {
	backend := tensor.DetectBackend()
	if backend == nil || !backend.Available() {
		t.Skip("no tensor backend available on this platform")
	}
	stream, err := backend.DefaultGPUStream()
	if err != nil {
		t.Fatalf("get stream: %v", err)
	}
	t.Setenv("SINTER_CACHE_DIR", t.TempDir())

	owner := newPrefixStore(fakeModelDir(t))
	other := newPrefixStore(fakeModelDir(t))
	if owner == nil || other == nil {
		t.Fatal("newPrefixStore returned nil with a valid model dir")
	}
	if owner.fingerprint == other.fingerprint {
		t.Fatal("two different model dirs produced the same fingerprint")
	}

	cache := testCache(t, backend, stream)
	defer cache.Free()
	entry := prefixStoreEntry{tokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	if err := owner.Save(entry, cache); err != nil {
		t.Fatalf("Save: %v", err)
	}
	// Place the file where the other model would look for it.
	if err := os.MkdirAll(other.dir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	src := owner.path(entry)
	dst := other.path(entry)
	if err := os.Rename(src, dst); err != nil {
		t.Fatalf("move slot: %v", err)
	}

	if _, _, ok := other.Find(entry.tokens, ""); !ok {
		t.Fatal("Find should still see the file (the gate is on Load)")
	}
	if _, err := other.Load(dst, 2, stream, backend); err == nil {
		t.Fatal("Load accepted a slot written by a different model")
	}
}

// TestPrefixStoreSkipsCorruptFiles guards the failure mode a crash can leave
// behind: a truncated or foreign file must be skipped, not loaded.
func TestPrefixStoreSkipsCorruptFiles(t *testing.T) {
	backend := tensor.DetectBackend()
	if backend == nil || !backend.Available() {
		t.Skip("no tensor backend available on this platform")
	}
	t.Setenv("SINTER_CACHE_DIR", t.TempDir())

	store := newPrefixStore(fakeModelDir(t))
	if store == nil {
		t.Fatal("newPrefixStore returned nil with a valid model dir")
	}
	if err := os.MkdirAll(store.dir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	junk := filepath.Join(store.dir, "prefix-deadbeef.bin")
	if err := os.WriteFile(junk, []byte("not a prefix slot"), 0o600); err != nil {
		t.Fatalf("write junk: %v", err)
	}
	if _, _, ok := store.Find([]int{1, 2, 3, 4, 5, 6, 7, 8}, ""); ok {
		t.Fatal("Find matched a corrupt file")
	}
}

// TestPrefixStoreRejectsWrongLayerCount guards the payload/model shape check:
// restoring a cache with the wrong layer count would fail later inside
// RestorePrefix, turning what should be a cache miss into a hard error.
func TestPrefixStoreRejectsWrongLayerCount(t *testing.T) {
	backend := tensor.DetectBackend()
	if backend == nil || !backend.Available() {
		t.Skip("no tensor backend available on this platform")
	}
	stream, err := backend.DefaultGPUStream()
	if err != nil {
		t.Fatalf("get stream: %v", err)
	}
	t.Setenv("SINTER_CACHE_DIR", t.TempDir())

	store := newPrefixStore(fakeModelDir(t))
	if store == nil {
		t.Fatal("newPrefixStore returned nil with a valid model dir")
	}
	cache := testCache(t, backend, stream)
	defer cache.Free()
	entry := prefixStoreEntry{tokens: []int{1, 2, 3, 4, 5, 6, 7, 8}}
	if err := store.Save(entry, cache); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, err := store.Load(store.path(entry), 3, stream, backend); err == nil {
		t.Fatal("Load accepted a payload whose layer count does not match the model")
	}
}

// TestPrefixStoreRejectsAbsurdHeader guards the header reader against a
// corrupt file: the declared token count is used to size an allocation, so an
// unchecked value would let one bad file exhaust memory.
func TestPrefixStoreRejectsAbsurdHeader(t *testing.T) {
	backend := tensor.DetectBackend()
	if backend == nil || !backend.Available() {
		t.Skip("no tensor backend available on this platform")
	}
	t.Setenv("SINTER_CACHE_DIR", t.TempDir())

	store := newPrefixStore(fakeModelDir(t))
	if store == nil {
		t.Fatal("newPrefixStore returned nil with a valid model dir")
	}
	if err := os.MkdirAll(store.dir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	var buf bytes.Buffer
	buf.Write(prefixStoreMagic[:])
	binary.Write(&buf, binary.LittleEndian, prefixStoreVersion)
	binary.Write(&buf, binary.LittleEndian, uint16(len(store.fingerprint)))
	buf.WriteString(store.fingerprint)
	binary.Write(&buf, binary.LittleEndian, uint16(0)) // imageKey
	binary.Write(&buf, binary.LittleEndian, int32(0))  // posDelta
	binary.Write(&buf, binary.LittleEndian, uint32(0xFFFFFFFF))
	path := filepath.Join(store.dir, "prefix-badc0de.bin")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, _, ok := store.Find([]int{1, 2, 3, 4, 5, 6, 7, 8}, ""); ok {
		t.Fatal("Find matched a header declaring an absurd token count")
	}
}

// TestPrefixStoreEvictsOldest guards the disk bound: a system prompt that
// changes between releases must not accumulate slots forever.
func TestPrefixStoreEvictsOldest(t *testing.T) {
	backend := tensor.DetectBackend()
	if backend == nil || !backend.Available() {
		t.Skip("no tensor backend available on this platform")
	}
	stream, err := backend.DefaultGPUStream()
	if err != nil {
		t.Fatalf("get stream: %v", err)
	}
	t.Setenv("SINTER_CACHE_DIR", t.TempDir())

	store := newPrefixStore(fakeModelDir(t))
	if store == nil {
		t.Fatal("newPrefixStore returned nil with a valid model dir")
	}
	cache := testCache(t, backend, stream)
	defer cache.Free()

	for i := 0; i < maxPersistedPrefixes+2; i++ {
		tokens := make([]int, 8)
		for j := range tokens {
			tokens[j] = i*100 + j
		}
		if err := store.Save(prefixStoreEntry{tokens: tokens}, cache); err != nil {
			t.Fatalf("Save %d: %v", i, err)
		}
	}
	files, err := filepath.Glob(filepath.Join(store.dir, "prefix-*.bin"))
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	if len(files) > maxPersistedPrefixes {
		t.Fatalf("store kept %d files, want at most %d", len(files), maxPersistedPrefixes)
	}
}
