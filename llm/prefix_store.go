//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package llm

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"github.com/sprout-foundry/sinter/tensor"
)

// A prefixStore keeps a model's warm system-prompt KV slot on disk so a
// fresh process can skip re-prefilling it. In-memory slots (see Model's
// prefixSlots) already make later turns cheap, but they die with the
// process: a restarted agent pays the full system-prompt prefill again.
//
// Only the warm system prefix is persisted — the static prompt prefix that
// WarmSystemPrefix computes and that every conversation shares. Persisting
// each conversation's own slot would rewrite hundreds of megabytes per turn
// for a hit the in-memory slots already serve, so the store holds the
// protected slot and nothing else.
//
// Safety is the whole design constraint. A slot is only served when the
// model fingerprint, the image key and the exact token sequence all match,
// because a DeltaNet layer's recurrent state cannot be rewound or repaired:
// restoring the wrong one produces plausible but wrong output.
const prefixStoreVersion = uint32(1)

// prefixCacheEpoch invalidates every persisted slot when the forward pass
// changes. A slot is a function of the weights AND of the code that computed
// it: a change to rope scaling, a norm, or the recurrent kernel leaves the
// fingerprint identical while making the stored K/V wrong. Bump this with any
// such change — the cost is one re-prefill, the alternative is silently wrong
// output.
const prefixCacheEpoch = 1

var prefixStoreMagic = [4]byte{'S', 'P', 'P', 'X'}

// prefixStoreEntry is the metadata that has to match before a persisted
// slot's K/V may be used.
type prefixStoreEntry struct {
	tokens   []int
	imageKey string
	posDelta int
}

// prefixStore is one model's on-disk prefix-slot directory. A nil
// *prefixStore is valid and means persistence is off.
type prefixStore struct {
	dir         string
	fingerprint string
}

// maxPersistedPrefixes bounds how many slots a model keeps on disk. The warm
// system prefix is normally the only one; the cap exists so a system prompt
// that changes between releases cannot accumulate stale files forever.
const maxPersistedPrefixes = 4

// maxHeaderTokens bounds the token count a header may declare. The store only
// ever holds a warm prefix, which WarmSystemPrefix already caps at the
// context length; the bound exists so a corrupt file cannot make the header
// reader allocate an arbitrary amount of memory.
const maxHeaderTokens = 1 << 20

// newPrefixStore resolves the cache directory for modelDir and fingerprints
// the model. Returns nil when persistence is disabled or the directory
// cannot be determined — callers treat nil as "no persistence", never as an
// error, since the in-memory cache path is fully correct on its own.
func newPrefixStore(modelDir string) *prefixStore {
	root, ok := prefixStoreRoot()
	if !ok {
		return nil
	}
	fp, err := modelFingerprint(modelDir)
	if err != nil {
		return nil
	}
	return &prefixStore{dir: filepath.Join(root, "prefix", fp), fingerprint: fp}
}

// prefixStoreRoot returns the directory that holds all persisted prefixes.
// SINTER_PREFIX_DISK=0 disables persistence entirely; SINTER_CACHE_DIR
// relocates the root (tests use this to stay hermetic).
func prefixStoreRoot() (string, bool) {
	if os.Getenv("SINTER_PREFIX_DISK") == "0" {
		return "", false
	}
	if d := os.Getenv("SINTER_CACHE_DIR"); d != "" {
		return d, true
	}
	if d := os.Getenv("XDG_CACHE_HOME"); d != "" {
		return filepath.Join(d, "sinter"), true
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", false
	}
	return filepath.Join(home, ".cache", "sinter"), true
}

// modelFingerprint identifies the weights and tokenizer a persisted slot was
// computed from. A slot is only valid against the exact same model: the K/V
// and recurrent state are functions of the weights, and the token ids are a
// function of the tokenizer. File sizes and modification times catch an
// in-place requant or replacement without hashing gigabytes of weights.
func modelFingerprint(modelDir string) (string, error) {
	h := sha256.New()
	fmt.Fprintf(h, "epoch\x00%d\x00", prefixCacheEpoch)
	for _, name := range []string{"config.json", "tokenizer.json"} {
		data, err := os.ReadFile(filepath.Join(modelDir, name))
		if err != nil {
			return "", fmt.Errorf("fingerprint %s: %w", name, err)
		}
		fmt.Fprintf(h, "%s\x00%d\x00", name, len(data))
		h.Write(data)
	}
	shards, err := filepath.Glob(filepath.Join(modelDir, "*.safetensors"))
	if err != nil {
		return "", err
	}
	if len(shards) == 0 {
		return "", fmt.Errorf("fingerprint: no safetensors in %s", modelDir)
	}
	sort.Strings(shards)
	for _, shard := range shards {
		st, err := os.Stat(shard)
		if err != nil {
			return "", fmt.Errorf("fingerprint %s: %w", shard, err)
		}
		fmt.Fprintf(h, "%s\x00%d\x00%d\x00", filepath.Base(shard), st.Size(), st.ModTime().UnixNano())
	}
	return hex.EncodeToString(h.Sum(nil))[:16], nil
}

// path returns the file a given entry is stored under. The name is derived
// from the entry's content, so re-warming the same system prefix overwrites
// nothing and never grows the directory.
func (s *prefixStore) path(e prefixStoreEntry) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\x00%s\x00%d\x00", s.fingerprint, e.imageKey, e.posDelta)
	for _, tok := range e.tokens {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], uint32(tok))
		h.Write(b[:])
	}
	return filepath.Join(s.dir, "prefix-"+hex.EncodeToString(h.Sum(nil))[:16]+".bin")
}

// Save writes cache to the store under entry, atomically (temp file plus
// rename) so a concurrent reader never sees a half-written slot and a crash
// cannot leave a truncated one. An existing file with the same content is
// left alone: the name is content-derived, so the bytes would be identical.
func (s *prefixStore) Save(e prefixStoreEntry, cache *KVCache) (err error) {
	if err := os.MkdirAll(s.dir, 0o700); err != nil {
		return fmt.Errorf("prefix store: mkdir: %w", err)
	}
	dst := s.path(e)
	if _, statErr := os.Stat(dst); statErr == nil {
		// The name is content-derived, so an existing file holds these exact
		// bytes. Verify it is actually readable before trusting it: a file
		// truncated by something outside this process would otherwise make
		// every future cold start a cache miss with no way to recover.
		if _, readErr := readPrefixHeader(dst); readErr == nil {
			return nil
		}
	}

	tmp, err := os.CreateTemp(s.dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("prefix store: temp file: %w", err)
	}
	defer func() {
		if err != nil {
			_ = os.Remove(tmp.Name())
		}
	}()

	w := bufio.NewWriter(tmp)
	if err = writePrefixHeader(w, s.fingerprint, e); err != nil {
		tmp.Close()
		return err
	}
	if err = cache.WriteSpill(w); err != nil {
		tmp.Close()
		return fmt.Errorf("prefix store: write payload: %w", err)
	}
	if err = w.Flush(); err != nil {
		tmp.Close()
		return fmt.Errorf("prefix store: flush: %w", err)
	}
	if err = tmp.Close(); err != nil {
		return fmt.Errorf("prefix store: close: %w", err)
	}
	if err = os.Rename(tmp.Name(), dst); err != nil {
		return fmt.Errorf("prefix store: rename: %w", err)
	}
	s.evict()
	return nil
}

// Find returns the persisted entry whose entire token sequence is a true
// prefix of tokenIDs and the file holding it, preferring the longest match.
// Only headers are read, so a directory of multi-hundred-megabyte slots
// costs a few kilobytes of I/O to search.
func (s *prefixStore) Find(tokenIDs []int, imageKey string) (prefixStoreEntry, string, bool) {
	matches, err := filepath.Glob(filepath.Join(s.dir, "prefix-*.bin"))
	if err != nil {
		return prefixStoreEntry{}, "", false
	}
	var best prefixStoreEntry
	bestPath := ""
	for _, path := range matches {
		e, err := readPrefixHeader(path)
		if err != nil || e.imageKey != imageKey || e.posDelta != 0 {
			continue
		}
		if len(e.tokens) > len(tokenIDs) {
			continue
		}
		if longestCommonPrefix(tokenIDs, e.tokens) != len(e.tokens) {
			continue
		}
		if len(e.tokens) > len(best.tokens) {
			best, bestPath = e, path
		}
	}
	return best, bestPath, bestPath != ""
}

// Load reads the slot at path. The header is re-verified against this
// store's fingerprint so a file copied in from another model's directory is
// rejected rather than restored, and the payload must describe exactly
// numLayers layers — a mismatch would otherwise surface later as a failed
// RestorePrefix, turning a cache miss into a hard generation error.
func (s *prefixStore) Load(path string, numLayers int, stream tensor.Stream, backend tensor.Backend) (*KVCache, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("prefix store: open: %w", err)
	}
	defer f.Close()

	r := bufio.NewReader(f)
	fp, _, err := readPrefixHeaderFrom(r)
	if err != nil {
		return nil, err
	}
	if fp != s.fingerprint {
		return nil, fmt.Errorf("prefix store: %s belongs to model %s, not %s", path, fp, s.fingerprint)
	}
	cache, err := ReadSpill(r, stream, backend)
	if err != nil {
		return nil, fmt.Errorf("prefix store: read payload: %w", err)
	}
	if len(cache.layers) != numLayers {
		cache.Free()
		return nil, fmt.Errorf("prefix store: %s has %d layers, model has %d", path, len(cache.layers), numLayers)
	}
	return cache, nil
}

// evict trims the directory to maxPersistedPrefixes, dropping the least
// recently written first. Failures are ignored: a full directory only costs
// disk, never correctness.
func (s *prefixStore) evict() {
	matches, err := filepath.Glob(filepath.Join(s.dir, "prefix-*.bin"))
	if err != nil || len(matches) <= maxPersistedPrefixes {
		return
	}
	type aged struct {
		path string
		mod  int64
	}
	files := make([]aged, 0, len(matches))
	for _, path := range matches {
		if st, err := os.Stat(path); err == nil {
			files = append(files, aged{path, st.ModTime().UnixNano()})
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].mod < files[j].mod })
	for _, f := range files[:len(files)-maxPersistedPrefixes] {
		_ = os.Remove(f.path)
	}
}

// writePrefixHeader writes the container header that precedes the spill
// payload.
func writePrefixHeader(w io.Writer, fingerprint string, e prefixStoreEntry) error {
	if _, err := w.Write(prefixStoreMagic[:]); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, prefixStoreVersion); err != nil {
		return err
	}
	if err := writePrefixString(w, fingerprint); err != nil {
		return err
	}
	if err := writePrefixString(w, e.imageKey); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, int32(e.posDelta)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(e.tokens))); err != nil {
		return err
	}
	for _, tok := range e.tokens {
		if err := binary.Write(w, binary.LittleEndian, int32(tok)); err != nil {
			return err
		}
	}
	return nil
}

func writePrefixString(w io.Writer, s string) error {
	if err := binary.Write(w, binary.LittleEndian, uint16(len(s))); err != nil {
		return err
	}
	_, err := io.WriteString(w, s)
	return err
}

// readPrefixHeader reads just the header of the file at path.
func readPrefixHeader(path string) (prefixStoreEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return prefixStoreEntry{}, err
	}
	defer f.Close()
	_, e, err := readPrefixHeaderFrom(bufio.NewReader(f))
	return e, err
}

func readPrefixHeaderFrom(r io.Reader) (string, prefixStoreEntry, error) {
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return "", prefixStoreEntry{}, err
	}
	if magic != prefixStoreMagic {
		return "", prefixStoreEntry{}, fmt.Errorf("prefix store: bad magic")
	}
	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return "", prefixStoreEntry{}, err
	}
	if version != prefixStoreVersion {
		return "", prefixStoreEntry{}, fmt.Errorf("prefix store: unsupported version %d", version)
	}
	fp, err := readPrefixString(r)
	if err != nil {
		return "", prefixStoreEntry{}, err
	}
	imageKey, err := readPrefixString(r)
	if err != nil {
		return "", prefixStoreEntry{}, err
	}
	var posDelta int32
	if err := binary.Read(r, binary.LittleEndian, &posDelta); err != nil {
		return "", prefixStoreEntry{}, err
	}
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", prefixStoreEntry{}, err
	}
	if n > maxHeaderTokens {
		return "", prefixStoreEntry{}, fmt.Errorf("prefix store: header declares %d tokens", n)
	}
	tokens := make([]int, n)
	for i := range tokens {
		var tok int32
		if err := binary.Read(r, binary.LittleEndian, &tok); err != nil {
			return "", prefixStoreEntry{}, err
		}
		tokens[i] = int(tok)
	}
	return fp, prefixStoreEntry{tokens: tokens, imageKey: imageKey, posDelta: int(posDelta)}, nil
}

func readPrefixString(r io.Reader) (string, error) {
	var n uint16
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	if n == 0 {
		return "", nil
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}
