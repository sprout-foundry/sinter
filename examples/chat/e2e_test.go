//go:build darwin || linux

// End-to-end test for the gomlx module: builds examples/chat as a real
// binary and drives it like an implementor would — a local model, a chat
// prompt, streamed tokens. This catches breakage that unit tests can't:
// cgo link errors, backend registration, template/prompt mismatches, the
// whole load -> prefill -> decode -> detokenize path.
//
// Skips (rather than fails) when no model is on disk, so `go test ./...`
// stays green on any machine; CI passes -args -model-dir to run it live.
package main_test

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// candidateModelDirs returns local model dirs, smallest first so the test
// runs fast on machines with several models downloaded.
func candidateModelDirs(t *testing.T) []string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	roots := []string{
		filepath.Join(home, ".cache", "sprout", "models"),
		filepath.Join(home, ".local", "share", "sprout", "models"),
		filepath.Join(home, ".auto-term", "models"),
	}
	var dirs []string
	for _, root := range roots {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			d := filepath.Join(root, e.Name())
			if _, err := os.Stat(filepath.Join(d, "model.safetensors")); err == nil {
				dirs = append(dirs, d)
			}
		}
	}
	return dirs
}

// isThinkingModel reports whether the model at dir uses Qwen3-style
// <think> blocks (and therefore needs the empty-think prefix to answer
// directly). Detected by tokenizer special tokens.
func isThinkingModel(dir string) bool {
	data, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return false
	}
	return strings.Contains(string(data), "<think>")
}

// ggmlPlatform reports whether this machine can run the engine via the
// GGML backend (linux/arm64 or linux/amd64 — includes Termux on Android,
// where GOOS is "android" but the libc/kernel are Linux).
func ggmlPlatform() bool {
	gpu := runtime.GOOS == "linux" && (runtime.GOARCH == "arm64" || runtime.GOARCH == "amd64")
	// GOOS is "android" on Termux; the linux build constraints in this
	// module accept it, so mirror that here.
	return gpu || (runtime.GOOS == "android" && runtime.GOARCH == "arm64")
}

// memAvailableBytes reads MemAvailable from /proc/meminfo (Linux/Android).
func memAvailableBytes() (uint64, bool) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, false
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "MemAvailable:") {
			kb, err := strconv.ParseUint(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(strings.TrimPrefix(line, "MemAvailable:")), "kB")), 10, 64)
			if err != nil {
				return 0, false
			}
			return kb * 1024, true
		}
	}
	return 0, false
}

// dirWeightsBytes sums the size of safetensors files in a model dir.
func dirWeightsBytes(dir string) uint64 {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	var total uint64
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".safetensors") {
			continue
		}
		if info, err := e.Info(); err == nil {
			total += uint64(info.Size())
		}
	}
	return total
}

func TestChatEndToEnd(t *testing.T) {
	if runtime.GOOS != "darwin" && !ggmlPlatform() {
		t.Skip("e2e generation requires the Metal backend (darwin/arm64) or GGML (linux)")
	}
	dirFlag := os.Getenv("SINTER_E2E_MODEL")
	if dirFlag == "" {
		cands := candidateModelDirs(t)
		if len(cands) == 0 {
			t.Skip("no local model found; set SINTER_E2E_MODEL or download one (see README)")
		}
		dirFlag = cands[0]
		t.Logf("using model: %s", dirFlag)
	}
	args := []string{"-model", dirFlag, "-prompt", "Say exactly: hello world", "-max-tokens", "60"}
	if isThinkingModel(dirFlag) {
		// Thinking models emit a <think> block first; prefilling an empty
		// one makes them answer immediately. Non-thinking models (e.g.
		// Qwen2.5) don't know the tokens, so only add it when present.
		args = append(args, "-thinking")
	}

	bin, err := goBuildExample(t)
	if err != nil {
		t.Fatalf("build example: %v", err)
	}

	// "Say exactly" keeps greedy output short and predictable: the model
	// should echo the phrase then stop (EOS) well before the cap.
	cmd := exec.Command(bin, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("sinter-chat: %v\n%s", err, out)
	}
	text := string(out)
	if !strings.Contains(strings.ToLower(text), "hello world") {
		t.Fatalf("output did not contain 'hello world':\n%s", text)
	}
	if strings.Contains(text, "tok/s") {
		t.Logf("%s", text[strings.LastIndex(text, "["):])
	}
}

// TestChatAllLocalFamilies drives the example against every model found on
// disk, so each architecture family (qwen2, llama/minicpm5, qwen3,
// qwen3_5, gemma4, lfm2) gets an end-to-end proof wherever its weights are
// present. Assertions are lenient — any non-empty generation — because tiny
// models don't always echo the exact phrase; the strict echo check lives in
// TestChatEndToEnd. Skips cleanly when no models are installed.
func TestChatAllLocalFamilies(t *testing.T) {
	if runtime.GOOS != "darwin" && !ggmlPlatform() {
		t.Skip("generation requires the Metal backend (darwin/arm64) or GGML (linux)")
	}
	dirs := candidateModelDirs(t)
	if len(dirs) == 0 {
		t.Skip("no local models found; download one (see README)")
	}
	if pinned := os.Getenv("SINTER_E2E_MODEL"); pinned != "" {
		// CI pins one model: the primary e2e already covers it.
		t.Skip("SINTER_E2E_MODEL set; TestChatEndToEnd covers it")
	}

	bin, err := goBuildExample(t)
	if err != nil {
		t.Fatalf("build example: %v", err)
	}

	for _, dir := range dirs {
		dir := dir
		t.Run(filepath.Base(dir), func(t *testing.T) {
			// Android LMK guard: the loader holds the safetensors blob plus
			// dequantized weights, so peak RSS is ~2x weights + KV + graph.
			// Observed on this device: 1.4 GB weights at 4.5 GB free = ok,
			// 2.3 GB weights at ~4.5 GB free = LMK kills the process. Skip
			// anything above 40% of currently-available memory.
			if avail, ok := memAvailableBytes(); ok {
				w := dirWeightsBytes(dir)
				if w > 0 && w*10 > avail*4 {
					t.Skipf("weights (%d MB) > 40%% of MemAvailable (%d MB): LMK risk on this device",
						w>>20, avail>>20)
				}
			}
			args := []string{"-model", dir, "-prompt", "Say exactly: hello world", "-max-tokens", "40", "-timeout", "4m"}
			if isThinkingModel(dir) {
				args = append(args, "-thinking")
			}
			out, err := exec.Command(bin, args...).CombinedOutput()
			if err != nil {
				t.Fatalf("sinter-chat: %v\n%s", err, out)
			}
			text := string(out)
			// Generated text is everything before the trailing [load ...]
			// stats line (stderr), so strip it before the emptiness check —
			// a zero-token run still prints stats and must fail here.
			gen := text
			if i := strings.LastIndex(text, "[load"); i >= 0 {
				gen = text[:i]
				t.Logf("%s", text[i:])
			} else {
				t.Logf("output (no stats line): %q", text)
			}
			if strings.TrimSpace(gen) == "" {
				t.Fatalf("no output generated (stats: %q)", text)
			}
		})
	}
}

// TestChatUsageExitCode pins the CLI contract: missing -model exits 2 with
// usage on stderr. Runs everywhere (no GPU, no model needed).
func TestChatUsageExitCode(t *testing.T) {
	bin, err := goBuildExample(t)
	if err != nil {
		t.Fatalf("build example: %v", err)
	}
	out, err := exec.Command(bin).CombinedOutput()
	if err == nil {
		t.Fatalf("expected non-zero exit, got 0:\n%s", out)
	}
	if ee, ok := err.(*exec.ExitError); !ok || ee.ExitCode() != 2 {
		t.Fatalf("expected exit code 2, got %v", err)
	}
	if !strings.Contains(string(out), "usage:") {
		t.Fatalf("expected usage on stderr:\n%s", out)
	}
}

func goBuildExample(t *testing.T) (string, error) {
	t.Helper()
	// The test lives in the same dir as the example, so "." is the main
	// package; building via `go build .` from the module root keeps this
	// working regardless of where `go test` was invoked from.
	root, err := findModuleRoot()
	if err != nil {
		return "", err
	}
	bin := filepath.Join(t.TempDir(), "sinter-chat")
	// Non-darwin needs the ggml build tag (backend selection + engine
	// files are gated on it; see llm/architecture.go).
	buildArgs := []string{"build", "-o", bin}
	if runtime.GOOS != "darwin" {
		buildArgs = append(buildArgs, "-tags", "ggml")
	}
	buildArgs = append(buildArgs, "./examples/chat")
	build := exec.Command("go", buildArgs...)
	build.Dir = root
	if out, err := build.CombinedOutput(); err != nil {
		return "", fmt.Errorf("go build: %v\n%s", err, out)
	}
	return bin, nil
}

func findModuleRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("go.mod not found")
		}
		dir = parent
	}
}
