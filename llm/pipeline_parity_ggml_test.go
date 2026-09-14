//go:build linux && (arm64 || amd64) && cgo && ggml

package llm_test

import (
	"context"
	"os"
	"testing"

	"github.com/sprout-foundry/sinter/llm"
	_ "github.com/sprout-foundry/sinter/llm/all"
)

// TestPipelinedDecodeParityLiveModelGGML mirrors the darwin parity test for
// the GGML backend. KNOWN BROKEN: the pipelined decode loop currently
// segfaults on GGML (see the usePipelined comment in model.go — crash in
// ggml_backend_graph_compute during the batch flush). This test is
// env-guarded (SINTER_PIPE_TEST_MODEL) rather than skip-guarded because a
// cgo segfault takes down the whole test process; normal `go test ./...`
// must never trip it. Run it explicitly while debugging the crash:
//
//	SINTER_PIPE_TEST_MODEL=~/.cache/sprout/models/qwen2.5-0.5b-4bit \
//	  go test -tags ggml -run TestPipelinedDecodeParityLiveModelGGML -v ./llm
//
// When it passes byte-for-byte across prompts AND survives, the crash is
// fixed and the GGML default flip can be revisited.
func TestPipelinedDecodeParityLiveModelGGML(t *testing.T) {
	dir := os.Getenv("SINTER_PIPE_TEST_MODEL")
	if dir == "" {
		t.Skip("SINTER_PIPE_TEST_MODEL not set (pipelined decode is known-broken on GGML; see model.go)")
	}

	prompts := []string{
		"Hello",
		"The capital of France is",
		"Write a short poem about the ocean.",
		"Count: one two three four five six seven.",
	}

	runOne := func(t *testing.T, prompt string) string {
		t.Helper()
		model, err := llm.NewModel(dir)
		if err != nil {
			t.Fatalf("NewModel(%q): %v", dir, err)
		}
		defer model.Close()

		cfg := llm.DefaultGenerateConfig()
		cfg.MaxTokens = 24
		cfg.Temperature = 0
		cfg.RepetitionPenalty = 0

		out, err := model.GenerateText(context.Background(), prompt, cfg)
		if err != nil {
			t.Fatalf("GenerateText(%q): %v", prompt, err)
		}
		return out
	}

	for _, prompt := range prompts {
		os.Setenv("SINTER_PIPELINE_DECODE", "0")
		plain := runOne(t, prompt)

		t.Setenv("SINTER_PIPELINE_DECODE", "1")
		pipelined := runOne(t, prompt)
		os.Unsetenv("SINTER_PIPELINE_DECODE")

		if pipelined != plain {
			t.Errorf("pipelined decode divergence for %q:\n  pipelined: %q\n  plain:     %q", prompt, pipelined, plain)
		} else {
			t.Logf("parity ok for %q: %q", prompt, plain)
		}
	}
}
