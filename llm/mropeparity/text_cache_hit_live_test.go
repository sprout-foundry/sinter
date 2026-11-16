//go:build ggml && cgo && ((darwin && arm64) || (linux && (arm64 || amd64)))

package mropeparity

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/sprout-foundry/sinter/llm"
	_ "github.com/sprout-foundry/sinter/llm/qwen35"
	_ "github.com/sprout-foundry/sinter/tensor/ggml"
)

// TestTextExactHitSkipsPrefill covers a text prompt whose whole prompt
// matches a retained slot. Nothing is left to prefill, so the slot's stored
// hidden state is the only way to get the last position's logits — the same
// mechanism image prompts use.
//
// Asserts both halves: the answer must not change, and the work must
// actually go away. Answer equality alone would pass if the hit silently
// fell back to a full prefill.
func TestTextExactHitSkipsPrefill(t *testing.T) {
	dir := os.Getenv("SINTER_VISION_MODEL")
	if dir == "" {
		t.Skip("set SINTER_VISION_MODEL to a qwen3.5 VL model directory")
	}
	m, err := llm.NewModel(dir)
	if err != nil {
		t.Skipf("load model: %v", err)
	}
	defer m.Close()

	prompt := "<|im_start|>user\nName three primary colours.<|im_end|>\n<|im_start|>assistant\n"
	run := func() (string, int64) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		gen := llm.DefaultGenerateConfig()
		gen.MaxTokens = 12
		gen.Temperature = 0
		var ids []int
		if err := m.Generate(ctx, prompt, gen, func(id int) { ids = append(ids, id) }); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return m.DecodeTokenIDs(ids), m.LastPromptMs()
	}

	first, coldMs := run()
	second, warmMs := run()
	t.Logf("prefill: cold=%dms warm=%dms", coldMs, warmMs)

	if first != second {
		t.Fatalf("cache hit changed the answer:\n first: %q\n again: %q", first, second)
	}
	if warmMs == 0 {
		t.Fatalf("warm prefill reported 0ms (cold %dms); the timer did not run", coldMs)
	}
	if warmMs*4 > coldMs {
		t.Fatalf("warm prefill %dms is not meaningfully below cold %dms: the cache hit did not skip the prefill", warmMs, coldMs)
	}
}

// TestTextMultiTurnStillDeltas guards the other side of the exact-hit
// branch: a growing conversation must keep taking the delta path. If the
// exact-match branch swallowed it, every turn would return the first turn's
// answer.
func TestTextMultiTurnStillDeltas(t *testing.T) {
	dir := os.Getenv("SINTER_VISION_MODEL")
	if dir == "" {
		t.Skip("set SINTER_VISION_MODEL to a qwen3.5 VL model directory")
	}
	m, err := llm.NewModel(dir)
	if err != nil {
		t.Skipf("load model: %v", err)
	}
	defer m.Close()

	base := "<|im_start|>user\nName three primary colours.<|im_end|>\n<|im_start|>assistant\n"
	turns := []string{
		base,
		base + "The three primary colors are Red, Green, and Blue.<|im_end|>\n<|im_start|>user\nNow name three secondary colours.<|im_end|>\n<|im_start|>assistant\n",
		base + "The three primary colors are Red, Green, and Blue.<|im_end|>\n<|im_start|>user\nNow name three secondary colours.<|im_end|>\n<|im_start|>assistant\nSecondary colors are Orange, Green, and Purple.<|im_end|>\n<|im_start|>user\nAnd one tertiary?<|im_end|>\n<|im_start|>assistant\n",
	}
	var answers []string
	for i, p := range turns {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		gen := llm.DefaultGenerateConfig()
		gen.MaxTokens = 12
		gen.Temperature = 0
		var ids []int
		err := m.Generate(ctx, p, gen, func(id int) { ids = append(ids, id) })
		cancel()
		if err != nil {
			t.Fatalf("turn %d: %v", i, err)
		}
		answers = append(answers, m.DecodeTokenIDs(ids))
		t.Logf("turn %d: %q", i, answers[i])
	}

	// Each turn asks something different, so identical answers would mean a
	// stale prefix was served.
	if answers[0] == answers[1] && answers[1] == answers[2] {
		t.Fatalf("all turns identical, cache likely served a stale prefix: %q", answers[0])
	}
	if answers[0] == answers[1] {
		t.Fatalf("turn 1 repeated turn 0's answer: %q", answers[1])
	}
}
