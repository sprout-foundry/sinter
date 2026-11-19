"""Apples-to-apples: identical PROMPT LENGTHS in both engines.

Corrects mlxlm_tokenexact.py, whose 'long' entry fed mlx-lm a 62-token
prompt (the generated ids) while sinter's 'long' prompt was 2541 tokens —
so the old table's 'mlx-lm is flat across context' was an artifact.
"""
import sys, time
from mlx_lm import load, stream_generate

MODEL = sys.argv[1]
model, tokenizer = load(MODEL)

sys_prefix = "You are a careful software engineering agent. Read before you edit, keep changes small, verify with tests. "
prompts = {
    "short": tokenizer.encode("<|im_start|>user\nWrite a paragraph about the sea.<|im_end|>\n<|im_start|>assistant\n"),
    "long": tokenizer.encode("<|im_start|>system\n" + sys_prefix * 120 + "<|im_end|>\n<|im_start|>user\nWrite a paragraph about the sea.<|im_end|>\n<|im_start|>assistant\n"),
}
print(f"prompt sizes: short={len(prompts['short'])} long={len(prompts['long'])}", flush=True)

best = {}
for label, ids in prompts.items():
    for rep in range(3):
        n = 0
        t_first = t_last = None
        for resp in stream_generate(model, tokenizer, ids, max_tokens=64, sampler=None):
            if t_first is None: t_first = time.perf_counter()
            t_last = time.perf_counter()
            n += 1
        dec = t_last - t_first
        tps = (n - 1) / dec
        best[label] = max(best.get(label, 0), tps)
        print(f"{label:6s} rep={rep} prompt={len(ids):5d} n={n} decode={dec*1000:7.1f}ms {tps:6.1f} tok/s", flush=True)

print()
print(f"mlx-lm best: short={best['short']:.1f} long={best['long']:.1f} -> long loses {100*(best['short']-best['long'])/best['short']:.1f}%")
