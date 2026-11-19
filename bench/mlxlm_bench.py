"""Apples-to-apples decode benchmark against sinter's TestScratchDecodePaths.

Same prompts, same 64-token budget, greedy. Decode time is measured from the
first streamed token to the last, so prefill is excluded on both sides.
"""

import sys
import time

from mlx_lm import load, stream_generate

MODEL = sys.argv[1]
REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 3

SHORT = (
    "<|im_start|>user\nWrite a paragraph about the sea.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
LONG = (
    "<|im_start|>system\n"
    + "You are a careful software engineering agent. Read before you edit, "
    "keep changes small, verify with tests. " * 120
    + "<|im_end|>\n"
    + SHORT
)

print(f"loading {MODEL}", flush=True)
model, tokenizer = load(MODEL)

for label, prompt in (("short", SHORT), ("long", LONG)):
    for rep in range(REPEAT):
        n = 0
        ids = []
        t_first = None
        t_last = None
        for resp in stream_generate(
            model, tokenizer, prompt, max_tokens=64, sampler=None
        ):
            if t_first is None:
                t_first = time.perf_counter()
            t_last = time.perf_counter()
            n += 1
            ids.append(resp.token)
            if resp.finish_reason == "stop":
                break
        decode = t_last - t_first
        # n-1 intervals between n tokens, matching sinter's (n-1)/decode.
        tps = (n - 1) / decode if decode > 0 and n > 1 else 0.0
        print(
            f"{label:6s} rep={rep} tokens={n} decode={decode * 1000:7.1f}ms "
            f"{tps:6.1f} tok/s ids={ids}",
            flush=True,
        )
