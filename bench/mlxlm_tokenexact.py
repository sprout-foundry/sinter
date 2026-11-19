"""Token-exact decode comparison against sinter.

Feeds mlx-lm the exact token ids sinter used, so both engines decode the same
sequence. Decode time is measured from the first generated token to the last,
excluding prefill, matching sinter's (n-1)/decode accounting.
"""

import sys
import time

from mlx_lm import load, stream_generate

MODEL = sys.argv[1]
REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 3

# Token ids sinter reported for each prompt (TestScratchDecodePaths).
PROMPTS = {
    "short": [
        198, 90700, 8340, 25, 271, 16, 13, 220, 2972, 2014, 53983, 279, 5952,
        64700, 198, 262, 348, 256, 31819, 25, 561, 9117, 13, 198, 262, 348,
        256, 14606, 25, 357, 13901, 13, 198, 262, 348, 256, 38622, 25, 9357,
        264, 51067, 14, 33756, 520, 6365, 883, 279, 9117, 13, 271, 17, 13, 220,
        2972, 35, 23544, 279, 64644, 321, 24985, 64700, 198, 262,
    ],
    "long": [
        271, 271, 760, 9117, 369, 264, 12401, 11, 8476, 495, 91159, 421, 47929,
        76045, 310, 279, 32946, 11, 1141, 7099, 72791, 286, 1172, 279, 6797,
        579, 20117, 42655, 466, 6121, 5830, 1083, 264, 5365, 11, 24577, 6105,
        1172, 279, 17257, 4009, 13, 1049, 11155, 287, 2199, 1083, 26706, 58801,
        11, 979, 71306, 440, 31883, 50255, 88559, 11, 74715, 54876, 11, 321, 279,
    ],
}

print(f"loading {MODEL}", flush=True)
model, tokenizer = load(MODEL)

for label, ids in PROMPTS.items():
    for rep in range(REPEAT):
        n = 0
        t_first = t_last = None
        for resp in stream_generate(
            model, tokenizer, ids, max_tokens=64, sampler=None
        ):
            if t_first is None:
                t_first = time.perf_counter()
            t_last = time.perf_counter()
            n += 1
            if resp.finish_reason == "stop":
                break
        decode = t_last - t_first
        tps = (n - 1) / decode if decode > 0 and n > 1 else 0.0
        print(
            f"{label:6s} rep={rep} prompt_tokens={len(ids)} out_tokens={n} "
            f"decode={decode * 1000:7.1f}ms {tps:6.1f} tok/s",
            flush=True,
        )
