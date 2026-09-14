# sinter

Local LLM inference engine for Go — pure Go (with cgo shims to MLX/GGML),
running Qwen and Gemma-class models fully on-device on Apple Silicon
(Metal) and Linux (GGML: Metal/CUDA/Vulkan/CPU).

Extracted from sprout and auto-term so both (and future projects) share
one engine instead of parallel forks. Auto-term's qwen2 architecture and
its GGML portability fixes are merged in; sprout's newer archs (qwen3,
qwen3.5 dense + MoE, gemma4, lfm2), MTP, prefix-cache, and delta-prefill
work live here now.

## Usage

```go
import (
    "github.com/sprout-foundry/sinter/llm"
    _ "github.com/sprout-foundry/sinter/llm/all" // registers all archs + backends
)

m, err := llm.NewModel(modelDir)
defer m.Close()

txt, err := m.GenerateText(ctx, prompt, llm.DefaultGenerateConfig())
```

For lean binaries, blank-import only what you need — e.g. just
`llm/qwen2` + `mlx` for a Qwen2.5-Coder tool on macOS.

**One GPU thread per process.** All inference is pinned to a single OS
thread per Model instance (the backends require it). Multiple Model
instances in one process work, but each runs its own GPU thread and
competes for memory — prefer one Model at a time. Documented here since
it's easy to trip over when embedding sinter in a larger app.

Model catalog with RAM tiers and auto-selection lives in `llm/catalog`
(separate from the engine so apps can keep their own list).

## Environment variables

| Var | Purpose |
|---|---|
| `SINTER_BACKEND` | Force a backend by logical name (`metal`, `ggml`) |
| `SINTER_GGML_CPU` | Force GGML to CPU device (testing on any platform) |
| `SINTER_ALLOW_OVERWEIGHT` | Skip the RAM gate for power users |
| `SINTER_PREFIX_CACHE_MAX` | Override prefix-cache slot sizing |
| `SINTER_PIPELINE_DECODE` | Pipelined decode: default ON (GGML fixed; MLX parity bug still open). `=0` opts out |
| `SINTER_LOCAL_DEBUG` / `SINTER_GEN_MEM` | Engine + generation memory debug logging |
| `SINTER_COMPILED_DECODE`, `SINTER_MTP_*` | Experimental decode-path opt-ins (parity-tested, MLX only) |
| `SINTER_GGML_BATCH=0` | Disable GGML batched eval (on by default; restores eager per-op eval) |

## Development

```bash
go build -tags ggml ./...   # GGML platforms (Linux, Termux/Android): needs libggml + libggml-base
go build ./...              # macOS (Apple Silicon): needs mlx-c (brew); stubs compile elsewhere
go test -tags ggml ./...    # or plain `go test ./...` on macOS
```

Live-model tests (weights on disk) skip when the model is absent.

### Termux (Android arm64)

GOOS is `android` there, which every `linux`-gated file in this module accepts.

1. `pkg install go clang` and `pkg install llama-cpp` for the ggml libs
   (or point cgo at your own build — see step 3).
2. `go build -tags ggml ./...` — Termux's clang finds ggml in `$PREFIX/lib`.
3. Termux's packaged libggml-cpu is built for baseline armv8-a, which halves
   quantized matmul throughput on dotprod/i8mm CPUs. Test:
   `go test -tags ggml -run TestCPUQuantFastPaths ./tensor/ggml`. To fix,
   build ggml with `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm`,
   install to a private prefix, and build with
   `CGO_CFLAGS=-I<prefix>/include CGO_LDFLAGS="-L<prefix>/lib -Wl,-rpath,<prefix>/lib"`.

## Example / e2e smoke test

[`examples/chat`](examples/chat) is a minimal implementor-facing example and
the module's end-to-end gate:

```bash
go run ./examples/chat -model ~/.cache/sprout/models/qwen3-0.6b
```

It loads a model, streams tokens, and prints tok/s. The e2e test builds the
example and drives it against any model found in ~/.cache/sprout/models,
~/.local/share/sprout/models, or ~/.auto-term/models (set SINTER_E2E_MODEL to
pin one; thinking models like qwen3 need the empty-think prefix, which the
test adds automatically). CI should run it after a model-fetch step.

MIT — see sprout-foundry org for the license file before publishing.
