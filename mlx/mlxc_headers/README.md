Vendored headers: ml-explore/mlx-c v0.7.0 (MIT, © 2023-2026 Apple Inc.),
mlx/c/*.h — copied 2026-10-09 from the Homebrew mlx-c 0.7.0 install
(/opt/homebrew/opt/mlx-c/include, built from tag v0.7.0;
https://github.com/ml-explore/mlx-c/archive/refs/tags/v0.7.0.tar.gz).

Headers only; no mlx-c sources are compiled here. sinter/mlx dlopens
libmlxc.dylib at runtime (see mlx_shim.c) instead of linking it, so these
headers replace the old `-I/opt/homebrew/include` build dependency.

v0.7.0 deltas that matter to callers (the 0.6.0-era ones):
- mlx_fast_scaled_dot_product_attention gained `bool force_fused` just
  before the stream. Callers must pass false (let MLX pick the
  implementation) to preserve pre-0.7.0 behavior; a stale 9-arg call
  against the 0.7.0 libmlxc corrupts the stream register (asserts
  "expected a non-empty mlx_stream").
- mlx_fast_cross_entropy is new (not referenced by sinter).
- Ops/stream/vector/fft/device headers carry further additive changes;
  the dispatch shim is regenerated from these headers.

To refresh: re-copy from the matching mlx-c tag and re-run
scripts/gen_mlxc_shim.py.
