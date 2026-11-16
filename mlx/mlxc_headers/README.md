Vendored headers: ml-explore/mlx-c v0.7.0 (MIT, © 2023-2024 Apple Inc.),
mlx/c/*.h — copied 2026-11-16 from tag v0.7.0
(https://github.com/ml-explore/mlx-c/archive/refs/tags/v0.7.0.tar.gz).

Headers only; no mlx-c sources are compiled here. sinter/mlx dlopens
libmlxc.dylib at runtime (see mlx_shim.c) instead of linking it, so these
headers replace the old `-I/opt/homebrew/include` build dependency.

The headers must match the installed libmlxc's major/minor, not just its API
family: mlx-c passes structs and enums by value across the boundary, so a
stale header silently misreads arguments rather than failing to link. v0.7.0
inserted a `bool force_fused` before the stream in
mlx_fast_scaled_dot_product_attention; under the v0.6.0 header the stream
handle landed in that slot and the real stream was read from garbage, which
surfaced as "no fused kernel is available: the fused kernels require a GPU
(Metal) stream" on every MLX model path.

To refresh: re-copy from the matching mlx-c tag and re-run
scripts/gen_mlxc_shim.py.
