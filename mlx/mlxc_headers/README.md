Vendored headers: ml-explore/mlx-c v0.6.0 (MIT, © 2023-2024 Apple Inc.),
mlx/c/*.h — copied 2026-09-24 from tag v0.6.0
(https://github.com/ml-explore/mlx-c/archive/refs/tags/v0.6.0.tar.gz).

Headers only; no mlx-c sources are compiled here. sinter/mlx dlopens
libmlxc.dylib at runtime (see mlx_shim.c) instead of linking it, so these
headers replace the old `-I/opt/homebrew/include` build dependency.

To refresh: re-copy from the matching mlx-c tag and re-run
scripts/gen_mlxc_shim.py.
