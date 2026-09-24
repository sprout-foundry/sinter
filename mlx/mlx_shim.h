//go:build darwin && arm64 && cgo

// Runtime dispatch state for the MLX C API.
//
// sinter/mlx does not link libmlxc.dylib at build time. The entry points are
// resolved at first use by mlx_shim.c (dlopen + dlsym) into mlx_dispatch;
// the generated wrappers in mlx_shim_dispatch_gen.c forward through it and
// degrade to "not loaded" results when the library is absent, so the binary
// builds and runs on Macs without the mlx-c Homebrew formula.
#ifndef MLX_SHIM_H
#define MLX_SHIM_H

#include "mlx_shim_dispatch_gen.h"

// Filled in by mlx_shim.c once the library loads (all-zero otherwise).
extern mlx_shim_dispatch_t mlx_dispatch;

// dlsym-resolves every entry point from handle into d.
// Defined in mlx_shim_resolve_gen.c; returns 1 on success, 0 if any
// entry point is missing (caller should drop the library).
int mlx_shim_resolve_dispatch(void* handle, mlx_shim_dispatch_t* d);

// Returns 1 if the MLX C library is loaded and all entry points resolved.
// Safe to call from any thread; dlopen happens once.
int mlx_shim_ready(void);

// Records a "library not loaded" message in the Go error handler (once per
// process) so checkRC surfaces an actionable error instead of rc=1.
void mlx_shim_report_notloaded(void);

#endif
