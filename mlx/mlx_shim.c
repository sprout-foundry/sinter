//go:build darwin && arm64 && cgo

// Runtime loader for the MLX C library.
//
// The package no longer links -lmlx/-lmlxc: the generated wrappers
// (mlx_shim_dispatch_gen.c) forward every entry point through mlx_dispatch,
// which this file populates via dlopen/dlsym on first use. Without the
// library, every entry point degrades (zero handles / rc=1) and
// Available() reports false, so the binary runs on Macs that lack the
// mlx-c Homebrew formula.
//
// Candidate paths, in order:
//   1. $MLX_C_LIB — explicit path to libmlxc.dylib (power users, vendors)
//   2. <executable dir>/lib/libmlxc.dylib and <executable dir>/libmlxc.dylib
//      — self-contained bundles (ship the dylib next to the binary)
//   3. /opt/homebrew/opt/mlx-c/lib/libmlxc.dylib — Apple-Silicon Homebrew
//   4. /usr/local/opt/mlx-c/lib/libmlxc.dylib — Intel-prefix Homebrew
//
// libmlxc's own dependency on libmlx (the C++ core) loads transitively via
// dyld. If a candidate loads but is missing entry points (version skew),
// it is dropped and the next candidate is tried.
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include "mlx_shim.h"

// Defined in mlx_error_shim.c; forwards to the Go error handler so the
// not-loaded message surfaces via lastMLXError() in checkRC.
void mlx_go_error_handler_shim(const char* msg, void* data);

mlx_shim_dispatch_t mlx_dispatch;

static pthread_once_t mlx_shim_once = PTHREAD_ONCE_INIT;
static int mlx_shim_ready_flag;
static int mlx_shim_notloaded_reported;

static void mlx_shim_init_once(void) {
  const char* candidates[5] = {
      getenv("MLX_C_LIB"),
      NULL,
      NULL,
      "/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib",
      "/usr/local/opt/mlx-c/lib/libmlxc.dylib",
  };

#if defined(__APPLE__)
  char exe_dir_candidate1[1200];
  char exe_dir_candidate2[1200];
  char exe_path[1024];
  uint32_t size = sizeof exe_path;
  if (_NSGetExecutablePath(exe_path, &size) == 0) {
    char* slash = strrchr(exe_path, '/');
    if (slash != NULL) {
      *slash = '\0';
      snprintf(exe_dir_candidate1, sizeof exe_dir_candidate1,
               "%s/lib/libmlxc.dylib", exe_path);
      snprintf(exe_dir_candidate2, sizeof exe_dir_candidate2,
               "%s/libmlxc.dylib", exe_path);
      candidates[1] = exe_dir_candidate1;
      candidates[2] = exe_dir_candidate2;
    }
  }
#endif

  for (int i = 0; i < 5; i++) {
    const char* path = candidates[i];
    if (path == NULL || path[0] == '\0') {
      continue;
    }
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
      continue;
    }
    if (mlx_shim_resolve_dispatch(handle, &mlx_dispatch)) {
      mlx_shim_ready_flag = 1;
      return;
    }
    dlclose(handle);
  }
}

int mlx_shim_ready(void) {
  pthread_once(&mlx_shim_once, mlx_shim_init_once);
  return mlx_shim_ready_flag;
}

void mlx_shim_report_notloaded(void) {
  if (mlx_shim_notloaded_reported) {
    return;
  }
  mlx_shim_notloaded_reported = 1;
  mlx_go_error_handler_shim(
      "mlx-c runtime library (libmlxc.dylib) not found; install with "
      "'brew install mlx-c' or point MLX_C_LIB at it",
      NULL);
}
