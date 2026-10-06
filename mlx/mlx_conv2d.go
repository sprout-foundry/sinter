//go:build darwin && arm64 && cgo

// Package mlx: Conv2D lives in its own file to keep the cgo import "C"
// beside the ops that use it. See mlx_cgo.go for the cgo flags (applied
// package-wide) and mlx_shim.c for the runtime dlopen.
package mlx

/*
#include <mlx/c/array.h>
#include <mlx/c/ops.h>
#include <mlx/c/stream.h>
*/
import "C"

// Conv2D applies a 2D convolution. input is [B, H, W, C_in] (MLX NHWC);
// weight is [C_out, kH, kW, C_in/groups]. stride/padding/dilation apply to
// both spatial axes (the C API takes per-axis values; the Go surface mirrors
// Conv1D's single int per parameter). groups is C_in/groups ("groups" in the
// torch sense: weight's last dim is C_in/groups).
func Conv2D(input, weight *Array, stride, padding, dilation, groups int, s *Stream) (*Array, error) {
	out := newOutput()
	rc := C.mlx_conv2d(&out, input.cHandle(), weight.cHandle(),
		C.int(stride), C.int(stride), C.int(padding), C.int(padding),
		C.int(dilation), C.int(dilation), C.int(groups), s.cHandle())
	return wrapResult(out, rc, "conv2d")
}
