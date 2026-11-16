//go:build darwin && arm64 && cgo

package mlx

/*
#include <mlx/c/stream.h>
#include <mlx/c/device.h>
*/
import "C"

import "fmt"

// DeviceType reports whether a device handle is the CPU or the GPU.
type DeviceType int

const (
	DeviceCPU DeviceType = DeviceType(C.MLX_CPU)
	DeviceGPU DeviceType = DeviceType(C.MLX_GPU)
)

func (d DeviceType) String() string {
	switch d {
	case DeviceCPU:
		return "cpu"
	case DeviceGPU:
		return "gpu"
	}
	return fmt.Sprintf("device(%d)", int(d))
}

// StreamDeviceType reports the device type a stream is bound to. The fused
// attention kernel requires a GPU stream, and its error message does not say
// which device it saw, so this is the only way to tell a genuinely-Metal
// stream apart from one that merely looks available.
func StreamDeviceType(s *Stream) (DeviceType, error) {
	var dev C.mlx_device
	if rc := C.mlx_stream_get_device(&dev, s.cHandle()); rc != 0 || dev.ctx == nil {
		return 0, fmt.Errorf("mlx: stream get_device failed (rc=%d)", rc)
	}
	defer C.mlx_device_free(dev)
	var t C.mlx_device_type
	if rc := C.mlx_device_get_type(&t, dev); rc != 0 {
		return 0, fmt.Errorf("mlx: device get_type failed (rc=%d)", rc)
	}
	return DeviceType(t), nil
}

// DefaultDeviceType reports the device MLX considers default. Streams created
// without an explicit device are bound to it, so a CPU default means a
// "GPU stream" request produces a CPU stream.
func DefaultDeviceType() (DeviceType, error) {
	var dev C.mlx_device
	if rc := C.mlx_get_default_device(&dev); rc != 0 || dev.ctx == nil {
		return 0, fmt.Errorf("mlx: get default device failed (rc=%d)", rc)
	}
	defer C.mlx_device_free(dev)
	var t C.mlx_device_type
	if rc := C.mlx_device_get_type(&t, dev); rc != 0 {
		return 0, fmt.Errorf("mlx: device get_type failed (rc=%d)", rc)
	}
	return DeviceType(t), nil
}
