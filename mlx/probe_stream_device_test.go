//go:build darwin && arm64 && cgo

package mlx

import (
	"runtime"
	"testing"
)

// TestProbeStreamDevice reports what device the GPU streams are actually on.
// The fused-attention error claims the stream is not Metal, which is worth
// checking directly rather than inferring from the error text.
func TestProbeStreamDevice(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if dt, err := DefaultDeviceType(); err != nil {
		t.Logf("default device: %v", err)
	} else {
		t.Logf("default device: %s", dt)
	}

	for name, get := range map[string]func() (*Stream, error){
		"DefaultGPUStream": DefaultGPUStream,
		"NewGPUStream":     NewGPUStream,
	} {
		s, err := get()
		if err != nil {
			t.Logf("%s: %v", name, err)
			continue
		}
		dt, err := StreamDeviceType(s)
		if err != nil {
			t.Logf("%s: %v", name, err)
			continue
		}
		t.Logf("%s: device=%s", name, dt)
	}
}
