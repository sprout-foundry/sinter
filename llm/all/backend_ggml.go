//go:build ((linux && (arm64 || amd64)) || android) && cgo && ggml

package all

// The GGML backend registers itself via init(). Kept in a separate file so
// importing it here never breaks darwin builds, where the package is gated
// out by the ggml build tag (see tensor/ggml/backend.go).
//
// GOOS is "android" on Termux; every linux-gated file in this module also
// accepts android, so the backend registration matches that here.
import (
	_ "github.com/sprout-foundry/sinter/tensor/ggml"
)
