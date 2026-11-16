//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package qwen35

import (
	"encoding/json"
	"fmt"
	"os"
)

// visionConfig is the `vision_config` section of a qwen3_5 wrapper config.
// Only the fields the tower reads are decoded; the rest are ignored.
type visionConfig struct {
	Depth                 int     `json:"depth"`
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumHeads              int     `json:"num_heads"`
	InChannels            int     `json:"in_channels"`
	PatchSize             int     `json:"patch_size"`
	TemporalPatchSize     int     `json:"temporal_patch_size"`
	SpatialMergeSize      int     `json:"spatial_merge_size"`
	NumPositionEmbeddings int     `json:"num_position_embeddings"`
	OutHiddenSize         int     `json:"out_hidden_size"`
	HiddenAct             string  `json:"hidden_act"`
	DeepstackVisualIdx    []int   `json:"deepstack_visual_indexes"`
	LayerNormEps          float64 `json:"layer_norm_eps"`
}

// HeadDim is the per-head width, derived the way the reference does.
func (c visionConfig) HeadDim() int { return c.HiddenSize / c.NumHeads }

// loadVisionConfig reads vision_config from the model's config.json. It
// returns nil (no error) when the checkpoint has no vision tower, so a
// text-only qwen3_5 still loads.
func loadVisionConfig(configPath string) (*visionConfig, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var wrapper struct {
		VisionConfig *visionConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(data, &wrapper); err != nil {
		return nil, fmt.Errorf("parse vision_config: %w", err)
	}
	vc := wrapper.VisionConfig
	if vc == nil || vc.Depth == 0 {
		return nil, nil
	}
	if vc.LayerNormEps == 0 {
		vc.LayerNormEps = 1e-6
	}
	if vc.HiddenAct == "" {
		vc.HiddenAct = "gelu_pytorch_tanh"
	}
	if vc.NumPositionEmbeddings == 0 {
		return nil, fmt.Errorf("vision_config: num_position_embeddings missing")
	}
	if vc.OutHiddenSize == 0 {
		return nil, fmt.Errorf("vision_config: out_hidden_size missing")
	}
	if len(vc.DeepstackVisualIdx) > 0 {
		return nil, fmt.Errorf("vision_config: deepstack_visual_indexes %v is not supported", vc.DeepstackVisualIdx)
	}
	return vc, nil
}
