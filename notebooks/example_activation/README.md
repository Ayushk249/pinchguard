# Example activation bundle (Qwen3-32B, layer 32)

One residual-stream activation captured via the production `HFCapturer`,
for validating the Assistant-Axis probe before the full M4 run exists.

## Facts
- **model_id**: `/datapool/analysis_data/tara/pinchguard/models`
- **layer**: `32`
- **hook**: `module-output of model.model.layers[32]`
- **token_position**: `last_input`
- **hidden_dim**: `5120`
- **activation_key**: `L32`
- **dtype**: `float16`
- **quantization**: `nf4`
- **capture_runtime**: `hf-eager`
- **step_id**: `2e03ddc8-eee6-4900-a3b2-eeb471d2698f`
- **npz**: `2e03ddc8-eee6-4900-a3b2-eeb471d2698f.npz`

## Load it
```python
import numpy as np
a = np.load("2e03ddc8-eee6-4900-a3b2-eeb471d2698f.npz")["L32"]  # shape (1, 5120), float16
```

## Open cross-check with Lion
`L32` here is the **module-output of decoder block index 32**
(`model.model.layers[32]`). Confirm the probe extracts the axis at the
same definition (vs `hidden_states[32]`, which is the *input* to block
32 / output of block 31) before trusting a projection.
