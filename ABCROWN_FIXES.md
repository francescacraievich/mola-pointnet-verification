# α,β-CROWN Fixes for PointNet Verification

This document describes the fixes applied to α,β-CROWN to enable MaxPool1d verification.

## Problem

α,β-CROWN's auto_LiRPA library doesn't support MaxPool1d (1D max pooling) out of the box.
The ONNX export of PyTorch's MaxPool1d creates a `pads` attribute with 1 element instead of 4,
causing IndexError in the verification process.

## Fixes Applied

### 1. Fix onnx2pytorch compatibility (`load_model.py:203-210`)

**File**: `alpha-beta-CROWN/complete_verifier/load_model.py`

**Issue**: The `quirks` parameter is not supported in newer versions of onnx2pytorch.

**Fix**:
```python
try:
    # Fix: onnx2pytorch in newer versions doesn't support 'quirks' parameter
    try:
        pytorch_model = onnx2pytorch.ConvertModel(
            onnx_model, experimental=True, quirks=quirks)
    except TypeError:
        # Fallback without quirks parameter
        pytorch_model = onnx2pytorch.ConvertModel(
            onnx_model, experimental=True)
except TypeError as e:
    # ... existing error handling ...
```

### 2. Fix MaxPool1d padding support (`pooling.py:29-40`)

**File**: `alpha-beta-CROWN/complete_verifier/auto_LiRPA/operators/pooling.py`

**Issue**: MaxPool1d in ONNX has `pads=[0]` (1 element) but auto_LiRPA expects `pads=[0,0,0,0]` (4 elements for 2D).

**Fix**:
```python
def __init__(self, attr=None, inputs=None, output_index=0, options=None):
    super().__init__(attr, inputs, output_index, options)
    # Fix for MaxPool1d: pads may have length 1 instead of 4 (for 2D pooling)
    assert ('pads' not in attr) or len(attr['pads']) <= 1 or (attr['pads'][0] == attr['pads'][2])
    assert ('pads' not in attr) or len(attr['pads']) <= 1 or (attr['pads'][1] == attr['pads'][3])

    self.requires_input_bounds = [0]
    self.kernel_size = attr['kernel_shape']
    self.stride = attr['strides']
    # Fix for MaxPool1d: pads may have only 1 element
    if 'pads' in attr and len(attr['pads']) == 1:
        self.padding = [attr['pads'][0], 0]  # 1D pooling, no padding in second dim
    else:
        self.padding = [attr['pads'][0], attr['pads'][1]] if 'pads' in attr else [0, 0]
```

## Verification Script Changes

### ONNX Export (`verify_with_abcrown_nsga3.py:159`)

**Change**: Use ONNX opset 11 instead of 17 for better compatibility.

```python
torch.onnx.export(
    model,
    dummy_input,
    str(output_path),
    export_params=True,
    opset_version=11,  # Opset 11 for onnx2pytorch compatibility (was 17)
    ...
)
```

**Reason**: Opset 17 includes `training_mode` attribute in BatchNorm which is not supported by onnx2pytorch.

## Testing

After applying these fixes, verification works correctly:

```bash
python scripts/verify_with_abcrown_nsga3.py --n-samples 5 --epsilon 0.005
# Result: 5/5 samples VERIFIED (100.0%)
```

## Installation Note

If you clone this repository and want to use α,β-CROWN verification:

1. Clone α,β-CROWN into the project directory
2. Apply the fixes above manually, OR
3. The fixes will be automatically applied when you run the verification script for the first time (future work)

## References

- α,β-CROWN: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- Issue with MaxPool1d: https://github.com/Verified-Intelligence/auto_LiRPA/issues/XX
