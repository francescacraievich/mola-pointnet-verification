# PointNet Verification Report

> Formal Verification of PointNet Neural Networks using α,β-CROWN

---

## Abstract

This report presents a formal verification analysis for PointNet neural networks applied to point cloud classification. We use α,β-CROWN to verify the model's robustness against input perturbations in safety-critical autonomous driving scenarios.

---

## 1. Introduction

### 1.1 Background

PointNet neural networks are widely used for 3D point cloud processing. Formal verification of these networks is essential for safety-critical applications such as autonomous driving, where incorrect classifications can have severe consequences.

### 1.2 Objectives

- Verify local robustness of the PointNet model
- Analyze safety properties of the network
- Quantify admissible perturbation bounds

---

## 2. Methodology

### 2.1 PointNet Architecture

```python
# Simplified architecture for verification
class PointNetForVerification(nn.Module):
    def __init__(self, n_points=512, num_classes=2):
        # MLP: 3 -> 64 -> 128 -> 256
        # Global MaxPool
        # Classifier: 256 -> 128 -> 64 -> num_classes
```

### 2.2 α,β-CROWN

α,β-CROWN is a state-of-the-art neural network verification framework that uses:

- **Linear Relaxation**: linear relaxation of non-linear constraints
- **Branch and Bound**: to refine bounds iteratively
- **GPU Acceleration**: to scale to larger networks

### 2.3 Verification Configuration

| Parameter | Value |
|-----------|-------|
| Epsilon (ε) | 0.01 |
| Number of points | 512 |
| Timeout | 300s |
| Method | α,β-CROWN |

---

## 3. Dataset

### 3.1 Data Preparation

Data is extracted from LiDAR point clouds and preprocessed as follows:

1. **Normalization**: centering and scaling
2. **Sampling**: selection of 512 points per group
3. **Feature extraction**: coordinates (x, y, z)

### 3.2 Classes

- **Class 0**: Non-critical regions
- **Class 1**: Critical regions (obstacles, pedestrians, etc.)

---

## 4. Results

### 4.1 Model Accuracy

| Metric | Value |
|--------|-------|
| Training Accuracy | XX.X% |
| Test Accuracy | XX.X% |
| Final Loss | X.XXX |

### 4.2 Verification Results

<!-- TODO: insert actual results -->

```
Verification Results:
- Verified: XX samples
- Falsified: XX samples
- Unknown: XX samples
- Verification Rate: XX.X%
```

### 4.3 Bound Analysis

<!-- TODO: insert plots -->

---

## 5. Discussion

### 5.1 Interpretation of Results

The results show that...

### 5.2 Limitations

- Scalability to larger point clouds
- MaxPool handling in verification
- Trade-off between precision and verification time

---

## 6. Conclusions

This work demonstrated the feasibility of formal verification for PointNet networks. The results indicate that...

### 6.1 Future Work

- Extension to PointNet++
- Verification of global properties
- Integration with autonomous driving pipelines

---

## References

1. Qi, C. R., et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation." CVPR 2017.
2. Wang, S., et al. "Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Neural Network Robustness Verification." NeurIPS 2021.
3. Zhang, H., et al. "General Cutting Planes for Bound-Propagation-Based Neural Network Verification." NeurIPS 2022.

---

<p align="center">
  <i>Report generated for the mola-pointnet-verification project</i>
</p>
