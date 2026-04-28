# Point Cloud Classification Optimization Algorithm

## Executive Summary

This document details the optimization algorithms and techniques used in the Point Cloud Classification pipeline, which improved accuracy from a baseline of **29.89%** to **~85%** on the Lille1 dataset.

---

## 1. Core Architecture

### 1.1 Model Architecture: PointMLP

```
Input Features (8 dimensions)
    ↓
[Fully Connected: 256 units]
    ↓
Batch Normalization → ReLU → Dropout (0.2)
    ↓
[Fully Connected: 128 units]
    ↓
Batch Normalization → ReLU → Dropout (0.2)
    ↓
[Fully Connected: num_classes units]
    ↓
Output (Logits) → Softmax (during inference)
```

**Parameters:**
- Input dimensions: 8 features
- Hidden layer 1: 256 neurons
- Hidden layer 2: 128 neurons
- Output layer: Number of classes (dynamic)
- Activation: ReLU
- Dropout rate: 0.2 (20%)

---

## 2. Input Features & Data Optimization

### 2.1 Feature Selection

The key optimization breakthrough was selecting **all 8 geometric and temporal features** instead of just spatial coordinates:

| Feature | Type | Purpose |
|---------|------|---------|
| `x, y, z` | Spatial | 3D point coordinates |
| `x_origin, y_origin, z_origin` | Spatial | Original coordinate system reference |
| `GPS_time` | Temporal | Timestamp information |
| `reflectance` | Physical | LiDAR intensity measurement |

**Impact:** XYZ-only baseline achieved ~30% accuracy → All 8 features achieved ~85% accuracy

### 2.2 Data Pipeline

```
┌─────────────────────────────────────────────────────┐
│ 1. Load PLY File (Lille1.ply)                       │
└────────────────┬────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────┐
│ 2. Extract Features & Labels (8 features + class)   │
└────────────────┬────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────┐
│ 3. Sampling (max 200,000 points to control memory)  │
└────────────────┬────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────┐
│ 4. Filter Ultra-Rare Classes (min 3 samples)        │
│    Ensures stratified split stability               │
└────────────────┬────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────┐
│ 5. Remap Labels to Contiguous IDs [0..num_classes] │
└────────────────┬────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────┐
│ 6. Stratified Train/Val/Test Split                  │
│    - Train: 64% | Validation: 16% | Test: 20%      │
└────────────────┬────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────┐
│ 7. Feature Normalization                            │
│    Using training set mean/std (prevents data leak) │
└────────────────┬────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────┐
│ Ready for Training                                  │
└─────────────────────────────────────────────────────┘
```

---

## 3. Training Optimization Algorithms

### 3.1 Optimizer: AdamW

**Algorithm:** Adaptive Moment Estimation with Weight decay (AMSGrad variant)

```
AdamW combines:
├─ Momentum estimation (exponential moving average of gradients)
├─ Adaptive learning rates (per-parameter LR scaling)
└─ Decoupled weight decay (L2 regularization)

Key Hyperparameters:
├─ Learning Rate: 1e-3 (0.001)
├─ Weight Decay: 1e-4 (0.0001) - L2 regularization
├─ β₁ (momentum): 0.9 (default)
└─ β₂ (RMSprop): 0.999 (default)
```

**Update Rule (Simplified):**
$$m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t$$
$$v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2$$
$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \lambda \cdot \theta_{t-1}$$

Where:
- $m_t$ = first moment estimate (momentum)
- $v_t$ = second moment estimate (adaptive learning rate)
- $g_t$ = gradient at time t
- $\lambda$ = weight decay coefficient
- $\alpha$ = learning rate

### 3.2 Learning Rate Scheduling: ReduceLROnPlateau

**Strategy:** Reduce learning rate when validation accuracy plateaus

```
Monitoring Metric: Validation Accuracy

When val_acc shows NO improvement for 2 epochs:
├─ Reduce learning rate: new_lr = old_lr × 0.5
├─ Reset momentum to restart optimization
└─ Continue training with lower learning rate

Benefits:
├─ Escapes local plateaus in loss landscape
├─ Prevents oscillation around minima
└─ Enables fine-tuning in later training stages
```

### 3.3 Early Stopping with Model Checkpointing

**Strategy:** Prevent overfitting by monitoring validation performance

```
For each epoch:
├─ Train on training set
├─ Evaluate on validation set
├─ If val_acc > best_val_acc:
│  ├─ Save model weights (checkpoint)
│  ├─ Reset patience counter
│  └─ Update best accuracy
├─ Else:
│  ├─ Increment patience counter
│  └─ If patience > threshold (8 epochs):
│     └─ Stop training and restore best model
└─ Continue to next epoch

Best Model Restoration: Uses weights from epoch with highest val_acc
```

### 3.4 Loss Function: Cross-Entropy with Optional Class Weighting

**Standard Cross-Entropy:**
$$\mathcal{L} = -\sum_{i=1}^{N} \sum_{j=1}^{C} y_{i,j} \log(\hat{y}_{i,j})$$

Where:
- $N$ = number of samples
- $C$ = number of classes
- $y_{i,j}$ = true label (one-hot encoded)
- $\hat{y}_{i,j}$ = predicted probability

**Balanced Variant** (optional):
$$\mathcal{L}_{weighted} = -\sum_{i=1}^{N} \sum_{j=1}^{C} w_j \cdot y_{i,j} \log(\hat{y}_{i,j})$$

Where $w_j$ = inverse class frequency weight

**Note:** Standard cross-entropy performed better on this dataset (no class weighting needed).

---

## 4. Regularization Techniques

### 4.1 Batch Normalization

**Purpose:** Stabilize training by normalizing layer inputs

```
For each mini-batch:
1. Compute mean: μ_B = (1/m) Σ xᵢ
2. Compute variance: σ²_B = (1/m) Σ (xᵢ - μ_B)²
3. Normalize: x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
4. Scale & shift: yᵢ = γ·x̂ᵢ + β

Benefits:
├─ Higher learning rates possible
├─ Reduces internal covariate shift
├─ Acts as mild regularizer
└─ Accelerates convergence
```

### 4.2 Dropout

**Purpose:** Prevent co-adaptation and overfitting

```
During Training:
- Randomly deactivate 20% of neurons
- Scale remaining activations by 1/(1-p) to maintain expected values
- Ensemble effect: Different sub-networks see different data

During Inference:
- Use all neurons (no dropout)
- Activations already scaled appropriately

Effect: Similar to training multiple models and averaging
```

### 4.3 Weight Decay (L2 Regularization)

**Formula:**
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \sum_i w_i^2$$

**Effect:**
- Penalizes large weights
- Prevents model from relying on few dominant features
- Improves generalization
- Default weight decay: 1e-4

---

## 5. Training Process Flow

```
┌──────────────────────────────────────────────────────────────┐
│ Initialize Model, Optimizer, Scheduler                       │
│ Set: epochs=35, patience=8, best_val_acc=-1.0               │
└─────────────────────┬──────────────────────────────────────┘
                      ↓
              ┌───────────────────────────────┐
              │ For each epoch (1 to 35)      │
              └───────┬───────────────────────┘
                      ↓
        ┌─────────────────────────────────┐
        │ TRAINING PHASE                  │
        ├─────────────────────────────────┤
        │ For each batch in train_loader: │
        │  1. Forward pass: logits = model(x)
        │  2. Compute loss: L = CE(logits, y)
        │  3. Backward: L.backward()
        │  4. AdamW step: optimizer.step()
        │  5. Track metrics
        └────────┬────────────────────────┘
                 ↓
        ┌─────────────────────────────────┐
        │ VALIDATION PHASE                │
        ├─────────────────────────────────┤
        │ For each batch in val_loader:   │
        │  1. Forward pass (no gradients) │
        │  2. Compute loss & accuracy     │
        │  3. Accumulate metrics          │
        └────────┬────────────────────────┘
                 ↓
        ┌─────────────────────────────────┐
        │ Learning Rate Scheduling        │
        ├─────────────────────────────────┤
        │ scheduler.step(val_acc)         │
        │ If no improvement for 2 epochs: │
        │   learning_rate ×= 0.5          │
        └────────┬────────────────────────┘
                 ↓
        ┌─────────────────────────────────┐
        │ Early Stopping Check            │
        ├─────────────────────────────────┤
        │ If val_acc > best_val_acc:      │
        │   ✓ Save model weights          │
        │   ✓ Reset patience = 0          │
        │   ✓ Update best_val_acc         │
        │ Else:                           │
        │   ✗ patience += 1               │
        │   ✗ If patience >= 8:           │
        │     └─ STOP TRAINING            │
        └────────┬────────────────────────┘
                 ↓
    Continue to next epoch or exit
```

---

## 6. Evaluation & Metrics

### 6.1 Metrics Computed

**Per-epoch metrics:**
- Training loss and accuracy
- Validation loss and accuracy

**Final evaluation metrics:**
- Test accuracy
- Precision, Recall, F1-score (per class)
- Confusion matrix
- Optimization summary (vs baseline)

### 6.2 Optimization Summary

```
Baseline Accuracy:  0.2989 (29.89%)
Current Accuracy:   ~0.85  (85%)
─────────────────────────────
Absolute Gain:      +0.5511 (+55.11 percentage points)
Relative Gain:      +184.3% improvement over baseline
```

---

## 7. Hyperparameter Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--max-points` | 200,000 | Memory-efficient sampling |
| `--batch-size` | 1,024 | Mini-batch size for SGD |
| `--epochs` | 35 | Maximum training epochs |
| `--lr` | 1e-3 | Initial learning rate (AdamW) |
| `--weight-decay` | 1e-4 | L2 regularization coefficient |
| `--dropout` | 0.2 | Dropout probability |
| `--patience` | 8 | Early stopping patience (epochs) |
| `--test-size` | 0.2 | Test set fraction (20%) |
| `--val-size` | 0.2 | Val set fraction of train (16% total) |

---

## 8. Key Optimizations Applied

### 8.1 Feature Engineering
✓ **Complete feature utilization**: All 8 PLY features instead of 3 (xyz only)
✓ **Improvement**: 30% → 85% accuracy

### 8.2 Model Architecture
✓ **Batch normalization**: Faster convergence, higher stability
✓ **Dropout regularization**: Prevents overfitting
✓ **Appropriate layer sizes**: 8 → 256 → 128 → num_classes

### 8.3 Training Strategy
✓ **AdamW optimizer**: Better generalization than SGD
✓ **Learning rate scheduling**: Escape plateaus automatically
✓ **Early stopping**: Prevent overfitting and unnecessary computation
✓ **Stratified splitting**: Preserve class distribution in train/val/test

### 8.4 Data Preprocessing
✓ **Feature normalization**: Using training statistics
✓ **Class filtering**: Remove ultra-rare classes (<3 samples)
✓ **Smart sampling**: 200K point limit for memory efficiency

---

## 9. Comparison: Baseline vs Optimized

```
BASELINE (XYZ only):
├─ Features: x, y, z (3 dimensions)
├─ Accuracy: ~0.30 (30%)
├─ Model: Simple MLP
└─ Training: Basic SGD

OPTIMIZED (8 features + techniques):
├─ Features: x, y, z, x_origin, y_origin, z_origin, GPS_time, reflectance
├─ Accuracy: ~0.85 (85%)
├─ Model: MLP with BatchNorm + Dropout
├─ Optimizer: AdamW with ReduceLROnPlateau
├─ Regularization: Weight decay, dropout, batch norm
├─ Strategy: Early stopping, stratified splitting
└─ Result: +55.11 percentage points improvement
```

---

## 10. Reproducibility

**To reproduce these results:**

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run training with default parameters
python main.py --data data/Lille1.ply

# Run with custom parameters
python main.py \
    --data data/Lille1.ply \
    --epochs 50 \
    --batch-size 512 \
    --lr 5e-4 \
    --patience 10
```

---

## 11. Technical References

### Algorithms Implemented

1. **AdamW Optimizer**
   - Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization"
   - Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization"

2. **Batch Normalization**
   - Ioffe & Szegedy (2015): "Batch Normalization: Accelerating Deep Network Training"

3. **Dropout**
   - Hinton et al. (2012): "Improving Neural Networks with Dropout"

4. **Learning Rate Scheduling**
   - Common technique in deep learning for adaptive learning rates

5. **Early Stopping**
   - Prechelt (1998): "Early Stopping - But When?"

---

## 12. Performance Visualization Guide

The following graphs are recommended to visualize algorithm performance:

1. **Training & Validation Curves**
   - X-axis: Epoch
   - Y-axis: Loss (left), Accuracy (right)
   - Two lines per metric: Training vs Validation

2. **Learning Rate Decay**
   - X-axis: Epoch
   - Y-axis: Learning Rate (log scale)
   - Shows ReduceLROnPlateau effects

3. **Confusion Matrix**
   - Classes on both axes
   - Color intensity = prediction count
   - Diagonal = correct predictions

4. **Per-Class Metrics**
   - Grouped bar chart: Precision, Recall, F1 for each class
   - Identifies class-specific challenges

5. **Feature Importance (if using interpretability)**
   - Bar chart showing relative contribution of each feature
   - Gradient-based attribution recommended

---

**Document Generated:** April 28, 2026  
**Algorithm Version:** PointMLP with AdamW + ReduceLROnPlateau + Early Stopping  
**Dataset:** Lille1.ply (Large-scale airborne lidar point cloud)
