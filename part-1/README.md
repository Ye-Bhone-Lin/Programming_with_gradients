# CPE 663 Special Topic III: Deep Learning — Lab 4 Part 1

## Linear Regression with Batch Gradient Descent

This part implements a linear regression model trained using batch gradient descent to predict medical insurance charges based on patient features.

## Course Information

- **Course**: CPE 663 Special Topic III: Deep Learning
- **Assignment**: Lab 4 — Part 1
- **Topic**: Linear Regression with Gradient Descent

## Overview

The goal is to build a linear regression model that predicts individual medical costs billed by health insurance (`charges`) using three input features:
- `age`: Age of the primary beneficiary
- `gender`: Gender (0 or 1 encoded)
- `bmi`: Body Mass Index

The model is trained from scratch using **batch gradient descent** (without using ML frameworks like PyTorch/TensorFlow) to understand the underlying optimization mechanics.

## Files

| File | Description |
|------|-------------|
| `regression_model.py` | Main Python script implementing gradient descent, training, evaluation, and prediction |
| `insurance_charges.csv` | Input dataset with columns: age, gender, bmi, charges |
| `lab4_part_1.sbatch` | SLURM batch submission script for running on the HPC cluster |

## Dataset

- **Source**: Insurance Charges dataset
- **Features**: age, gender, bmi (3 features)
- **Target**: charges (medical cost)
- **Preprocessing**: Standardization (z-score normalization) applied to both features and target

## Implementation Details

### Model

Linear regression hypothesis with bias term:

```
y = w0 + w1*x1 + w2*x2 + w3*x3
```

In matrix form (with design matrix including bias column):
```
y = X @ w
```

### Loss Function

Mean Squared Error (MSE), averaged over samples and halved for gradient convenience:

```
L(w) = (1 / (2n)) * Σ (y_pred_i - y_i)^2
```

### Optimization

**Batch Gradient Descent**:
- Uses the full training set to compute the gradient at each step
- Weight update: `w = w - α * ∇L(w)`
- Learning rate (α) experimented across multiple values

### Learning Rate Sweep

The script tests 6 different learning rates:
- `[0.001, 0.01, 0.1, 0.15, 0.5, 1.0]`

This demonstrates how the learning rate affects:
- Convergence speed
- Stability (divergence if α is too large)
- Final loss value

### Prediction

After training with α=0.01, the model makes predictions on 4 new sample patients and saves the results to CSV.

## How to Run

### Local Execution

```bash
# Ensure dependencies are installed (numpy, pandas, matplotlib)
pip install numpy pandas matplotlib

# Run the script
python regression_model.py
```

### Cluster Execution (SLURM)

```bash
sbatch lab4_part_1.sbatch
```

The sbatch script:
- Requests 4 CPUs, 8GB memory, 1 GPU (gpu4090 partition)
- 30-minute time limit
- Activates the `env_dl` virtual environment
- Saves stdout/stderr to `logs/%j.out` and `logs/%j.err`

**Note**: You may need to update the file paths inside `regression_model.py` and `lab4_part_1.sbatch` to match your local or cluster environment paths.

## Output

All outputs are saved to the `output/` directory:

| File | Description |
|------|-------------|
| `loss_history_alpha_0.01.png` | Training loss curve over iterations for α=0.01 |
| `loss_history_multi_lr.png` | Overlaid loss curves (log-scale) for all 6 learning rates |
| `predictions.csv` | Predicted insurance charges for 4 sample patients |

