# CPE 663 Special Topic III: Deep Learning — Lab 4 Part 2

## Binary Classification with Logistic Regression (Batch GD)

This part implements a binary logistic regression classifier trained using batch gradient descent to distinguish between two Iris flower species.

## Course Information

- **Course**: CPE 663 Special Topic III: Deep Learning
- **Assignment**: Lab 4 — Part 2
- **Topic**: Logistic Regression with Batch Gradient Descent

## Overview

The task is binary classification on a reduced Iris dataset, distinguishing between:
- **Class 0**: Iris-setosa
- **Class 1**: Iris-versicolor

Using four sepal/petal measurement features:
- sepal length
- sepal width
- petal length
- petal width

The logistic regression model is implemented from scratch (no ML frameworks) using **sigmoid activation** and **binary cross-entropy loss**, optimized via batch gradient descent.

## Files

| File | Description |
|------|-------------|
| `classification_model.py` | Main script: logistic regression, training, evaluation, confusion matrix |
| `iris_2class_train.csv` | Training set (Iris 2-class subset) |
| `iris_2class_test.csv`  | Test set (Iris 2-class subset) |
| `lab4_part_2.sbatch`    | SLURM batch submission script |

## Dataset

- **Source**: Modified Iris dataset (2 classes only)
- **Classes**: Iris-setosa (0), Iris-versicolor (1)
- **Features**: 4 numeric measurements (sepal/petal length & width)
- **Split**: Separate train/test CSV files provided

## Implementation Details

### Model

Logistic regression (linear + sigmoid):

```
z = w0 + w1*x1 + w2*x2 + w3*x3 + w4*x4
p = σ(z) = 1 / (1 + e^(-z))
```

`p` is the predicted probability of class 1 (versicolor).
Class prediction: `ŷ = 1 if p ≥ 0.5 else 0`

### Loss Function

**Binary Cross-Entropy (BCE)**, averaged over samples:

```
L(w) = -(1/m) * Σ [ y * log(p) + (1-y) * log(1-p) ]
```

A small epsilon (`1e-12`) is added inside the log for numerical stability.

### Optimization

**Batch Gradient Descent** with:
- Learning rate α = 0.1
- 2000 iterations
- Weights initialized to zero (including bias)

Gradient derivation gives:
```
∇L(w) = (1/m) * Xᵀ · (σ(Xw) - y)
```

### Evaluation Metrics

1. **Training loss history** — plotted over iterations
2. **Confusion Matrix** on the test set:
   - TP / TN / FP / FN counts
3. **Accuracy** on the test set:
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

## How to Run

### Local Execution

```bash
pip install numpy pandas matplotlib
python classification_model.py
```

### Cluster Execution (SLURM)

```bash
sbatch lab4_part_2.sbatch
```

**Note**: Update file paths inside `classification_model.py` and `lab4_part_2.sbatch` to match your environment.

## Output

All outputs are saved to the `output/` directory:

| File | Description |
|------|-------------|
| `loss_history.png` | BCE training loss curve over 2000 iterations (α=0.1) |
| `confusion_matrix.png` | Heatmap confusion matrix with test accuracy annotation |
| `loss_history.csv` | Iteration vs. loss values (CSV) |

