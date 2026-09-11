# CPE 663 Special Topic III: Deep Learning

## Lab 4 — Programming with Gradients

An implementation-only lab focused on building and understanding gradient-based optimization algorithms from scratch — no deep learning frameworks. All models are implemented with pure NumPy.

---

## Course Information

| Detail | |
|--------|---|
| **Course** | CPE 663 Special Topic III: Deep Learning |
| **Assignment** | Lab 4 — Programming with Gradients |
---

## Overview

This lab explores the fundamentals of gradient-based learning by implementing three core ML components entirely from scratch (NumPy only — no PyTorch/TensorFlow). The goal is to develop intuition for how gradients drive optimization, how different gradient descent variants behave, and how the choice of hyperparameters (learning rate, batch size) impacts training.

The lab progresses from regression to classification, and from full-batch GD to mini-batch GD:

1. **Part 1** — Linear Regression with Batch Gradient Descent
2. **Part 2** — Binary Logistic Classification with Batch GD
3. **Part 3** — Mini-Batch GD Ablation Study (Batch Size Comparison)

---

## Repository Structure

```
.
├── part-1/                          # Linear Regression
│   ├── README.md                    # Part 1 detailed documentation
│   ├── regression_model.py          # Batch GD linear regression + LR sweep
│   ├── insurance_charges.csv        # Input: insurance cost dataset
│   ├── lab4_part_1.sbatch           # SLURM submission script
│   └── output/
│       ├── loss_history_alpha_0.01.png
│       ├── loss_history_multi_lr.png
│       └── predictions.csv
│
├── part-2/                          # Binary Classification (Batch GD)
│   ├── README.md                    # Part 2 detailed documentation
│   ├── classification_model.py      # Logistic regression + BCE + confusion matrix
│   ├── iris_2class_train.csv        # Iris train (Setosa vs Versicolor)
│   ├── iris_2class_test.csv         # Iris test
│   ├── lab4_part_2.sbatch           # SLURM submission script
│   └── output/
│       ├── loss_history.png
│       └── confusion_matrix.png
│
└── part-3/                          # Mini-Batch GD Comparison
    ├── README.md                    # Part 3 detailed documentation
    ├── mini_batch_gd.py             # MBGD + batch size ablation (4,8,16,32,64)
    ├── lab4_part_3.sbatch           # SLURM submission script
    └── output/
        └── mini_batch_loss_history.png
```

---

## Part-by-Part Summary

### Part 1 — Linear Regression with Batch Gradient Descent

- **Task**: Predict insurance charges from `(age, gender, bmi)`
- **Model**: Linear regression with bias (design matrix trick)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Batch gradient descent
- **Key experiment**: Learning rate sweep over `[0.001, 0.01, 0.1, 0.15, 0.5, 1.0]` to demonstrate convergence / divergence behavior
- **Deliverables**: Loss curves, predictions on 4 new sample patients

→ See [part-1/README.md](part-1/README.md) for full details.

---

### Part 2 — Binary Classification with Logistic Regression (Batch GD)

- **Task**: Classify Iris Setosa (0) vs. Iris Versicolor (1) from 4 sepal/petal features
- **Model**: Logistic regression → linear + sigmoid
- **Loss**: Binary Cross-Entropy (BCE / log loss)
- **Optimizer**: Batch gradient descent (α = 0.1, 2000 iters)
- **Key experiment**: Evaluate on a held-out test set using a confusion matrix and accuracy
- **Deliverables**: BCE training curve, confusion matrix heatmap with accuracy

→ See [part-2/README.md](part-2/README.md) for full details.

---

### Part 3 — Mini-Batch Gradient Descent (Batch Size Study)

- **Task**: Same 2-class Iris classification as Part 2, now with mini-batches
- **Model**: Identical logistic regression (reused)
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Mini-batch gradient descent with shuffling per epoch
- **Key experiment**: Ablate batch size over `[4, 8, 16, 32, 64]` and compare convergence speed, loss smoothness, and test accuracy
- **Deliverables**: Overlaid loss curves per batch size, final loss + accuracy summary table

→ See [part-3/README.md](part-3/README.md) for full details.


## How to Run

Each part is self-contained. Detailed instructions are in each part's own `README.md`.

Quick local run (requires `numpy`, `pandas`, `matplotlib`):

```bash
# Install dependencies (once)
pip install numpy pandas matplotlib

# Part 1
cd part-1 && python regression_model.py

# Part 2
cd ../part-2 && python classification_model.py

# Part 3
cd ../part-3 && python mini_batch_gd.py
```

For SLURM cluster execution, use the provided `lab4_part_*.sbatch` scripts:
```bash
sbatch part-1/lab4_part_1.sbatch
sbatch part-2/lab4_part_2.sbatch
sbatch part-3/lab4_part_3.sbatch
```
---

## License

See [LICENSE](LICENSE).
