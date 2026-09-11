# CPE 663 Special Topic III: Deep Learning — Lab 4 Part 3

## Mini-Batch Gradient Descent — Batch Size Comparison

This part compares the performance of **mini-batch gradient descent** with different batch sizes on the same 2-class Iris classification task from Part 2.

## Course Information

- **Course**: CPE 663 Special Topic III: Deep Learning
- **Assignment**: Lab 4 — Part 3
- **Topic**: Mini-Batch Gradient Descent (Batch Size Ablation Study)

## Overview

Building on the logistic regression model from Part 2, this part replaces **batch gradient descent** (full dataset per step) with **mini-batch gradient descent**, and systematically evaluates how batch size affects:

- Convergence speed (loss per epoch)
- Training stability (smoothness of loss curve)
- Final training loss
- Test set accuracy

The same Iris 2-class dataset (Setosa vs. Versicolor) and model architecture are used.

## Files

| File | Description |
|------|-------------|
| `mini_batch_gd.py` | Main script: mini-batch GD, batch size sweep, plots, and accuracy summary |
| `lab4_part_3.sbatch` | SLURM batch submission script |

**Note**: The Iris CSV data files are referenced from the `part-2` directory (shared dataset).

## Dataset

Same 2-class Iris dataset as Part 2 (loaded from `../part-2/` paths):
- Classes: Iris-setosa (0) vs. Iris-versicolor (1)
- Features: sepal length, sepal width, petal length, petal width
- Train/test split provided as separate CSVs

## Implementation Details

### Model

Same logistic regression as Part 2:
- Linear layer + sigmoid activation
- Binary Cross-Entropy (BCE) loss
- 4 input features + 1 bias term (5 weights total)
- Weights initialized to zero

### Mini-Batch Gradient Descent Algorithm

For each epoch:
1. Shuffle the entire training set (random permutation)
2. Divide the shuffled data into batches of size `batch_size`
3. For each batch:
   - Compute gradient using only the mini-batch
   - Update weights: `w = w - α * ∇L_batch(w)`
4. Record the **full training-set loss** at the end of the epoch

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate (α) | 0.1 |
| Epochs | 100 |
| Batch sizes tested | [4, 8, 16, 32, 64] |

### Batch Size Interpretations

- **Size 4–8**: Closer to Stochastic Gradient Descent (SGD) — noisy gradients, fast per-epoch progress
- **Size 16–32**: Typical mini-batch range — balance of noise and efficiency
- **Size 64**: Approaches batch GD (full dataset if n ≈ 64) — smoother gradients, slower per iteration

### Evaluation

For each batch size, the script reports:
1. Final training BCE loss after 100 epochs
2. Learned weight vector
3. Test-set accuracy (threshold = 0.5)
4. A summary table combining loss and accuracy for all batch sizes

## How to Run

### Local Execution

```bash
pip install numpy pandas matplotlib
python mini_batch_gd.py
```

### Cluster Execution (SLURM)

```bash
sbatch lab4_part_3.sbatch
```

**Note**: Update file paths inside `mini_batch_gd.py` and `lab4_part_3.sbatch` to match your environment. The script expects the Iris CSVs to exist under the `part-2` directory from Lab 4.

## Output

All outputs are saved to the `output/` directory:

| File | Description |
|------|-------------|
| `mini_batch_loss_history.png` | Overlaid loss curves (BCE vs. epoch) for all 5 batch sizes |

Additionally, results are printed to stdout:
- Final loss per batch size
- Final weight vectors per batch size
- Test accuracy per batch size
- Summary DataFrame (batch_size, final_loss, accuracy)
