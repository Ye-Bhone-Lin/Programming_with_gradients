import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('/storage/slurm/home/69070701715@cpe.kmutt.ac.th/deep_learning/src/lab_4/part_2/iris_2class_train.csv')
test  = pd.read_csv('/storage/slurm/home/69070701715@cpe.kmutt.ac.th/deep_learning/src/lab_4/part_2/iris_2class_test.csv')

feature_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']

label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1}

X_train = train[feature_cols].values.astype(float)
y_train = train['class'].map(label_map).values.astype(float)

X_test = test[feature_cols].values.astype(float)
y_test = test['class'].map(label_map).values.astype(float)

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

Xb_train = add_bias(X_train)
Xb_test  = add_bias(X_test)

n_train, n_features = Xb_train.shape  # n_features includes bias

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def compute_loss(X, y, w):
    """Binary cross-entropy loss (average over samples)."""
    m = X.shape[0]
    z = X @ w
    p = sigmoid(z)
    eps = 1e-12  # numerical stability
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return loss

alpha = 0.1
n_iterations = 2000

w = np.zeros(n_features)  # w = [w0 (bias), w1, w2, w3, w4] initialized to 0
loss_history = []
loss_history.append(compute_loss(Xb_train, y_train, w))

m = n_train
for it in range(n_iterations):
    z = Xb_train @ w
    p = sigmoid(z)
    grad = (Xb_train.T @ (p - y_train)) / m
    w = w - alpha * grad
    loss_history.append(compute_loss(Xb_train, y_train, w))

print("Final weight vector:")
print(w)

print(f"\nFinal training loss (BCE): {loss_history[-1]:.6f}")
print(f"Initial training loss (BCE): {loss_history[0]:.6f}")

plt.figure(figsize=(7,5))
plt.plot(range(0, n_iterations+1), loss_history, color='#2563eb')
plt.xlabel('Iteration')
plt.ylabel('Binary Cross-Entropy Loss')
plt.title('Training Loss History (Gradient Descent, α=0.1)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/storage/slurm/home/69070701715@cpe.kmutt.ac.th/deep_learning/src/lab_4/part_2/output/loss_history.png', dpi=150)
plt.close()

pd.DataFrame({'iteration': range(0, n_iterations+1), 'loss': loss_history}).to_csv(
    'loss_history.csv', index=False)

z_test = Xb_test @ w
p_test = sigmoid(z_test)
y_pred = (p_test >= 0.5).astype(int)

TP = np.sum((y_pred == 1) & (y_test == 1))
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))

confusion = np.array([[TN, FP],
                       [FN, TP]])

accuracy = (TP + TN) / len(y_test)

print("\nConfusion Matrix (rows=Actual, cols=Predicted)")
print("                 Pred: Setosa   Pred: Versicolor")
print(f"Actual: Setosa        {TN:^10d}   {FP:^15d}")
print(f"Actual: Versicolor    {FN:^10d}   {TP:^15d}")

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Save confusion matrix plot
fig, ax = plt.subplots(figsize=(5,4.5))
im = ax.imshow(confusion, cmap='Blues')
labels = ['Setosa', 'Versicolor']
ax.set_xticks([0,1]); ax.set_xticklabels(labels)
ax.set_yticks([0,1]); ax.set_yticklabels(labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix (Accuracy = {accuracy*100:.2f}%)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(confusion[i, j]), ha='center', va='center',
                 color='white' if confusion[i,j] > confusion.max()/2 else 'black',
                 fontsize=16, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('/storage/slurm/home/69070701715@cpe.kmutt.ac.th/deep_learning/src/lab_4/part_2/output/confusion_matrix.png', dpi=150)
plt.close()