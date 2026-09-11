import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("//storage/slurm/home/69070701715@cpe.kmutt.ac.th/deep_learning/src/lab_4/part_2/iris_2class_train.csv")
test = pd.read_csv("//storage/slurm/home/69070701715@cpe.kmutt.ac.th/deep_learning/src/lab_4/part_2/iris_2class_test.csv")

feature_cols = [
    'sepal length',
    'sepal width',
    'petal length',
    'petal width'
]

label_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1
}

X_train = train[feature_cols].values.astype(float)
y_train = train['class'].map(label_map).values.astype(float)

X_test = test[feature_cols].values.astype(float)
y_test = test['class'].map(label_map).values.astype(float)

def add_bias(X):
    return np.hstack([
        np.ones((X.shape[0], 1)),
        X
    ])


Xb_train = add_bias(X_train)
Xb_test = add_bias(X_test)

n_train, n_features = Xb_train.shape

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def compute_loss(X, y, w):

    z = X @ w
    p = sigmoid(z)

    eps = 1e-12

    loss = -np.mean(
        y * np.log(p + eps)
        + (1 - y) * np.log(1 - p + eps)
    )

    return loss

def mini_batch_gradient_descent(
    X,
    y,
    alpha=0.1,
    n_epochs=100,
    batch_size=16
):

    n_samples, n_features = X.shape

    w = np.zeros(n_features)

    loss_history = []

    loss_history.append(
        compute_loss(X, y, w)
    )

    for epoch in range(n_epochs):

        indices = np.random.permutation(n_samples)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, n_samples, batch_size):

            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            z = X_batch @ w
            p = sigmoid(z)

            actual_batch_size = X_batch.shape[0]

            gradient = (
                X_batch.T @ (p - y_batch)
            ) / actual_batch_size

            w = w - alpha * gradient

        loss = compute_loss(X, y, w)

        loss_history.append(loss)

    return w, loss_history

alpha = 0.1
n_epochs = 100

batch_sizes = [4, 8, 16, 32, 64]

results = {}


for batch_size in batch_sizes:

    w, loss_history = mini_batch_gradient_descent(
        Xb_train,
        y_train,
        alpha=alpha,
        n_epochs=n_epochs,
        batch_size=batch_size
    )

    results[batch_size] = {
        "weights": w,
        "loss": loss_history
    }

    print(
        f"Batch size = {batch_size:<3} "
        f"Final loss = {loss_history[-1]:.6f}"
    )

print("\n================================================")
print("FINAL WEIGHT VECTORS")
print("================================================")

for batch_size in batch_sizes:

    w = results[batch_size]["weights"]

    print(f"\nBatch size = {batch_size}")
    print(w)

plt.figure(figsize=(8, 6))

for batch_size in batch_sizes:

    loss_history = results[batch_size]["loss"]

    plt.plot(
        range(len(loss_history)),
        loss_history,
        label=f"Batch size = {batch_size}"
    )

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title(
    "Mini-Batch Gradient Descent "
    "(Different Batch Sizes)"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("/storage/slurm/home/69070701715@cpe.kmutt.ac.th/deep_learning/src/lab_4/part_3/output/mini_batch_loss_history.png", dpi=150)
plt.close()
print("\n================================================")
print("TEST ACCURACY")
print("================================================")

for batch_size in batch_sizes:

    w = results[batch_size]["weights"]

    z_test = Xb_test @ w
    p_test = sigmoid(z_test)

    y_pred = (p_test >= 0.5).astype(int)

    accuracy = np.mean(y_pred == y_test)

    print(
        f"Batch size = {batch_size:<3} "
        f"Accuracy = {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

summary = []

for batch_size in batch_sizes:

    w = results[batch_size]["weights"]
    loss_history = results[batch_size]["loss"]

    z_test = Xb_test @ w
    p_test = sigmoid(z_test)

    y_pred = (p_test >= 0.5).astype(int)

    accuracy = np.mean(y_pred == y_test)

    summary.append({
        "batch_size": batch_size,
        "final_loss": loss_history[-1],
        "accuracy": accuracy
    })

summary_df = pd.DataFrame(summary)

print(summary_df)