import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/storage/slurm/home/69070701715@cpe.kmutt.ac.th/deep_learning/src/lab_4/part_1/insurance_charges.csv")
# print("Dataset shape:", df.shape)
# print(df.head())

X = df[["age", "gender", "bmi"]].to_numpy(dtype=float)
y = df["charges"].to_numpy(dtype=float)

n_samples, n_features = X.shape

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
y_mean, y_std = y.mean(), y.std()

X_scaled = (X - X_mean) / X_std
y_scaled = (y - y_mean) / y_std

X_design = np.hstack([np.ones((n_samples, 1)), X_scaled])  

def gradient_descent(X, y, alpha=0.01, n_iters=1000, w_init=None):
    n, d = X.shape
    w = np.zeros(d) if w_init is None else w_init.copy()
    loss_history = []
    for i in range(n_iters):
        y_pred = X @ w
        error = y_pred - y
        loss = (1 / (2 * n)) * np.sum(error ** 2)  
        loss_history.append(loss)
        grad = (1 / n) * (X.T @ error)
        w = w - alpha * grad
    return w, loss_history

alpha = 0.01
n_iters = 1000
w_final, loss_history = gradient_descent(X_design, y_scaled, alpha=alpha, n_iters=n_iters)

print("\n=== Results with alpha = 0.01 ===")
print("Final weight vector:")
print(w_final)
print(f"Initial loss: {loss_history[0]:.6f}")
print(f"Final loss: {loss_history[-1]:.6f}")

plt.figure(figsize=(7, 5))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.title(f"Training Loss History (alpha = {alpha})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/storage/slurm/home/69070701715@cpe.kmutt.ac.th/deep_learning/src/lab_4/part_1/output/loss_history_alpha_0.01.png", dpi=150)
plt.close()

learning_rates = [0.001, 0.01, 0.1, 0.15, 0.5, 1.0]

results = {}

plt.figure(figsize=(8, 6))

for lr in learning_rates:

    w_lr, hist_lr = gradient_descent(X_design, y_scaled, alpha=lr, n_iters=n_iters)

    results[lr] = (w_lr, hist_lr)

    print(f"alpha = {lr:<6} " f"final_loss = {hist_lr[-1]}")

    hist_plot = np.nan_to_num(
        np.array(hist_lr),
        nan=1e6,
        posinf=1e6,
        neginf=1e6
    )

    hist_plot = np.clip(hist_plot,1e-6,1e6)

    plt.plot(hist_plot, label=f"alpha={lr}")

plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss for Different Learning Rates")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/storage/slurm/home/69070701715@cpe.kmutt.ac.th/deep_learning/src/lab_4/part_1/output/loss_history_multi_lr.png", dpi=150)
plt.close()

new_data = pd.DataFrame({
    "age":    [31, 43, 28, 50],
    "gender": [0, 1, 1, 0],
    "bmi":    [22, 19, 33, 28],
})

new_X_scaled = (new_data[["age", "gender", "bmi"]].to_numpy(dtype=float) - X_mean) / X_std
new_X_design = np.hstack([np.ones((new_X_scaled.shape[0], 1)), new_X_scaled])

pred_scaled = new_X_design @ w_final
pred_charges = pred_scaled * y_std + y_mean 

new_data["predicted_charges"] = pred_charges
print("\n=== Predictions ===")
print(new_data)

new_data.to_csv("/storage/slurm/home/69070701715@cpe.kmutt.ac.th/deep_learning/src/lab_4/part_1/output/predictions.csv", index=False)
