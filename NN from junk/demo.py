import numpy as np
import pandas as pd
from neural_network import NeuralNetwork

rng = np.random.default_rng(1)

# ---- synthetic 2D "two moons"-ish dataset (numpy only) ---- #
n = 500
theta = rng.uniform(0, np.pi, n)
class0 = np.column_stack([np.cos(theta), np.sin(theta)]) + rng.normal(0, 0.1, (n, 2))
class1 = np.column_stack([1 - np.cos(theta), 1 - np.sin(theta) - 0.5]) + rng.normal(0, 0.1, (n, 2))

X = np.vstack([class0, class1])
y = np.array([0] * n + [1] * n)

# shuffle
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]

# put in a DataFrame to prove pandas interop works
df = pd.DataFrame(X, columns=["x1", "x2"])
df["label"] = y

split = int(0.8 * len(df))
train_df, test_df = df.iloc[:split], df.iloc[split:]

X_train, y_train = train_df[["x1", "x2"]], train_df["label"].to_numpy().reshape(-1, 1)
X_test, y_test = test_df[["x1", "x2"]], test_df["label"].to_numpy()

nn = NeuralNetwork(
    layer_sizes = [2, 16, 8, 1],
    hidden_activation = "relu",
    output_activation = "sigmoid",
    loss="binary_crossentropy",
    l2 = 1e-4,
)

nn.fit(X_train, y_train, epochs = 200, batch_size = 32, lr = 0.1, momentum = 0.9, validation_data=(X_test, y_test.reshape(-1, 1)), print_every = 40)

acc = nn.score_accuracy(X_test, y_test)
print(f"\nTest accuracy: {acc:.3f}")
