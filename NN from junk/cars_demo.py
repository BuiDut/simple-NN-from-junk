# ==================
# = Import Library =
# ==================

import numpy as np
import pandas as pd
from neural_network import NeuralNetwork

# ==================
# = load & inspect =
# ==================

df = pd.read_csv("cars.csv")
df = df.drop(columns=["stt"])  # just a row index column

# ===================================================================
# = Ordinal encoding                                                =
# = All feature columns are ordered categories, so we map them to   =
# = integers that preserve that order (better for a NN than random  =
# = integer codes, and avoids the extra columns one-hot would add). =
# ===================================================================

maps = {
    "buying":   {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "maint":    {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "doors":    {"2": 0, "3": 1, "4": 2, "5more": 3},
    "persons":  {"2": 0, "4": 1, "more": 2},
    "lug_boot": {"small": 0, "med": 1, "big": 2},
    "safety":   {"low": 0, "med": 1, "high": 2},
}
for col, m in maps.items():
    df[col] = df[col].map(m)

target_map = {"unacc": 0, "acc": 1}
df["label"] = df["acceptability"].map(target_map)

feature_cols = list(maps.keys())
X_raw = df[feature_cols].to_numpy(dtype=float)
y = df["label"].to_numpy(dtype=float).reshape(-1, 1)

print("Class balance:\n", df["acceptability"].value_counts(), "\n")

# ====================
# = train/test split =
# ====================

rng = np.random.default_rng(0)
n = len(df)
idx = rng.permutation(n)
split = int(0.8 * n)
train_idx, test_idx = idx[:split], idx[split:]

# =======================================================================
# = standardize features                                                =
# = fit scaler on TRAIN ONLY, then apply to both, to avoid leaking test =
# = statistics into training                                            =
# =======================================================================

mu = X_raw[train_idx].mean(axis=0)
sigma = X_raw[train_idx].std(axis=0)
sigma[sigma == 0] = 1.0  # guard against constant columns
X_scaled = (X_raw - mu) / sigma

X_train, y_train = X_scaled[train_idx], y[train_idx]
X_test, y_test = X_scaled[test_idx], y[test_idx]

# =========
# = train =
# =========

nn = NeuralNetwork(layer_sizes = [X_train.shape[1], 16, 8, 1], hidden_activation = "relu", output_activation = "sigmoid", loss = "binary_crossentropy", l2 = 1e-3, seed = 42,)
nn.fit(X_train, y_train, epochs = 600, batch_size = 16, lr = 0.1, momentum = 0.9, validation_data=(X_test, y_test), print_every = 30,)

# ============
# = evaluate =
# ============

preds = nn.predict_classes(X_test)
y_true = y_test.ravel().astype(int)

acc = np.mean(preds == y_true)
tp = np.sum((preds == 1) & (y_true == 1))
tn = np.sum((preds == 0) & (y_true == 0))
fp = np.sum((preds == 1) & (y_true == 0))
fn = np.sum((preds == 0) & (y_true == 1))

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print(f"\nTest accuracy:  {acc:.3f}")
print(f"Confusion matrix -> TP:{tp} TN:{tn} FP:{fp} FN:{fn}")
print(f"Precision (acc):  {precision:.3f}")
print(f"Recall    (acc):  {recall:.3f}")
print(f"F1        (acc):  {f1:.3f}")

baseline_acc = max(y_true.mean(), 1 - y_true.mean())
print(f"\nBaseline (always predict majority class): {baseline_acc:.3f}")