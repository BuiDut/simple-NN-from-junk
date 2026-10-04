# ==================================================================================
# = A from-scratch feedforward neural network using ONLY numpy + pandas.           =
# =                                                                                =
# = Supports:                                                                      =
# =  - Arbitrary number of layers / layer sizes                                    =
# =  - Activations: relu, leaky_relu, sigmoid, tanh, linear, softmax (output only) =    
# =  - Losses: mse (regression), binary_crossentropy, categorical_crossentropy     =
# =  - Mini-batch gradient descent with optional momentum                          =
# =  - L2 regularization                                                           =
# =  - pandas DataFrame/Series input (auto-converted to numpy internally)          =
# ==================================================================================

# ====================
# = Import Libraries =
# ====================

import numpy as np
import pandas as pd

# =========================
# = Activations Functions =
# =========================

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(z.dtype)

def leaky_relu(z, alpha = 0.001):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z, alpha = 0.001):
    return np.where(z > 0, 1.0, alpha)

def sigmoid(z):
    z = np.clip(z, -400, 400)   # Prevent overfloat
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.tanh(z)**2

def linear(z):
    return z

def linear_deriv(z):
    return np.ones_like(z)

def softmax(z):
    z = z - np.max(z, axis = 1, keepdims = True)    # prvent big number like e^10 -> inf
    e = np.exp(z)
    return e / np.sum(e, axis = 1, keepdims = True)

ACTIVATIONS = {
    "relu": (relu, relu_deriv),
    "leaky_relu": (leaky_relu, leaky_relu_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
    "linear": (linear, linear_deriv),
}

# ==================
# = Neural Network =
# ==================

class NeuralNetwork:
    
# =====================================================================
# =   Parameters                                                      =
# =====================================================================
# =   layer_sizes: list[int]                                          =
# =       e.g. [n_features, 16, 8, n_outputs]                         =
# =                                                                   =
# =   hidden_activation: str                                          =
# =       one of ACTIVATIONS keys, used for all hidden layers         =
# =                                                                   =
# =   output_activation: str                                          =
# =       "linear" (regression),                                      =
# =       "sigmoid" (binary classification),                          =
# =       or "softmax" (multi-class classification)                   =
# =                                                                   =
# =   loss: str                                                       =
# =       "mse", "binary_crossentropy", or "categorical_crossentropy" =
# =====================================================================

    def __init__(self, layer_sizes, hidden_activation = "relu", output_activation = "linear", loss = "mse", l2 = 0.0, seed = 42):
        assert len(layer_sizes) >= 2, "Need at least input and output layer"    # Just a sanity check
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1       # Number of weight layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.l2 = l2

        rng = np.random.default_rng(seed)   # Renerate random number based on seed
        self.W, self.b = [], []
        for i in range(self.L):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            
            # He init for relu-family, Xavier otherwise
            if hidden_activation in ("relu", "leaky_relu") and i < self.L - 1:
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
                
            self.W.append(rng.normal(0, scale, size = (fan_in, fan_out)))
            self.b.append(np.zeros((1, fan_out)))

        # momentum buffers (bias)
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

        self.history = {"loss": [], "val_loss": []}

# ==========
# = Helper =
# ==========

    @staticmethod
    def _to_numpy(X):   # Convert pandas to numpy to calculate
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy(dtype = float)
        return np.asarray(X, dtype = float)

    def _activate(self, z, is_output):      # Activate the fuctions
        if is_output:
            if self.output_activation == "softmax":
                return softmax(z)
            act, _ = ACTIVATIONS[self.output_activation]
            return act(z)
        act, _ = ACTIVATIONS[self.hidden_activation]
        return act(z)

    def _activate_deriv(self, z, is_output):
        if is_output:
            # softmax handled jointly with cross-entropy in backprop
            _, deriv = ACTIVATIONS.get(self.output_activation, (None, linear_deriv))
            return deriv(z)
        _, deriv = ACTIVATIONS[self.hidden_activation]
        return deriv(z)

    # ===========
    # = Forward =
    # ===========

    def forward(self, X):       # Returns (activations, pre_activations) for every layer, include input.
        A = [X]         # Memory, only add, the output of every layer
        Zs = []         # Pre-activation, raw wx+b
        for i in range(self.L):
            z = A[-1] @ self.W[i] + self.b[i]
            Zs.append(z)
            is_output = (i == self.L - 1)
            A.append(self._activate(z, is_output))
        return A, Zs

    def predict(self, X):       # Still forward, but only return the final output of the last layer
        X = self._to_numpy(X)
        A, _ = self.forward(X)
        return A[-1]

    # ================
    # = Loss Fuction =
    # ================

    def compute_loss(self, y_pred, y_true):         # As mentioned before, only mse, binary crossentropy and categorical crossentropy is accepted at the moment
        m = y_true.shape[0]
        if self.loss == "mse":
            data_loss = np.mean((y_pred - y_true)**2)
            
        elif self.loss == "binary_crossentropy":
            eps = 1e-12
            p = np.clip(y_pred, eps, 1 - eps)
            data_loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            
        elif self.loss == "categorical_crossentropy":
            eps = 1e-12
            p = np.clip(y_pred, eps, 1 - eps)
            data_loss = -np.mean(np.sum(y_true * np.log(p), axis = 1))
            
        else:
            raise ValueError(f"Unknown loss {self.loss}")       # Just a sanity check

        if self.l2 > 0:
            reg = sum(np.sum(w ** 2) for w in self.W)
            data_loss = data_loss + (self.l2 / (2 * m)) * reg
        return data_loss

    # ============
    # = Backward =
    # ============

    def backward(self, A, Zs, y_true):      # Where gradent can shine
        m = y_true.shape[0]
        grads_W = [None] * self.L
        grads_b = [None] * self.L

        y_pred = A[-1]

        # Output-layer delta.
        # For (softmax + categorical_crossentropy) and (sigmoid + binary_crossentropy) and (linear + mse), the delta simplifies to (y_pred - y_true).
        simplifies = (
            (self.output_activation == "softmax" and self.loss == "categorical_crossentropy") or
            (self.output_activation == "sigmoid" and self.loss == "binary_crossentropy") or
            (self.output_activation == "linear" and self.loss == "mse")
        )
        if simplifies:
            if self.loss == "mse":
                delta = 2 * (y_pred - y_true) / m
            else:
                delta = (y_pred - y_true) / m
        else:
            # generic fallback (dLoss/dA * dA/dZ), assumes mse-style gradient (i'm lazy for complex shit now)
            dA = 2 * (y_pred - y_true) / m
            delta = dA * self._activate_deriv(Zs[-1], is_output = True)

        for i in reversed(range(self.L)):       # Back-propagation
            grads_W[i] = A[i].T @ delta + (self.l2 / m) * self.W[i]
            grads_b[i] = np.sum(delta, axis = 0, keepdims = True)
            if i > 0:
                dA_prev = delta @ self.W[i].T
                delta = dA_prev * self._activate_deriv(Zs[i - 1], is_output = False)

        return grads_W, grads_b

    # =======
    # = Fit =
    # =======
    
    # ========================================================================
    # =     for each epoch:                                                  =
    # =         shuffle data                                                 =
    # =         for each mini-batch:                                         =
    # =             forward  → get predictions + cached A, Zs                =
    # =             backward → get gradients                                 =
    # =             update weights (with momentum)                           =
    # =         measure + log loss on full train set (and val set, if given) =
    # =         print progress                                               =
    # ========================================================================

    def fit(self, X, y, epochs = 200, batch_size = 32, lr = 0.001, momentum = 0.95, validation_data = None, verbose = True, print_every = 10):
        X = self._to_numpy(X)
        y = self._to_numpy(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n = X.shape[0]
        rng = np.random.default_rng(0)

        for epoch in range(1, epochs + 1):
            idx = rng.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]     # Ensure randomness, the order of the shuffle in both X and y are the same

            for start in range(0, n, batch_size):       # Loop in the batch, one mini-batch at a time
                end = start + batch_size
                Xb, yb = X_shuf[start:end], y_shuf[start:end]

                A, Zs = self.forward(Xb)
                gW, gb = self.backward(A, Zs, yb)

                for i in range(self.L):     # Update
                    self.vW[i] = momentum * self.vW[i] - lr * gW[i]
                    self.vb[i] = momentum * self.vb[i] - lr * gb[i]
                    self.W[i] += self.vW[i]
                    self.b[i] += self.vb[i]

            # epoch metrics
            y_pred_full = self.predict(X)
            train_loss = self.compute_loss(y_pred_full, y)
            self.history["loss"].append(train_loss)

            val_msg = ""    
            if validation_data is not None:     # Only if there is validation data
                Xv, yv = validation_data
                Xv = self._to_numpy(Xv)
                yv = self._to_numpy(yv)
                if yv.ndim == 1:
                    yv = yv.reshape(-1, 1)
                val_pred = self.predict(Xv)
                val_loss = self.compute_loss(val_pred, yv)
                self.history["val_loss"].append(val_loss)
                val_msg = f" - val_loss: {val_loss:.4f}"

            if verbose and (epoch % print_every == 0 or epoch == 1):    # Print every print_every epoch
                print(f"epoch {epoch:4d}/{epochs} - loss: {train_loss:.4f}{val_msg}")

        return self.history

    # ===============
    # = Quick asset =
    # ===============

    def predict_classes(self, X):
        p = self.predict(X)
        if self.output_activation == "softmax":
            return np.argmax(p, axis=1)
        return (p >= 0.5).astype(int).ravel()

    def score_accuracy(self, X, y_true_labels):
        preds = self.predict_classes(X)
        y_true_labels = self._to_numpy(y_true_labels).ravel()
        return float(np.mean(preds == y_true_labels))