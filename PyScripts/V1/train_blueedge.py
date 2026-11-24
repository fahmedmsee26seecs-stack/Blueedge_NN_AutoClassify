"""
BlueEdge-style data type classifier: 40 features -> 64 -> 32 -> 14 (softmax)
Run from venv:  python PyScripts\train_blueedge.py
Folder layout expected:
    BlueEdge_Project\
        Dataset\improved_dataset v4.csv
        PyScripts\train_blueedge.py   (this file)
        Models\                       (created automatically)
"""
import os, re, math, random
from pathlib import Path
from collections import Counter

SEED = 42
# SEED = 43
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Datasets" / "improved_dataset v4.csv"
MODEL_DIR = ROOT / "Models"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- features
PATTERNS = [
    r"^[\w.+-]+@[\w-]+\.[\w.]+$",                                    # 0 email
    r"^\+?\d{0,3}[\s-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$",        # 1 phone
    r"^(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,4}年\d{1,2}月\d{1,2}日|\d{1,2}-[A-Za-z]{3}(-\d{2,4})?)$",  # 2 date
    r"^\d{3}-\d{2}-\d{4}$",                                          # 3 ssn
    r"^((https?|ftp)://\S+|www\.\S+)$",                              # 4 url
    r"^\d+$",                                                        # 5 numeric id
    r"^[$€£¥₨]\s?[\d,]+(\.\d+)?$",                                   # 6 currency
    r"^(\d+(\.\d+)?\s?%|[\w-]+\s+percent)$",                         # 7 percentage
    r"^(true|false|yes|no|y|n|active|inactive|on|off|0|1)$",         # 8 binary
    r"^((Mr|Mrs|Ms|Dr|Prof)\.?\s)?([A-Z][a-z]+[-'.]?[A-Za-z]*\s?)+(Jr\.?|Sr\.?|II|III|IV)?$",  # 9 name-like
    r"^-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+$",                        # 10 coordinates
    r"^\d+\s+\w+(\s\w+)*",                                           # 11 address-like
]
IGNORE_CASE = {0, 4, 7, 8}

KEYWORDS = ["street", "road", "ave", "unit", "box", "apt", "@", "http", "www",
            "percent", "dr.", "prof", "mr", "active", "inactive"]


def entropy(s):
    n = len(s)
    if n == 0:
        return 0.0
    return -sum(v / n * math.log2(v / n) for v in Counter(s).values())


def extract(text):
    s = str(text).strip()
    n = max(len(s), 1)
    digits = sum(ch.isdigit() for ch in s)
    alpha = sum(ch.isalpha() for ch in s)
    spaces = sum(ch.isspace() for ch in s)
    special = len(s) - digits - alpha - spaces
    upper = sum(ch.isupper() for ch in s)
    lower = sum(ch.islower() for ch in s)

    length = [len(s), len(s.split()), digits, alpha, spaces, special]            # 6
    pattern = [1 if re.match(p, s, re.I if i in IGNORE_CASE else 0) else 0
               for i, p in enumerate(PATTERNS)]                                  # 12
    stat = [digits / n, alpha / n, upper / (lower + 1), special / n,
            spaces / n, len(set(s)) / n, entropy(s)]                             # 7
    low = s.lower()
    content = [1 if k in low else 0 for k in KEYWORDS]                           # 15
    return length + pattern + stat + content


assert len(extract("test@mail.com")) == 40

# ---------------------------------------------------------------- data
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df["sample"] = df["sample"].fillna("").astype(str)
X = np.array([extract(t) for t in df["sample"]], dtype=np.float32)
le = LabelEncoder()
y = le.fit_transform(df["data_type"])
print("Features:", X.shape, "Labels:", y.shape)
print("Classes :", list(le.classes_))

# stratified split: 70 / 15 / 15  (test = 210 samples, same as paper)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=SEED)
print("Train/Val/Test:", len(X_train), len(X_val), len(X_test))

# scale: fit on train only
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------- model
L2 = regularizers.l2(0.001)
model = keras.Sequential([
    layers.Input(shape=(40,)),
    layers.Dense(64, activation="relu", kernel_initializer="he_normal", kernel_regularizer=L2),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu", kernel_initializer="he_normal", kernel_regularizer=L2),
    layers.Dropout(0.2),
    layers.Dense(14, activation="softmax", kernel_regularizer=L2),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
]

# ---------------------------------------------------------------- tracker
EPOCHS = 100   # quick check run; set 100 for full training


class Tracker(keras.callbacks.Callback):
    """Record batch loss, test loss/accuracy per epoch, weight snapshots."""

    def __init__(self, X_t, y_t):
        super().__init__()
        self.X_t, self.y_t = X_t, y_t
        self.batch_loss, self.test_loss, self.test_acc = [], [], []
        self.snapshots = []          # index 0 = initial weights

    def _snap(self):
        return [w.copy() for w in self.model.get_weights()]   # W1,b1,W2,b2,W3,b3

    def on_train_begin(self, logs=None):
        self.snapshots.append(self._snap())

    def on_train_batch_end(self, batch, logs=None):
        self.batch_loss.append(logs["loss"])

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.X_t, self.y_t, verbose=0)
        self.test_loss.append(loss)
        self.test_acc.append(acc)
        self.snapshots.append(self._snap())


tracker = Tracker(X_test_s, y_test)   # test data only watched, never used for stopping
callbacks.append(tracker)

# ---------------------------------------------------------------- train
history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=EPOCHS, batch_size=32,
    callbacks=callbacks, verbose=2,
)
pd.DataFrame(history.history).to_csv(MODEL_DIR / "training_history.csv", index=False)

# ---------------------------------------------------------------- visualise
PLOT_DIR = MODEL_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)
h = history.history
ep = np.arange(1, len(h["loss"]) + 1)
names = ["L1 weights", "L1 biases", "L2 weights", "L2 biases", "L3 weights", "L3 biases"]
snaps = tracker.snapshots


def small_ticks(a):
    if len(ep) <= 20:
        a.set_xticks(ep)


# Figure 1: cost + accuracy + overfit/underfit view
fig1, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(ep, h["loss"], "o-", label="train")
ax[0, 0].plot(ep, h["val_loss"], "s-", label="validation")
ax[0, 0].plot(ep, tracker.test_loss, "^-", label="test")
ax[0, 0].set(title="Cost per epoch", xlabel="epoch", ylabel="loss")
ax[0, 0].legend(); small_ticks(ax[0, 0])

ax[0, 1].plot(ep, h["accuracy"], "o-", label="train")
ax[0, 1].plot(ep, h["val_accuracy"], "s-", label="validation")
ax[0, 1].plot(ep, tracker.test_acc, "^-", label="test")
ax[0, 1].set(title="Accuracy per epoch", xlabel="epoch", ylabel="accuracy")
ax[0, 1].legend(); small_ticks(ax[0, 1])

ax[1, 0].plot(tracker.batch_loss)
ax[1, 0].set(title="Training cost per batch step", xlabel="batch step", ylabel="loss")

gap = np.array(h["val_loss"]) - np.array(h["loss"])
ax[1, 1].bar(ep, gap)
ax[1, 1].axhline(0, color="k", lw=0.8)
ax[1, 1].set(title="Gap = validation loss - train loss", xlabel="epoch", ylabel="gap")
small_ticks(ax[1, 1])
fig1.tight_layout()
fig1.savefig(PLOT_DIR / "1_cost_accuracy.png", dpi=120)

# Figure 2: how weights and biases move
upd = np.array([[np.mean(np.abs(snaps[e][i] - snaps[e - 1][i])) for i in range(6)]
                for e in range(1, len(snaps))])           # epochs x 6
norm = np.array([[np.linalg.norm(snaps[e][i]) for i in range(6)]
                 for e in range(len(snaps))])             # (epochs+1) x 6
ep0 = np.arange(0, len(snaps))

fig2, ax = plt.subplots(2, 2, figsize=(12, 8))
for i in (0, 2, 4):
    ax[0, 0].plot(ep, upd[:, i], "o-", label=names[i])
ax[0, 0].set(title="Mean |weight change| per epoch", xlabel="epoch", ylabel="mean |dW|")
ax[0, 0].legend(); small_ticks(ax[0, 0])
for i in (1, 3, 5):
    ax[0, 1].plot(ep, upd[:, i], "o-", label=names[i])
ax[0, 1].set(title="Mean |bias change| per epoch", xlabel="epoch", ylabel="mean |db|")
ax[0, 1].legend(); small_ticks(ax[0, 1])
for i in (0, 2, 4):
    ax[1, 0].plot(ep0, norm[:, i], "o-", label=names[i])
ax[1, 0].set(title="Weight norm (epoch 0 = initial)", xlabel="epoch", ylabel="L2 norm")
ax[1, 0].legend(); ax[1, 0].set_xticks(ep0) if len(ep0) <= 21 else None
for i in (1, 3, 5):
    ax[1, 1].plot(ep0, norm[:, i], "o-", label=names[i])
ax[1, 1].set(title="Bias norm (epoch 0 = initial)", xlabel="epoch", ylabel="L2 norm")
ax[1, 1].legend(); ax[1, 1].set_xticks(ep0) if len(ep0) <= 21 else None
fig2.tight_layout()
fig2.savefig(PLOT_DIR / "2_weight_bias_updates.png", dpi=120)

# Figure 3: distributions, initial vs final
fig3, ax = plt.subplots(2, 3, figsize=(14, 7))
for col, i in enumerate((0, 2, 4)):
    ax[0, col].hist(snaps[0][i].ravel(), bins=40, alpha=0.6, label="initial")
    ax[0, col].hist(snaps[-1][i].ravel(), bins=40, alpha=0.6, label="final")
    ax[0, col].set(title=names[i] + " distribution", xlabel="value", ylabel="count")
    ax[0, col].legend()
for col, i in enumerate((1, 3, 5)):
    ax[1, col].plot(snaps[0][i], "o-", label="initial")
    ax[1, col].plot(snaps[-1][i], "s-", label="final")
    ax[1, col].set(title=names[i] + " per neuron", xlabel="neuron", ylabel="value")
    ax[1, col].legend()
fig3.tight_layout()
fig3.savefig(PLOT_DIR / "3_weight_bias_distributions.png", dpi=120)

# Fit diagnosis (rule of thumb)
tr_l, va_l, te_l = h["loss"][-1], h["val_loss"][-1], tracker.test_loss[-1]
tr_a, va_a = h["accuracy"][-1], h["val_accuracy"][-1]
print("\n=== Fit diagnosis ===")
print(f"Final loss  train {tr_l:.4f} | val {va_l:.4f} | test {te_l:.4f}")
print(f"Final acc   train {tr_a:.3f} | val {va_a:.3f}")
print(f"Gap (val - train): {va_l - tr_l:.4f}")
if tr_a < 0.85 and (va_l - tr_l) < 0.15:
    print("Verdict: UNDERFIT or too few epochs (train cost still high, small gap)")
elif (va_l - tr_l) > 0.15 or (len(h["val_loss"]) > 2 and h["val_loss"][-1] > min(h["val_loss"]) + 0.05):
    print("Verdict: OVERFIT (validation cost above train cost or rising)")
else:
    print("Verdict: GOOD FIT (train and validation close, accuracy high)")
if EPOCHS < 10:
    print(f"Note: only {EPOCHS} epochs, verdict unreliable. Run longer for real answer.")

# ---------------------------------------------------------------- evaluate
probs = model.predict(X_test_s, verbose=0)
pred = probs.argmax(axis=1)
conf = probs.max(axis=1)

print("\nTest accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%  (paper: 94.2%)")
print("\nClassification report:")
print(classification_report(y_test, pred, target_names=le.classes_, digits=3))
print("Confusion matrix (rows = true, cols = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, pred), index=le.classes_, columns=le.classes_))

# confidence threshold 0.6: below -> flag for manual review
THRESH = 0.6
keep = conf >= THRESH
print(f"\nFlagged for review (conf < {THRESH}): {(~keep).sum()} of {len(conf)}")
if keep.any():
    print("Accuracy on accepted samples:",
          round(accuracy_score(y_test[keep], pred[keep]) * 100, 2), "%")

# misclassified samples, inspect for label noise
_, idx_test = train_test_split(np.arange(len(X)), test_size=0.15, stratify=y, random_state=SEED)
wrong = idx_test[pred != y_test]
out = df.iloc[wrong].copy()
out["predicted"] = le.inverse_transform(pred[pred != y_test])
out.to_csv(MODEL_DIR / "misclassified.csv", index=False, encoding="utf-8-sig")
print("Misclassified samples saved:", len(out))

# ---------------------------------------------------------------- save
model.save(MODEL_DIR / "blueedge_model.keras")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
print("Saved model, scaler, label encoder in:", MODEL_DIR)
print("Plots saved in:", PLOT_DIR)
plt.show()