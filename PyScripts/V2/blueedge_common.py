"""Shared module: paths, features, model, training. Single source of truth for features."""
import os, re, math, random
from pathlib import Path
from collections import Counter

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "Datasets"
V4 = DATA_DIR / "improved_dataset v4.csv"
UNI_RAW = DATA_DIR / "Datasets final.csv"
CHALLENGE = DATA_DIR / "challenge_set.csv"
UNI_TEST = DATA_DIR / "university_test.csv"
OUT = ROOT / "Results"
OUT.mkdir(exist_ok=True)

SEEDS = [5, 123, 456, 789]          # paper seeds
DEFAULTS = dict(lr=0.001, batch=32, dropout=0.2, l2=0.001, beta2=0.999)  # paper says beta2 0.9999 (likely typo)

# ------------------------------------------------------------------ features (40)
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
    r"^((Mr|Mrs|Ms|Dr|Prof)\.?\s)?([A-Z][a-z]+[-'.]?[A-Za-z]*\s?)+(Jr\.?|Sr\.?|II|III|IV)?$",  # 9 name
    r"^-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+$",                        # 10 coordinates
    r"^\d+\s+\w+(\s\w+)*",                                           # 11 address-like
]
IGNORE_CASE = {0, 4, 7, 8}
KEYWORDS = ["street", "road", "ave", "unit", "box", "apt", "@", "http", "www",
            "percent", "dr.", "prof", "mr", "active", "inactive"]
assert len(KEYWORDS) == 15 and len(PATTERNS) == 12

GROUPS = {"length": list(range(0, 6)), "pattern": list(range(6, 18)),
          "statistical": list(range(18, 25)), "content": list(range(25, 40))}
ALL = list(range(40))


def entropy(s):
    n = len(s)
    return 0.0 if n == 0 else -sum(v / n * math.log2(v / n) for v in Counter(s).values())


def extract(text):
    s = str(text).strip()
    n = max(len(s), 1)
    digits = sum(c.isdigit() for c in s)
    alpha = sum(c.isalpha() for c in s)
    spaces = sum(c.isspace() for c in s)
    special = len(s) - digits - alpha - spaces
    upper = sum(c.isupper() for c in s)
    lower = sum(c.islower() for c in s)
    length = [len(s), len(s.split()), digits, alpha, spaces, special]
    pattern = [1 if re.match(p, s, re.I if i in IGNORE_CASE else 0) else 0 for i, p in enumerate(PATTERNS)]
    stat = [digits / n, alpha / n, upper / (lower + 1), special / n, spaces / n, len(set(s)) / n, entropy(s)]
    low = s.lower()
    content = [1 if k in low else 0 for k in KEYWORDS]
    return length + pattern + stat + content


assert len(extract("test@mail.com")) == 40


def featurize(strings):
    return np.array([extract(t) for t in strings], dtype=np.float32)


# ------------------------------------------------------------------ data
def load_v4():
    df = pd.read_csv(V4, encoding="utf-8")
    df["sample"] = df["sample"].fillna("").astype(str)
    X = featurize(df["sample"])
    le = LabelEncoder()
    y = le.fit_transform(df["data_type"])
    return df, X, y, le


def holdout_idx(y, seed):
    """70/15/15 style: returns (train_idx, test_idx) with 15% test (210 of 1400)."""
    return train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=seed)


# ------------------------------------------------------------------ model
def build_model(n_in, p, seed):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    L2 = regularizers.l2(p["l2"])
    init = keras.initializers.HeNormal(seed=seed)
    m = keras.Sequential([
        layers.Input(shape=(n_in,)),
        layers.Dense(64, activation="relu", kernel_initializer=init, kernel_regularizer=L2),
        layers.Dropout(p["dropout"], seed=seed),
        layers.Dense(32, activation="relu", kernel_initializer=init, kernel_regularizer=L2),
        layers.Dropout(p["dropout"], seed=seed),
        layers.Dense(14, activation="softmax", kernel_regularizer=L2),
    ])
    m.compile(optimizer=keras.optimizers.Adam(p["lr"], beta_1=0.9, beta_2=p["beta2"], epsilon=1e-8),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def set_seed(seed):
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def fit_model(Xtr, ytr, seed, params=None, epochs=100, verbose=0):
    """Scaler fit on train only. 15% of train held for early stopping."""
    from tensorflow import keras
    p = {**DEFAULTS, **(params or {})}
    set_seed(seed)
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=seed)
    sc = StandardScaler().fit(Xa)
    m = build_model(Xa.shape[1], p, seed)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)]
    m.fit(sc.transform(Xa), ya, validation_data=(sc.transform(Xv), yv), epochs=epochs,
          batch_size=p["batch"], callbacks=cbs, verbose=verbose)
    return m, sc


def predict_probs(m, sc, X):
    return m.predict(sc.transform(X), verbose=0)


# ------------------------------------------------------------------ regex baseline
RULE_ORDER = [(3, "ssn"), (0, "email"), (4, "url"), (10, "geographic_coordinates"), (7, "percentage"),
              (6, "currency"), (2, "date"), (1, "phone_number"), (8, "binary"), (5, "numeric_id"),
              (11, "address"), (9, "person_name")]


def regex_classify(s):
    s = str(s).strip()
    for i, t in RULE_ORDER:
        if re.match(PATTERNS[i], s, re.I if i in IGNORE_CASE else 0):
            return t
    return "text_description"


# ------------------------------------------------------------------ perturbations
def add_noise(s, p, rng):
    chars = list(str(s))
    pool = "abcdefghijklmnopqrstuvwxyz0123456789-_/.@ "
    return "".join(rng.choice(pool) if rng.random() < p else c for c in chars)


def drop_chars(s, keep, rng):
    out = "".join(c for c in str(s) if rng.random() < keep)
    return out if out else str(s)[:1]
