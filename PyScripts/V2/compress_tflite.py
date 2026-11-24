"""Compression pipeline per paper S4: INT8 quantization, 15% magnitude pruning + 10 epoch fine-tune, Huffman.
Run: python PyScripts\\compress_tflite.py
"""
import time, heapq
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from blueedge_common import *


def tflite_convert(model, rep_X, int8):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([x[None].astype(np.float32)] for x in rep_X)
    return conv.convert()


def tflite_eval(blob, X, y, n_lat=200):
    it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors()
    i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    preds, t = [], []
    for x in X:
        it.set_tensor(i_in, x[None].astype(np.float32))
        t0 = time.perf_counter(); it.invoke(); t.append((time.perf_counter() - t0) * 1000)
        preds.append(it.get_tensor(i_out).argmax())
    return accuracy_score(y, preds) * 100, float(np.mean(t[:n_lat]))


def huffman_bits(data):
    c = Counter(data.tolist())
    if len(c) == 1: return len(data)
    heap = [(w, i, {s: 0}) for i, (s, w) in enumerate(c.items())]; heapq.heapify(heap); n = len(heap)
    while len(heap) > 1:
        w1, _, d1 = heapq.heappop(heap); w2, _, d2 = heapq.heappop(heap)
        heapq.heappush(heap, (w1 + w2, n, {s: l + 1 for s, l in {**d1, **d2}.items()})); n += 1
    lens = heap[0][2]
    return sum(c[s] * lens[s] for s in c)


def int8_weights(model):
    out = []
    for w in model.get_weights():
        sc = np.abs(w).max() / 127 or 1.0
        out.append(np.round(w / sc).astype(np.int8).ravel())
    return np.concatenate(out)


df, X, y, le = load_v4()
tr, te = holdout_idx(y, SEEDS[0])
m, sc = fit_model(X[tr], y[tr], SEEDS[0])
Xtr_s, Xte_s = sc.transform(X[tr]), sc.transform(X[te]); yte = y[te]
rows = []

acc = accuracy_score(yte, m.predict(Xte_s, verbose=0).argmax(1)) * 100
nparam = m.count_params()
rows.append(dict(stage="float32 keras", acc=acc, size_bytes=nparam * 4, latency_ms=np.nan))

blob = tflite_convert(m, Xtr_s[:200], int8=False); a, l = tflite_eval(blob, Xte_s, yte)
rows.append(dict(stage="tflite float32", acc=a, size_bytes=len(blob), latency_ms=l))
blob8 = tflite_convert(m, Xtr_s[:200], int8=True); a, l = tflite_eval(blob8, Xte_s, yte)
rows.append(dict(stage="tflite int8 (PTQ)", acc=a, size_bytes=len(blob8), latency_ms=l))

# magnitude prune 15% of kernel weights, fine-tune 10 epochs keeping mask
kern = [i for i, w in enumerate(m.get_weights()) if w.ndim == 2]
W = m.get_weights(); masks = {}
allk = np.concatenate([np.abs(W[i]).ravel() for i in kern]); thr = np.percentile(allk, 15)
for i in kern:
    masks[i] = (np.abs(W[i]) > thr).astype(np.float32); W[i] = W[i] * masks[i]
m.set_weights(W)


class Mask(tf.keras.callbacks.Callback):
    def on_batch_end(self, b, logs=None):
        w = self.model.get_weights()
        for i, mk in masks.items(): w[i] = w[i] * mk
        self.model.set_weights(w)


m.fit(Xtr_s, y[tr], epochs=10, batch_size=32, callbacks=[Mask()], verbose=0)
pacc = accuracy_score(yte, m.predict(Xte_s, verbose=0).argmax(1)) * 100
nz = sum(int((w != 0).sum()) for w in m.get_weights())
rows.append(dict(stage=f"pruned 15% (nonzero {nz}/{nparam})", acc=pacc, size_bytes=nz * 4, latency_ms=np.nan))

blobp = tflite_convert(m, Xtr_s[:200], int8=True); a, l = tflite_eval(blobp, Xte_s, yte)
rows.append(dict(stage="pruned + tflite int8", acc=a, size_bytes=len(blobp), latency_ms=l))

q = int8_weights(m)
hb = huffman_bits(q) / 8 + 14 * 2  # + rough codebook/scale overhead
rows.append(dict(stage="pruned int8 + Huffman (estimate, raw weights only)", acc=a, size_bytes=int(hb), latency_ms=np.nan))

r = pd.DataFrame(rows); r["size_KB"] = (r["size_bytes"] / 1024).round(2)
print(r.to_string()); r.to_csv(OUT / "compression.csv", index=False)
print("\nPaper: float ~20 KB -> 4.3 KB final (~85% reduction), accuracy drop < 1%, <100 ms inference.")
print("NOTE: TFLite flatbuffer carries metadata overhead, so file size > raw weight bytes. Report both.")
