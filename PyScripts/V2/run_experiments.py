"""Unified experiment runner. Usage (from project root, venv active):
    python PyScripts\\run_experiments.py audit
    python PyScripts\\run_experiments.py cv        [--repeats 1]
    python PyScripts\\run_experiments.py grid      [--n-configs 12 --folds 3]
    python PyScripts\\run_experiments.py ablation
    python PyScripts\\run_experiments.py noise
    python PyScripts\\run_experiments.py baseline
    python PyScripts\\run_experiments.py challenge
Outputs in Results\\.
"""
import argparse, itertools, random, time, re
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from blueedge_common import *


# ------------------------------------------------------------------ helpers
def template(s):
    s = re.sub(r"[A-Z]", "A", str(s)); s = re.sub(r"[a-z]", "a", s); return re.sub(r"\d", "0", s)


def cv_run(X, y, cols, params, seed, n_splits=10):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    accs, oof = [], np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        m, sc = fit_model(X[tr][:, cols], y[tr], seed, params)
        pred = predict_probs(m, sc, X[te][:, cols]).argmax(1)
        oof[te] = pred; accs.append(accuracy_score(y[te], pred))
    return np.array(accs), oof


def mcnemar(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b)); c = int(np.sum(~correct_a & correct_b))
    if b + c == 0: return b, c, 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, stat, float(chi2.sf(stat, 1))


def holdout_model(seed=SEEDS[0]):
    df, X, y, le = load_v4()
    tr, te = holdout_idx(y, seed)
    m, sc = fit_model(X[tr], y[tr], seed)
    return df, X, y, le, tr, te, m, sc


# ------------------------------------------------------------------ commands
def cmd_audit(a):
    df, X, y, le = load_v4()
    print("Class counts:\n", df["data_type"].value_counts().to_string())
    print("\nExact duplicate samples (same text):", df.duplicated("sample").sum())
    print("Same text, different label:", (df.groupby("sample")["data_type"].nunique() > 1).sum())
    df["tpl"] = df["sample"].map(template)
    div = df.groupby("data_type")["tpl"].nunique().rename("unique_templates")
    div = pd.concat([div, df.groupby("data_type")["sample"].nunique().rename("unique_samples")], axis=1)
    print("\nTemplate diversity per class (low = templated/too easy):\n", div.to_string())
    tr, te = holdout_idx(y, SEEDS[0])
    tr_t, te_t = set(df.iloc[tr]["tpl"]), df.iloc[te]["tpl"]
    leak = te_t.isin(tr_t).mean()
    print(f"\nTest samples whose template also appears in train: {leak:.1%}")
    print("Test samples with exact text in train:", df.iloc[te]["sample"].isin(set(df.iloc[tr]["sample"])).sum())
    # regex-only decodability
    pred = [regex_classify(s) for s in df["sample"]]
    print(f"Pure regex baseline accuracy on all 1400: {accuracy_score(df['data_type'], pred):.3f}  (high => data trivially easy)")
    div.to_csv(OUT / "audit_template_diversity.csv")


def cmd_cv(a):
    df, X, y, le = load_v4()
    rows = []
    for seed in SEEDS[:a.repeats]:
        t = time.time(); accs, oof = cv_run(X, y, ALL, {}, seed)
        print(f"seed {seed}: mean {accs.mean()*100:.2f} +- {accs.std()*100:.2f}  min {accs.min()*100:.2f}  max {accs.max()*100:.2f}  ({time.time()-t:.0f}s)")
        rows.append(dict(seed=seed, mean=accs.mean(), std=accs.std(), min=accs.min(), max=accs.max()))
        if seed == SEEDS[0]:
            rng = np.random.default_rng(0); ok = (oof == y)
            bs = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(1000)]
            print(f"  bootstrap 95% CI (n=1000): [{np.percentile(bs,2.5)*100:.2f}, {np.percentile(bs,97.5)*100:.2f}]")
            print(f"  CV coefficient of variation: {accs.std()/accs.mean():.4f}  (paper 0.014)")
            print(pd.DataFrame(confusion_matrix(y, oof), index=le.classes_, columns=le.classes_).to_string())
    pd.DataFrame(rows).to_csv(OUT / "cv_results.csv", index=False)
    print("Paper target: 94.2 +- 1.3")


def cmd_grid(a):
    df, X, y, le = load_v4()
    full = list(itertools.product([1e-4, 5e-4, 1e-3, 5e-3, 1e-2], [16, 32, 64, 128],
                                  [0.1, 0.2, 0.3, 0.4, 0.5], [1e-4, 1e-3, 1e-2]))
    print("Full grid size:", len(full))
    cfgs = random.Random(0).sample(full, a.n_configs)
    rows, trainings, t0 = [], 0, time.time()
    for lr, bs, dr, l2 in cfgs:
        accs, _ = cv_run(X, y, ALL, dict(lr=lr, batch=bs, dropout=dr, l2=l2), SEEDS[0], a.folds)
        trainings += a.folds
        rows.append(dict(lr=lr, batch=bs, dropout=dr, l2=l2, mean_acc=accs.mean(), std=accs.std()))
        print(rows[-1])
    per = (time.time() - t0) / trainings
    print(f"\nSeconds per training: {per:.1f}")
    print(f"Projected full grid x 10-fold: {per*len(full)*10/3600:.2f} h   (paper: 4.8 h CPU / 0.75 h GPU)")
    print(f"Projected x 10 repeats: {per*len(full)*10*10/3600:.2f} h")
    pd.DataFrame(rows).sort_values("mean_acc", ascending=False).to_csv(OUT / "grid_subset.csv", index=False)


def cmd_ablation(a):
    df, X, y, le = load_v4()
    cfg = {"full model": ALL}
    for g, idx in GROUPS.items():
        cfg[f"without {g}"] = [i for i in ALL if i not in idx]
    cfg["only pattern"] = GROUPS["pattern"]; cfg["only statistical"] = GROUPS["statistical"]
    paper = {"full model": 94.2, "without length": 91.8, "without pattern": 83.5, "without statistical": 92.7,
             "without content": 90.3, "only pattern": 78.2, "only statistical": 67.5}
    rows = []
    for name, cols in cfg.items():
        accs, _ = cv_run(X, y, cols, {}, SEEDS[0], a.folds)
        rows.append(dict(config=name, acc=accs.mean() * 100, std=accs.std() * 100, paper=paper[name]))
        print(rows[-1])
    r = pd.DataFrame(rows); r["delta_vs_full"] = r["acc"] - r.loc[0, "acc"]; r["paper_delta"] = r["paper"] - 94.2
    print(r.to_string()); r.to_csv(OUT / "ablation.csv", index=False)


def cmd_noise(a):
    df, X, y, le, tr, te, m, sc = holdout_model()
    strs = df.iloc[te]["sample"].tolist(); yt = y[te]; names = le.classes_
    rows = []
    for kind, levels in (("noise", [0, .05, .10, .20, .30]), ("completeness", [1.0, .8, .6, .4])):
        for lv in levels:
            accs, racc = [], []
            for s in SEEDS:
                rng = random.Random(s)
                pert = [add_noise(t, lv, rng) if kind == "noise" else drop_chars(t, lv, rng) for t in strs]
                p = predict_probs(m, sc, featurize(pert)).argmax(1)
                accs.append(accuracy_score(yt, p))
                racc.append(np.mean([regex_classify(t) == names[k] for t, k in zip(pert, yt)]))
            rows.append(dict(kind=kind, level=lv, nn=np.mean(accs) * 100, regex=np.mean(racc) * 100))
            print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "robustness.csv", index=False)
    print("Paper: >80% at 20% noise. Completeness 80%: 91.7, 60%: 85.3, 40%: 72.6 (rule 76.3/59.6/35.2, regex 68.2/42.1/21.4)")


def cmd_baseline(a):
    df, X, y, le, tr, te, m, sc = holdout_model()
    names = le.classes_; strs = df.iloc[te]["sample"].tolist(); yt = y[te]
    nn = predict_probs(m, sc, X[te]).argmax(1)
    rg = np.array([list(names).index(regex_classify(s)) for s in strs])
    print(f"NN holdout acc:    {accuracy_score(yt, nn)*100:.2f}")
    print(f"Regex baseline:    {accuracy_score(yt, rg)*100:.2f}")
    b, c, stat, p = mcnemar(nn == yt, rg == yt)
    print(f"McNemar NN vs regex: b={b} c={c} chi2={stat:.2f} p={p:.4g}  (paper vs RegEx: chi2 52.1)")
    print(classification_report(yt, rg, labels=range(14), target_names=names, zero_division=0, digits=3))


def cmd_challenge(a):
    df, X, y, le = load_v4()
    m, sc = fit_model(X, y, SEEDS[0])
    for name, path in (("challenge", CHALLENGE), ("university", UNI_TEST)):
        if not path.exists():
            print(f"{path} missing. Run build_eval_sets.py first."); continue
        d = pd.read_csv(path, encoding="utf-8", dtype=str).fillna("")
        known = d["data_type"].isin(le.classes_); d = d[known].reset_index(drop=True)
        probs = predict_probs(m, sc, featurize(d["sample"])); pred = probs.argmax(1); conf = probs.max(1)
        yt = le.transform(d["data_type"])
        print(f"\n=== {name}: n={len(d)}  acc={accuracy_score(yt, pred)*100:.2f}%  (paper challenge: 88.6%)")
        print(f"flagged conf<0.6: {(conf < 0.6).sum()}   acc on accepted: {accuracy_score(yt[conf>=.6], pred[conf>=.6])*100:.2f}%")
        rg = np.array([regex_classify(s) for s in d["sample"]])
        print(f"regex baseline acc: {np.mean(rg == d['data_type'].values)*100:.2f}%")
        d["predicted"] = le.inverse_transform(pred); d["confidence"] = conf.round(3)
        wrong = d[d["predicted"] != d["data_type"]]
        wrong.to_csv(OUT / f"{name}_errors.csv", index=False, encoding="utf-8-sig")
        print(wrong.head(25).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "cv", "grid", "ablation", "noise", "baseline", "challenge"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--n-configs", type=int, default=12)
    ap.add_argument("--folds", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "grid" and "--folds" not in __import__("sys").argv: a.folds = 3
    globals()["cmd_" + a.cmd](a)
