import joblib
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from sklearn.utils import shuffle

from .config import MODEL_DIR, RESULTS_DIR, RANDOM_STATE, VAL_SPLIT, TEST_SPLIT
from .data import load_raw, prepare_features
from .features import add_ratio_features




def split_indices(n_rows: int, val_split: float, test_split: float):

    train_frac = 1.0 - val_split - test_split
    cut1 = int(train_frac * n_rows)
    cut2 = int((train_frac + val_split) * n_rows)
    idx = np.arange(n_rows)
    return idx[:cut1], idx[cut1:cut2], idx[cut2:]


def choose_threshold_on_validation(y_val, s_val):
    
    prec, rec, thr_grid = precision_recall_curve(y_val, s_val)
    best_f1, best_thr, best_p, best_r = -1.0, None, 0.0, 0.0
    for i in range(len(thr_grid)):
        p, r = float(prec[i]), float(rec[i])
        f1 = 0.0 if (p + r) <= 0 else (2 * p * r) / (p + r)
        if f1 > best_f1:
            best_f1, best_thr, best_p, best_r = f1, float(thr_grid[i]), p, r
    if best_thr is None:
        best_thr = float(np.quantile(s_val, 0.99))
    return best_thr, best_f1, best_p, best_r


def safe_ap(y, s):

    if np.unique(y).size < 2:
        return float("nan")
    return average_precision_score(y, s)


def grid_search_if(X_tr_s, y_tr, X_va_s, y_va, random_state):
   
    param_grid = [
        {"n_estimators": 400, "max_samples": "auto", "max_features": 1.0, "contamination": float(y_tr.mean())},
        {"n_estimators": 400, "max_samples": 0.5,   "max_features": 1.0, "contamination": float(y_tr.mean())},
        {"n_estimators": 600, "max_samples": 0.5,   "max_features": 1.0, "contamination": float(y_tr.mean())},
        {"n_estimators": 600, "max_samples": 0.5,   "max_features": 0.75,"contamination": float(y_tr.mean())},
        {"n_estimators": 800, "max_samples": 0.5,   "max_features": 1.0, "contamination": float(y_tr.mean())},
        {"n_estimators": 1000, "max_samples": 0.5,   "max_features": 0.75, "contamination": float(y_tr.mean())}
    ]

    best = None
    for params in param_grid:
        iso = IsolationForest(
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            max_features=params["max_features"],
            contamination=params["contamination"],
            random_state=random_state,
            n_jobs=-1,
        ).fit(X_tr_s)

        s_va = -iso.decision_function(X_va_s)
        ap_va = safe_ap(y_va, s_va)
        print(f"[GRID] {params} -> val AUC-PR = {ap_va:.3f}")

        if best is None or ap_va > best["ap_va"]:
            best = {
                "params": params,
                "iso": iso,
                "ap_va": ap_va,
                "s_va": s_va,
            }

    print(f"[GRID] Best params: {best['params']} with val AUC-PR={best['ap_va']:.3f}")
    return best["iso"]

# ----------------------------- Main -----------------------------
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading & preparing data")
    raw = load_raw()
    X, y, _ = prepare_features(raw)
    X = add_ratio_features(X)

    X, y = shuffle(X, y, random_state=RANDOM_STATE)

    print("Splitting Data")
    tr, va, te = split_indices(len(X), VAL_SPLIT, TEST_SPLIT)
    X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]
    y_tr, y_va, y_te = y.iloc[tr], y.iloc[va], y.iloc[te]
    print(f"Shapes: train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}")
    print(
        f"Attack rate: train={float(y_tr.mean()):.3f}, "
        f"val={float(y_va.mean()):.3f}, test={float(y_te.mean()):.3f}"
    )

    print("Scaling")
    scaler = RobustScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    print("Training IsolationForest (grid search)")
    iso = grid_search_if(X_tr_s, y_tr, X_va_s, y_va, RANDOM_STATE)


    print("Scoring")
    s_tr = -iso.decision_function(X_tr_s)
    s_va = -iso.decision_function(X_va_s)
    s_te = -iso.decision_function(X_te_s)

    ap_tr = safe_ap(y_tr, s_tr)
    ap_va = safe_ap(y_va, s_va)
    ap_te = safe_ap(y_te, s_te)
    tr_str = f"{ap_tr:.3f}" if np.isfinite(ap_tr) else "N/A"
    print(f"AUC-PR  train={tr_str}  val={ap_va:.3f}  test={ap_te:.3f}")

    print("Choosing threshold on validation")
    best_thr, f1, p, r = choose_threshold_on_validation(y_va, s_va)
    print(f"VAL best: F1={f1:.3f}, P={p:.3f}, R={r:.3f}, thr={best_thr:.6f}")

    print("Test evaluation")
    yhat_te = (s_te >= best_thr).astype(int)
    cm = confusion_matrix(y_te, yhat_te)
    print("Confusion matrix:\n", cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print("Classification report:\n", classification_report(y_te, yhat_te, digits=3))

    print("Saving bundle and scores")
    joblib.dump(
        {"scaler": scaler, "iso": iso, "thr": float(best_thr), "num_cols": list(X.columns)},
        MODEL_DIR / "iforest.joblib",
    )
    np.save(RESULTS_DIR / "scores_test.npy", s_te)
    np.save(RESULTS_DIR / "labels_test.npy", y_te.values)

    print("▶ Plotting results")

    # 1) PR Curve 
    prec, rec, thr_grid = precision_recall_curve(y_va, s_va)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AUC={ap_va:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Validation)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pr_curve_val.png", dpi=150)
    plt.close()

    # 2) Score Density Plot (Test)
    plt.figure(figsize=(6, 4))

    sorted_scores = np.sort(s_te)
    density, bins = np.histogram(sorted_scores, bins=200, density=True)
    center = (bins[:-1] + bins[1:]) / 2

    plt.plot(center, density, color="steelblue", linewidth=2, label="Score Density")

    plt.axvline(float(best_thr), color="red", linestyle="--", linewidth=1.5,
                label=f"Threshold = {float(best_thr):.4f}")

    plt.title("Anomaly Score Density (Test)", fontsize=12, weight="bold")
    plt.xlabel("Anomaly Score (higher = more anomalous)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "score_density_test.png", dpi=150)
    plt.close()

    # 3) Confusion Matrix
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f"{val}", ha="center", va="center", color="black")
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix_test.png", dpi=150)
    plt.close()

    print("Done. Plots saved in 'results/'")


if __name__ == "__main__":
    main()
