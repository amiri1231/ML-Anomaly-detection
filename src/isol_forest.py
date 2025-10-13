import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report, confusion_matrix

from .config import MODEL_DIR, RESULTS_DIR, RANDOM_STATE, VAL_SPLIT, TEST_SPLIT
from .data import load_raw, prepare_features
from .features import add_ratio_features

#params are number of samples, percentage of validation and testing data
def split_indices(n_rows, val_split, test_split):
     # 1 - 0.2 - 0.2 = 0.6
    train_frac = 1.0 - val_split - test_split
    #if n = 1000, cut 1 will be 600 and cut2 will be 800
    cut1 = int(train_frac * n_rows)
    cut2 = int((train_frac + val_split) * n_rows)
    idx = np.arange(n_rows)
    #agar n = 10, cut 1 = 1 - 2 - 2 = 6
    #cut 2 = 1 - 0.2 = 0.8
    #idx 1 ache az 0 ta 6, idx 2 ache az 6 ta 8, idx 3 ache az 8 ta 10
    return idx[:cut1], idx[cut1:cut2], idx[cut2:]
#-------------------------------------------------------------------------------

#s_val for anoma. scores from iso forest
#y_val for true labels (1 for anom, 0 for normal)
def choose_threshold_on_validation(y_val, s_val):
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    best_f1 = -1.0
    best_thr = None
    best_p = 0.0
    best_r = 0.0
    #looping ovr thresheolds
    for i in range(len(thr)):
        p, r = float(prec[i]), float(rec[i])
        #F1 score = the harmonic mean of precision and recall.
        f1 = 0.0 if (p + r) <= 0 else 2*p*r/(p+r)
        if f1 > best_f1:
            best_f1, best_thr, best_p, best_r = f1, float(thr[i]), p, r
    #case if all the labels are 0s
    #set everthing to 99th percentile fo the scores, top 1% will be flagged
    if best_thr is None:
        best_thr = float(np.quantile(s_val, 0.99))
    return best_thr, best_f1, best_p, best_r

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading & preparing data")
    raw = load_raw()
    X, y, _ = prepare_features(raw)
    X = add_ratio_features(X)

    print("Splitting Data")
    #train, val, test , count them as row numbers
    tr, va, te = split_indices(len(X), VAL_SPLIT, TEST_SPLIT)
    #iloc toi python abe location indexing (teke Panda)
    #abe row entekhab bekne az numeric postion sho
    #X.iloc[tr] gives all rows from X whose integer positions are in the array tr
    X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]
    y_tr, y_va, y_te = y.iloc[tr], y.iloc[va], y.iloc[te]
    print(f"Shapes: train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}")

    #Scaling with RobustScaler
    #robust scaler is good for data with outliers
    #.fit(X_tr) , computes the median and IQR of each column
    #.transform(X) , uses those stored Median/IQR values to scale every sample ind ataset(subtract median, divide by IQR)
    print("Scaling")
    scaler = RobustScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)


    print("Training IsolationForest")
    contamination = max(0.1, float(y_tr.mean()))
    iso = IsolationForest(n_estimators=200, max_samples="auto",
                          contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1).fit(X_tr_s)

    print("Scoring")
    s_tr = -iso.decision_function(X_tr_s)
    s_va = -iso.decision_function(X_va_s)
    s_te = -iso.decision_function(X_te_s)

    ap_tr = average_precision_score(y_tr, s_tr)
    ap_va = average_precision_score(y_va, s_va)
    ap_te = average_precision_score(y_te, s_te)
    print(f"AUC-PR  train={ap_tr:.3f}  val={ap_va:.3f}  test={ap_te:.3f}")

    print("Choosing threshold on validation")
    thr, f1, p, r = choose_threshold_on_validation(y_va, s_va)
    print(f"VAL best: F1={f1:.3f}, P={p:.3f}, R={r:.3f}, thr={thr:.6f}")

    print("Test evaluation")
    yhat_te = (s_te >= thr).astype(int)
    print("Confusion matrix:\n", confusion_matrix(y_te, yhat_te))
    print("Classification report:\n", classification_report(y_te, yhat_te, digits=3))

    print("Saving bundle and scores")
    joblib.dump({"scaler": scaler, "iso": iso, "thr": thr, "num_cols": list(X.columns)}, MODEL_DIR / "iforest.joblib")
    np.save(RESULTS_DIR / "scores_test.npy", s_te)
    np.save(RESULTS_DIR / "labels_test.npy", y_te.values)
    print("Done.")

if __name__ == "__main__":
    main()
