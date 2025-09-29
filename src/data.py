from pathlib import Path
import pandas as pd
import numpy as np
from .config import DATA_DIR, DATA_FILES, LABEL_COL

DROP_COLS = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp",
    "Src Port", "Dst Port", "Protocol", #fwd header?
]

def load_raw() -> pd.DataFrame:
    dfs = []
    for fname in DATA_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path.resolve()}")
        df = pd.read_csv(path, low_memory=False)
        # normalize headers (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)

    # Find the label column case-insensitively, then rename to LABEL_COL
    lower_map = {c.lower(): c for c in raw.columns}
    if LABEL_COL.lower() not in lower_map:
        raise ValueError(f"Label column '{LABEL_COL}' not found after stripping headers. "
                         f"Available columns include: {list(raw.columns[-5:])}")
    true_label_name = lower_map[LABEL_COL.lower()]
    if true_label_name != LABEL_COL:
        raw = raw.rename(columns={true_label_name: LABEL_COL})

    # ensure label is string
    raw[LABEL_COL] = raw[LABEL_COL].astype(str)
    return raw


def prepare_features(raw: pd.DataFrame):
    # drop non-features if present
    to_drop = [c for c in DROP_COLS if c in raw.columns]
    df = raw.drop(columns=to_drop, errors="ignore").copy()

    # numeric-only features
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if LABEL_COL in num_cols:
        num_cols.remove(LABEL_COL)
    if len(num_cols) == 0:
        raise ValueError("No numeric feature columns found after dropping identifiers.")

    X = (df[num_cols]
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0)
         .astype(float))

    # binary labels: BENIGN -> 0, anything else -> 1
    y_list = []
    for v in df[LABEL_COL]:
        y_list.append(0 if isinstance(v, str) and "BENIGN" in v.upper() else 1)
    y = pd.Series(y_list, index=df.index)

    return X, y, num_cols
