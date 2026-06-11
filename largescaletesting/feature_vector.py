# features.py
import os
import numpy as np
import pandas as pd

# Where to persist features across runs
#CSV_PATH = "features.csv"
CSV_PATH = "features_ig_CD_HIT_90_final.csv"
# In-memory store for the current process, initialized from CSV
def _load_super_dict_from_csv(csv_path: str = CSV_PATH) -> dict:
    """Load the CSV into a dict-of-dicts."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path, index_col="PIN")
    # convert to dict of dicts, dropping NaNs
    return {
        pin: {k: v for k, v in row.items() if pd.notna(v)}
        for pin, row in df.to_dict(orient="index").items()
    }

super_dict = _load_super_dict_from_csv()

# ---------- helpers ----------
def _coerce_value(v):
    """Unwrap singleton sets; coerce numpy scalars to Python floats for CSV."""
    if isinstance(v, set):
        if len(v) == 1:
            v = next(iter(v))
        else:
            raise ValueError(f"Ambiguous multi-element set: {v}")
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    # Last try: cast to float if possible; otherwise keep as-is (string/object)
    try:
        return float(v)
    except Exception:
        return v

def _upsert_csv(pin: str, pin_features: dict, csv_path: str = CSV_PATH):
    """Upsert one PIN's features into the CSV, unioning columns."""
    new_row = pd.DataFrame(
        [{**{"PIN": pin}, **{k: _coerce_value(v) for k, v in pin_features.items()}}]
    ).set_index("PIN")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col="PIN")
        # unionize columns (include any new features)
        all_cols = sorted(set(df.columns) | set(new_row.columns))
        df = df.reindex(columns=all_cols)
        new_row = new_row.reindex(columns=all_cols)
        # upsert (overwrite existing row for this PIN)
        df.loc[pin] = new_row.loc[pin]
    else:
        df = new_row

    # stable column order is nice
    df = df.reindex(columns=sorted(df.columns))
    df.to_csv(csv_path)

# ---------- public API ----------
def extract_feature(PIN, feature_name, feature_value):
    """Record one feature and persist to CSV immediately."""
    pin_dict = super_dict.get(PIN, {})
    pin_dict[feature_name] = feature_value
    super_dict[PIN] = pin_dict

    # Persist/update this PIN's row on disk so future runs accumulate
    _upsert_csv(PIN, pin_dict)

def get_features():
    """Return the current in-memory dictionary (this run only)."""
    return super_dict
#print(super_dict)
