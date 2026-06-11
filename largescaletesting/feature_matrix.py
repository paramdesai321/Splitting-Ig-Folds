import numpy as np
import feature_vector
import main
import plotting
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import sys

PIN = sys.argv[1]

import pandas as pd

def reorder_csv_columns(csv_path: str, feature_order: list[str], output_path: str | None = None):
    df = pd.read_csv(csv_path)

    # keep first column (e.g., 'PIN') fixed in place
    first_col = df.columns[0]

    # only keep features that actually exist in the CSV
    ordered_features = [f for f in feature_order if f in df.columns]

    # preserve any leftover columns not mentioned in feature_order
    leftovers = [c for c in df.columns if c not in ordered_features and c != first_col]

    # build the new column order
    new_col_order = [first_col] + ordered_features + leftovers

    df = df[new_col_order]

    out_path = output_path or csv_path
    df.to_csv(out_path, index=False)

    return df

def read_pin_order(path, column="PIN"):
    import pandas as pd
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df[column].astype(str).tolist()

def _to_scalar(v):
    """Coerce supported values to a numeric scalar."""
    # unwrap one-element sets like {np.float64(...)}
    if isinstance(v, set):
        if len(v) == 1:
            v = next(iter(v))
        else:
            raise ValueError(f"Ambiguous set with {len(v)} elements: {v}")

    # numpy scalars → float
    if isinstance(v, (np.floating, np.integer)):
        return float(v)

    # plain Python numerics
    if isinstance(v, (int, float, bool)):
        return float(v)
    elif  isinstance(v,(str)):
        return str(v)
    # last resort: try float() (will fail for strings, dicts, etc.)
    try:
        return float(v)
    except Exception as e:
        raise TypeError(f"Unsupported value type {type(v)} for {v}") from e


#def vectorize_super_dict(super_dict, fill_value=np.nan, sort_features=True, sort_pins=True):
#    """
#    Returns:
#        X : np.ndarray, shape (n_pins, n_features)
#        pins : list[str]
#        feature_names : list[str]
#    """
#    # 1) stable orderings
#    pins = list(super_dict.keys())
#    if sort_pins:
#        pins = sorted(pins)
#
#    feature_set = set()
#    for pin in pins:
#        feature_set.update(super_dict[pin].keys())
#
#    feature_names = sorted(feature_set) if sort_features else list(feature_set)
#
#    # 2) build matrix
#    X = np.full((len(pins), len(feature_names)), fill_value, dtype=float)
#
#    feat_idx = {f: j for j, f in enumerate(feature_names)}
#    for i, pin in enumerate(pins):
#        for f, v in super_dict[pin].items():
#            j = feat_idx[f]
#            X[i, j] = _to_scalar(v)
#
#    return X, pins, feature_names
#
def vectorize_super_dict(
    super_dict, 
    fill_value=np.nan, 
    sort_features=True, 
    sort_pins=True, 
    pin_order=None,          # <- NEW
    strict_order=False       # <- NEW: if True, only keep pins present in pin_order
):
    """
    Returns:
        X : np.ndarray, shape (n_pins, n_features)
        pins : list[str]           (in the desired order)
        feature_names : list[str]
    """
    # --- choose pin order ---
    if pin_order is not None:
        pin_order = [str(p) for p in pin_order]  # normalize to str
        sd_keys = set(map(str, super_dict.keys()))
        # pins that exist in super_dict, in the exact order from pin_order
        pins = [p for p in pin_order if p in sd_keys]
        if not strict_order:
            # append any leftover pins from super_dict that weren’t listed
            pins += [str(p) for p in super_dict.keys() if str(p) not in pin_order]
    else:
        # original behavior
        pins = list(map(str, super_dict.keys()))
        if sort_pins:
            pins = sorted(pins)

    # --- collect features ---
    feature_set = set()
    for pin in pins:
        feature_set.update(super_dict[str(pin)].keys())

    feature_names = sorted(feature_set) if sort_features else list(feature_set)

    # --- build matrix ---
    X = np.full((len(pins), len(feature_names)), fill_value, dtype=float)

    feat_idx = {f: j for j, f in enumerate(feature_names)}
    for i, pin in enumerate(pins):
        for f, v in super_dict[str(pin)].items():
            j = feat_idx[f]
            X[i, j] = _to_scalar(v)

    return X, pins, feature_names
feature_order = main.sort_features_by_module_order(feature_vector.super_dict[PIN])
input_csv = reorder_csv_columns("features_ig_CD_HIT_90_final.csv", feature_order)
pin_order = read_pin_order("features_ig_CD_HIT_90_final.csv", column="PIN")


#print(pin_order)
X, pins, feature_names = vectorize_super_dict(
    feature_vector.super_dict,
    sort_pins=False,      
    sort_features=True,
    pin_order=pin_order,   
    strict_order=True     
)

def export_X():
    return np.array(X)
X = np.array(X)
X= X.T
print(X[1:7])
print(X[7:11])
#print(feature_names)
#print(X.T)
#print(np.argwhere(np.isnan(X.T)))
print(X.shape)
#print(X[3])
#print(feature_vector.super_dict)
#print(pins)
#print(pins[4253])          # ['1A4K_L_3_to_107']
#print(feature_names) # ordered list of all features
#print(feature_names)
#print(X.shape)       # (1, n_features)
#print(X)          # the numeric vector for that PIN
#dummy = list(X)
#X = X[~np.isnan(X).any(axis=1)]
#print(X.shape)
#Example array
#print(np.isnan(X))
data = X
#print(f"{data[:5]}")
data1 = data[0]
data2 = data[1]
data2 = data[2]
data3 = data[3]
#print(f"data3:{data3}")
data4 = data[4]
data5= data[5]
#print(f"data4:{data4}")

#print(f"data 4: {data4}")

#print(data.shape)
#print("Data for violin plot:", data)
arr = np.asarray(data[0]*180/np.pi, dtype=float).ravel()         # ensure 1-D float
arr = arr[np.isfinite(arr)]                         # drop any NaN/Inf silently
arr1 = np.asarray(data[1]*180/np.pi, dtype=float).ravel()         # ensure 1-D float
arr1 = arr1[np.isfinite(arr1)]
arr2 = np.asarray(data[2]*180/np.pi, dtype=float).ravel()         # ensure 1-D float
arr2 = arr2[np.isfinite(arr2)]
arr3 = np.asarray(data[3]*180/np.pi, dtype=float).ravel()         # ensure 1-D float
arr3 = arr3[np.isfinite(arr3)]
arr4 = np.asarray(data[4]*180/np.pi, dtype=float).ravel()         # ensure 1-D float
arr4 = arr4[np.isfinite(arr4)]
arr5 = np.asarray(data[5]*180/np.pi, dtype=float).ravel()         # ensure 1-D float
arr5 = arr5[np.isfinite(arr5)]
#print(arr)
#print(X[200:])

#print(f"min arr0: {np.min(arr)}")
#print(f"data3:{data3}")

#print(f"arr: {arr}")
distances = X.T[-4:]
#print(f"X :{X.shape}")
#print(f"X.T : {X.T.shape}")

projections = X.T[5:9]
#print(f"distances : {distances[0].shape}")
#print(f"projections: {projections.shape}")
#print(f"data : {data.shape}")
#print(f"arr3:{arr3}")
#print(f"arr4:{arr4}")

#if arr.size < 2:
# raise ValueError("Need at least 2 finite values for a violin plot.")

vmin, vmax = float(arr.min()), float(arr.max())
spread = vmax - vmin
pad = 0.1 * spread if spread > 0 else 1.0           # fallback if constant

fig, ax = plt.subplots()


#ax.violinplot([data[10]*180/np.pi], positions=[1],                  # wrap in list => 1 dataset
#          showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([data[13]*(180/np.pi)], positions=[4],                  # wrap in list => 1 dataset
#         showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([data[14]*(180/np.pi)], positions=[7],                  # wrap in list => 1 dataset
#          showmeans=True, showmedians=True, showextrema=True)
#
#ax.violinplot([data[19]*(180/np.pi)], positions=[10],                  # wrap in list => 1 dataset
#          showmeans=True, showmedians=True, showextrema=True)

#ax.violinplot([arr], positions=[1],                  # wrap in list => 1 dataset
#       showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([arr1], positions=[3],                  # wrap in list => 1 dataset
#       showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([arr2], positions=[5],                  # wrap in list => 1 dataset
#       showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([arr3], positions=[7],                  # wrap in list => 1 dataset
#       showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([arr4], positions=[9],                  # wrap in list => 1 dataset
#         showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([arr5], positions=[11],                  # wrap in list => 1 dataset
#         showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([projections[0]*180/np.pi], positions=[1],                  # wrap in list => 1 dataset
#ax.violinplot([projections[0]*180/np.pi], positions=[1],                  # wrap in list => 1 dataset
#          showmeans=True, showmedians=True, showextrema=True)
#
#ax.violinplot([projections[1]*180/np.pi], positions=[2],                  # wrap in list => 1 dataset
#           showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([projections[2]*180/np.pi], positions=[3],                  # wrap in list => 1 dataset
#           showmeans=True, showmedians=True, showextrema=True)

#ax.violinplot([distances[0]], positions=[1],                  # wrap in list => 1 dataset
#           showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([distances[1]], positions=[2],                  # wrap in list => 1 dataset
#           showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([distances[2]], positions=[3],                  # wrap in list => 1 dataset
#           showmeans=True, showmedians=True, showextrema=True)
#ax.violinplot([distances[3]], positions=[4],                  # wrap in list => 1 dataset
#           showmeans=True, showmedians=True, showextrema=True)
#
#ax.set_xlim(0,12)
#ax.set_ylim(0,360)
#ax.set_xticks([1,3,5,7,9,11])
#ax.set_xticklabels(["B and C","B and E","B and F","C and E","C and F","E and F"])
#ax.set_xlabel("Strand Pairs")
#ax.set_ylabel("Angle(degrees)")
#ax.set_title("Angle between strands")
#plt.show()

#g1 = np.asarray(data[0], dtype=float).ravel()
#g2 = np.asarray(data[1], dtype=float).ravel()
#
#
#fig, ax = plt.subplots()
#
#p1 = ax.violinplot([g1], positions=[1], showmeans=True, showmedians=True)
#p2 = ax.violinplot([g2], positions=[1], showmeans=True, showmedians=True)
#
#for b in p1['bodies']: b.set_alpha(0.45)
#for b in p2['bodies']: b.set_alpha(0.45)
#
#ax.set_xticks([1]); ax.set_xticklabels(['g1 vs g2'])
#plt.show()
#
#updated_feature_order = main.sort_features_by_module_order(main.updated_super_dict[PIN])
#updated_input_csv = reorder_csv_columns("features_ig_CD_HIT_90.csv", feature_order)
#updated_pin_order = read_pin_order("features_ig_CD_HIT_90.csv", column="PIN")
#updated_X, pins, feature_names = vectorize_super_dict(
#    main.updated_super_dict,
#    sort_pins=False,      
#    sort_features=True,
#    pin_order=pin_order,   
#    strict_order=True     
#)
#updated_X  = np.array(updated_X)
#print(updated_X.shape)
#
