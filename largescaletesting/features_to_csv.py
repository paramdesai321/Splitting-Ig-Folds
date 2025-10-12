import numpy as np
import pandas as pd
import main
def superdict_to_csv(super_dict, filename="features.csv"):
    rows = []
    for pin, features in super_dict.items():
        row = {"PIN": pin}
        for feat, val in features.items():
            # unwrap one-element sets like {np.float64(...)}
            if isinstance(val, set) and len(val) == 1:
                val = next(iter(val))
            # convert numpy scalars to Python floats
            if isinstance(val, (np.floating, np.integer)):
                val = float(val)
            row[feat] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index("PIN", inplace=True)
    df.to_csv(filename)
    print(f"✅ Features written to {filename}")

super_dict = main.super_dict
if __name__ == "__main__":
    superdict_to_csv(super_dict, "features_ig.csv")

