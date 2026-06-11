import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('matched_and_merged_CD_HIT_90_final.csv')

ig_type = df['igtype']

positions_dict = (
    df.reset_index(drop=True)
      .groupby("igtype")
      .apply(lambda g: g.index.tolist())
      .to_dict()
)

pdb_by_ig = (
    df.groupby("igtype")["PIN"]
      .apply(list)
      .to_dict()
)

def type_changed_indices(dictionary):
    result = []
    keys = dictionary.keys()
    for key in keys:
        result.append(dictionary[key][-1])
    return result
#print(positions_dict.keys())
print(len(pdb_by_ig.items()))
indices = type_changed_indices(positions_dict)
#print(indices)

keys = list(positions_dict.keys())
#print(keys)
cmap = plt.get_cmap('tab20')
color_map = {k: cmap(i % 20) for i, k in enumerate(keys)}  # dict: igtype -> RGBA
#print(color_map)

# row-wise color for each record in df
row_colors = df['igtype'].map(color_map)
#print(row_colors)
