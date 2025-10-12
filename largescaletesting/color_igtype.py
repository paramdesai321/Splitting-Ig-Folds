import pandas as pd

df = pd.read_csv('consolidated_pdbs_umesh_noCD19.csv')

ig_type = df['igtype']

positions_dict = (
    df.reset_index(drop=True)
      .groupby("igtype")
      .apply(lambda g: g.index.tolist())
      .to_dict()
)

pdb_by_ig = (
    df.groupby("igtype")["pdb"]
      .apply(list)
      .to_dict()
)

def type_changed_indices(dictionary):
    result = []
    keys = dictionary.keys()
    for key in keys:
        result.append(dictionary[key][-1])
    return result
print(positions_dict.keys())
#print(pdb_by_ig)
indices = type_changed_indices(positions_dict)
print(indices)
