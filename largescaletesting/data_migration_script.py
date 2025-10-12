import pandas as pd

# Define file paths
PATH1 = "input_data.xlsx"
PATH2 = "features_ig.csv"
PATH3 = "consolidated_pdbs_umesh_dummy.csv"

# Read the files (use read_excel if Excel)
df1 = pd.read_excel(PATH1)
df2 = pd.read_csv(PATH2)

# Ensure first column names
col1 = df1.columns[0]
col2 = df2.columns[0]

# Extract first 4 letters
df1["prefix"] = df1[col1].astype(str).str[:4]
df2["prefix"] = df2[col2].astype(str).str[:4]

# Merge on prefix (so we can take col2 instead of col1)
merged = pd.merge(df1, df2[[col2, "prefix"]], on="prefix", how="inner")

# Replace first column of df1 with col2 from df2
cols = [col2] + [c for c in df1.columns if c not in [col1, "prefix"]]
result = merged[cols]

# Save result
result.to_csv(PATH3, index=False)
