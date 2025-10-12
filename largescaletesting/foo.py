import numpy as np
import pandas as pd

df = pd.read_csv('features_ig.csv')
print(df.loc[["PIN", "Angle between B and C"]])
#first_col = df.loc[["PIN","Angle between B and C"]]
#print(first_col)
