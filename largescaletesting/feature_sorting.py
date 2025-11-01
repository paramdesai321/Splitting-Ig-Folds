import pandas as pd
import numpy as np

df = pd.read_csv("features_ig.csv")

print(df.columns)
sorted_df = df.sort_values(by='Angle between B and C')
print(df.iloc[:,0:6])

outliers_angle_B_C = df['Angle between B and C'] > 2.6 
outliers_angle_B_C  =df['Angle between B and C'] == True
print(outliers_angle_B_C )
