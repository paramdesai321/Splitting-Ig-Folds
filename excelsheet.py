import os
import pandas as pd

# Set the directory you want to scan
folder_path = './largescaletesting/BestFitPlane'  # Replace with your folder path

# Get list of file names (excluding directories)
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Convert to DataFrame
df = pd.DataFrame(file_names, columns=['File Name'])

# Define output Excel path
output_excel = os.path.join(folder_path, 'file_list.xlsx')

# Write to Excel
df.to_excel(output_excel, index=False)

print(f"Excel file written with {len(file_names)} files at: {output_excel}")

