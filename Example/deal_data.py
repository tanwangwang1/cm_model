import pandas as pd
import os
file_path = '/home/matteo/github_2/clustering_module/Example/all_scores_3blobs_5centroids.csv'
df = pd.read_csv(file_path, header=None)

# Set column names for clarity
df.columns = ["Alpha", "Seed", "ACC", "SS", "DBI", "ARI", "NMI", "HS", "CS", "VM"]
df_sorted = df.sort_values(by="Alpha")  # Sort by Alpha for clear grouping

# Create a directory to save the 10 files, one for each unique Alpha
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

# Save each Alpha group to a separate CSV file
file_paths = []
for alpha, group in df_sorted.groupby("Alpha"):
    file_path = f"{output_dir}/alpha_{alpha}.csv"
    group.to_csv(file_path, index=False)
    file_paths.append(file_path)