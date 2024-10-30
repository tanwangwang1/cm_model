# import pandas as pd
# import os
# file_path = '/home/matteo/github_2/clustering_module/Example/all_scores_3blobs_5centroids.csv'
# df = pd.read_csv(file_path, header=None)

# # Set column names for clarity
# df.columns = ["Alpha", "Seed", "ACC", "SS", "DBI", "ARI", "NMI", "HS", "CS", "VM"]
# df_sorted = df.sort_values(by="Alpha")  # Sort by Alpha for clear grouping

# # Create a directory to save the 10 files, one for each unique Alpha
# output_dir = './'
# os.makedirs(output_dir, exist_ok=True)

# # Save each Alpha group to a separate CSV file
# file_paths = []
# for alpha, group in df_sorted.groupby("Alpha"):
#     file_path = f"{output_dir}/alpha_{alpha}.csv"
#     group.to_csv(file_path, index=False)
#     file_paths.append(file_path)

# import pandas as pd

# # Load the uploaded CSV file to check for rows with New_alpha == 0.8
# file_path = '/home/matteo/github_2/clustering_module/Example/all_scores_3blobs_5centroids - Sheet1.csv'
# df = pd.read_csv(file_path)

# # Filter rows where New_alpha equals 0.8
# new_alpha_08_rows = df[df['New_alpha'] == 0.8]
# import ace_tools as tools; tools.display_dataframe_to_user(name="New Alpha = 0.8 Rows", dataframe=new_alpha_08_rows)
import pandas as pd

# Load the uploaded CSV file
file_path = '/home/matteo/github_2/clustering_module/Example/3blobs5centroids.csv'
data = pd.read_csv(file_path)

# Sort and group by the 'Alpha' column
sorted_data = data.sort_values(by='Alpha')
sorted_data = sorted_data.round(3)
# Save the sorted data to a new CSV file
output_path = './sorted_by_alpha.csv'
sorted_data.to_csv(output_path, index=False)

