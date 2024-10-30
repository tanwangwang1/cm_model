import pandas as pd

file_path = "/home/matteo/github_2/clustering_module/Example/sorted_by_alpha.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Filter data for a specific Alpha value (e.g., 1.04)
alpha_list = [1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18,1.2]
for alpha_value in alpha_list:
  data_alpha = data[data['Alpha'] == alpha_value]

  # Calculate means and standard deviations, excluding 'Alpha' and 'Seed'
  means = data_alpha.iloc[:, 2:].mean(numeric_only=True).round(3)
  stds = data_alpha.iloc[:, 2:].std(numeric_only=True).round(3)
  # print('\n')
  # print("Alpha", alpha_value)
  # print("平均值:")
  print(", ".join(map(str, means.values)))

  # print("标准差:")
  # print("& ".join(map(str, stds.values)))
  # Interleave means and standard deviations for display
  # interleaved_output = f"{alpha_value} & " + " & ".join(f"{mean} & {std}" for mean, std in zip(means.values, stds.values)) + "\\\\"

  # print(interleaved_output)

