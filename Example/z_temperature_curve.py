import pandas as pd
import matplotlib.pyplot as plt

# Data for plotting
data = {
    'Temperature': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    'ACC': [1.0, 0.951, 0.973, 1.0, 0.977, 0.988, 1.0, 1.0, 0.936],
    'SS': [0.846, 0.661, 0.656, 0.846, 0.666, 0.676, 0.846, 0.846, 0.66],
    'DBI': [0.214, 0.632, 0.653, 0.214, 0.57, 0.561, 0.214, 0.214, 0.68],
    'ARI': [1.0, 0.936, 0.962, 1.0, 0.968, 0.983, 1.0, 1.0, 0.921],
    'NMI': [1.0, 0.94, 0.959, 1.0, 0.964, 0.977, 1.0, 1.0, 0.931],
    'HS': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'CS': [1.0, 0.887, 0.921, 1.0, 0.93, 0.956, 1.0, 1.0, 0.871],
    'VM': [1.0, 0.94, 0.959, 1.0, 0.964, 0.977, 1.0, 1.0, 0.931]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set up plot
plt.figure(figsize=(10, 6))
metrics = df.columns[1:]  # Exclude 'Temperature' column

# Plot each metric
for metric in metrics:
    plt.plot(df['Temperature'], df[metric], marker='o', label=metric)

# Customize plot settings
plt.xlabel("Temperature", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.title("Metrics Scores Across Different Temperatures", fontsize=16)
plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(df['Temperature'])  # Set x-axis ticks to the specific Temperature values
#plt.grid(True)

plt.tight_layout()
savepath = savepath = "/home/matteo/github_2/experiments_d1104/with_strategy/3blobs5centroids/means_across_temp.png"
plt.savefig(savepath)
