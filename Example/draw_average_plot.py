# import pandas as pd
# import matplotlib.pyplot as plt

# # Data for plotting
# # data = {
# #     "Alpha": [1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2],
# #     "ACC": [78.53, 72.05, 70.94, 67.48, 68.67, 68.37, 65.77, 69.86, 65.09, 67.43],
# #     # "SS": [0.484, 0.439, 0.429, 0.417, 0.428, 0.416, 0.404, 0.412, 0.408, 0.406],
# #     # "DBI": [1.026, 1.316, 1.404, 1.363, 1.411, 1.347, 1.317, 1.508, 1.319, 1.286],
# #     # "ARI": [0.736, 0.698, 0.695, 0.663, 0.673, 0.672, 0.648, 0.681, 0.646, 0.658],
# #     # "NMI": [0.794, 0.763, 0.756, 0.745, 0.748, 0.749, 0.74, 0.75, 0.74, 0.742],
# #     # "HS": [0.896, 0.898, 0.9, 0.899, 0.898, 0.898, 0.897, 0.893, 0.898, 0.892],
# #     # "CS": [0.721, 0.667, 0.652, 0.636, 0.642, 0.644, 0.63, 0.648, 0.63, 0.635],
# #     # "VM": [0.794, 0.763, 0.756, 0.745, 0.748, 0.749, 0.74, 0.75, 0.74, 0.742]
# # }
# data = {'New_alpha':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09],
#         #'ACC': [93.5, 94.5, 94.45, 94.51, 94.32, 94.1, 93.9, 93.33, 77.52], 
#         'SS': [0.648, 0.711, 0.686, 0.726, 0.722, 0.698, 0.692, 0.693, 0.387], 
#         'DBI': [0.446, 0.336, 0.364, 0.301, 0.304, 0.341, 0.337, 0.357, 0.61],
#         'ARI': [0.84, 0.849, 0.847, 0.847, 0.842, 0.837, 0.831, 0.823, 0.563], 
#         'NMI': [0.824, 0.832, 0.83, 0.831, 0.827, 0.823, 0.818, 0.811, 0.644], 
#         'CS': [0.817, 0.832, 0.831, 0.833, 0.83, 0.824, 0.821, 0.811, 0.698], 
#         'VW': [0.824, 0.832, 0.83, 0.831, 0.827, 0.823, 0.818, 0.811, 0.644]}


# df = pd.DataFrame(data)

# # Plotting each column as a line on the graph
# plt.figure(figsize=(8, 6))
# for column in df.columns[1:]:  # Exclude 'Alpha' column
#     plt.plot(df["Alpha"], df[column], label=column, marker='o')
    
#     # Highlight the point where Alpha = 1.14 in red
#     alpha_1_14_value = df[df["Alpha"] == 1.02][column].values[0]
#     plt.plot(1.02, alpha_1_14_value, 'ro')  # 'ro' for red circle marker

# # Setting labels and title
# plt.xlabel("Alpha Values", fontsize=14)
# plt.ylabel("Mean Values", fontsize=14)
# plt.title("Mean Values for ACC Across Alpha Values", fontsize=16)
# plt.legend(title="Metrics", loc="best")

# # Use only the specific Alpha values as x-axis ticks
# plt.xticks(df["Alpha"])

# # Save the figure before showing it
# plt.savefig("3blobs5centroids_acc_fig.png", bbox_inches="tight")  # Save the plot with bounding box

# # Show the plot
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Data for plotting
data = {
    'New_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'ACC': [76.81, 76.54, 78.27, 88.89, 95.04, 95.78, 98.08, 96.98, 99.9]}
    # 'SS': [0.562, 0.553, 0.526, 0.578, 0.562, 0.728, 0.751, 0.758, 0.829], 
    # 'DBI': [0.954, 0.934, 1.003, 0.894, 0.721, 0.491, 0.444, 0.405, 0.246], 
    # 'ARI': [0.799, 0.789, 0.785, 0.877, 0.934, 0.958, 0.975, 0.966, 0.999], 
    # 'NMI': [0.865, 0.859, 0.85, 0.902, 0.937, 0.965, 0.975, 0.972, 0.998], 
    # 'HS': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
    # 'CS': [0.763, 0.756, 0.741, 0.822, 0.883, 0.935, 0.952, 0.948, 0.996], 
    # 'VW': [0.865, 0.859, 0.85, 0.902, 0.937, 0.965, 0.975, 0.972, 0.998]}

df = pd.DataFrame(data)

# Plotting each column as a line on the graph
plt.figure(figsize=(8, 6))
for column in df.columns[1:]:  # Exclude 'New_alpha' column
    plt.plot(df["New_alpha"], df[column], label=column, marker='o')
    
    # Highlight the point where New_alpha = 0.01 in red (as an example point)
    new_alpha_0_01_value = df[df["New_alpha"] == 0.9][column].values[0]
    plt.plot(0.9, new_alpha_0_01_value, 'ro')  # 'ro' for red circle marker

# Setting labels and title
plt.xlabel("New_alpha Values", fontsize=14)
plt.ylabel("Mean Values", fontsize=14)
plt.title("Mean Values of ACC Across New_alpha Values", fontsize=16)
plt.legend(title="Metrics", loc="best")

# Use only the specific New_alpha values as x-axis ticks
plt.xticks(df["New_alpha"])

# Save the figure before showing it
plt.savefig("3blobs5centroids_acc_fig.png", bbox_inches="tight")  # Save the plot with bounding box

# Show the plot
plt.show()




