import pandas as pd
import matplotlib.pyplot as plt
# 读取原始数据文件
df = pd.read_csv('/home/matteo/github_2/experiments_d1104/with_strategy/3blobs5centroids/seed_67/all_scores_3blobs_5centroids.csv')

df.columns = df.columns.str.strip()

# 按照 'New_alpha' 排序并保留小数点后三位
df_sorted = df.sort_values(by='New_alpha').round(3)

# 按照 'New_alpha' 分组并计算最后8列的均值和标准差
mean_std_df = df_sorted.groupby("New_alpha")[["ACC", "SS", "DBI", "ARI", "NMI", "HS", "CS", "VM"]].agg(['mean', 'std'])

# 格式化输出为指定的 & 格式，并在每行结尾加上两个反斜杠
output_str = "\n".join([
    f"{index:.2f} & " + " & ".join(f"{mean_val:.3f} & {std_val:.3f}" 
                                   for mean_val, std_val in zip(row[::2], row[1::2])) + " \\\\"
    for index, row in mean_std_df.iterrows()
])

# 提取不同 New_alpha 的均值并构建新的字典
mean_values_dict = {col: [round(val, 3) for val in mean_std_df[(col, 'mean')]] for col in mean_std_df.columns.levels[0]}



# 打印输出
print("Latex Format in order to draw mean_std figure:")
print(output_str)
print("\nPrepaire for drawing the curve:")
print(mean_values_dict)
mean_values_dict["Alpha"] = [1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20]

df = pd.DataFrame(mean_values_dict)

colors = {
    'ACC': 'blue',
    'SS': 'green',
    'DBI': 'orange',
    'ARI': 'purple',
    'NMI': 'cyan',
    'HS': 'brown',
    'CS': 'pink',
    'VM': 'gray'
}

# Plotting with fixed x-axis ticks
plt.figure(figsize=(10, 6))
for column, color in colors.items():  # Use specified color for each metric
    plt.plot(df['Alpha'], df[column], marker='o', label=column, color=color)
    # Highlight the point at Alpha = 1.12
    alpha_1_12_value = df.loc[df['Alpha'] == 1.02, column].values[0]  # Get the y value for Alpha = 1.12
    plt.scatter(1.02, alpha_1_12_value, color='red', zorder=5)  # Plot the red point on top

# Customize plot settings
plt.xlabel("Alpha", fontsize=14)
plt.ylabel("Mean Values", fontsize=14)
plt.title("Mean Values of Metrics Across Alpha Values", fontsize=16)
plt.xticks(df['Alpha'])  # Set x-axis ticks to the specific Alpha values
plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.grid(True)

plt.tight_layout()
savepath = "/home/matteo/github_2/experiments_d1104/with_strategy/3blobs5centroids/means_across_alpha.png"
plt.savefig(savepath,dpi=300)