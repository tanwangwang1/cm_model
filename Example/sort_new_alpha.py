import pandas as pd

# 读取原始数据文件
df = pd.read_csv('/home/matteo/github_2/clustering_module/Example/new_alpha_plot_67.csv')

# # 按照 'New_alpha' 排序并保留小数点后三位
# df_sorted = df.sort_values(by='New_alpha').round(3)

# # 按照 'New_alpha' 分组并计算最后8列的均值和标准差
# mean_std_df = df_sorted.groupby('New_alpha').agg(['mean', 'std']).iloc[:, -16:]

# # 格式化输出为指定的 & 格式，并在每行结尾加上两个反斜杠
# output_str = "\n".join([
#     f"{index:.2f} & " + " & ".join(f"{mean_val:.3f} & {std_val:.3f}" 
#                                    for mean_val, std_val in zip(row[::2], row[1::2])) + " \\\\"
#     for index, row in mean_std_df.iterrows()
# ])
# print(output_str)
# 去除列名中的多余空格
df.columns = df.columns.str.strip()

# 按照 'New_alpha' 排序并保留小数点后三位
df_sorted = df.sort_values(by='New_alpha').round(3)

# 按照 'New_alpha' 分组并计算最后8列的均值和标准差
mean_std_df = df_sorted.groupby("New_alpha")[["ACC", "SS", "DBI", "ARI", "NMI", "HS", "CS", "VW"]].agg(['mean', 'std'])

# 格式化输出为指定的 & 格式，并在每行结尾加上两个反斜杠
output_str = "\n".join([
    f"{index:.2f} & " + " & ".join(f"{mean_val:.3f} & {std_val:.3f}" 
                                   for mean_val, std_val in zip(row[::2], row[1::2])) + " \\\\"
    for index, row in mean_std_df.iterrows()
])

# 提取不同 New_alpha 的均值并构建新的字典
mean_values_dict = {col: [round(val, 3) for val in mean_std_df[(col, 'mean')]] for col in mean_std_df.columns.levels[0]}



# 打印输出
print("Formatted Table Output:")
print(output_str)
print("\nMean values dictionary:")
print(mean_values_dict)









