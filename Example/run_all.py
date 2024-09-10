import csv
import subprocess
import pandas as pd
# 读取 CSV 文件
csv_file = '/home/matteo/github/clustering_module/experiment_all_metric/all_scores.csv'  # CSV 文件路径
python_script = 'Blobs_fixed_ahcene_all_metrics.py'  # 要执行的 Python 脚本路径
seed_csv = "/home/matteo/github/clustering_module/Example/a_center4.csv"
df= pd.read_csv(seed_csv)
seed_list = df['seed'].tolist()
for seed_ in seed_list:
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # 确保行非空
                # 假设 CSV 文件的每行数据结构为：param1,param2,param3
                #import pdb;pdb.set_trace()
                alpha, c_alpha, temp= float(row[1].lstrip('\ufeff')),float(row[2]),float(row[3]) # 分割逗号分隔的参数
                output_path = f'/home/matteo/github/clustering_module/experiment_all_metric/center_04/seed_{seed_}/a_{alpha}_c_{c_alpha}_t_{temp}'
                command = ['python', python_script, '-a', alpha, '-ca', c_alpha, '-t', temp, '-s', seed_, '-o', output_path]
                command = [str(item) for item in command]
                # 执行 Python 脚本命令
                subprocess.run(command)