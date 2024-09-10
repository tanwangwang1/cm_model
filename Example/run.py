import csv
import subprocess
import numpy as np 
# 读取 CSV 文件
csv_file = 'd0712_v0_30.csv'  # CSV 文件路径
python_script = 'Blobs_fixed_ahcene_all_metrics.py'  # 要执行的 Python 脚本路径
np.random.seed(0)
seed_ = np.random(range(201), size=100, replace=False)
for seed_ in range(1,201):
    alpha = 3.1
    c_alpha = 0.6
    temp = 7
    output_path = f'./images_d0727_new/seed_{seed_}/a_{alpha}_c_{c_alpha}_t_{temp}'
    command = ['python', python_script, '-a', alpha, '-ca', c_alpha, '-t', temp, '-s', seed_, '-o', output_path]
    command = [str(item) for item in command]
    # 执行 Python 脚本命令
    subprocess.run(command)