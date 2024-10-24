import csv
import subprocess
import numpy as np 
# 读取 CSV 文件
python_script = 'draw_all_blobs_3.py'  # 要执行的 Python 脚本路径
seeds = [100, 110, 170, 195, 70, 101, 113, 174, 20, 75, 103, 116, 17, 31, 79, 104, 119, 185, 38, 7, 
105, 148, 188, 51, 91, 109, 161, 189, 54, 10, 167, 192, 67]
for seed in seeds:
    for alpha in range(1,11)
        alpha = 3.1
        c_alpha = 0.6
        temp = 7
        output_path = f'./center4'
        command = ['python', python_script, '-a', alpha, '-ca', c_alpha, '-t', temp, '-s', seed_, '-o', output_path]
        command = [str(item) for item in command]
        # 执行 Python 脚本命令
        subprocess.run(command)