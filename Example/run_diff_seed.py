import csv
import subprocess
import numpy as np 
# 读取 CSV 文件
python_script = 'draw_all_blobs_2.py'  # 要执行的 Python 脚本路径

for seed_ in range(0,201):
    alpha = 3.1
    c_alpha = 0.6
    temp = 7
    output_path = f'./center5'
    command = ['python', python_script, '-a', alpha, '-ca', c_alpha, '-t', temp, '-s', seed_, '-o', output_path]
    command = [str(item) for item in command]
    # 执行 Python 脚本命令
    subprocess.run(command)