import csv
import subprocess
import numpy as np
import os

python_script = 'code_without_strategy_d1024.py'  # 要执行的 Python 脚本路径
seeds_5blobs = [100, 110, 170, 195, 70, 101, 113, 174, 20, 75, 103, 116, 17, 31, 79, 104, 119, 185, 38, 7, 105, 148, 188, 51, 91, 109, 161, 189, 54, 10, 167, 192, 67]
seeds_3blobs = [100, 139, 175, 26, 77, 101, 141, 176, 29, 78, 103, 142, 179, 31, 79, 104, 144, 17, 37, 7, 
105, 146, 181, 38, 81, 106, 147, 183, 3, 83, 108, 148, 184, 41, 84, 10, 152, 187, 42, 87, 
111, 153, 189, 45, 8, 112, 155, 18, 47, 92, 113, 160, 191, 48, 95, 116, 161, 192, 49, 96, 
118, 163, 193, 51, 97, 119, 166, 198, 56, 98, 128, 167, 199, 60, 9, 130, 168, 20, 67, 131, 
170, 21, 70, 133, 172, 23, 71, 138, 174, 24, 75]
seeds = [7, 10, 17, 20, 148, 31, 161, 38, 167, 170, 174, 51, 189, 192, 67, 70, 75, 79, 100, 101, 103, 104, 105, 113, 116, 119]
alpha_list = [1.02,1.04,1.06,1.08,1.10,1.12,1.14,1.16,1.18,1.20]
selected_seed_55 = [7,10,20,67,116,119,161,167,170,192]
selected_seed_35 = [17,51,75,100,101,104,116,161,170,192]
seeds_51 = [67]
for seed in seeds_51:
    for alpha in alpha_list:
        output_path = f'/home/matteo/github_2/experiments_d1029/without_strategy/3blobs5centroids/seed_{seed}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        command = ['python', python_script, '-a', alpha,'-s', seed, '-o', output_path]
        command = [str(item) for item in command]
        # 执行 Python 脚本命令
        subprocess.run(command)