#!/bin/bash

# 生成的种子文件

t=5
n=0.1

while [ $(echo "$n <= 1" | bc) -eq 1 ]; do
    t=1  # 重置 t
    while [ $(echo "$t < 15" | bc) -eq 1 ]; do
        # 生成输出目录
        output_dir="/home/matteo/github/clustering_module/Example/txt_file/n_${n}_t_${t}"
        # 调用 Python 脚本
        python3 cm_mnist.py  -n $n -t $t > "$output_dir"
        # 增加 t 参数值
        t=$(echo "$t + 1" | bc)
    done
    
    # 增加 ca 参数值
    n=$(echo "$n + 0.1" | bc)
done
