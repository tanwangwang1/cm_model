#!/bin/bash

# 生成的种子文件
SEEDS_FILE="a_center5.txt"

a=1.1
c=5
t=1
ca=0.1

# 读取种子并循环
while IFS= read -r seed; do
    # 对每个种子，循环所有 a 和 t 的组合
    a=1.1  # 重置 a
    while [ $(echo "$a < 10" | bc) -eq 1 ]; do
        t=1  # 重置 t
        while [ $(echo "$t < 10" | bc) -eq 1 ]; do
            # 生成输出目录
            output_dir="/home/matteo/github/clustering_module/experiment_all_metric/center_5/${seed}_${a}_${t}"
            mkdir -p "$output_dir"

            # 生成输出文件名
            output_file="${output_dir}/seed_${seed}"

            # 调用 Python 脚本
            python3 Blobs_fixed_ahcene_all_metrics.py -a $a -s ${seed} -c $c -ca $ca -t $t -o "$output_dir"

            # 增加 ca 参数值
            ca=$(echo "$ca + 0.1" | bc)
        done
        ca=0.1  # 重置 ca
        # 增加 a 参数值
        a=$(echo "$a + 0.5" | bc)
    done
done < "$SEEDS_FILE"

