#!/bin/bash

# 生成的种子文件

seed=71
a=4.1
c=5
t=1
ca=0.1

while [ $(echo "$a < 10" | bc) -eq 1 ]; do
    # 对于每个 a 值，ca 从 0 到 1 循环
    ca=0.1  # 初始化 ca
    while [ $(echo "$ca <= 1" | bc) -eq 1 ]; do
        t=1  # 重置 t
        while [ $(echo "$t < 10" | bc) -eq 1 ]; do
            # 生成输出目录
            output_dir="/home/matteo/github/clustering_module/experiment_all_metric/center_5/${seed}_${a}_${t}_${ca}"
            mkdir -p "$output_dir"

            # 生成输出文件名
            output_file="${output_dir}/seed_${seed}"

            # 调用 Python 脚本
            python3 Blobs_fixed_ahcene_all_metrics.py -a $a -s ${seed} -c $c -ca $ca -t $t -o "$output_dir"

            # 增加 t 参数值
            t=$(echo "$t + 1" | bc)
        done
        
        # 增加 ca 参数值
        ca=$(echo "$ca + 0.2" | bc)
    done

    # 增加 a 参数值
    a=$(echo "$a + 1" | bc)
done
