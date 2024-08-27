#!/bin/bash
seed=71
a=7.1
c=5
t=4
output_dir="/home/matteo/thesis_2/clustering-Module-main/d08012/images_${seed}_${a}_${t}"
ca=0.6
while [ $(echo "$a < 10" | bc) -eq 1 ]; do
    while [ $(echo "$t < 10" | bc) -eq 1 ]; do
        # 循环调用 Python 脚本
        while [ $(echo "$ca <= 1" | bc) -eq 1 ]; do
            # 生成输出文件名
            output_file="${output_dir}/ca_${ca}"

            # 调用 Python 脚本
            python3 Blobs_fixed_ahcene_d0812.py -a $a -s ${seed} -c $c -ca $ca -t $t -o "/home/matteo/thesis_2/clustering_module/d0815/images_${a}/images_${seed}_${a}_${t}"

            # 增加 ca 参数值
            ca=$(echo "$ca + 0.1" | bc)
        done
        ca=0.1
        # 增加 t 参数值
        t=$(echo "$t + 1" | bc)
    done
    t=1
    a=$(echo "$a + 0.5" | bc)
done
