#!/bin/bash
seed=71
a=5.1
c=5
t=8
ca=0.9

            # 调用 Python 脚本
python3 Blobs_fixed_ahcene_d0731.py -a $a -s ${seed} -c $c -ca $ca -t $t -o "/home/matteo/thesis_2/clustering_module/d0816/images_${a}/images_${seed}_${a}_${t}"
