#!/bin/bash

# 循环从1到200的seed

seed=1
while [ $seed -le 200 ]  # 注意中括号内的空格
do
  # 运行Python脚本并传递当前的seed值
  python3 draw_all_blobs_4.py --seed ${seed}

  # 打印调试信息
  echo "Running experiment with seed=${seed}"
  
  # 将 seed 增加1
  seed=$((seed + 1))
done
