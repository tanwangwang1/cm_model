#!/bin/bash

# 生成的种子文件


new_alpha=0.10
temperature=60

# savefile="/home/matteo/github_2/clustering_module/Example/experiments_d1103_mnist/alpha_1.04_new_alpha_${new_alpha}_temp_${temperature}.txt"
# python3 cm_mnist_latest_d1102.py  -n $new_alpha -t $temperature > $savefile

while [ $(echo "$temperature< 110" | bc) -eq 1 ]; do
  new_alpha=0.10
  # while [ $(echo "$new_alpha< 0.5" | bc) -eq 1 ]; do
      # 调用 Python 脚本
    savefile="/home/matteo/github_2/clustering_module/Example/experiments_d11040_mnist/alpha_1.04_new_alpha_${new_alpha}_temp_${temperature}.txt"
    python3 cm_mnist_latest_d1102_odd.py  -n $new_alpha -t $temperature > $savefile
      # 增加 t 参数值
  #     new_alpha=$(echo "$new_alpha + 0.1" | bc)
  # done
  temperature=$(echo "$temperature + 10" | bc)
done 



# #!/bin/bash
# #!/bin/bash
# # 定义参数的可能值
# alpha_values=(1 5 10 15 20)
# new_alpha_values=(0.1 0.3 0.5 0.8)
# temperature_values=(5 10 15 20 25 30)

# # 随机选择alpha, new_alpha, 和 temperature的值
# alpha=${alpha_values[$RANDOM % ${#alpha_values[@]}]}
# new_alpha=${new_alpha_values[$RANDOM % ${#new_alpha_values[@]}]}
# temperature=${temperature_values[$RANDOM % ${#temperature_values[@]}]}

# # 定义保存文件的路径
# savefile="/home/matteo/github_2/clustering_module/Example/experiments_d1022_mnist/alpha_${alpha}_new_alpha_${new_alpha}_temp_${temperature}.txt"

# # 运行 Python 脚本并将输出保存到文件
# python3 cm_mnist.py -a ${alpha} -n ${new_alpha} -t ${temperature} > ${savefile}
