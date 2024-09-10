import random

# 生成1到200中的100个不重复的随机种子
seeds = random.sample(range(1, 201), 100)

# 将种子写入文件
with open('seeds.txt', 'w') as f:
    for seed in seeds:
        f.write(f"{seed}\n")
