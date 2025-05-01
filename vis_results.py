import scipy.io as sio
import json
from performance import performance_O
from costs import total_C
import matplotlib.pyplot as plt
import numpy as np

# 读取.mat文件
data = sio.loadmat('Results/Apr29.mat')

# 获取各个变量
#para = data['para']
allo4 = data['allo4']
obj_final4 = data['obj_final4']
obj_list4 = data['obj_list4'] 
iter4 = data['iter4']
ind_op = data['ind_op']

# print(allo4)
# print(obj_final4)
# print(iter4)

with open('Results/Apr29_para.json', 'r', encoding='utf-8') as f:
    config = json.load(f)    
para = config['para']
C_para = ['c1', 'c2', 'c3', 'alpha3', 's3', 'c4', 'c5']
O_para = ['w1', 'q1', 'r1', 'w2', 'q2', 'r2', 'u2', 'w3', 'beta3', 'w4', 'w5', 'w12', 'w34', 'w35']
para_C = {key: para[key] for key in C_para}
para_O = {key: para[key] for key in O_para}


out = obj_list4[0,ind_op]
out = out[0][0]
#print(out)
out = out.flatten()

# 创建图形和两个y轴
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()  # 创建第二个y轴

# 计算x轴数据
x = np.arange(len(out)-1)

# 在左轴上画第一条线
line1, = ax1.plot(x, np.array(out[-1]-out[:-1]), 'b-', label=r'out[-1]-out[:-1]')
ax1.set_ylabel('Original Scale', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 在右轴上画第二条线
line2, = ax2.plot(x, np.log10(np.array(out[-1]-out[:-1])), 'r-', label=r'log10')
ax2.set_ylabel('Log Scale', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 添加网格线
ax1.grid(True, linestyle='--', alpha=0.7)

# 设置标题
plt.title('Convergence', fontsize=14)
ax1.set_xlabel('Iterations', fontsize=12)

# 合并两个线的图例
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('Results/Convergence.png', dpi=300, bbox_inches='tight')
plt.show()
# plt.figure(figsize=(10, 6))
# # Converence to the optimal value 
# plt.plot(np.arange(len(out)-1),np.array(out[-1]-out[:-1]),'b')
# plt.plot(np.arange(len(out)-1),np.log10( np.array(out[-1]-out[:-1]) ),'b')
# plt.legend((r'out[-1]-out[:-1]',r'log10'), loc = 'upper right')
# plt.title('Convergence')
# plt.savefig('Results/Convergence.png', dpi=300, bbox_inches='tight')


fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()  # 创建第二个y轴

x_labels = ['a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'O', 'C', 'TO']
x = np.arange(len(x_labels))
colors = ['red', 'blue', 'green', 'purple', 'orange']
line_names = ['00', '01', '10', '11']

# 存储所有线的引用以便后面统一添加图例
lines1 = []
lines2 = []

for i in range(4):
    yi = np.zeros(len(x_labels))
    yi[0:5] = allo4[i]
    yi[5] = performance_O(*allo4[i].tolist(), **para_O)  # 计算性能
    yi[6] = total_C(*allo4[i].tolist(), **para_C)  # 计算成本
    yi[7] = obj_final4[0,i]
    
    # 分别画前5点和后3点
    line1, = ax1.plot(x[:5], yi[:5], color=colors[i], marker='o', linestyle='-', 
                      markersize=8)
    line2, = ax2.plot(x[5:], yi[5:], color=colors[i], marker='o', linestyle='-', 
                      markersize=8)
    
    lines1.append(line1)
    lines2.append(line2)

# 设置x轴标签
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)

# 设置标题和标签
plt.title('Optimization Results', fontsize=14)
ax1.set_xlabel('Parameters', fontsize=12)
ax1.set_ylabel('Allocation Values', fontsize=12)
ax2.set_ylabel('Performance Metrics', fontsize=12)

# 添加网格线（只对左轴添加）
ax1.grid(True, linestyle='--', alpha=0.7)

# 合并两个轴的图例
all_lines = lines1 + lines2
ax1.legend(lines1, line_names, loc='upper left')

# 调整两个y轴的颜色
ax1.tick_params(axis='y', labelcolor='black')
ax2.tick_params(axis='y', labelcolor='black')

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('Results/Op_Res.png', dpi=300, bbox_inches='tight')
plt.show()

# x_labels = ['a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'O', 'C', 'TO']  # x轴标签
# x = np.arange(len(x_labels))  # x轴坐标 [0,1,2,3,4,5,6,7]
# colors = ['red', 'blue', 'green', 'purple', 'orange']  # 每条线的颜色
# line_names = ['00', '01', '10', '11']  # 图例名称

# para_trunc = para.delete()

# plt.figure(figsize=(10, 6))
# for i in range(4):
#     yi = np.zeros(len(x_labels))  # 初始化y坐标
#     yi[0:5] = allo4[i]
#     yi[5] = performance_O(*allo4[i], **para)  # 计算性能
#     yi[6] = total_C(*allo4[i], **para)  # 计算成本
#     yi[7] = obj_final4[i]  # 计算目标函数值
#     plt.plot(x, yi, color=colors[i], marker='o', linestyle='-', 
#              label=line_names[i], markersize=8)

# plt.xticks(x, x_labels)
# plt.title('Optimization Results', fontsize=14)
# plt.xlabel('X Label', fontsize=12)
# plt.ylabel('Value', fontsize=12)


# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()

# plt.savefig('Results/Op_Res.png', dpi=300, bbox_inches='tight')
