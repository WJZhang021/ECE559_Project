import scipy.io as sio
import json
import numpy as np

from total_objective import TO, TO_array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
print(obj_final4)
obj_final4 = obj_final4[0]
obj_final4 = obj_final4.flatten()
print(obj_final4)

with open('Results/Apr29_para.json', 'r', encoding='utf-8') as f:
    config = json.load(f)    
para = config['para']
C_para = ['c1', 'c2', 'c3', 'alpha3', 's3', 'c4', 'c5']
O_para = ['w1', 'q1', 'r1', 'w2', 'q2', 'r2', 'u2', 'w3', 'beta3', 'w4', 'w5', 'w12', 'w34', 'w35']
para_C = {key: para[key] for key in C_para}
para_O = {key: para[key] for key in O_para}
TO_para = {**para_C, **para_O, **{'B': para['B'], 'lambda_val': para['lambda_val']}}
#print(para)
    
allo = allo4[ind_op]
allo = allo[0][0]
# print(allo)
# print(allo[2])

plt.figure(figsize=(10, 6))
a = np.linspace(0, 1, 1000)
N = len(a)
ALLO = np.repeat([allo], N, axis=0)

for i in range(3):
    allo_temp = ALLO.copy()
    allo_temp[:, i] = a
    to = TO_array(allo_temp[:, 0], allo_temp[:, 1], allo_temp[:, 2], allo_temp[:, 3], allo_temp[:, 4], **TO_para)
    plt.plot(a, to, label=f'a{i+1}')
    
    x = allo[i]
    y = obj_final4[ind_op]
    plt.scatter(x,y, s=100, marker='*', label='_nolegend_')

plt.ylim([-15, 30])
plt.legend((r'a_1', r'a_2', r'a_3'), loc = 'lower right')
plt.title('objective landscape around optimal - 1D')
plt.savefig('Results/TOls_1D.png', dpi=300, bbox_inches='tight')



a = np.linspace(0, 1, 100)
xx, yy = np.meshgrid(a, a)
N = len(a)

fig = plt.figure(figsize=(18, 6))

# 设置colorbar的范围
vmin, vmax = 0, 25

# 第一张图：移动a1和a2
ax1 = fig.add_subplot(131, projection='3d')
Z1 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        allo_temp = allo.copy()
        allo_temp[0] = xx[i,j]
        allo_temp[1] = yy[i,j]
        Z1[i,j] = TO(allo_temp[0], allo_temp[1], allo_temp[2], 
                          allo_temp[3], allo_temp[4], **TO_para)

ax1.scatter(allo[0], allo[1], obj_final4[ind_op], 
           color='red', s=200, marker='*')
surf1 = ax1.plot_surface(xx, yy, Z1, cmap='viridis', 
                        linewidth=0.5, antialiased=False,
                        vmin=vmin, vmax=vmax)
ax1.plot_wireframe(xx, yy, Z1, color='gray', alpha=0.5, linewidth=0.5)

ax1.set_xlabel('a1')
ax1.set_ylabel('a2')
ax1.set_zlim([vmin, vmax])
ax1.set_title('Objective Landscape (a1-a2)')

# 第二张图：移动a1和a3
ax2 = fig.add_subplot(132, projection='3d')
Z2 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        allo_temp = allo.copy()
        allo_temp[0] = xx[i,j]
        allo_temp[2] = yy[i,j]
        Z2[i,j] = TO(allo_temp[0], allo_temp[1], allo_temp[2], 
                          allo_temp[3], allo_temp[4], **TO_para)

ax2.scatter(allo[0], allo[2], obj_final4[ind_op], 
           color='red', s=200, marker='*')
surf2 = ax2.plot_surface(xx, yy, Z2, cmap='viridis', 
                        linewidth=0.5, antialiased=False,
                        vmin=vmin, vmax=vmax)
ax2.plot_wireframe(xx, yy, Z2, color='gray', alpha=0.5, linewidth=0.5)

ax2.set_xlabel('a1')
ax2.set_ylabel('a3')
ax2.set_zlim([vmin, vmax])
ax2.set_title('Objective Landscape (a1-a3)')

# 第三张图：移动a2和a3
ax3 = fig.add_subplot(133, projection='3d')
Z3 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        allo_temp = allo.copy()
        allo_temp[1] = xx[i,j]
        allo_temp[2] = yy[i,j]
        Z3[i,j] = TO(allo_temp[0], allo_temp[1], allo_temp[2], 
                          allo_temp[3], allo_temp[4], **TO_para)

ax3.scatter(allo[1], allo[2], obj_final4[ind_op], 
           color='red', s=200, marker='*')
surf3 = ax3.plot_surface(xx, yy, Z3, cmap='viridis', 
                        linewidth=0.5, antialiased=False,
                        vmin=vmin, vmax=vmax)
ax3.plot_wireframe(xx, yy, Z3, color='gray', alpha=0.5, linewidth=0.5)

ax3.set_xlabel('a2')
ax3.set_ylabel('a3')
ax3.set_zlim([vmin, vmax])
ax3.set_title('Objective Landscape (a2-a3)')

# 只在第三张图右侧添加colorbar
cbar = fig.colorbar(surf3, ax=ax3)
cbar.set_label('Objective Value')

plt.tight_layout()
plt.savefig('Results/TOls_2D.png', dpi=300, bbox_inches='tight')
plt.show()
