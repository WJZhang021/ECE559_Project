import scipy.io as sio

# 读取.mat文件
data = sio.loadmat('Results/Apr29.mat')

# 获取各个变量
para = data['para']
allo4 = data['allo4']
obj_final4 = data['obj_final4']
obj_list4 = data['obj_list4'] 
iter4 = data['iter4']
ind_op = data['ind_op']

print(allo4)
print(obj_final4)
print(iter4)


# for key in data.keys():
#     if not key.startswith('__'):  # 跳过mat文件的内置变量
#         print(f"\n{key}:")
#         print(f"Type: {type(data[key])}")
#         print(f"Shape: {data[key].shape}")
#         print(f"Content: {data[key]}")