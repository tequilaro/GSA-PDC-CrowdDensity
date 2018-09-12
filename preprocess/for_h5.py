import numpy as np
import h5py

def cut(name):
    f = h5py.File(name+'.h5', 'r')
    den = np.array(f['density'])
    temp_1, temp_2 = np.array_split(den, 2, axis=0)
    den_list = np.array_split(temp_1, 2, axis=1)
    den_list += np.array_split(temp_2, 2, axis=1)

    for i in range(4):
        new_f = h5py.File(name + '_'+ str(i+1) +'.h5', 'w')
        new_f.create_dataset("density", data=den_list[i])


image_num = int(input("输入文件数量："))
print("当前指定文件数量：" + str(image_num))
image_path = input("输入文件路径：")
print("当前指定文件路径：" + image_path)
image_name = input("输入文件统一命名前缀：")
print("文件命名前缀为：" + image_name)
for i in range(image_num):
    name = image_name + '_' + str(i+1)
    cut(image_path+name)

print("all done!")