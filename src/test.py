import numpy as np

def apply_mask_to_data(data, mask):
    masked_data = np.ma.masked_array(data, mask=1-mask)
    flattened_data = masked_data.compressed()
    return flattened_data

# 测试数据
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

flattened_data = apply_mask_to_data(data, mask)
print(flattened_data)

import numpy as np

def restore_data_from_flattened(flattened_data, mask):
    h, w = mask.shape
    restored_data = np.zeros((h, w))
    restored_data[mask == 1] = flattened_data
    return restored_data

# 测试数据
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

flattened_data = np.array([1, 3, 5, 7, 9])  # 用之前生成的展平数据作为测试数据

restored_data = restore_data_from_flattened(flattened_data, mask)
print(restored_data)