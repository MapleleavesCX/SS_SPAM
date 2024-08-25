import time
import numpy as np


def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
        print(f"{func.__name__} 函数用时：{elapsed_time:.3f} ms")
        return result
    return wrapper


# 按照百分比(小数)划分数据集,默认为按列划分全集
def Divide_X(data, start=0.0, end=1.0, direction='c'):
    # 如果传入的是列表或其他非 NumPy 数组类型，将其转换为 NumPy 数组
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    rows, cols = data.shape
    if direction == 'c':
        return data[:, int(start*cols):int(end*cols)]
        
    elif direction == 'r':
        return data[int(start*rows):int(end*rows), :]
        
    else:
        raise ValueError(f"No direction found for '{direction}'")


def Divide_y(label, start=0.0, end=1.0):
    # 如果传入的是列表或其他非 NumPy 数组类型，将其转换为 NumPy 数组
    if not isinstance(label, np.ndarray):
        label = np.array(label)

    length = label.shape[0]
    sliced_label = label[int(start*length):int(end*length)]
    
     # 确保返回的始终是一维 NumPy 数组
    if sliced_label.ndim > 1 and sliced_label.shape[1] == 1:
        sliced_label = sliced_label.ravel()
    
    return sliced_label

# 示例使用
if __name__ == '__main__':
    None