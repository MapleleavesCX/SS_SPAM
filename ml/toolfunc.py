import time
import numpy as np
import pandas as pd


# 终端色彩打印
class TerminalColors:
    END = '\033[0m'   # 颜色生效终止符
    
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'



def timing(func):
    '''
    计时函数 装饰器：
    - @timing
    '''
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
        print(TerminalColors.BRIGHT_WHITE + 
              f"{func.__name__} 函数用时：{elapsed_time:.3f} ms"
                + TerminalColors.END)
        return result
    return wrapper


# 按照百分比(小数)划分数据集,默认为按列划分全集
def Divide_X(data, start=0.0, end=1.0, direction='c'):
    '''按照百分比(小数)划分特征矩阵'''
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
    '''按照百分比(小数)划分标签'''
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
