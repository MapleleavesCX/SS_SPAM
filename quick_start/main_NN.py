import sys
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import ray
import spu
import secretflow as sf
from secretflow.device.driver import wait, reveal
from secretflow.data import FedNdarray, PartitionWay

sys.path.append('../')
from Ss_spam import ss_spam

from ml.toolfunc import Divide_X, Divide_y

'''注意修改路径'''

# 示例使用
if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # 文本处理参数集合
    _input_file_path = '../data/SpamData.csv'
    # 模型选择参数
    # 假设用户选择了模型ID为3的 SecureNN 模型
    model_id = 3

    # *****************************************SPU设置*********************************************
    

    sf.init(['alice', 'bob', 'server'], address='local')

    # 初始化 SPU ,导入节点
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'server']))

    # 初始化 PYU
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    server = sf.PYU('server')
    clients = [alice, bob]
    print('初始化隐私保护垃圾邮件过滤器...')
    run = ss_spam(model_id, Server=server, Clients=clients)

    print('文本处理中...')
    X, y = run.Textprocessor(_input_file_path=_input_file_path)

    # ***************************************数据集划分*******************************************

    print('alice / bob 划分给每个节点1/2的 X 部分(默认按列)...')
    # alice / bob 每个节点都拥有1/2的 X 部分(默认按列)
    X_data = FedNdarray(
        partitions={
            alice: alice(Divide_X)(X, 0, 0.5, "r"),
            bob: bob(Divide_X)(X, 0.5, 1, "r")
        },
        partition_way=PartitionWay.HORIZONTAL,
    )


    # 把 y 分给alice
    def read_y(data, start=0.0, end=1.0):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        length = data.shape[0]
        sliced_label = data[int(start * length):int(end * length)]

        # 重塑为二维数组 (n, 1)
        return sliced_label.reshape(-1, 1)


    # 更新 y_data 的分配
    y_data = FedNdarray(
        partitions={
            alice: alice(read_y)(y, 0, 0.5),
            bob: bob(read_y)(y, 0.5, 1)
        },
        partition_way=PartitionWay.HORIZONTAL,
    )

    # 等待IO
    wait([p.data for p in X_data.partitions.values()])
    wait([p.data for p in y_data.partitions.values()])

    # 划分训练集和测试集
    print('划分训练集和测试集...')
    _x1, _x2, _y1, y_test = run.train_test_split(X_data, y_data)

    # *******************************************训练*************************************************
    print('训练模型...')
    run.train()

    # *******************************************预测*************************************************

    # 评估模型
    print('评估模型...')
    yhat = reveal(run.predict())
    y_ = reveal(y_test.partitions[alice])

    alice_key = next(iter(yhat.keys()))  # 假设 yhat 中只有一个 key，即 PYURuntime(alice)
    yhat_alice = yhat.get(alice_key, [])

    # 检查提取的结果
    # print(f"yhat_alice: {yhat_alice}")

    # 确保 yhat_alice 包含张量，才能进行下一步操作
    if len(yhat_alice) > 0:
        # 2. 将这些张量转换为 NumPy 数组
        yhat_values = [tensor.numpy() for tensor in yhat_alice]

        # 3. 合并成一个完整的 NumPy 数组
        yhat_values = np.concatenate(yhat_values)

    # 打印转换后的结果
    # print(f"yhat_values: {yhat_values}")
    else:
        print("yhat_alice 为空，无法进行 concatenate 操作")

    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

    auc_score = roc_auc_score(y_, yhat_values)
    logging.info(f"auc: {auc_score}")

    # 将预测结果二值化：> 0.5 则为 1，否则为 0
    binary_class_results = np.where(yhat_values > 0.5, 1, 0)

    # 计算准确率
    accuracy = accuracy_score(y_, binary_class_results)
    logging.info(f"acc: {accuracy}")

    # 获取分类报告
    print("classification report:")
    print(classification_report(y_, binary_class_results))
