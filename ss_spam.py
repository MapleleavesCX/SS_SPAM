# -*- coding: utf-8 -*-
# @Date    : 2024/09/12
# @Author  : MapleleavesCX

# 这里是 主调类 pp_spam_filter
'''
**************************************************************************************************
垃圾邮件数据集格式：
    - 任意，但需要用户自己标识出特征列与文本列

**************************************************************************************************
依赖：
secretflow v1.5.0b0
scikit-learn
pandas
numpy
nltk  
(
    初次使用需要下载处理文本的停用词等内容，内置自动下载函数 check_nltk_resources 请保证网络条件通畅；
    如 check_nltk_resources 自动化下载失败，可以尝试手动下载，步骤如下：
    https://www.nltk.org/nltk_data/
    - 下载文件:
        id: stopwords
        id: punkt
        id: punkt_tab
    - 解压后放置到根目录文件夹，各文件路径如下：
        stopwords: nltk_data/corpora/
        punkt 和 punkt_tab: nltk_data/tokenizers/
    - 在根目录启动命令行，输入指令，把相关内容路径添加到 nltk 路径：
        export NLTK_DATA='/path/to/SS_SPAM-0.2.0/nltk_data'
)

**************************************************************************************************
'''

import sys
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

import ray
import spu
import secretflow as sf
from secretflow.data.split import train_test_split
from secretflow.device.driver import wait, reveal
from secretflow.data import FedNdarray, PartitionWay

from ml.header import model_selector
from ml.newTextPreprocessor import check_nltk_resources, TextPreprocessor
from ml.toolfunc import Divide_X, Divide_y, timing, TerminalColors


# ************************************************************************************************
class pp_spam_filter:
    '''隐私保护的垃圾邮件过滤器'''

    @timing
    def __init__(self, model_id, extract_feat_method, reduce_dim_method,
                 parties=[], ray_addr='local', cluster_def={},
                 _n_components=1000, _ngram_range=(1, 1), _min_df=5,
                 *args, **kwargs):
        '''
        初始化
        - model_id:   模型选择 （必选）
            - 1001: SecureLogisticRegression(spu),
            - 1002: SecureDecisionTree(spu),
            - 1003: SecureSSGLM(spu),
            - 2001: SecureNN(Server, Clients, others),
            - 2099:SecureXGboost(Server, Clients)        模型不兼容！不可用
        - extract_feat_method 特征工程处理方法选择（必选）
            - 'bow'
            - 'tfidf'
        - reduce_dim_method 降维方法（必选）
            - 'pca'
            - 'svd'
        - parties     成员列表 （必选）
        - ray_addr    ray地址 （可选）
        - cluster_def 集群网络配置（可选）

        - _n_components 降维后拥有的特征量个数（可选）
        - _ngram_range  
        - _min_df       
        '''

        ray.shutdown()

        # 获取成员个数
        self.num = len(parties)
        if self.num < 2:
            raise ValueError(TerminalColors.BRIGHT_RED + 
                             "[E]The number of participating nodes should be at least two"
                               + TerminalColors.END)

        # 初始化 secreetflow
        sf.init(parties=parties, address=ray_addr)

        # 设置 SPU
        if ray_addr == 'local':  # 本地单节点仿真
            _spu = sf.SPU(sf.utils.testing.cluster_def(parties))
        else:  # 集群仿真
            cluster_def['runtime_config'] = {
                    'protocol': spu.spu_pb2.SEMI2K,
                    'field': spu.spu_pb2.FM128,
                    'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
                }
            _spu = sf.SPU(cluster_def=cluster_def)

        # 初始化 PYU：形成键值对：'alice':alice
        self.PYU = {}
        for key in parties:
            self.PYU[key] = sf.PYU(key)
        self.model_id = model_id
        # 初始化模型: 10开头为 spu 类，20开头为 server-clients 类
        if model_id > 1000 and model_id < 2000:
            self.model = model_selector(model_id, _spu, *args, **kwargs)

        elif model_id > 2000 and model_id < 3000:
            Server = self.PYU[parties[0]]
            Clients = [self.PYU[parties[i]] for i in range(1, self.num)]
            self.model = model_selector(model_id, Server, Clients, *args, **kwargs)
        else:
            raise ValueError(TerminalColors.BRIGHT_RED + 
                             f"[E]No model found for id {model_id}" + TerminalColors.END)

        # 调用函数检查文本处理相关资源
        check_nltk_resources()
        # 初始化文本处理器
        self.preprocessor = TextPreprocessor(
            extract_feat_method=extract_feat_method, reduce_dim_method=reduce_dim_method,
            n_components=_n_components, ngram_range=_ngram_range, min_df=_min_df)

    # ************************************************************************************************
    @timing
    def Textprocessor(self, _input_dataset=None, _input_file_path='', _output_file_path='',
                      message_column_name='', labels_column_name='',
                      classification={},
                      Sep=',', Encoding='utf-8'
                      ):
        '''
        文本处理器
        - 必选参数：
            - _input_dataset         传入数据集
            - _input_file_path       传入数据集文件路径
            (以上二者必须有一个传参，若都传入参数则默认 _input_file_path 优先)
            - message_column_name    文本列名称
            - labels_column_name     标签列名称
        - 可选参数：
            - classification         分类特征字典
            - Sep                    文本列分隔符
            - Encoding               编码方式
        - 返回值：
            - X   经过文本处理后的数据集特征
            - y   经过文本处理后的数据集标签
        '''

        # 加载数据
        if _input_file_path != '':
            self.preprocessor.load_data(input_file_path=_input_file_path, Sep=Sep, Encoding=Encoding)
        elif _input_dataset != None and _input_file_path == '':
            self.preprocessor.load_data(input_data=_input_dataset, Sep=Sep, Encoding=Encoding)
        else:
            raise ValueError(TerminalColors.BRIGHT_RED + 
                             "[E]No input file path specified or no dataset input." + TerminalColors.END)

        # 提取特征（内置预处理数据），可自定义提取方法method
        self.X = self.preprocessor.Processing_features(message_column_name)
        # 标签标准化
        self.y = self.preprocessor.Processing_labels(labels_column_name, classification)

        if _output_file_path != '':
            # 保存处理后的数据到CSV文件
            self.preprocessor.save_to_csv(output_file_path=_output_file_path)

        return self.X, self.y

    # ************************************************************************************************
    @timing
    def data_divider(self,
                     direction, X_partitions={}, y_partitions={},
                     split_factor=0.8, random_state=1234):
        '''
        数据集划分器
        - direction     数据集划分方向(纵向'c'或横向'r')
        - X_partitions  数据特征 X 划分设置
        - y_partitions  数据标签 y 划分设置
        - split_factor  划分因子(即 训练数据所占比例)
        - random_state  随机度
        '''
        self.X_partitions = X_partitions
        self.par_way = None
        if direction == 'r':
            self.par_way = PartitionWay.HORIZONTAL
        elif direction == 'c':
            self.par_way = PartitionWay.VERTICAL
        else:
            raise ValueError(TerminalColors.BRIGHT_RED + 
                             "[E]The direction of data \
                             partitioning is either row \
                             wise ('r') or column wise ('c')" + TerminalColors.END)

        parX = {}
        for key in X_partitions:
            parX[self.PYU[key]] = self.PYU[key](Divide_X)(
                self.X, *X_partitions[key], direction=direction)

        if direction == 'c':
            parY = {}
            for key in y_partitions:
                parY[self.PYU[key]] = self.PYU[key](Divide_y)(
                    self.y, *y_partitions[key])
        else:
            def read_y(data, start=0.0, end=1.0):
                if not isinstance(data, np.ndarray):
                    data = np.array(data)

                length = data.shape[0]
                sliced_label = data[int(start * length):int(end * length)]

                # 重塑为二维数组 (n, 1)
                return sliced_label.reshape(-1, 1)

                # 更新 y_data 的分配

            parY = {}
            for key in y_partitions:
                parY[self.PYU[key]] = self.PYU[key](read_y)(
                    self.y, *y_partitions[key])

        FedNdarray_X = FedNdarray(partitions=parX, partition_way=self.par_way)

        FedNdarray_y = FedNdarray(partitions=parY, partition_way=self.par_way)

        # 等待IO
        wait([p.data for p in FedNdarray_X.partitions.values()])
        wait([p.data for p in FedNdarray_y.partitions.values()])

        # 划分训练集和测试集
        self.X_train, self.X_test = train_test_split(
            FedNdarray_X, train_size=split_factor, random_state=random_state)
        self.y_train, self.y_test = train_test_split(
            FedNdarray_y, train_size=split_factor, random_state=random_state)

        return self.X_train, self.X_test, self.y_train, self.y_test

    # ************************************************************************************************
    @timing
    def train(self, *args, **kwargs):
        '''模型训练'''
        self.model.train(self.X_train, self.y_train, *args, **kwargs)

    # ************************************************************************************************
    @timing
    def evaluation(self, memberKey):
        '''
        模型评估
        - memberKey   选择解密的数据所属结点名称
        '''
        # 预测结果的解密
        yhat = reveal(self.model.predict(self.X_test))

        # 检查 yhat 是字典还是 NumPy 数组
        if isinstance(yhat, dict):
            # 情况1：yhat 是一个字典
            alice_key = next(iter(yhat.keys()))  # 获取字典中的第一个 key
            yhat_alice = yhat.get(alice_key, [])

            if len(yhat_alice) > 0:
                # 将张量转换为 NumPy 数组并合并
                yhat_values = [tensor.numpy() for tensor in yhat_alice]
                output_y = np.concatenate(yhat_values)
            else:
                print("yhat_alice 为空，无法进行 concatenate 操作")
                return
        else:
            # 情况2：yhat 直接是一个 NumPy 数组
            output_y = np.array(yhat)

        # 解密真实标签
        initial_y = reveal(self.y_test.partitions[self.PYU[memberKey]])

        # 检查 initial_y 是字典还是 NumPy 数组
        if isinstance(initial_y, dict):
            # 如果 initial_y 是字典
            initial_y = list(initial_y.values())
            initial_y = np.concatenate([tensor.numpy() for tensor in initial_y])
        else:
            # 如果 initial_y 是 NumPy 数组
            initial_y = np.array(initial_y)

        # 输出尺寸以供调试
        print('initial_y 的尺寸', initial_y.shape)
        print('output_y 的尺寸', output_y.shape)

        # 计算 AUC 分数
        auc_score = roc_auc_score(initial_y, output_y)
        logging.info(TerminalColors.BRIGHT_GREEN + f"auc: {auc_score}" + TerminalColors.END)

        # 将预测结果二值化：> 0.5 则为 1，否则为 0
        binary_class_results = np.where(output_y > 0.5, 1, 0)

        # 计算准确率
        accuracy = accuracy_score(initial_y, binary_class_results)
        logging.info(TerminalColors.BRIGHT_GREEN + f"acc: {accuracy}" + TerminalColors.END)

        # 获取分类报告
        print(TerminalColors.BRIGHT_GREEN + "classification report:" + TerminalColors.END)
        print(TerminalColors.BRIGHT_CYAN + 
              classification_report(initial_y, binary_class_results) + TerminalColors.END)


    # ************************************************************************************************
    @timing
    def filter(self, input_text_path='',
               message_column_name='', labels_column_name='', classify={},
               _Sep=',', _Encoding='utf-8'):
        '''
        本地过滤器 （请先完成对数据集的文本处理）
        - input_text_path     文本地址
        - message_column_name 文本列的列名称
        - labels_column_name  特征列名称（可选）： 当此参数不为 None 则进入测试模式，输出分类结果评估
        - classify            特征标准化处理设置，为{}则直接输出
        - Sep                 文本列分隔符
        - Encoding            编码方式
        '''
        self.preprocessor.load_data(input_text_path, Sep=_Sep, Encoding=_Encoding)

        input_X = self.preprocessor.get_column(message_column_name)
        features = self.preprocessor.Processing_features(message_column_name)

        parX = {}
        for key in self.X_partitions:
            parX[self.PYU[key]] = self.PYU[key](Divide_X)(
                features, *self.X_partitions[key], direction='r')
                
        FedNdarray_X = FedNdarray(partitions=parX, partition_way=self.par_way)

        # 预测结果的解密和转换
        yhat = reveal(self.model.predict(FedNdarray_X))

        # 检查 yhat 是否为字典
        if isinstance(yhat, dict):
            # 遍历所有的 key，将所有方的数据合并
            yhat_values = []
            for key, value in yhat.items():
                if len(value) > 0:
                    # 将张量转换为 NumPy 数组
                    yhat_part = [tensor.numpy() for tensor in value]
                    yhat_values.append(np.concatenate(yhat_part))
                else:
                    print(f"{key} 为空，无法进行 concatenate 操作")

            # 合并所有参与方的数据
            if len(yhat_values) > 0:
                output_y = np.concatenate(yhat_values).reshape(-1, 1)
            else:
                print("所有参与方的预测结果都为空")
                return
        else:
            # 如果 yhat 是 NumPy 数组
            output_y = np.array(yhat).reshape(-1, 1)

        # 将预测结果二值化：> 0.5 则为 1，否则为 0
        output_y = np.where(output_y > 0.5, 1, 0)

        if output_y.shape[0] != input_X.shape[0]:
            raise ValueError(TerminalColors.BRIGHT_RED + 
                             "[E]The number of rows for 'output_y' and 'input_X' does not match."
                              + TerminalColors.END)

        path1 = "./output/spam.txt"
        path2 = "./output/ham.txt"

        with open(path1, 'w') as file_class_1, open(path2, 'w') as file_class_0:
            for label, sample in zip(output_y, input_X):
                if label == 1:
                    file_class_1.write(''.join(map(str, sample)) + '\n')
                elif label == 0:
                    file_class_0.write(''.join(map(str, sample)) + '\n')
                else:
                    raise ValueError(TerminalColors.BRIGHT_RED + 
                                     "[E]Label should be either 0 or 1." + TerminalColors.END)

        # 以下为额外测试代码
        if labels_column_name is None:
            return

        # 处理 initial_y
        initial_y = self.preprocessor.Processing_labels(labels_column_name, classify)

        # 检查 initial_y 是否为字典
        if isinstance(initial_y, dict):
            initial_y = list(initial_y.values())
            initial_y = np.concatenate([tensor.numpy() for tensor in initial_y])
        else:
            initial_y = np.array(initial_y)

        print(TerminalColors.BRIGHT_YELLOW + 'initial_y 的尺寸' + TerminalColors.END, initial_y.shape)
        print(TerminalColors.BRIGHT_YELLOW + 'output_y 的尺寸' + TerminalColors.END, output_y.shape)

        # 计算 AUC 分数
        auc_score = roc_auc_score(initial_y, output_y)
        logging.info(TerminalColors.BRIGHT_GREEN + f"auc: {auc_score}" + TerminalColors.END)

        # 计算准确率
        accuracy = accuracy_score(initial_y, output_y)
        logging.info(TerminalColors.BRIGHT_GREEN + f"acc: {accuracy}" + TerminalColors.END)

        # 获取分类报告
        print(TerminalColors.BRIGHT_GREEN + "classification report:" + TerminalColors.END)
        print(TerminalColors.BRIGHT_CYAN + classification_report(initial_y, output_y) + TerminalColors.END)


