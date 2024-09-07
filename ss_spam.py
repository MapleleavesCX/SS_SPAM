# -*- coding: utf-8 -*-
# @Date    : 2024/08/25
# @Author  : MapleleavesCX

# 这里是 主调类 ss_spam
'''
考虑多种模型与多种应用情况,本篇对代码标准作统一说明:
**************************************************************************************************
以下为模型选择的常量 model_id 初始化输入参数 定义：

（各个模型的具体参数介绍请见对应的模型函数文件）

1: SecureLogisticRegression(spu),      逻辑回归
2: SecureDecisionTree(spu),            决策树
3: SecureNN(Server, Clients, others),  神经网络
4: SecureSSGLM(spu),                   广义线性模型

99: SecureXGboost(Server, Clients), # 不兼容当前格式的函数， 目前仅测试用， 不可调用predict

PS: 添加新模型请到 .ml/header.py 中的 model_selector 类

**************************************************************************************************
统一未处理的初始数据集格式：

第一列              第二列
判断是否为垃圾邮件   邮件文本内容

**************************************************************************************************

依赖：
secretflow v1.5.0b0
scikit-learn
pandas
nltk  
(
    初次使用需要下载处理文本的停用词等内容，内置自动下载函数 check_nltk_resources 请保证网络条件通畅；
    如 check_nltk_resources 自动化下载失败，可以尝试手动下载：
    https://www.nltk.org/nltk_data/
     -下载文件ID:
        id: stopwords
        id: punkt
)


**************************************************************************************************
'''

import sys
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import ray
import spu
import secretflow as sf
from secretflow.data.split import train_test_split
from secretflow.device.driver import wait, reveal
from secretflow.data import FedNdarray, PartitionWay

from ml.header import model_selector
from ml.TextPreprocessor import check_nltk_resources, TextPreprocessor
from ml.toolfunc import Divide_X, Divide_y

class ss_spam:
    '''隐私保护的垃圾邮件过滤器'''
    def __init__(self, model_id, *args, **kwargs):
        
        # 初始化模型
        self.model = model_selector(model_id, *args, **kwargs)
        # 调用函数检查资源
        check_nltk_resources()
        # 初始化文本处理器
        self.preprocessor = TextPreprocessor()
    
    def Textprocessor(self, _input_dataset=None, _input_file_path='', _output_file_path='', 
                      extract_feat_method='tfidf', reduce_dim_method='pca', 
                      _n_components=1000, _ngram_range=(1, 1), _min_df=5):   
        '''
        文本处理器
        必选参数： 
            _input_dataset    传入数据集
            _input_file_path  传入数据集文件路径
            (以上二者必须有一个传参，若都传入参数则默认 _input_file_path 优先)
        可选参数： 
            _output_file_path 传出数据集文件路径(=''则默认若不保存)  
            extract_feat_method 特征工程处理方法选择(默认tfidf)
            reduce_dim_method 降维方法(默认pca)
            _n_components 降维后拥有的的特征量个数(默认1000)
            _ngram_range
            _min_df
        '''   

        # 加载数据
        if _input_file_path != '':
            self.preprocessor.load_data(input_file_path=_input_file_path)
        elif _input_dataset != None and _input_file_path == '':
            self.preprocessor.load_data(input_data=_input_dataset)
        else:
            print("No input file path specified or no dataset input.")

        # 提取特征（内置预处理数据）
        self.preprocessor.extract_features(method=extract_feat_method, 
                                      ngram_range=_ngram_range, 
                                      min_df=_min_df)
        # 降维,可自设定降维方式method和压缩后的特征数量n_components
        self.preprocessor.reduce_dimensions(method=reduce_dim_method, 
                                       n_components=_n_components)
        # 返回输出特征矩阵和标签
        self.X, y = self.preprocessor.get_X_y()

        # LR的训练数据集必须标准化或规范化
        scaler = StandardScaler()
        # 调用 scaler.fit_transform(x) 方法对数据进行拟合和转换。
        X = scaler.fit_transform(self.X)

        if _output_file_path != '':
            # 保存处理后的数据到CSV文件
            self.preprocessor.save_to_csv(output_file_path=_output_file_path)

        return X, y

    def train_test_split(self, X_data, y_data, split_factor=0.8, random_state=1234):
        # 划分训练集和测试集
        self.X_train, self.X_test = train_test_split(X_data, train_size=split_factor, random_state=random_state)
        self.y_train, self.y_test = train_test_split(y_data, train_size=split_factor, random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, *args, **kwargs):
        self.model.train(self.X_train, self.y_train, *args, **kwargs)

    def predict(self, new_X_test=None):
        if new_X_test == None:
            return reveal(self.model.predict(self.X_test))
        else:
            return reveal(self.model.predict(new_X_test))



# 示例使用
if __name__ == '__main__':


    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


    # 文本处理参数集合
    _input_file_path = './data/SpamData.csv'
    # 模型选择参数
    # 假设用户选择了模型ID为1的 SecureLogisticRegression 模型
    model_id = 1

    # 参与节点列表
    member = ['alice', 'bob']
    # 初始化 PYU
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')


    #*****************************************SPU设置*********************************************
    
    sf.init(parties=member, address='local')

    spu = sf.SPU(sf.utils.testing.cluster_def(member))

    print('初始化隐私保护垃圾邮件过滤器...')
    run = ss_spam(model_id, spu)

    print('文本处理中...')
    X, y = run.Textprocessor(_input_file_path=_input_file_path)

    #***************************************数据集划分*******************************************

    print('alice / bob 划分给每个节点1/2的 X 部分(默认按列)...')
    # alice / bob 每个节点都拥有1/2的 X 部分(默认按列)
    X_data = FedNdarray(
        partitions={
            alice: alice(Divide_X)(X, 0, 0.5),
            bob: bob(Divide_X)(X, 0.5, 1),
        },
        partition_way=PartitionWay.VERTICAL,
    )

    # 把 y 分给alice
    y_data = FedNdarray(
        partitions={alice: alice(Divide_y)(y)},
        partition_way=PartitionWay.VERTICAL,
    )

    # 等待IO
    wait([p.data for p in X_data.partitions.values()])
    wait([p.data for p in y_data.partitions.values()])

    # 划分训练集和测试集
    print('划分训练集和测试集...')
    _x1, _x2, _y1, y_test = run.train_test_split(X_data, y_data)

    #*******************************************训练*************************************************
    print('训练模型...')
    run.train()

    #*******************************************预测*************************************************

    # 评估模型
    print('评估模型...')
    yhat = run.predict()
    y_ = reveal(y_test.partitions[alice])


    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

    # 获得分类的曲线下面积（auc）得分
    logging.info(f"auc: {roc_auc_score(y_, yhat)}")
    binary_class_results = np.where(yhat > 0.5, 1, 0)
    # 获得分类的准确度得分
    logging.info(f"acc: {accuracy_score(y_, binary_class_results)}")
    # 获取分类报告
    print("classification report:")
    print(classification_report(y_, binary_class_results))
