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
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

import ray
import spu
import secretflow as sf
from secretflow.data.split import train_test_split
from secretflow.device.driver import wait, reveal
from secretflow.data import FedNdarray, PartitionWay

from ml.header import model_selector
from ml.newTextPreprocessor import check_nltk_resources, TextPreprocessor
from ml.toolfunc import Divide_X, Divide_y, timing


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

        # 获取成员个数
        self.num = len(parties)
        if self.num < 2:
            raise ValueError("[E]The number of participating nodes should be at least two")

        # 初始化 secreetflow
        sf.init(parties=parties, address=ray_addr)

        # 设置 SPU
        if ray_addr=='local':   # 本地单节点仿真
            spu = sf.SPU(sf.utils.testing.cluster_def(parties))
        else:                   # 集群仿真
            spu = sf.SPU(cluster_def=cluster_def)

        # 初始化 PYU：形成键值对：'alice':alice
        self.PYU = {}
        for key in parties:
            self.PYU[key] = sf.PYU(key)
        
        # 初始化模型: 10开头为 spu 类，20开头为 server-clients 类
        if model_id > 1000 and model_id < 2000:
            self.model = model_selector(model_id, spu, *args, **kwargs)

        elif model_id > 2000 and model_id < 3000:
            server = self.Parties[parties[0]]
            clients = [self.Parties[parties[i]] for i in range(1,self.num)]
            self.model = model_selector(model_id, server, clients, *args, **kwargs)
        else:
            raise ValueError(f"[E]No model found for id {model_id}")

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
            raise ValueError("[E]No input file path specified or no dataset input.")
        
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

        self.par_way = None
        if direction == 'r':
            self.par_way = PartitionWay.HORIZONTAL
        elif direction == 'c':
            self.par_way = PartitionWay.VERTICAL
        else:
            raise ValueError("[E]The direction of data \
                             partitioning is either row \
                             wise ('r') or column wise ('c')")
        
        parX = {}
        for key in X_partitions:
            parX[self.PYU[key]] = self.PYU[key](Divide_X)(
                self.X, *X_partitions[key], direction=direction)
        
        parY = {}
        for key in y_partitions:
            parY[self.PYU[key]] = self.PYU[key](Divide_y)(
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
        yhat = reveal(self.model.predict(self.X_test))

        initial_y = reveal(self.y_test.partitions[self.PYU[memberKey]])
        
        initial_y = np.array(initial_y)
        output_y = np.array(yhat)

        print('initial_y 的尺寸', initial_y.shape)
        print('output_y  的尺寸', output_y.shape)

        auc_score = roc_auc_score(initial_y, output_y)
        logging.info(f"auc: {auc_score}")

        # 将预测结果二值化：> 0.5 则为 1，否则为 0
        binary_class_results = np.where(output_y > 0.5, 1, 0)

        # 计算准确率
        accuracy = accuracy_score(initial_y, binary_class_results)
        logging.info(f"acc: {accuracy}")

        # 获取分类报告
        print("classification report:")
        print(classification_report(initial_y, binary_class_results))

# ************************************************************************************************
    @timing
    def filter(self, memberKey='', input_text_path='', 
               message_column_name='', labels_column_name='', classify={}, 
               Sep=',', Encoding='utf-8'
               ):
        '''
        本地过滤器 （请先完成对数据集的文本处理）
        - memberKey           选择执行节点名称
        - input_text_path     文本地址
        - message_column_name 文本列的列名称
        - labels_column_name  特征列名称（可选）： 当此参数不为 None 则进入测试模式，输出分类结果评估
        - classify            特征标准化处理设置，为{}则直接输出
        - Sep                 文本列分隔符
        - Encoding            编码方式
        '''
        self.preprocessor.load_data(input_text_path, Sep=Sep, Encoding=Encoding)

        input_X = self.preprocessor.get_column(message_column_name)

        features = self.preprocessor.Processing_features(message_column_name)

        parX = {}
        parX[self.PYU[memberKey]] = self.PYU[memberKey](lambda x:x)(features)
        FedNdarray_X = FedNdarray(partitions=parX, partition_way=self.par_way)
        
        output_y = np.array(reveal(self.model.predict(FedNdarray_X))).reshape(-1,1)

        # 将预测结果二值化：> 0.5 则为 1，否则为 0
        output_y = np.where(output_y > 0.5, 1, 0)

        print('output_y 的尺寸', output_y.shape)
        print('input_X  的尺寸', input_X.shape)

        if output_y.shape[0] != input_X.shape[0]:
            raise ValueError("[E]The number of rows for 'output_y' and 'input_X' does not match.")
    
        path1 = "./output/spam.txt"
        path2 = "./output/ham.txt"

        with open(path1, 'w') as file_class_1, open(path2, 'w') as file_class_0:
            for label, sample in zip(output_y, input_X):
                if label == 1:
                    file_class_1.write(''.join(map(str, sample)) + '\n')
                elif label == 0:
                    file_class_0.write(''.join(map(str, sample)) + '\n')
                else:
                    raise ValueError("[E]Label should be either 0 or 1.")
                
        
        # 以下为额外测试代码
        if labels_column_name == None:
            return
        
        initial_y = self.preprocessor.Processing_labels(labels_column_name,classify)
        initial_y = np.array(initial_y)

        print('initial_y 的尺寸', initial_y.shape)
        print('output_y  的尺寸', output_y.shape)

        auc_score = roc_auc_score(initial_y, output_y)
        logging.info(f"auc: {auc_score}")

        # 将预测结果二值化：> 0.5 则为 1，否则为 0
        binary_class_results = np.where(output_y > 0.5, 1, 0)

        # 计算准确率
        accuracy = accuracy_score(initial_y, binary_class_results)
        logging.info(f"acc: {accuracy}")

        # 获取分类报告
        print("classification report:")
        print(classification_report(initial_y, binary_class_results))


# 示例使用
if __name__ == '__main__':

    # 文本地址
    spam1_path = './data/spam1.csv'
    column_X_spam1='Message'
    column_y_spam1='Category'
    classif_spam1={'ham':0, 'spam':1}
    encoding_spam1='ctf-8'

    spam2_path = './data/spam2.csv'
    classif_spam2={}
    column_X_spam2='text'
    column_y_spam2='spam'
    encoding_spam2='ISO-8859-1'
    
    # 这里由于 spam2.csv 中的标签列已经标准化，故直接返回，传入 {} 即可

    # 模型选择参数
    model_id = 1001
    # 成员设置
    parties = ['alice', 'bob']
    # 成员数据集划分： alice / bob 划分给每个节点1/2的 X 部分(默认按列)
    X_par = {
        'alice':(0,0.5),
        'bob':(0.5,1)
    }
    Y_par = {
        'alice':(0,1)
    }
    # ********************************************初始化***********************************************
    print('初始化隐私保护垃圾邮件过滤器...')
    run = pp_spam_filter(
        model_id, parties=parties, extract_feat_method='tfidf', reduce_dim_method='pca')
    # *******************************************文本处理**********************************************
    print('文本处理...')
    run.Textprocessor(_input_file_path=spam2_path, 
                      message_column_name='text', labels_column_name='spam', 
                      classification={}, Encoding='ISO-8859-1')

    # ******************************************数据集划分*********************************************
    print('划分数据...')
    run.data_divider(direction='c', X_partitions=X_par, y_partitions=Y_par)

    # *******************************************训练*************************************************
    print('训练模型...')
    run.train()

    # *******************************************评估*************************************************
    print('评估模型...')
    run.evaluation('alice')

    # *******************************************输出*************************************************
    print('过滤器测试：')
    run.filter(memberKey='alice', input_text_path=spam1_path, 
            message_column_name='Message', labels_column_name='Category', classify=classif_spam1)
    



