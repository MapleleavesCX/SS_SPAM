# -*- coding: utf-8 -*-
# @Date    : 2024/09/12
# @Author  : MapleleavesCX

'''
**********************************************************************
文本处理类型TextPreprocessor说明:
    - __init__            初始化文本处理器，定义特征方法与降维方法
        - extract_fe_method 特征提取方法参数
            - 'bow'
            - 'tfidf'
        - reduce_dim_method 降维方法参数
            - 'pca'
            - 'svd'
    - load_data           负责从CSV文件中加载数据
    - get_column          将读取到的原始数据集按列名返回
    - preprocess_text     用于单个文本的预处理，是 preprocess_data 的前置
    - preprocess_data     对整个数据集应用预处理，是 preprocess_data 的前置
    - Processing_features 特征提取与降维
    - Processing_labels   标签标准化
    - save_to_csv         将处理后的数据保存到CSV文件中
**********************************************************************
'''

import sys

# 首先检查nltk是否安装，未安装则自动安装
try:
    import nltk
except ImportError:
    # 尝试安装 nltk
    print("nltk is not installed. Attempting to install it...")
    try:
        import pip
        pip.main(['install', 'nltk'])
        print("nltk has been successfully installed.")
    except Exception as e:
        print(f"Failed to install nltk: {e}")
        sys.exit(1)

import nltk
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler


# 检查nltk相关停用词等资源是否已下载，未下载则自动下载
def check_nltk_resources():
    """检查并自动下载nltk资源"""
    try:
        # 尝试获取停用词
        stopwords.words('english')
        print("[None]Stopwords are already downloaded.")

    except LookupError:
        print("Downloading stopwords...")

        try:
            nltk.download('stopwords', quiet=False)
            stopwords.words('english')
            print("Success!")

        except Exception as e:
            print(f"[E] Failed to download stopwords: {e}")
            sys.exit(1)

    try:
        # 尝试获取Punkt tokenizer
        nltk.data.find('tokenizers/punkt')
        print("[None]Punkt tokenizer is already downloaded.")

    except LookupError:
        print("Downloading Punkt tokenizer...")
        try:
            nltk.download('punkt', quiet=False)
            nltk.data.find('tokenizers/punkt')
            print("Success!")

        except Exception as e:
            print(f"[E] Failed to download Punkt tokenizer: {e}")
            sys.exit(1)




class TextPreprocessor:
    """文本处理器"""
    def __init__(self, extract_feat_method, reduce_dim_method, 
                         n_components=1000, ngram_range=(1, 1), min_df=5):
        """
        初始化
        - extract_features_method  特征提取方式
        - reduce_dim_method        降维方式
        - n_components             降维参数：保留的特征量
        - ngram_range              文本处理参数：定义连续的 n 个词或字符序列最小和最大长度
        - min_df                   文本处理参数：最小文档频率
        """
        self.data = None
        self.preprocessed_data = None
        self.features = None
        self.labels = None

        self.extract_features = {
            'bow':CountVectorizer,
            'tfidf':TfidfVectorizer
        }
        self.reduce_dimension = {
            'pca':PCA,
            'svd':TruncatedSVD
        }

        extract = self.extract_features.get(extract_feat_method)
        reduce = self.reduce_dimension.get(reduce_dim_method)

        if not extract:
            raise ValueError(f"No model found for id '{extract_feat_method}'")
        if not reduce:
            raise ValueError(f"No reduce_dimension found for '{reduce_dim_method}'")

        self.vectorizer = extract(ngram_range=ngram_range, min_df=min_df)
        self.reducer = reduce(n_components=n_components)

    

    def load_data(self, input_file_path='', input_data=None, Sep=',', Encoding='utf-8'):
        """加载数据"""
        # 既未指定数据所在路径，也没有任何数据的输入
        if input_file_path == '' and input_data == None:
            print("[E]No input file path specified or no dataset input.")
            return None
        elif input_file_path != '':
            self.data = pd.read_csv(input_file_path, sep=Sep, encoding=Encoding)
            return self.data
        elif input_file_path == '' and input_data != None:
            self.data = input_data


    def get_column(self, column_name):
        """
        将读取到的原始数据集按列名返回
        """   
        # 提取列
        raw_data_column = self.data.get(column_name, None)
        
        # 检查是否有 column_name 列
        if raw_data_column is None:
            raise ValueError(f"[E]Cannot find column named '{column_name}'.")
        
        return np.array(raw_data_column)


    def preprocess_text(self, text):
        """单文本预处理函数(原则上不允许直接调用，为 preprocess_data 的前置函数)"""
        # 转换为小写
        text = text.lower()
        # 移除标点符号
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 分词
        words = word_tokenize(text)
        # 移除停用词
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        # 返回处理后的单词列表
        return filtered_words


    def preprocess_data(self, key):
        """批量预处理数据集(原则上不允许直接调用，为 Processing_features 的前置函数)"""
        self.data['processed_text'] = self.data[key].apply(self.preprocess_text)
        self.preprocessed_data = self.data.copy()


    def Processing_features(self, message_column_name):
        """
        features处理:提取特征与降维
        - message_column_name   文本列名称
        """
        
        # 预处理数据集
        self.preprocess_data(message_column_name)

        processed_text_str = self.preprocessed_data['processed_text'].apply(
            lambda x: ' '.join(x))

        raw_features = self.vectorizer.fit_transform(processed_text_str)

        self.features = self.reducer.fit_transform(raw_features.toarray())

        # LR的训练数据集必须标准化或规范化
        scaler = StandardScaler()
        # 调用 scaler.fit_transform(x) 方法对数据进行拟合和转换。
        self.features = scaler.fit_transform(self.features)

        return self.features


    def Processing_labels(self, labels_column_name, classification={}):
        """
        labels处理: 标签标准化
        - labels_column_name     标签列名称
        - classification         分类特征字典，若为空{},则不做处理，直接返回该列
        """
        if classification == {}:
            self.labels = self.preprocessed_data[labels_column_name]
        else:
            self.labels = self.preprocessed_data[labels_column_name].map(classification)

        return self.labels



    def save_to_csv(self, output_file_path=None):
        """将处理后的数据保存到CSV文件"""

        if output_file_path == None:
            print("No output file path specified.")
            return None
        if self.features is not None:
            # 将特征向量转换为 DataFrame
            feature_df = pd.DataFrame(self.reduced_features,
                columns=[f'feature_{i}' for i in range(self.reduced_features.shape[1])])
            
            if self.labels != None:
                # 有标签情况
                result_df = pd.concat([self.labels, feature_df], axis=1)
                result_df.to_csv(output_file_path, index=False)
            else:
                # 无标签情况
                result_df = pd.concat([feature_df], axis=1)
                result_df.to_csv(output_file_path, index=False)

        else:
            print("[E]Features have not been extracted.")


# 示例使用
if __name__ == '__main__':
    # 调用函数检查资源
    check_nltk_resources()

    # 调用文本处理,填入数据集路径名称<data_name>.csv与输出路径文件名称<new_data_name>.csv
    preprocessor = TextPreprocessor(extract_features_method='tfidf', 
                                    reduce_dim_method='pca', n_components=1000)

    # 加载数据
    print("加载数据...")
    data = preprocessor.load_data(input_file_path='../data/SpamData.csv')

    # 提取特征（内置预处理数据）
    print("提取特征与降维...")
    features = preprocessor.Processing_features('Message')

    print('标签标准化...')
    labels = preprocessor.Processing_labels('Category', {'ham': 0, 'spam': 1})

    # 保存处理后的数据到CSV文件
    print("保存中...")
    preprocessor.save_to_csv(output_file_path='../data/new_SpamData.csv')

    print("over!")

    # 输出特征和标签
    '''
    print("Features:")
    print(features)
    print("Labels:")
    print(labels)
    '''
