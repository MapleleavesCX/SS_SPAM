# -*- coding: utf-8 -*-
# @Date    : 2024/07/27
# @Author  : MapleleavesCX

'''
**********************************************************************
文本处理类型TextPreprocessor说明:
    load_data           负责从CSV文件中加载数据
    preprocess_text     用于单个文本的预处理
    preprocess_data     对整个数据集应用预处理
    extract_features    可以选择不同的特征提取方法
    reduce_dimensions   对特征提取后的数据集进行降维
    save_to_csv         将处理后的数据保存到CSV文件中
**********************************************************************
通过改变extract_features方法中的method参数来选择不同的特征提取方法,可选:
    bow,tfidf
**********************************************************************
词袋模型(Bag of Words, BoW):
    描述:词袋模型是最简单的文本特征提取方法之一,它将文本表示为单词的集合,
        忽略单词出现的顺序。
    实现:使用CountVectorizer类,它可以将文本转换为词频矩阵。
    优点:简单易懂,易于实现。
    缺点:忽略了单词之间的顺序关系,可能会丢失一些上下文信息。

TF-IDF向量化:
    描述:TF-IDF(Term Frequency-Inverse Document Frequency)不仅考虑了单
        词在文档中的频率,还考虑了单词在整个文档集合中的频率。
    实现:使用TfidfVectorizer类。
    优点:可以减轻常用词的影响,强调稀有词的重要性。
    缺点:仍然忽略了单词间的顺序。
**********************************************************************
通过改变reduce_dimensions方法中的method参数来选择不同的特征提取方法,可选:
    pca,svd
通过改变reduce_dimensions方法中的n_components参数确定降维后的特征数量
    默认1000
**********************************************************************
TruncatedSVD:
    这是一种针对稀疏矩阵的有效 SVD（奇异值分解）实现。它类似于 PCA 但适用于稀
    疏数据，并且通常用于文本数据的降维。

SparsePCA:
    一种基于稀疏主成分的 PCA 方法。它试图找到具有较少非零系数的主成分，从而获得
    更易于解释的特征。
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
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD


# 检查nltk相关停用词等资源是否已下载，未下载则自动下载
def check_nltk_resources():
    """检查并自动下载nltk资源"""
    try:
        # 尝试获取停用词
        stopwords.words('english')
        print("Stopwords are already downloaded.")

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
        print("Punkt tokenizer is already downloaded.")

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
    def __init__(self):
        """初始化"""
        self.data = None
        self.preprocessed_data = None
        self.features = None
        self.labels = None
        self.reduced_features = None

    def load_data(self, input_file_path='', input_data=None):
        """加载数据"""
        # 既未指定数据所在路径，也没有任何数据的输入
        if input_file_path == '' and input_data == None:
            print("No input file path specified or no dataset input.")
            return None
        elif input_file_path != '':
            self.data = pd.read_csv(input_file_path)
            return self.data
        elif input_file_path == '' and input_data != None:
            self.data = input_data

    def preprocess_text(self, text):
        """文本预处理函数"""
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

    def preprocess_data(self):
        """自动预处理数据集"""

        self.data['processed_text'] = self.data['Message'].apply(self.preprocess_text)
        self.preprocessed_data = self.data.copy()


    def extract_features(self, method='bow', ngram_range=(1, 1), min_df=5):
        """提取特征"""
        
        # 预处理数据集
        self.preprocess_data()

        processed_text_str = self.preprocessed_data['processed_text'].apply(
            lambda x: ' '.join(x))

        vectorizer = None
        if method == 'bow':
            vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
        elif method == 'tfidf':
            vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)

        self.features = vectorizer.fit_transform(processed_text_str)
        self.labels = self.preprocessed_data['Category'].map({'ham': 0, 'spam': 1})


    def reduce_dimensions(self, method='pca', n_components=1000):
        """降维特征"""

        if self.features is None:
            raise ValueError("Features have not been extracted.")

        # 根据选择的方法进行降维
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components)
        else:
            raise ValueError("Invalid dimension reduction method.")

        self.reduced_features = reducer.fit_transform(self.features.toarray())


    def save_to_csv(self, output_file_path=None):
        """将处理后的数据保存到CSV文件"""

        if output_file_path == None:
            print("No output file path specified.")
            return None
        if self.reduced_features is not None:
            # 将特征向量转换为 DataFrame
            feature_df = pd.DataFrame(self.reduced_features,
                columns=[f'feature_{i}' for i in range(self.reduced_features.shape[1])])
            # 将特征向量附加到原始 DataFrame
            result_df = pd.concat([self.labels, feature_df], axis=1)
            result_df.to_csv(output_file_path, index=False)
        elif self.features is not None:
            # 如果没有进行降维，则直接保存原始特征
            feature_df = pd.DataFrame(self.features.toarray(),
                columns=[f'feature_{i}' for i in range(self.features.shape[1])])
            result_df = pd.concat([self.labels, feature_df], axis=1)
            result_df.to_csv(output_file_path, index=False)
        else:
            print("Features have not been extracted.")

    def get_X_y(self):
        """获取特征矩阵和标签"""
        if self.reduced_features is not None:
            # 如果进行了降维
            features = self.reduced_features
        elif self.features is not None:
            # 如果没有进行降维
            features = self.features.toarray()
        else:
            raise ValueError("Features have not been extracted.")
        labels = self.labels.values
        return features, labels


# 示例使用
if __name__ == '__main__':
    # 调用函数检查资源
    check_nltk_resources()

    # 调用文本处理,填入数据集路径名称<data_name>.csv与输出路径文件名称<new_data_name>.csv
    preprocessor = TextPreprocessor()

    # 加载数据
    print("加载数据...")
    data = preprocessor.load_data(input_file_path='../data/SpamData.csv')

    # 提取特征（内置预处理数据），可自定义提取方法method
    print("预处理，并提取特征...")
    preprocessor.extract_features(method='tfidf', ngram_range=(1, 1), min_df=5)

    # 降维,可自设定降维方式method和压缩后的特征数量n_components
    print("降维...")
    preprocessor.reduce_dimensions(method='pca', n_components=1000)
    
    # 返回输出特征矩阵和标签
    features, labels = preprocessor.get_X_y()

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
