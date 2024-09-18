# SS_SPAM 基于隐语的隐私保护垃圾邮件过滤器-2.0

## 目录
- [简介](#简介)
- [环境](#环境)
- [数据集格式](#数据集格式)
- [文件结构](#文件结构)
- [安装](#安装)
- [使用](#使用)
- [贡献指南](#贡献指南)
- [联系方式](#联系方式)
- [致谢](#致谢)



## 简介

实现包含多种隐私计算机器学习模型可选的垃圾邮件过滤器统一框架 v0.2.0版本相比v0.1.0版本改进如下：

- 进一步提升了用户友好性，实现了从集群设置到邮件分类的全流程封装，用户无需自己调用 secreflow 的相关接口，即可轻松使用本框架；
- 更简洁清晰的文本处理器代码与更合理的API设置
- 任意类型的垃圾邮件文本数据集的处理与读取



## 环境

- secretflow 1.5.0b0
  - https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.5.0b0/source/secretflow
- python >=3.10
- scikit-learn v1.5.1
- nltk v3.8.1  
- pandas v1.5.1

文本处理需要下载停用词，所以需要保证网络环境通畅否则文本处理器会报错



## 数据集格式

​	\- 任意，但需要用户自己标识出特征列与文本列

## 文件结构

```
SS_SPAM-0.20
├── data
│   ├── spam1.csv
│   └── Spam2.csv
├── ml
│   ├── model #可选模型
│   |	├── __init__.py
│   |	├── DecisionTree.py 
│   |	├── XGboost.py 
│   |	├── NN.py
│   |	├── SecureLogisticRegression.py
|	|	├── SSGLM.py
|	|	└── new_model_example.py
│   ├── __init__.py
│   ├── header.py
│   ├── newTextPreprocessor.py #新的文本处理器
│   └── toolfunc.py 
├── __init__.py
├── main.py#使用demo
├── LICENSE
├── README.md
├── setup.py
└── ss_spam.py#主调类
```



### ss_spam.py

隐私保护的垃圾邮件过滤器

```python
class pp_spam_filter:
```

pp_spam_filte是主调类，负责统一不同模块之间不同调用的方式，这样用同一个api就可以调用不同的模块。

ss_spam下定义了多个函数。

- _init___用于初始化模型 

  ```python
      def __init__(self, model_id, extract_feat_method, reduce_dim_method, 
                  parties=[], ray_addr='local', cluster_def={}, 
                   _n_components=1000, _ngram_range=(1, 1), _min_df=5,
                  *args, **kwargs):
  ```

  ##### 参数说明

  - model_id  模型选择（必选）
    - 1: SecureLogisticRegression(spu)       逻辑回归
    - 2: SecureDecisionTree(spu)                 决策树
    - 3: SecureNN(Server, Clients, others)  神经网络
    - 4: SecureSSGLM(spu)                            广义线性模型
  - extract_feat_method 特征工程处理方法选择（必选）
    - 'bow'
    - 'tfidf'
  -  reduce_dim_method 降维方法（必选）
    - 'pca'
    - 'svd'
  - parties   成员列表 （必选）
  - ray_addr   ray地址 （可选）
  - cluster_def 集群网络配置（可选）
  - _n_components 降维后拥有的特征量个数（可选）
  - _ngram_range 
  - _min_df
  
- Textprocessor 用于处理数据集

  从本地加载完数据集后对数据进行处理，然后再对数据进行标准化和规范化。

  ```python
      def Textprocessor(self, _input_dataset=None, _input_file_path='', _output_file_path='', 
                        message_column_name='', labels_column_name='',
                        classification={},
                        Sep=',', Encoding='utf-8'):
  ```

  ##### 参数说明  

  必选参数： 

  - _input_dataset    传入数据集

  - _input_file_path  传入数据集文件路径

    (以上二者必须有一个传参，若都传入参数则默认 _input_file_path 优先)
    
  - message_column_name   文本列名称

  - \- labels_column_name   标签列名称

  可选参数： 

  - classification     分类特征字典
  - Sep           文本列分隔符
  - Encoding        编码方式

  返回值：

  - X  经过文本处理后的数据集特征
  - y  经过文本处理后的数据集标签

  

- data_divider用与划分处理好的数据集

  将数据集随机划分为训练集和测试集，分别用来训练和测试模型。

  ```
      def data_divider(self, 
                       direction, X_partitions={}, y_partitions={}, 
                       split_factor=0.8, random_state=1234):
  ```

  ##### 参数说明

  - direction   数据集划分方向(纵向'c'或横向'r')
  - X_partitions  数据特征 X 划分设置
  - y_partitions  数据标签 y 划分设置
  - split_factor  划分因子(即 训练数据所占比例)
  - random_state  随机度

- train 训练函数，根据选择的模型对数据进行训练

  ```python
  def train(self, *args, **kwargs):
  ```

- evaluation 评估函数，对训练好的模型进行评估

  ```python
  def evaluation(self, memberKey):
  ```

- filter本地过滤器，使用训练好的模型对处理过的数据集进行过滤

  ```python
      def filter(self, memberKey='', input_text_path='', 
                 message_column_name='', labels_column_name='', classify={}, 
                 Sep=',', Encoding='utf-8'
                 ):
  ```

  **参数说明**

  - memberKey     					     选择执行节点名称
  - input_text_path                        文本地址
  - message_column_name         文本列的列名称
  - labels_column_name(可选)     特征列名称（ 当此参数不为 None 则进入测试模式，输出分类结果评估）
  - classify                                        特征标准化处理设置，为{}则直接输出
  - Sep                                               文本列分隔符
  - Encoding                                     编码方式

#### 调用方法

首先初始化模型

```
run = pp_spam_filter(
     model_id, parties=parties, extract_feat_method='tfidf', reduce_dim_method='pca')
```

之后即可调用 Textprocessor 处理数据

```
run.Textprocessor(_input_file_path=spam1_path, 
                  message_column_name='Message', labels_column_name='Category', 
                  classification=classif_spam1)
```

数据集划分

```
run.data_divider(direction='c', X_partitions=X_par, y_partitions=Y_par)
```

对模型进行训练

```
run.train()
```

接着再对模型进行预测评估

```
run.evaluation('alice')
```

最后使用训练完的过滤器对数据进行过滤

```
run.filter(memberKey='alice', input_text_path=spam1_path, 
            message_column_name='Message', labels_column_name='Category', classify=classif_spam1)
```



### newTextPreprocessor.py

文本处理器用来对文本数据进行处理，由于邮件信息大多是文本信息，所以使用TextPreprocessor来对数据集进行处理。

```
class TextPreprocessor
```

定义 TextPreprocessor类来对文本数据进行处理

#### 调用方法

首先调用函数检查资源

```
check_nltk_resources()
```

调用文本处理器

```
preprocessor = TextPreprocessor(extract_features_method='tfidf', 
                                    reduce_dim_method='pca', n_components=1000)
```

加载数据

```
data = preprocessor.load_data(input_file_path='../data/SpamData.csv')
```

对数据进行处理

```
features = preprocessor.Processing_features('Message')#提取特征与降维
labels = preprocessor.Processing_labels('Category', {'ham': 0, 'spam': 1})#标签标准化
```

保存处理后的数据到CSV文件

```
preprocessor.save_to_csv(output_file_path='../data/new_SpamData.csv')
```



### LICENSE

该项目签署了[Apache License 2.0](https://github.com/MapleleavesCX/SS_SPAM/blob/v0.1.0/LICENSE)授权许可.



## 安装

git clone https://github.com/MapleleavesCX/SS_SPAM.git



## 使用

​	**from ss_spam import pp_spam_filter**



#### 添加新模型

​		1.以SS_SPAM/ml/model/new_model_example.py提供的模型标准创建新的模型代码添加到SS_SPAM/ml/model下。

​		2.在SS_SPAM/ml/header.py中import新的模型。

​		3.修改model_selector类下的初始化函数，添加新的模型及对应的编号。



## 贡献指南

- **欢迎加入**：我们非常欢迎您的贡献！无论是修复 bug、改进文档还是添加新功能，您的每一份努力都将使项目更加完善。
- **贡献流程**
  1. 克隆仓库到本地。
  2. 创建一个新的分支：`git checkout -b your-feature-name`。
  3. 实现您的更改。
  4. 提交更改：`git commit -m "Add some feature"`。
  5. 推送至远程仓库：`git push origin your-feature-name`。
  6. 在 GitHub 上发起 Pull Request。
- **代码规范**：请确保您的代码符合 编码标准。
- **问题反馈**：如果您发现了 bug 或有改进建议，请在 Issues 页面提交问题。
- **许可证**：本项目遵循 apache 2.0 许可证。
- **联系我们**：如有任何疑问，请发送邮件至 3218391825@qq.com。



## 联系方式

[MapleleavesCX](https://github.com/MapleleavesCX)   3218391825@qq.com



## 致谢

首先非常感谢山东大学网络空间安全学院的李增鹏老师对于我们小组的指导和隐语平台提供的资源，其次也感谢数据直通队内每一个队员的辛苦付出，以下排名不分先后。

[MapleleavesCX](https://github.com/MapleleavesCX)

[Rex-7](https://github.com/Rex-7)

[sdushushu](https://github.com/sdushushu)

[Taorich](https://github.com/Taorich)

