# SS_SPAM 基于隐语的隐私保护垃圾邮件过滤器 v0.1.0

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

本工具基于隐语框架，实现包含多种隐私计算机器学习模型可选的垃圾邮件过滤器，统一接口，统一格式，结合文本处理与隐语的隐私计算模型框架，功能上方便用户使用同时、也方便其他开发者增加新的模型或者算法，即：一个多模型、易扩展、用户友好的过滤器。



## 环境

- secretflow 1.5.0b0
  - https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.5.0b0/source/secretflow
- python >=3.10
- scikit-learn v1.5.1
- nltk v3.8.1  
- pandas v1.5.1

文本处理需要下载停用词，所以需要保证网络环境通畅否则文本处理器会报错



## 数据集格式

统一未处理的初始数据格式

|       第一列       |    第二列    |
| :----------------: | :----------: |
| 判断是否为垃圾邮件 | 邮件文本内容 |



## 文件结构

```
SS_SPAM-0.10
├── data
│   └── SpamData.csv
├── ml
│   ├── model #可选模型
│   |	├── __init__.py
│   |	├── DecisionTree.py 
│   |	├── XGboost.py 
│   |	├── NN.py
│   |	├── SecureLogisticRegression.py
|   |   ├── SSGLM.py
|   |   └── new_model_example.py
│   ├── __init__.py
│   ├── header.py
│   ├── TextPreprocessor.py #文本处理
│   └── toolfunc.py 
├── quick_star#使用demo
│   ├── main_ClusterSimulation.py
│   ├── main_NN.py
│   └── main_SingleNode.py
├── __init__.py
├── LICENSE
├── README.md
├── setup.py
└── ss_spam.py
```



### ss_spam.py

```python
class pp_spam_filter:
```

ss_spam是主调类，负责统一不同模块之间不同调用的方式，这样用同一个api就可以调用不同的模块。

ss_spam下定义了多个函数。

- _init___用于初始化模型 

  ```python
  def __init__(self, model_id, *args, **kwargs):
  ```
  
  ##### 参数说明
  
  - model_id  模型选择
  
    - 1: SecureLogisticRegression(spu)       逻辑回归
    - 2: SecureDecisionTree(spu)                 决策树
    - 3: SecureNN(Server, Clients, others)  神经网络
    - 4: SecureSSGLM(spu)           				 广义线性模型
  
    
  
- Textprocessor 用于处理数据集

  从本地加载完数据集后对数据进行处理，然后再对数据进行标准化和规范化。

  ```python
  def Textprocessor(self, _input_dataset=None, _input_file_path='', _output_file_path='', 
                        extract_feat_method='tfidf', reduce_dim_method='pca', 
                        _n_components=1000, _ngram_range=(1, 1), _min_df=5):   
  
  ```
  
  ##### 参数说明  
  
  必选参数： 
  
  - _input_dataset    传入数据集
  
  - _input_file_path  传入数据集文件路径
  
    (以上二者必须有一个传参，若都传入参数则默认 _input_file_path 优先)
  
  可选参数： 
  
  - _output_file_path 传出数据集文件路径(=''则默认若不保存)  
  
  - extract_feat_method 特征工程处理方法选择，有bow和tfidf两种方法(默认tfidf)
  - reduce_dim_method 降维方法，有pca和svd两种方法(默认pca)
  - _n_components 降维后拥有的的特征量个数(默认1000)
  - _ngram_range
  - _min_df
  
  
  
- train_test_split 用与划分处理好的数据集

  将数据集随机划分为训练集和测试集，分别用来训练和测试模型。

  ```
  def train_test_split(self, X_data, y_data, split_factor=0.8, random_state=1234):
  ```
  
  ##### 参数说明
  
  - random_state     控制数据在划分前的随机排序，默认为1234
  - split_factor            控制数据划分的大小，默认为0.8
  
- train 训练函数，根据选择的模型对数据进行训练

  ```python
  def train(self, *args, **kwargs):
  ```

- predict 预测函数，对训练好的模型进行测试

  ```python
  def predict(self, new_X_test=None):
  ```

#### 调用方法

首先初始化spu

```
spu = sf.SPU(cluster_def=cluster_def)
```

然后调用ss_spam初始化模型

```
run = ss_spam(model_id, spu)
```

之后即可调用 Textprocessor 处理数据

```
X, y = run.Textprocessor(_input_file_path=_input_file_path)
```

划分数据集和训练集

```
_x1, _x2, _y1, y_test = run.train_test_split(X_data, y_data)
```

准备工作完成后即可训练模型

```
run.train()
```

接着再对模型进行预测评估

```
run.predict()
```



### TextPreprocessor.py

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

初始化文本处理器

```
preprocessor = TextPreprocessor()
```

加载数据

```
data = preprocessor.load_data(input_file_path='SpamData.csv')
```

对数据进行处理

```
preprocessor.extract_features(method='tfidf', ngram_range=(1, 1), min_df=5)
preprocessor.reduce_dimensions(method='pca', n_components=1000)
```

返回输出矩阵和标签

```
features, labels = preprocessor.get_X_y()
```

保存处理后的数据

```
preprocessor.save_to_csv(output_file_path='new_SpamData.csv')
```



### quick_start

该目录下给出了SecureLogisticRegression 模型的单节点仿真和集群仿真的示例，SecureNN 模型的单节点仿真示例。

**main_SingleNode.py**

**main_ClusterSimulation.py**

**main_NN.py**



### LICENSE

该项目签署了[Apache License 2.0](https://github.com/MapleleavesCX/SS_SPAM/blob/v0.1.0/LICENSE)授权许可.



## 安装

git clone https://github.com/MapleleavesCX/SS_SPAM.git



## 使用

**from SS_SPAM.ss_spam import ss_spam**



### 添加新模型

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

