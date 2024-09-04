# SS_SPAM 基于隐语的隐私保护垃圾邮件过滤器



## 简介

本工具基于隐语框架，对邮箱中的垃圾邮件进行过滤。

针对不同用户的需求，实现更加个性化的过滤系统，同时保护用户隐私安全，对电子邮箱中的垃圾邮件进行过滤。



## 环境

- secretflow 1.5.0b0
  - https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.5.0b0/source/secretflow
- python >=3.10
- sklearn v1.1.2
- nltk v3.8.1  
- pandas v1.5.1

文本处理需要下载停用词，所以需要保证网络环境通畅否则文本处理器会报错



## 统一未处理的初始数据集格式

​           第一列          				第二列

判断是否为垃圾邮件          邮件文本内容



## 文件结构

### ss_spam.py

```python
class ss_spam:
```

ss_spam是主调类，负责统一不同模块之间不同调用的方式，这样用同一个api就可以调用不同的模块。

ss_spam下定义了多个函数。

- _init___用于初始化模型 

  ```python
  def __init__(self, model_id, *args, **kwargs):
          
          # 初始化模型
          self.model = model_selector(model_id, *args, **kwargs)
          # 调用函数检查资源
          check_nltk_resources()
          # 初始化文本处理器
          self.preprocessor = TextPreprocessor()
  ```

  ##### 参数说明

  - model_id  模型选择

    - 1: SecureLogisticRegression(spu)   逻辑回归
    - 2: SecureDecisionTree(spu)       决策树
    - 3: SecureNN(Server, Clients, others)  神经网络
    - 4: ssglm(spu)            广义线性模型

    

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

### ml

#### model

该目录下包含此次所实现的几个模型

- ##### DecisionTree.py

  定义了类SecureDecisionTree，

  ```
  class SecureDecisionTree:
  ```

  在类下定义了初始化函数_init__,训练函数fit和预测函数predict。

  ```
  def __init__(self, spu)
  
  def fit(self, X_train, y_train, params={
              'num_boost_round': 5,
              'max_depth': 5,
              'learning_rate': 0.1,
              'sketch_eps': 0.08,
              'objective': 'logistic',
              'reg_lambda': 0.1,
              'subsample': 1,
              'colsample_by_tree': 1,
              'base_score': 0.5,
          })
          
  def predict(self, X_test):
  ```

- ##### NN.py

  首先定义了函数create_conv_model来创建模型，

  ```
  def create_conv_model(input_shape, num_classes, name='model'):
  ```

  接着定义了SecureNN类，

  ```
  class SecureNN:
  ```

  在类下定义了初始化函数_init__,训练函数fit和预测函数predict。

  ```
  def __init__(self, 
                   Server, Clients, 
                   Input_shape=1,
                   Num_classes=(1000, ),
                   others={
                      'strategy':"fed_avg_w",
                      'backend':"tensorflow",
                   }):
         
  def fit(self, X_train, y_train, 
                params={
                  'epochs':10,
                  'sampler_method':"batch",
                  'batch_size':128,
                  'aggregate_freq':1,
              }):
              
  def predict(self, X_test, _batch_size=32):
  ```

- ##### SecureLogisticRegression.py

  定义了类SecureLogisticRegression，

  ```
  class SecureLogisticRegression:
  ```

  在类下定义了初始化函数_init__,训练函数fit和预测函数predict。

  ```
  def __init__(self, spu):
  
  def fit(self, X_train, y_train, params = {
              'epochs':5,
              'learning_rate':0.3,
              'batch_size':32,
              'sig_type':'t1',
              'reg_type':'logistic',
              'penalty':'l2', 
              'l2_norm':0.1
              }
              
  def predict(self, X_test):
  ```

- ##### SSGLM.py

  定义了类SecureSSGLM，

  ```
  class SecureSSGLM:
  ```

  在类下定义了初始化函数_init__,训练函数fit和预测函数predict。

  ```
  def __init__(self, spu):
  
  def fit(self, X_train, y_train, option='sgd', params = {
                      'learning_rate': 0.1,
                      'batch_size': 1024,
                      'eps': 1e-4,
                      'epochs': 100,
                      'l2_lambda': 0.01
                  }):
                  
  def predict(self, X_test):
  ```

- ##### XGBoost.py

  定义了类SecureXGboost，

  ```
  class SecureXGboost:
  ```

  在类下定义了初始化函数_init__和训练函数fit。

  ```
  def __init__(self, Server, Clients):
  
  def fit(self, X_train, X_test, params={
              'max_depth': 4,
              'eta': 0.3,
              'objective': 'binary:logistic',
              'min_child_weight': 2,
              'max_bin': 10,
              'eval_metric': 'auc',
              'verbosity':1,
              'hess_key': 'hess',
              'grad_key': 'grad',
              'label_key': 'Category',
              'num_boost_round':6  # 轮数
          }):
   
  ```

- ##### new_model_example.py

  添加新模型标准格式的示例。

#### header.py

header下统一导入了实现的所有模型， 再定义了model_selector 类来初始化所选的模型，接着调用训练和预测函数。

```
class model_selector:
```

在model_selector 类下定义了模型初始化函数，训练函数和预测函数，同时调用timging函数来计时。

```
    @timing
    def __init__(self, model_id=int, *args, **kwargs):
```

```
    @timing
    def train(self, X_train, y_train, *args, **kwargs):
```

```
    @timing
    def predict(self, X_test, *args, **kwargs):
```



#### TextPreprocessor.py

文本处理器用来对文本数据进行处理，由于邮件信息大多是文本信息，所以使用TextPreprocessor来对数据集进行处理。

```
class TextPreprocessor
```

定义 TextPreprocessor类来对文本数据进行处理

##### 调用方法

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



#### toolfunc.py

这里是一些过程中用到的函数

timing 函数用来计算模型训练所需要的时间，用于评估模型

```
def timing(func):
```

 Divide_X函数用于对数据集进行划分，用于把数据集划分给不同的节点，可以按水平的方式划分，也可以按竖直的方式划分，默认以数值的方式划分

```python
def Divide_X(data, start=0.0, end=1.0, direction='c'):
```

**参数说明**

- data 需要划分的数据
- start 划分数据部分的开头
- end 划分数据结尾
- direction 划分数据的方式

Divide_y函数用于划分数据集的标签

```python
def Divide_y(label, start=0.0, end=1.0):
```



### quick_start

该目录下给出了两个使用的例子

##### main_ClusterSimulation.py

##### main_SingleNode.py



### data

数据集文件夹，包含实例**SpamData.csv**



### LICENSE

Apache License 2.0



## 安装

git clone https://github.com/MapleleavesCX/SS_SPAM.git



## 使用

**from SS_SPAM.ss_spam import ss_spam**



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

[
MapleleavesCX](https://github.com/MapleleavesCX)   3218391825@qq.com



## 致谢

首先非常感谢山东大学网络空间安全学院的李增鹏老师对于我们小组的指导，其次也感谢数据直通队内每一个队员的辛苦付出，以下排名不分先后。

[MapleleavesCX](https://github.com/MapleleavesCX)

[Rex-7](https://github.com/Rex-7)

[sdushushu](https://github.com/sdushushu)

[Taorich](https://github.com/Taorich)

