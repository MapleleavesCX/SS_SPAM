# SS_SPAM 基于隐语的隐私保护垃圾邮件过滤器 v0.2.0



## 简介

实现包含多种隐私计算机器学习模型可选的垃圾邮件过滤器统一框架
##2.0版本相比1.0版本改进如下：
* 进一步提升了用户友好性，实现了从集群设置到邮件分类的全流程封装，用户无需自己调用 secreflow 的相关接口，即可轻松使用本框架；
* 更简洁清晰的文本处理器代码与更合理的API设置
* 任意类型的垃圾邮件文本数据集的处理与读取



## 环境

- secretflow 1.5.0b0
  - https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.5.0b0/source/secretflow
- python >=3.10
- scikit-learn v1.5.1
- nltk v3.8.1  
- pandas v2.2.2

文本处理需要下载停用词，所以需要保证网络环境通畅否则文本处理器会报错


.......



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

