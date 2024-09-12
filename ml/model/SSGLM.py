# -*- coding: utf-8 -*-
# @Date    : 2024/08/25
# @Author  : Taorich

import numpy as np
from secretflow.ml.linear.ss_glm import SSGLM

# train训练额外参数参考列表：
'''

params = {
    'offset':None,         # 可选参数，指定一个列作为偏移量使用
    'weight':None,         # 可选参数，指定一个列来用作观测权重
    'epochs': 100,         # 可选参数，迭代轮数，可以根据实际情况调整
    'l2_lambda': 0.01      # 可选参数， L2 正则化系数，可以适当调整以防止过拟合
}
'''

class SecureSSGLM:

    def __init__(self, spu):
        self.model = SSGLM(spu)

    def fit(self, X_train, y_train, option='sgd', sgd_LR=0.1, sgd_BatchSize=1024, params = {
                    'offset':None,
                    'weight':None,
                    'epochs': 30,
                    'l2_lambda': 0.01
                }):
        # 根据官方文档：
        # https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.5.0b0/source/secretflow.ml.linear.ml.linear.ss_glm#secretflow.ml.linear.ss_glm.SSGLM.fit_irls
        # 只有 fit_sgd 有参数 learning_rate 和 batch_size ，故调整如下：
        if option == 'sgd':
            self.model.fit_sgd(X_train, y_train, link='Logit', dist='Bernoulli', 
                               learning_rate=sgd_LR, batch_size=sgd_BatchSize, **params)
        elif option == 'irls':
            self.model.fit_irls(X_train, y_train, link='Logit', dist='Bernoulli', **params)
        else:
            raise ValueError(f"No mode found for '{option}'")
        

    def predict(self, X_test):
        return self.model.predict(X_test)


