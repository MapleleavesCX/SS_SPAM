# -*- coding: utf-8 -*-
# @Date    : 2024/08/24
# @Author  : TaoRich


from secretflow.ml.linear.ss_glm import SSGLM

# train训练额外参数参考列表：
'''
params = {
    'learning_rate': 0.1,  # 可选参数，默认值，控制每轮迭代中模型更新的程度
    'batch_size': 1024,    # 可选参数，默认值，每轮计算使用的样本数量
    'eps': 1e-4,           # 可选参数，默认值，用于判断收敛的标准
    'epochs': 100,         # 可选参数，迭代轮数，可以根据实际情况调整
    'l2_lambda': 0.01      # 可选参数， L2 正则化系数，可以适当调整以防止过拟合
}
'''

class ssglm:

    def __init__(self, spu):
        self.model = SSGLM(spu)

    def fit(self, X_train, y_train, option='sgd', params = {
                    'learning_rate': 0.1,
                    'batch_size': 1024,
                    'eps': 1e-4,
                    'epochs': 100,
                    'l2_lambda': 0.01
                }):
        
        if option == 'sgd':
            self.model.fit_sgd(X_train, y_train, **params)
        elif option == 'irls':
            self.model.fit_irls(X_train, y_train, **params)
        else:
            raise ValueError(f"No mode found for '{option}'")
        

    def predict(self, X_test):
        return self.model.predict(X_test)



