# -*- coding: utf-8 -*-
# @Date    : 2024/08/10
# @Author  : MapleleavesCX

from secretflow.ml.linear.ss_sgd import SSRegression

# train训练额外参数参考列表：
'''
params = {
    'epochs':5,
    'learning_rate':0.3,
    'batch_size':32,
    'sig_type':'t1',
    'reg_type':'logistic',
    'penalty':'l2', 
    'l2_norm':0.1
}
'''


class SecureLogisticRegression:

    def __init__(self, spu):
        self.model = SSRegression(spu)

    def fit(self, X_train, y_train, params = {
            'epochs':5,
            'learning_rate':0.3,
            'batch_size':32,
            'sig_type':'t1',
            'reg_type':'logistic',
            'penalty':'l2', 
            'l2_norm':0.1
            }
        ):
        self.model.fit(X_train, y_train, **params)

    def predict(self, X_test):
        return self.model.predict(X_test)


# 示例使用
if __name__ == '__main__':
    None
