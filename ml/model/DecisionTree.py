# -*- coding: utf-8 -*-
# @Date    : 2024/08/18
# @Author  : sdushushu

from secretflow.ml.boost.ss_xgb_v import Xgb

# train训练额外参数参考列表：
'''
params = {
    # for more detail, see Xgb API doc
    'num_boost_round': 5,
    'max_depth': 5,
    'learning_rate': 0.1,
    'sketch_eps': 0.08,
    'objective': 'logistic',
    'reg_lambda': 0.1,
    'subsample': 1,
    'colsample_by_tree': 1,
    'base_score': 0.5,
}
'''

class SecureDecisionTree:
    def __init__(self, spu):
        self.Xgb = Xgb(spu)

    def fit(self, X_train, y_train, params = {
	    'num_boost_round': 2,
	    'max_depth': 2,
	    'learning_rate': 0.1,
	    'sketch_eps': 0.08,
	    'objective': 'logistic',
	    'reg_lambda': 0.1,
	    'subsample': 1,
	    'colsample_by_tree': 1,
	    'base_score': 0.5,
	}):
        self.model=self.Xgb.train(dtrain=X_train, label=y_train, params=params)

    def predict(self, X_test):
        return self.model.predict(X_test)


# 示例使用
if __name__ == '__main__':
    None
