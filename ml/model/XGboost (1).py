# -*- coding: utf-8 -*-
# @Date    : 2024/08/22
# @Author  : Sdushushu


import secretflow as sf
from secretflow.ml.boost.homo_boost import SFXgboost

# train训练额外参数参考列表：
'''
inside_params = {
         'max_depth': 4, # 决策树的最大深度
         'eta': 0.3, # 学习率。控制每次提升权重时步长的大小。
         'objective': 'binary:logistic', # 定义模型的训练目标,"binary:logistic","reg:logistic","multi:softmax","multi:softprob","reg:squarederror"
         'min_child_weight': 2, # 决定进一步分割叶节点时所需的最小样本权重总和。较大的值可以防止模型过拟合。
         #'lambda': 0.1, # 权重的 L2 正则化系数。用来控制模型的复杂度，防止过拟合。
         #'alpha': 0, # 权重的 L1 正则化系数。可以用于特征选择，较大的值会增加稀疏性。
         'max_bin': 10, # 直方图算法中的最大分桶数。较小的值会加快训练速度，但可能降低模型精度。
         #'num_class':6, # 仅在多分类问题中使用，表示类别的数量。
         #'gamma': 0, # 分裂节点时所需的最小增益。如果值为0，则不做限制；较大的值可以避免过拟合。
         #'subsample': 1.0, # 用于控制每棵树的随机样本比例。较小的值会使模型更保守，但有助于防止过拟合。
         #'colsample_by_tree': 1.0, # 控制每棵树使用的特征的随机子样本比例。
         #'colsample_bylevel': 1.0, # 控制每一层使用的特征的随机子样本比例。
         'eval_metric': 'auc',  # 评价指标。supported eval metric：
                                    # 1. rmse
                                    # 2. rmsle
                                    # 3. mape
                                    # 4. logloss
                                    # 5. error
                                    # 6. error@t
                                    # 7. merror
                                    # 8. mlogloss
                                    # 9. auc
                                    # 10. aucpr
        'verbosity':1,
         # Special params in SFXgboost
         # Required
         'hess_key': 'hess', # Required, Mark hess columns, optionally choosing a column name that is not in the data set
         'grad_key': 'grad', # Required，Mark grad columns, optionally choosing a column name that is not in the data set
         'label_key': 'Category', # Required，ark label columns, optionally choosing a column name that is not in the data set
        
        'num_boost_round':6  # 轮数
     }
'''

class SecureXGboost:

    def __init__(self, Server, Clients):
        self.model = SFXgboost(Server, Clients)
    
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
        self.model.fit(train_hdf=X_train, valid_hdf=X_test, **params)

    '''
    # 目前没有找到预测函数
    def predict(self, X_test):
        return self.model.predict(X_test)
    '''


# 示例使用
if __name__ == '__main__':
    None
