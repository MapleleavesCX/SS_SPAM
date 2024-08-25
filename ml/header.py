

from .toolfunc import timing

# 其他模型的导入...
from .model.SecureLogisticRegression import SecureLogisticRegression
from .model.DecisionTree import SecureDecisionTree
from .model.XGboost import SecureXGboost
from .model.NN import SecureNN
from .model.SSGLM import ssglm


class model_selector:
    @timing
    def __init__(self, model_id=int, *args, **kwargs):
        '''初始化传参内容：
        1: SecureLogisticRegression(spu),
        2: SecureDecisionTree(spu),
        3: SecureNN(Server, Clients, others),
        4: ssglm(spu),

        99:SecureXGboost(Server, Clients)
        '''
        
        self.models = {
            1: SecureLogisticRegression,
            2: SecureDecisionTree,
            3: SecureNN,
            4: ssglm,
            # 其他模型的映射...

            99: SecureXGboost,
            
        }
        self.model_class = self.models.get(model_id)
        if not self.model_class:
            raise ValueError(f"No model found for id {model_id}")
        
        self.model = self.model_class(*args, **kwargs)

    @timing
    def train(self, X_train, y_train, *args, **kwargs):
        '''训练函数'''
        self.model.fit(X_train, y_train ,*args, **kwargs)

    @timing
    def predict(self, X_test, *args, **kwargs):
        '''预测函数'''
        return self.model.predict(X_test, *args, **kwargs)
    
    # # 以下函数对部分模型不支持
    # @timing
    # def load_model(self, model_path):
    #     '''模型加载函数（部分模型不支持）'''
    #     None
    
    # @timing
    # def save_model(self, model_path):
    #     '''模型保存函数（部分模型不支持）'''
    #     return None
