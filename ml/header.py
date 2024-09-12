

from .toolfunc import timing

# 其他模型的导入...
from .model.SecureLogisticRegression import SecureLogisticRegression
from .model.DecisionTree import SecureDecisionTree
from .model.XGboost import SecureXGboost
from .model.NN import SecureNN
from .model.SSGLM import SecureSSGLM


class model_selector:
    def __init__(self, model_id=int, *args, **kwargs):
        '''初始化传参内容：
        1001: SecureLogisticRegression(spu),
        1002: SecureDecisionTree(spu),
        1003: SecureSSGLM(spu),
        2011: SecureNN(Server, Clients, others),
        2099:SecureXGboost(Server, Clients)
        '''
        
        self.models = {
            1001: SecureLogisticRegression,
            1002: SecureDecisionTree,
            1003: SecureSSGLM,
            2001: SecureNN,
            2099: SecureXGboost,
            # 其他模型的映射...
            
        }
        self.model_class = self.models.get(model_id)
        if not self.model_class:
            raise ValueError(f"No model found for id {model_id}")
        
        self.model = self.model_class(*args, **kwargs)

    def train(self, X_train, y_train, *args, **kwargs):
        '''训练函数'''
        self.model.fit(X_train, y_train ,*args, **kwargs)

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
