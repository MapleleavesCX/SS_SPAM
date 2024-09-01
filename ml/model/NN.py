# -*- coding: utf-8 -*-
# @Date    : 2024/08/23
# @Author  :Rex

from tensorflow import keras
from tensorflow.keras import layers
import secretflow as sf
from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import SecureAggregator

def create_conv_model(input_shape, num_classes, name='model'):
    def create_model():
        # Create model
        model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Dense(256, activation="relu",kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu",kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="sigmoid"),  # 使用sigmoid适合二分类
    ])
        # Compile model
        model.compile(
            loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"]
        )
        return model

    return create_model


# __init__初始化额外参数参考列表：
'''
others = {
    'input_shape':1,
    'num_classes':(1000, ),
    'strategy':"fed_avg_w",
    'backend':"tensorflow",
}
'''
# train训练额外参数参考列表：
'''
params = {
    'epochs':10,
    'sampler_method':"batch",
    'batch_size':32,
    'aggregate_freq':1,
}
'''


class SecureNN:

    def __init__(self, 
                 Server, Clients, 
                 Input_shape=1,
                 Num_classes=(1000, ),
                 others={
                    'strategy':"fed_avg_w",
                    'backend':"tensorflow",
                 }):
        
        _model = create_conv_model(Input_shape, Num_classes)
        secure_aggregator = SecureAggregator(Server, Clients)

        self.model = FLModel(
            server=Server,
            device_list=Clients,
            model=_model,
            aggregator=secure_aggregator,
            **others
        )
    
    def fit(self, X_train, y_train, 
              params={
                'epochs':10,
                'sampler_method':"batch",
                'batch_size':32,
                'aggregate_freq':1,
            }):
        self.model.fit(
            X_train,
            y_train,
            **params
        )

   
    def predict(self, X_test, _batch_size=32):
        return self.model.predict(X_test, batch_size=_batch_size)
'''
predictions = self.model.predict(X_test, batch_size=32)
alice_predictions = sf.reveal(predictions[alice])
bob_predictions = sf.reveal(predictions[bob])
'''

# 示例使用
if __name__ == '__main__':
    None
