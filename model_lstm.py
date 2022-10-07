import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.layers import Layer


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = tf.keras.initializers.get('glorot_uniform')
        # W_regularizer: 权重上的正则化
        # b_regularizer: 偏置项的正则化
        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)
        # W_constraint: 权重上的约束项
        # b_constraint: 偏置上的约束项
        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1]),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (inputs.shape[0],-1, step_dim))

        if self.bias:
            eij *= self.b


        a = K.exp(eij)
        '''
        keras.backend.cast(x, dtype): 将张量转换到不同的 dtype 并返回
        '''
        if mask is not None:
            a *= K.cast(mask, K.floatx())

        '''
        keras.backend.epsilon(): 返回浮点数
        '''
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        weighted_input = tf.concat([a,x],axis=-1)

        # return K.sum(weighted_input, axis=1)
        return weighted_input
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
class lstm(Layer):
    def __init__(self):
        super().__init__()
        #self.reshape=Reshape((3000,20))
        self.reshape2 = Reshape((3000,200))
        self.reshape1=Reshape((10000,500))
        self.GRU=GRU(256, return_sequences=True)
        self.dropout=Dropout(0.2)
        self.dropout1=Dropout(0.2)
        self.normal=BatchNormalization()
        self.d1=Dense(64,activation="relu")
        self.d2=Dense(500*10000,activation="relu")
        self.glob=GlobalAveragePooling1D()
        self.at=Attention(1)
    def call(self, inputs, **kwargs):
       # x=self.reshape(inputs)
        x=inputs
        x=self.reshape2(x)
        x=self.GRU(x)
        x=self.dropout(x)

        x = self.at(x)
        x=self.normal(x)
        x=self.d1(x)
        x=self.dropout1(x)
        x=self.glob(x)
        x=self.d2(x)
        x=self.reshape1(x)
        return x
inputs = Input(shape=(3000,100, 2),batch_size=2)
# # # # inputs1=Reshape((3000,20))(inputs)
# # # # x = GRU(256, return_sequences=True)(inputs1)
# # # # x = Dropout(0.2)(x)
# # # # x = BatchNormalization()(x)
# # # #
# # # # x = Dense(32, activation="relu")(x)
# # # # x = Dropout(0.2)(x)
# # # # x=GlobalAveragePooling1D()(x)
# # # # x = Dense(500*10000, activation="relu")(x)
# # # # print(x)
# # # # x=Reshape((10000,500))(x)
x=lstm()(inputs)
model_lstm_attention = tf.keras.Model(inputs=inputs, outputs=x)
# # model_lstm_attention.compile(loss='mean_squared_error', optimizer='adam')
model_lstm_attention.summary()
