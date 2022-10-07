import tensorflow as tf
from tensorflow.keras.layers import (Dense, Conv2D, LayerNormalization,
                                     Layer, Dropout, Input, GlobalAveragePooling2D, Embedding,Reshape,GlobalAveragePooling1D)
from tensorflow.keras import Sequential, Model
from GCN import GCNConv
# def loss_m(x,y):


class Identity(Layer):
    # usage:
    # 首先实例化, attn = Identity()
    # 然后传入tensor, out = attn(a_tensor)
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs


# class PatchEmbedding(Layer):
#     # imag_size=[224,224],in_channels=3, patch_size=7
#     # embed_dim=16,
#     def __init__(self, patch_size, embed_dim, dropout=0.):
#         super().__init__()
#         self.patch_embed = Conv2D(embed_dim, patch_size, patch_size)
#         self.dropout = Dropout(dropout)
#
#     def call(self, inputs):
#         # [batch,224,224,3] -> [batch,32,32,16]
#         x = self.patch_embed(inputs)
#
#         # [batch,32,32,16] -> [batch,32*32,16]
#         x = tf.reshape(x, shape=[x.shape[0], x.shape[1] * x.shape[2], x.shape[3]])
#
#         x = self.dropout(x)
#
#         return x


class MLP(Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.fc1 = Dense(int(embed_dim * mlp_ratio))
        self.fc2 = Dense(embed_dim)
        self.dropout = Dropout(rate=dropout)

    def call(self, inputs):
        # [batch,h,w,embed_dims] -> [batch,h,w,embed_dims*mlp_ratio]
        x = self.fc1(inputs)
        x = tf.nn.relu(x)  # 激活函数
        x = self.dropout(x)

        # [batch,h,w,embed_dims*mlp_ratio] -> [batch,h,w,embed_dims]
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class Encoder(Layer):
    def __init__(self, embed_dims):
        super().__init__()
        self.atten = Identity()  # TODO
        self.atten_norm = LayerNormalization()
        self.mlp = MLP(embed_dims)
        self.mlp_norm = LayerNormalization()

    def call(self, inputs):
        # [batch, h'*w', embed_dims] -> [batch, h'*w', embed_dims]
        h = inputs
        x = self.atten_norm(inputs)  # 先做层标准化
        x = self.atten(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h

        return x
class Spatial(Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = Dense(64,activation='relu')
        self.fc2 = Dense(32, activation='relu')
        self.fc3=Dense(2, activation='relu')
    def call(self,inputs):
        x=self.fc1(inputs)
        x=self.fc2(x)
        x=self.fc3(x)
        return x
class Temporal(Layer):
    def __init__(self):
        super().__init__()
        self.emb=Embedding(
            input_dim=2,
            output_dim=1,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
            input_length=32,
        )

    def call(self, inputs):

       return tf.squeeze(self.emb(inputs))
class transformer_st_matching(Layer):
    def __init__(self, embed_dims, encoder_length=5, num_classes=2):
        super().__init__()
      #  self.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dims)
        # encoder list
        layer_list = []
        layer_list = [Encoder(embed_dims=embed_dims) for i in range(encoder_length)]
        self.encoders = Sequential(layer_list)
        #self.head = Dense(num_classes )
        self.head = Dense(num_classes*10000)
        self.re=Reshape(target_shape=(10000,500))
        self.avgpool = GlobalAveragePooling2D()
        self.layernorm = LayerNormalization()
        self.Sp=Spatial()
        self.tmp=Temporal()
        self.Gr1=GCNConv(2)
    def call(self, inputs):
        # [batch, h, w, embed_dims] -> [batch, h'*w', embed_dims]
      #  x = self.patch_embed(inputs)
        x=self.Sp(inputs)
        y=self.tmp(inputs)
        self.Gr1([x,y])
        # 通过encoder_length层encoder
        x_y=tf.concat([x,y],2)
        x = self.encoders(inputs)

        # layernorm, 对embed_dims维度做归一化
        x = self.layernorm(x)

        # [batch, h'*w', embed_dims] -> [batch,embed_dims]
        x = self.avgpool(x)

        # [batch, embed_dims] -> [batch, num_classes]
        x = self.head(x)
        print(x)
        return self.re(x)
        #return x

if __name__ == '__main__':
    inputs = Input(shape=(3000,10,2), batch_size=2)
    model_st_matching= transformer_st_matching(embed_dims=1, encoder_length=7, num_classes=500)
    out = model_st_matching(inputs)
    model = Model(inputs=inputs, outputs=out, name='transformers-tf2')
    model.summary()
    tf.keras.utils.plot_model(
        model,
        to_file='model.png',
        show_shapes=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )
    # inputs = Input(shape=(32, 2), batch_size=4)
    # model=Temporal()
    # out=model(inputs)
    # model1=Model(inputs=inputs, outputs=out, name='vit-tf2')
    # model1.summary()
# inputs = Input(shape=(3000,2), batch_size=32)
# model_st_matching= transformer_st_matching(embed_dims=2, encoder_length=5, num_classes=1000)
# out = model_st_matching(inputs)
# model = Model(inputs=inputs, outputs=out, name='transformers-tf2')
