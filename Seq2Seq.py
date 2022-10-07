import tensorflow as tf
from tensorflow.keras.layers import (Dense, Conv2D, LayerNormalization,
                                     Layer, Dropout, Input, GlobalAveragePooling2D, Embedding,Reshape,GlobalAveragePooling1D,LSTM,GRU)
from tensorflow.keras import Sequential, Model
# import seq2seq
# from seq2seq.models import SimpleSeq2Seq
#
# model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8)
# model.summary()
from GCN import GCNConv
from test3 import main_dataset2
import setting
from tqdm import *
from dataprocess import main_dataset
# Basic Parameters
batch_size = 1  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.

num_decoder_tokens=500
num_encoder_tokens=3000
# x1,x2,y=main_dataset2()
x1,x2,y=main_dataset()
print(tf.convert_to_tensor(y).shape)
encoder_inputs  = Input(shape=(3000, 2),batch_size=1)
# re1=Reshape((3000, 200))
encoder = LSTM(latent_dim, return_state=True)
# encoder_outputs, state_h, state_c = encoder(re1(encoder_inputs))
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs =  Input(shape=(10000,500),batch_size=1)
# decoder_inputs=tf.expand_dims(decoder_inputs,axis=0)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim ,return_state=True,return_sequences=True)

decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='relu')
decoder_outputs = decoder_dense(decoder_outputs)
re2=Reshape((10000,500))
decoder_outputs=re2(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
# @tf.function
# class lstm_seq_seq(Layer):
#     def __init__(self):
#         super().__init__()
#         #self.reshape=Reshape((3000,20))
#
#         # self.GRU=GRU(256, return_sequences=True)
#         self.GRU2=GRU(256,return_sequences=True)
#         self.dropout=Dropout(0.2)
#         self.dropout1=Dropout(0.2)
#
#         self.d1=Dense(64,activation="relu")
#         self.d2=Dense(500*10000,activation="relu")
#         self.glob=GlobalAveragePooling1D()
#
#     def call(self, inputs, **kwargs):
#        # x=self.reshape(inputs)
#         encoder_input=inputs[0]
#         decoder_input=inputs[1]
#         encoder_outputs, state_h, state_c=self.GRU(encoder_input)
#
#         encoder_states = [state_h, state_c]
#         x=self.GRU2(decoder_input,initial_state=encoder_states)
#         x= self.d1(x)
#         x=self.d2(x)
#         x=self.reshape2(x)
#         return x
# model1=lstm_seq_seq()
# encoder_inputs=Input((3000,2))
# decoder_inputs=Input((10000,500))
# decoder_outputs=model1([encoder_inputs, decoder_inputs])
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
#Run training
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
train_loss = tf.keras.losses.CategoricalCrossentropy()
train_accuracy=tf.keras.metrics.Accuracy()
@tf.function
def train_one_step(x,y,z):
    """
    一次迭代过程
    """
    # 求loss
    with tf.GradientTape() as tape:
        predictions = model([x,z])
        train_accuracy.update_state(y_true=y, y_pred=predictions)
        loss1 = train_loss(y_true=y,y_pred=predictions)
    # 求梯度
    grad = tape.gradient(loss1,  model.trainable_variables)
    # 梯度下降，更新噪声图片
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return loss1
model.compile(optimizer='rmsprop', loss=train_loss,
              metrics=['accuracy'])
model.fit([x1,x2],y,
          batch_size=batch_size,
          epochs=epochs,
         )
# for epoch in range(setting.epoch):
#     # 使用tqdm提示训练进度
#     with tqdm(total=len(x1) / setting.batch,
#               desc='Epoch {}/{}'.format(epoch, setting.epoch)) as pbar:
#         # 每个epoch训练settings.STEPS_PER_EPOCH次
#         for x, y,z in zip(x1,x2,y):
#             loss2 = train_one_step(x, z,y)
#             pbar.set_postfix(loss='%.4f' % float(loss2[len(loss2) - 1]), acc=float(train_accuracy.result()))
#             pbar.update(1)