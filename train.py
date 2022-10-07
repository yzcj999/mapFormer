from model import transformer_st_matching
from model_lstm import lstm as lstm_model
import setting
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import (Dense, Conv2D, LayerNormalization,
                                     Layer, Dropout, Input, GlobalAveragePooling1D, Embedding,Reshape)
from tensorflow.keras import Sequential, Model
# from dataprocess import main_dataset
from test3 import main_dataset
from eval import HAM
from tqdm import *
@tf.function
def MPJPE(y_actual,y_pred):
    print("y_actual",y_actual)
    print("y_pred",y_pred)


    with tf.compat.v1.Session():
        y_actual =  y_actual
        print(y_actual)
        y_pred =  y_pred

        print(y_actual)
        print( tf.reduce_mean(tf.norm(y_actual-y_pred,axis=len(y_pred.shape) - 1)))
        return tf.reduce_mean(tf.norm(y_actual-y_pred,axis=len(y_pred.shape) - 1))

def loss(x,y):
    # print(type(x))
    # x=tf.reduce_mean(tf.square(x-y))
    # print(x.shape)
    # return x
    number = tf.convert_to_tensor(x, dtype=tf.float32)+1
    number1 = tf.convert_to_tensor(y, dtype=tf.float32)+1
    number = tf.math.log(number)+1e-5
    number_log = number / (number + 1)

    x = (1 - number1) * (1 - number_log)
    z = tf.math.log(number_log) * number1
    loss1 = -(x + z)

    x=tf.reduce_sum(loss1,axis=-1)
    X=tf.reduce_sum(x,axis=-1)
    return x
    # print(loss1.l)
    # try:
    #     return tf.keras.losses.MSE(loss1)
    # except:
    #     return 1e-5
    #return
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
train_loss = tf.keras.losses.CategoricalCrossentropy()
# train_accuracy =tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_accuracy =HAM()
inputs = Input(shape=(3000, 100, 2), batch_size=2)
# model_st_matching= transformer_st_matching(embed_dims=2, encoder_length=5, num_classes=500)
model_st_matching = lstm_model()
out = model_st_matching(inputs)
model = Model(inputs=inputs, outputs=out)

def train_one_step(x,y):
    """
    一次迭代过程
    """
    # 求loss
    with tf.GradientTape() as tape:
        predictions = model(x)
        train_accuracy.update_state(y_true=y, y_pred=predictions)
        loss1 = train_loss(y_true=y,y_pred=predictions)
    # 求梯度
    grad = tape.gradient(loss1,  model.trainable_variables)
    # 梯度下降，更新噪声图片
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return loss1
def train():
    train_dataset,val_dataset=main_dataset()
    print(train_dataset)
    train_dataset_size=80
    val_dataset_size=20


    # #Input(shape=(3000,2), batch_size=32)
    # model.build(input_shape=(3000,2))
    # print(model)
#    model.load_weights(os.path.join(setting.save_path, 'transformers_{0}.h5'.format(str(setting.initial_epoch))))
    callbacks = [
        # 模型保存
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(setting.save_path, "lstm-at{epoch}.h5"),
            monitor='val_loss',
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=20,
                                         restore_best_weights=True),
       # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00000001)
    ]
    # 查看模型结构
    model.summary()
    model.compile(
        optimizer="rmsprop",
        loss=train_loss,
        metrics=['accuracy']
    )
    model.fit(
        train_dataset,
        epochs=setting.epoch,
        steps_per_epoch=train_dataset_size // setting.batch,
       # initial_epoch=setting.initial_epoch,
        validation_data=val_dataset,
        validation_steps=val_dataset_size // setting.batch,
       callbacks=callbacks,

    )
    # model.fit
    # for epoch in range(setting.epoch):
    #     # 使用tqdm提示训练进度
    #     with tqdm(total=len(train_dataset) / setting.batch,
    #               desc='Epoch {}/{}'.format(epoch, setting.epoch)) as pbar:
    #         # 每个epoch训练settings.STEPS_PER_EPOCH次
    #         for x, y in train_dataset:
    #             loss2 = train_one_step(x, y)
    #             pbar.set_postfix(loss='%.4f' % float(loss2[len(loss2) - 1]), acc=float(train_accuracy.result()))
    #             pbar.update(1)

train()