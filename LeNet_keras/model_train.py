import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 读取训练集、验证集路径
data_train = pathlib.Path("/data/ssd0/chenxiaolong/LeNet/data/train")
data_val   = pathlib.Path("/data/ssd0/chenxiaolong/LeNet/data/val")

# 类别名称（文件夹名称要和这里一致）
class_names = ['Cr', 'In', 'Pa', 'PS', 'Rs', 'Sc']

# 设置图片大小和批次数
batch_size = 64
img_h = 32
img_w = 32

# 用 image_dataset_from_directory 生成训练集 / 验证集
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_train,
    labels="inferred",
    label_mode="categorical",         # 如果你后面用 one-hot，就用 categorical；否则用 "int"
    class_names=class_names,          # 类别顺序按你给的列表
    image_size=(img_h, img_w),
    color_mode="rgb",                 # 如果是灰度图改成 "grayscale"
    batch_size=batch_size,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_val,
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    image_size=(img_h, img_w),
    color_mode="rgb",                 # 同上
    batch_size=batch_size,
    shuffle=False
)

# 归一化层：等价于之前的 rescale=1./255
normalization_layer = layers.Rescaling(1.0 / 255.0)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 利用keras搭建CNN
model = keras.Sequential()
model.add(Conv2D(filters=6, kernel_size=5, input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=5, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=120, kernel_size=5, activation='relu'))
model.add(Flatten())
model.add(Dense(84, activation='relu'))
model.add(Dense(6, activation='softmax'))

# 编译CNN
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# 传入数据集进行训练
history = model.fit(train_ds, validation_data=val_ds, epochs=50)

# 保存训练好的模型
model.save("model.h5")

# 绘制loss图
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("CNN神经网络loss值")
plt.legend()
plt.show()

# 绘制准确率
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("CNN神经网络accuracy值")
plt.legend()
plt.show()
