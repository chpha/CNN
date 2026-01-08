import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2

# 类别名称（文件夹名称要和这里一致）
class_names = ['Cr', 'In', 'Pa', 'PS', 'Rs', 'Sc']

# 设置图片大小和批次数
img_h = 32
img_w = 32

# 加载模型
model = load_model("model.h5")

# 数据读取与预处理
src = cv2.imread("/data/ssd0/chenxiaolong/LeNet/data/train/In/In_4.bmp")
src = cv2.resize(src, (32, 32))
src = src.astype("int32")
src = src/255

# 扩充数据维度
test_img = tf.expand_dims(src, 0)

preds = model.predict(test_img)
score = preds[0]

print("模型结果为{}，概率为{}".format(class_names[np.argmax(score)],np.max(score)))