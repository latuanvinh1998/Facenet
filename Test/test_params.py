
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

import sys 
sys.path.insert(1, "../Source/Models")

from inception_resnet_v2 import Inception_Resnet_V2
from squeeze_net_v1_1 import SqueezeNet
from mobile_net_v1 import MobileNetV1
from mobile_net_v2 import MobileNetV2



img_input = Input(shape=(160, 160, 3), name='input')

model1 = Inception_Resnet_V2(128)
model = Model(img_input, model1(img_input ,True), name='Inception Resnet v2')
model.summary()

model2 = SqueezeNet(128)
model = Model(img_input, model2(img_input, True), name='Squeeze Net')
model.summary()

model3 = MobileNetV1(128)
model = Model(img_input, model3(img_input ,True), name='Mobile Net V1')
model.summary()

model4 = MobileNetV2(128)
model = Model(img_input, model4(img_input ,True), name='Mobile Net V2')
model.summary()

import cv2

img = cv2.imread('../Dataset/Processed/Phat/1.jpg')
img = np.array(img, np.float32)

emb = model1(img, False)
emb1 = model2(img, False)
emb2 = model3(img, False)
emb3 = model4(img, False)

# print(emb)
# print(emb1)
# print(emb2)
print(emb3)