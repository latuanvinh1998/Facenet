import numpy as np 
import os
import sys
import torch
import pickle
import cv2
import tensorflow as tf
sys.path.insert(1, "../Source/Models")
from mobile_net_v2 import MobileNetV2
sys.path.insert(1, "../Source/MTCNN")
from mtcnn import MTCNN


def img_resize(img, scale_percent):
	width = int(img.shape[1] * scale_percent)
	height = int(img.shape[0] * scale_percent)
	dim = (width, height)
	img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return img

TRAINED_MODEL = '../TrainedModels/MobileNetV2/201105_1535/'

SVM_MODEL =  "../TrainedModels/SVM/facenet.pkl"

IMAGE_PATH1  = '../Dataset/Test/3.jpg'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(image_size=160, device=device)
model = MobileNetV2(128, alpha=1.0)
model.load_weights(TRAINED_MODEL).expect_partial()

with open(SVM_MODEL, 'rb') as infile:
	(svm_model, class_names) = pickle.load(infile)

img = cv2.imread(IMAGE_PATH1)
if max(img.shape[0], img.shape[1]) > 900:
	scale_percent = 900/max(img.shape[0], img.shape[1])
	img = img_resize(img, scale_percent)

#faces = detector.align(img=img, select_largest=False, save_path='../../Test/')
faces = detector.align(img=img, select_largest=False)

if faces is not None:
	for face in faces:
		face = np.float32(face)
		face_embedding = model(face, True)
		predictions = svm_model.predict_proba(face_embedding)
		best_class_idxs = np.argmax(predictions, axis=1)
		best_class_probabilities = predictions[np.arange(len(best_class_idxs)), best_class_idxs]
		for i in range(len(best_class_idxs)):
			print('%4d  %s: %.3f' % (i, class_names[best_class_idxs[i]], best_class_probabilities[i]))




