import numpy as np
import torch
import cv2
import sys
sys.path.insert(1, '../Source/Models')
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
IMAGE_PATH1  = '../Dataset/Test/3.jpg'
IMAGE_PATH2  = '../Dataset/Test/4.jpg'

img1 = cv2.imread(IMAGE_PATH1)
if max(img1.shape[0], img1.shape[1]) > 900:
	scale_percent = 900/max(img1.shape[0], img1.shape[1])
	img1 = img_resize(img1, scale_percent)

img2 = cv2.imread(IMAGE_PATH2)
if max(img2.shape[0], img2.shape[1]) > 900:
	scale_percent = 900/max(img2.shape[0], img2.shape[1])
	img2 = img_resize(img2, scale_percent)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(image_size=160, device=device)
model = MobileNetV2(128, alpha=1.0)
model.load_weights(TRAINED_MODEL)

faces1 = detector.align(img=img1, select_largest=True)
faces2 = detector.align(img=img2, select_largest=True)

if faces1 is not None:
	for face1 in faces1:
		face1 = np.float32(face1)
		face_embedding1 = model(face1, True)
		print(face_embedding1)

if faces2 is not None:
	for face2 in faces2:
		face2 = np.float32(face2)
		face_embedding2 = model(face2, True)
		print(face_embedding2)


