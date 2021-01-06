import numpy as np
import cv2
import torch
import time
import os
import sys
sys.path.insert(1, "../Source/MTCNN")
from mtcnn import MTCNN


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(device=device)

img_path = "../Dataset/Test/6.jpg"

img = cv2.imread(img_path)
boxes, points = detector.detect(img, select_largest=False, proba=False, landmarks=True)
if boxes is not None:
	for box, point in zip(boxes, points):
		cv2.rectangle(img, tuple((np.int32(box[0]), np.int32(box[1]))), tuple((np.int32(box[2]), np.int32(box[3]))), (255, 255, 255), 1)
		for pt in point:
			cv2.circle(img, tuple(np.int32(pt)), 1, (255, 255, 255), 6)

cv2.imshow('preview',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
