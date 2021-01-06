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



cap = cv2.VideoCapture("../../1.mp4")
if not (cap.isOpened()):
	print("Could not open video device")

while(True): 
	ret, frame = cap.read()
	if frame is None:
		break
	start = time.time()
	img = cv2.flip(frame, 1)
	boxes, points = detector.detect(img, select_largest=True, landmarks=True)
	if boxes is not None:
		for box, point in zip(boxes, points):
			cv2.rectangle(img, tuple((np.int32(box[0]), np.int32(box[1]))), tuple((np.int32(box[2]), np.int32(box[3]))), (255, 255, 255), 1)		
	print("FPS: ", np.int32(1/(time.time() - start)))
	cv2.imshow('preview',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()




