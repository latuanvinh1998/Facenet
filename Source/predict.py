import numpy as np 
import os
import sys
import torch
import pickle
import cv2
import time
import argparse
import math
import tensorflow as tf
from Models.mobile_net_v2 import MobileNetV2
sys.path.insert(1, "MTCNN")
from mtcnn import MTCNN
from detect_face import extract_face

def img_resize(img, scale_percent):
	width = int(img.shape[1] * scale_percent)
	height = int(img.shape[0] * scale_percent)
	dim = (width, height)
	img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return img 

FONT = cv2.FONT_HERSHEY_SIMPLEX 

SVM_MODEL =  "../TrainedModels/SVM/facenet.pkl"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(image_size=160, device=device)
model = MobileNetV2(128, alpha=1.0)
with open(SVM_MODEL, 'rb') as infile:
	(svm_model, class_names) = pickle.load(infile)

def ImageMain(args):
	TRAINED_MODEL = '../TrainedModels/MobileNetV2/'+ args.TRAINED_MODEL +'/facenet.ckpt'
	model.load_weights(TRAINED_MODEL).expect_partial()
	
	image = cv2.imread(args.image_path)
	if max(image.shape[0], image.shape[1]) > 900:
		scale_percent = 900/max(image.shape[0], image.shape[1])
		image = img_resize(image, scale_percent)

	boxes = detector.detect(img=image, select_largest=args.select_largest, proba=False, landmarks=False)
	if boxes is not None:
		for box in boxes:
			cv2.rectangle(image, tuple((np.int32(box[0]), np.int32(box[1]))), tuple((np.int32(box[2]), np.int32(box[3]))), (255, 255, 255), 1)
			face = extract_face(image, box, image_size=160)
			face_embedding = model(np.float32(face), True)

			predictions = svm_model.predict_proba(face_embedding)
			best_class_idxs = np.argmax(predictions, axis=1)
			best_class_probabilities = predictions[np.arange(len(best_class_idxs)), best_class_idxs]
			for i in range(len(best_class_idxs)):
				print('%4d  %s: %.3f' % (i, class_names[best_class_idxs[i]], best_class_probabilities[i]))
			predict_str = '%s: %.3f'%(class_names[best_class_idxs[i]], best_class_probabilities[i])
			pos = tuple((np.int32(box[0]), np.int32(box[1])))
			cv2.putText(image, predict_str, pos, FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.imshow('preview',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def VideoMain(args):
	TRAINED_MODEL = '../TrainedModels/MobileNetV2/'+ args.TRAINED_MODEL +'/facenet.ckpt'
	model.load_weights(TRAINED_MODEL).expect_partial()

	if args.MODE == 'VIDEO':
		cap = cv2.VideoCapture(args.video_path)
	elif args.MODE == 'WEBCAM':
		cap = cv2.VideoCapture("http://192.168.1.6:4747/video")

	if not (cap.isOpened()):
		print("Could not open video device")

	while(True): 
		ret, frame = cap.read()
		start = time.time()
		image = cv2.flip(frame, 1)
		boxes, points = detector.detect(image, select_largest=args.select_largest, proba=False, landmarks=True)

		if (boxes is not None):
			for box in boxes:
				cv2.rectangle(image, tuple((np.int32(box[0]), np.int32(box[1]))), tuple((np.int32(box[2]), np.int32(box[3]))), (255, 255, 255), 1)
				face = extract_face(image, box, image_size=160)
				face_embedding = model(np.float32(face), True)
				predictions = svm_model.predict_proba(face_embedding)

				best_class_idxs = np.argmax(predictions, axis=1)
				best_class_probabilities = predictions[np.arange(len(best_class_idxs)), best_class_idxs]
				
				for i in range(len(best_class_idxs)):
					print('%4d  %s: %.3f' % (i, class_names[best_class_idxs[i]], best_class_probabilities[i]))
				predict_str = '%s: %.3f'%(class_names[best_class_idxs[i]], best_class_probabilities[i])
				pos = tuple((np.int32(box[0]), np.int32(box[1])))
				cv2.putText(image, predict_str, pos, FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
		
		fps_str = 'fps: %3.f'%(int(1/(time.time() - start)))
		cv2.putText(image, fps_str, (10, 50), FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.imshow('preview',image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('TRAINED_MODEL', type=str,
		help='Trained model xxxxxx_xxxx')

	parser.add_argument('--select_largest', type=bool,
		help='Select the largest face in image/video',
		default=True)

	subparsers = parser.add_subparsers(dest='MODE',
		help='IMAGE/VIDEO/WEBCAM')

	image_parser = subparsers.add_parser('IMAGE',
		help='Predict people in a image')

	image_parser.add_argument('--image_path', type=str,
		help='Image path',
		default='../Dataset/Test/3.jpg')


	video_parser = subparsers.add_parser('VIDEO',
		help='Predict people in a video')

	video_parser.add_argument('--video_path', type=str,
		help='Video path',
		default='../../1.mp4')


	webcam_parser = subparsers.add_parser('WEBCAM',
		help='Predict people thought a webcam')


	args = parser.parse_args()

	if args.MODE == 'IMAGE':
		ImageMain(args)
	if args.MODE == 'VIDEO' or args.MODE == 'WEBCAM':
		VideoMain(args)