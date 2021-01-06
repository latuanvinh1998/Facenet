import os
import time 
import argparse
import pickle
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from Facenet import load_dataset
from Models.mobile_net_v2 import MobileNetV2


def main(args):
	BATCH_SIZE = 4

	EMBEDDING_SIZE = 128

	MODEL_DIR = '../TrainedModels/MobileNetV2/' + args.pretrained_model + '/'

	SVM_MODEL_DIR = '../TrainedModels/SVM/'

	if not os.path.exists(MODEL_DIR):
		raise ValueError("Could not found pretrained model directory!")

	os.makedirs(SVM_MODEL_DIR, exist_ok=True)

	#LOAD DATASET
	dataset = load_dataset.load_dataset(args.dataset_dir)
	if dataset is None:
		raise ValueError("Could not load dataset!")

	nrof_classes = len(dataset)
	print("Load Data: Done!!")
	print("Number of Classes:", nrof_classes)
	for i in range(len(dataset)):
		print('Class', dataset[i].name, end='')
		print(',', len(dataset[i].paths))

	#LOAD MODEL
	model = MobileNetV2(EMBEDDING_SIZE, alpha=1.0)
	MODEL_DIR = os.path.expanduser(MODEL_DIR)
	model.load_weights(MODEL_DIR)

	#FEED FORWARD TO GET EMBEDDING ARRAY
	start = time.time()
	print("Running Feed Forward: ", end='')
	image_paths_array = []
	labels_array = []
	for classes in dataset:
		for path in classes.paths:
			image_paths_array.append(path)
			labels_array.append(classes.name)

	posistion = np.arange(len(image_paths_array))
	ff_data = tf.data.Dataset.from_tensor_slices((image_paths_array, posistion))
	ff_data = ff_data.repeat().batch(BATCH_SIZE).prefetch(1)

	nrof_batches = int(np.ceil(len(image_paths_array)/BATCH_SIZE))
	emb_array = np.zeros((len(image_paths_array), EMBEDDING_SIZE))
	
	for (x_batch, y_batch) in ff_data.take(nrof_batches):
		images = []
		for filename in tf.unstack(x_batch):
			file_contents = tf.io.read_file(filename)
			image = tf.image.decode_image(file_contents, channels=3, dtype=tf.dtypes.float32)
			images.append(image)
		embedding = model(images, False)
		embedding = embedding.numpy()
		emb_array[y_batch,:] = embedding
	print('%3.fs'%(time.time() - start))

	svm_model = SVC(kernel='linear', probability=True)
	svm_model.fit(emb_array, labels_array)

	classifier_filename_exp = os.path.expanduser(SVM_MODEL_DIR+ 'facenet.pkl')
	with open(classifier_filename_exp, 'wb') as outfile:
		pickle.dump((svm_model, labels_array), outfile)
	print('Saved classifier model to file "%s"' % classifier_filename_exp)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset_dir', type=str, 
		help='Directory contain processed images',
		default='../Dataset/Processed')

	parser.add_argument('pretrained_model', type=str,
		help='Pretrained model directory',
		default='xxxxxx_xxxx/')

	args = parser.parse_args()
	main(args)