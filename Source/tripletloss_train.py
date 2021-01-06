import os.path
import shutil
import time
from datetime import date
from datetime import datetime
import argparse
import itertools
import tensorflow as tf
import numpy as np
from Facenet import evaluate
from Facenet.load_dataset import load_dataset, load_people_per_batch
from Models.mobile_net_v2 import MobileNetV2


def delete_dir_if_exist(path):
	if not isinstance(path, str):
		path = os.path.join(*path)
	if os.path.exists(path):
		shutil.rmtree(path)

def display_time(start, end):
	hours = 0
	minutes = 0
	seconds = end - start
	while seconds > 60:
		seconds-=60
		minutes+=1
	while minutes > 60:
		minutes -= 60
		hours += 1
	if hours > 0:
		print("Total Training Time: %dh%dm%ds" % (hours, minutes, seconds))
		return
	if minutes > 0:
		print("Total Training Time: %dm%ds" % (minutes, seconds))
		return 
	print("Total Training Time: %ds" % (seconds))
	return

@tf.function
def triplet_loss(anchor, positive, negative, alpha):
	with tf.name_scope('triplet_loss'):
		pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
		neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

		basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
		loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
	return loss


def select_triplets(image_paths, emb_array, people_per_batch, images_per_person, alpha):
	np.warnings.filterwarnings('ignore')
	start_idx_of_class = 0
	triplets = []
	num_trips = 0

	for i in range(people_per_batch):
		nrof_images = int(images_per_person[i])
		for j in range(1,nrof_images):
			a_idx = start_idx_of_class + j - 1
			all_dists_sqr = np.sum(np.square(emb_array[a_idx] - emb_array), 1)
			all_dists_sqr[start_idx_of_class:start_idx_of_class+nrof_images] = np.NaN
			for pair in range (j, nrof_images):
				p_idx = start_idx_of_class + pair
				pos_dist_sqr = np.sum(np.square(emb_array[a_idx] - emb_array[p_idx]))
				all_neg = np.where(all_dists_sqr - pos_dist_sqr < alpha)[0]
				nrof_random_negs = all_neg.shape[0]
				if nrof_random_negs > 0:
					rnd_idx = np.random.randint(nrof_random_negs)
					n_idx = all_neg[rnd_idx]
					triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
				num_trips += 1
		start_idx_of_class += nrof_images
	np.random.shuffle(triplets)
	return triplets, num_trips, len(triplets)


def main(args):

	NROF_EPOCHS = 1000

	BATCH_SIZE = 4 

	LEARNING_RATE = 0.001

	PEOPLE_PER_BATCH = 40

	IMAGE_PER_PERSON = 45

	EMBEDDING_SIZE = 128

	ALPHA = 0.8

	LOG_DIR = '../TensorBoardLogs/facenet'

	MODEL_SAVE_DIR = '../TrainedModels/MobileNetV2/'

	OPTIMIZER = 'ADAM'

	LFW_PAIR = 'Facenet/pairs.txt'

	LFW_DIR = '../../Datasets/lfw'

	LFW_NROF_FOLD = 10

	PRETRAINED_MODEL_DIR = '../TrainedModels/MobileNetV2/'

	#-------------------------TENSORBOARD Summary Writer.-----------------------------
	delete_dir_if_exist(LOG_DIR)
	os.makedirs(os.path.dirname(LOG_DIR) + "/", exist_ok=True)
	summary_writer = tf.summary.create_file_writer(LOG_DIR)

	#------------------------------LOAD DATASET------------------------------------------
	dataset = load_dataset(args.dataset_dir)
	if dataset is None:
		raise ValueError('Unable to load dataset')
	nrof_classes = len(dataset)
	print("Load Data: Done!!")
	print("Number of Classes:", nrof_classes)
	for i in range(len(dataset)):
		print('Class', dataset[i].name, end='')
		print(',', len(dataset[i].paths))


	#-------------------------------LOAD MODEL----------------------------------------
	model = MobileNetV2(EMBEDDING_SIZE, alpha=1.0)
	if args.pretrain and os.path.exists(PRETRAINED_MODEL_DIR):
		model.load_weights(PRETRAINED_MODEL_DIR)

	learning_rate = LEARNING_RATE
	with tf.name_scope('Optimizer'):
		if OPTIMIZER =='ADAGRAD':
			optimizer = tf.optimizers.Adagrad(learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
		elif OPTIMIZER =='ADADELTA':
			optimizer = tf.optimizers.Adadelta(learning_rate, rho=0.95, epsilon=1e-07)
		elif OPTIMIZER =='ADAM':
			optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
		elif OPTIMIZER =='RMSPROP':
			optimizer = tf.optimizers.RMSprop(learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)
		else:
			raise ValueError('Invalid optimization algorithm')


	epoch = 1
	global_step = 1
	global_start = time.time()
	while epoch < NROF_EPOCHS:

		#----------------------------------------FEED-FORWARD-----------------------------------------------------------
		people_per_batch = min(PEOPLE_PER_BATCH, nrof_classes)
		image_paths, num_per_class = load_people_per_batch(dataset, people_per_batch, IMAGE_PER_PERSON)
		nrof_train_example = np.sum(num_per_class)
		labels_array = np.arange(nrof_train_example)
		
		ff_data = tf.data.Dataset.from_tensor_slices((image_paths, labels_array))
		ff_data = ff_data.repeat().shuffle(5000).batch(BATCH_SIZE).prefetch(1)

		nrof_batches = int(np.ceil(nrof_train_example/BATCH_SIZE))
		emb_array = np.zeros((nrof_train_example, EMBEDDING_SIZE))

		for step, (x_batch, y_batch) in enumerate(ff_data.take(nrof_batches), 1):
			if (global_step == 1) and (step == 1):
				tf.summary.trace_on(graph=True, profiler=True)
			images = []
			for filename in tf.unstack(x_batch):
				file_contents = tf.io.read_file(filename)
				image = tf.image.decode_image(file_contents, channels=3, dtype=tf.dtypes.float32)
				images.append(image)
			embedding = model(images, True)
			embedding = embedding.numpy()
			emb_array[y_batch,:] = embedding
			if (global_step == 1) and (step == 1):
				with summary_writer.as_default():
					tf.summary.trace_export(name="trace",step=0,profiler_outdir=LOG_DIR)


		#---------------------FIND ALL TRIPLET-PAIRs FROM FEED-FORWARD ARRAY---------------------------------------------
		triplets, nrof_random_negs, nrof_triplets = select_triplets(image_paths, emb_array, people_per_batch, num_per_class, ALPHA)
		print("Nrof negative: %d, Nrof triplets: %d" %(nrof_random_negs, nrof_triplets))
		triplet_paths = list(itertools.chain(*triplets))
		train_paths = np.array(triplet_paths).reshape(-1, 3)
		train_data = tf.data.Dataset.from_tensor_slices((train_paths))
		train_data = train_data.repeat().shuffle(5000).batch(BATCH_SIZE).prefetch(1)

		#----------------------------------TRAIN MODEL-----------------------------------------------------------------
		nrof_batches = int(np.ceil(nrof_triplets/BATCH_SIZE))
		loss_array = []
		print("Batch size: %d, Number of batches: %d" %(BATCH_SIZE, nrof_batches))
		start = time.time()
		for x_batch in train_data.take(nrof_batches):
			images = []
			for filenames in tf.unstack(x_batch):
				for filename in tf.unstack(filenames):
					file_contents = tf.io.read_file(filename)
					image = tf.image.decode_image(file_contents, channels=3, dtype=tf.dtypes.float32)
					images.append(image)
			with tf.GradientTape() as g:
				embedding = model(images, True)
				anchor, positive, negative = tf.unstack(tf.reshape(embedding, (-1,3,EMBEDDING_SIZE)), 3, 1)
				loss = triplet_loss(anchor, positive, negative, ALPHA)
				loss_array.append(loss)
				total_loss = tf.reduce_mean(loss_array)

			trainable_variables = model.trainable_variables
			gradients = g.gradient(total_loss, trainable_variables)
			optimizer.apply_gradients(zip(gradients, trainable_variables))
			
			#---------------------------SUMARRY LOSS-----------------------------------------------------------------------
			if global_step % 100 == 0:
				print("Epoch: %d/%d, Global_Step: %d, Time %.3fs, Loss: %f" % (epoch, NROF_EPOCHS, global_step, time.time()-start, total_loss))
				display_time(global_start, time.time())
				start = time.time()
				with summary_writer.as_default():
					tf.summary.scalar('loss', total_loss, step=global_step)
					for w in trainable_variables:
						tf.summary.histogram(w.name, w.value(), step=global_step)

			#------------------------SAVE & EVALUATE-------------------------------------------------------------------
			if global_step % 1000 == 0:
				print("Saving model ...")
				day = date.today()
				now = datetime.now()
				model_dir = MODEL_SAVE_DIR + day.strftime("%y%m%d_")+now.strftime("%H%M/")
				model.save_weights(model_dir + "facenet.ckpt")
				print("Model saved at", model_dir)
				if args.evaluate:
					if os.path.exists(LFW_DIR):
						print("LFW directory: %s" % (LFW_DIR))
						_, _, acc = evaluate.model_evaluate(model, LFW_PAIR, LFW_DIR, batch_size=BATCH_SIZE, nrof_fold=LFW_NROF_FOLD)
						print('Accuracy: %1.3f+-%1.3f' % (np.mean(acc), np.std(acc)))
						with summary_writer.as_default():
							tf.summary.scalar('accuracy', np.mean(acc), step=global_step)
						time.sleep(5)
					else:
						print("Could not found LFW directory to evaluate ...")
				else:
					print('Disabled evaluate the model!!')
					time.sleep(3)
			global_step += 1
		epoch += 1


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset_dir', type=str, 
		help='Directory contain processed images',
		default='../Dataset/Processed')

	parser.add_argument('--pretrain', type=bool,
		help='Use pretrained model',
		default=False)

	parser.add_argument('--evaluate', type=bool,
		help='Evaluate model every 10000 step',
		default=True)

	args = parser.parse_args()
	main(args)