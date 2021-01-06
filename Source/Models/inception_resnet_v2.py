import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda, Layer
from tensorflow.keras import backend
from tensorflow.keras.models import Model

class conv2d(Layer):
	def __init__(self, numfilt, filtsz, strides=1, pad='same', act=True, name=None):
		super(conv2d, self).__init__()
		self.act = act
		self.convolution = Conv2D(filters=numfilt,
						kernel_size=filtsz,
						strides=strides,
						padding=pad,
						name=name+'conv2d',
						data_format='channels_last',
						use_bias=False)
		
		self.bn = BatchNormalization(axis=3, scale=False, name=name+'conv2d'+'bn')
		self.relu = Activation('relu', name=name+'conv2d'+'act')

	@tf.function
	def call(self, inputs, training=None):
		x = self.convolution(inputs)
		x = self.bn(x, training=training)
		if self.act: 
			x = self.relu(x)
		return x

class blockA(Layer):
	def __init__(self, scale, name=''):
		super(blockA, self).__init__()
		self.scale = scale
		pad = 'same'
		self.branch0 = conv2d(32, 1, 1, pad, True, name=name+'b0')
		self.branch1_1 = conv2d(32, 1, 1, pad, True, name=name+'b1_1')
		self.branch1_2 = conv2d(32, 3, 1, pad, True, name=name+'b1_2')
		self.branch2_1 = conv2d(32, 1, 1, pad, True, name=name+'b2_1')
		self.branch2_2 = conv2d(48, 3, 1, pad, True, name=name+'b2_2')
		self.branch2_3 = conv2d(64, 3, 1, pad, True, name=name+'b2_3')
		self.concat = Concatenate(axis=3, name=name+'mixed')
		self.filt_exp = conv2d(384, 1, 1, pad, False, name=name+'filt_exp_1x1')
	
	@tf.function
	def call(self, x, training=None):
		with tf.name_scope('Block_A'):
			with tf.name_scope('Branch_0'):
				branch0 = self.branch0(x, training=training)
			with tf.name_scope('Branch_1'):
				branch1 = self.branch1_1(x, training=training)
				branch1 = self.branch1_2(branch1, training=training)
			with tf.name_scope('Branch_2'):
				branch2 = self.branch2_1(x, training=training)
				branch2 = self.branch2_2(branch2, training=training)
				branch2 = self.branch2_3(branch2, training=training)
			branches = [branch0, branch1, branch2]
			mixed = self.concat(branches)
			filt_exp_1x1 = self.filt_exp(mixed, training=training)
			final_lay = tf.keras.layers.add([filt_exp_1x1 * self.scale, x])
		return tf.nn.relu(final_lay)


class blockB(Layer):
	def __init__(self, scale, name=''):
		super(blockB, self).__init__()
		self.scale = scale
		pad='same'
		self.branch0 = conv2d(192, 1, 1, pad, True, name=name+'b0')
		self.branch1_1 = conv2d(128, 1, 1, pad, True, name=name+'b1_1')
		self.branch1_2 = conv2d(160, [1, 7], 1, pad, True, name=name+'b1_2')
		self.branch1_3 = conv2d(192, [7, 1], 1, pad, True, name=name+'b1_3')
		self.concat = Concatenate(axis=3, name=name+'mixed')
		self.filt_exp = conv2d(1152, 1, 1, pad, False, name=name+'filt_exp_1x1')

	@tf.function
	def call(self, x, training=None):
		with tf.name_scope('Block_B'):
			with tf.name_scope('Branch_0'):
				branch0 = self.branch0(x, training=training)
			with tf.name_scope('Branch_1'):
				branch1 = self.branch1_1(x, training=training)
				branch1 = self.branch1_2(branch1, training=training)
				branch1 = self.branch1_3(branch1, training=training)
			branches = [branch0, branch1]
			mixed = self.concat(branches)
			filt_exp_1x1 = self.filt_exp(mixed, training=training)
			final_lay = tf.keras.layers.add([filt_exp_1x1 * self.scale, x])
		return tf.nn.relu(final_lay)


class blockC(Layer):
	def __init__(self, scale, name=''):
		super(blockC, self).__init__()
		self.scale = scale
		pad='same'
		self.branch0 = conv2d(192, 1, 1, pad, True, name=name+'b0')
		self.branch1_1 = conv2d(192, 1, 1, pad, True, name=name+'b1_1')
		self.branch1_2 = conv2d(224, [1, 3], 1, pad, True, name=name+'b1_2')
		self.branch1_3 = conv2d(256, [3, 1], 1, pad, True, name=name+'b1_3')
		self.concat = Concatenate(axis=3, name=name+'mixed')
		self.filt_exp = conv2d(2144, 1, 1, pad, False, name=name+'filt_exp_1x1')

	@tf.function
	def call(self, x, training=None):
		with tf.name_scope('Block_C'):
			with tf.name_scope('Branch_0'):
				branch0 = self.branch0(x, training=training)
			with tf.name_scope('Branch_1'):
				branch1 = self.branch1_1(x, training=training)
				branch1 = self.branch1_2(branch1, training=training)
				branch1 = self.branch1_3(branch1, training=training)
			branches = [branch0, branch1]
			mixed = self.concat(branches)
			filt_exp_1x1 = self.filt_exp(mixed, training=training)
			final_lay = tf.keras.layers.add([filt_exp_1x1 * self.scale, x])
		return tf.nn.relu(final_lay)

class stem(Layer):
	def __init__(self, name=''):
		super(stem, self).__init__()
		self.b_1 = conv2d(32, 3, 2, 'valid', True, name='conv1')
		self.b_2 = conv2d(32, 3, 1, 'valid', True, name='conv1')
		self.b_3 = conv2d(64, 3, 1, 'same', True, name='conv3')
		self.b1_1 = MaxPooling2D(3, strides=1, padding='valid', name='stem_br_11'+'_maxpool_1')
		self.b1_2 = conv2d(64, 3, 1, 'valid', True, name='br_12')
		self.concat_1 = Concatenate(axis=3, name='concat_1')
		self.b2_1_1 = conv2d(64, 1, 1, 'same', True, name='br_211')
		self.b2_1_2 = conv2d(64, [1,7], 1, 'same', True, name='br_212')
		self.b2_1_3 = conv2d(64, [7,1], 1, 'same', True, name='br_213')
		self.b2_1_4 = conv2d(96, 3, 1, 'valid', True, name='br_214')
		self.b2_2_1 = conv2d(64, 1, 1, 'same', True, name='br_221')
		self.b2_2_2 = conv2d(96, 3, 1, 'valid', True, name='br_222')
		self.concat_2 = Concatenate(axis=3, name='concat_2')
		self.b3_1 = conv2d(192, 3, 1, 'valid', True, name='br_31')
		self.b3_2 = MaxPooling2D(3,strides=1,padding='valid',name='br_32'+'_maxpool_2')
		self.concat_3 = Concatenate(axis=3, name='concat_3')

	@tf.function
	def call(self, x, training=None):
		with tf.name_scope('Stem_Block'):
			x = self.b_1(x, training=training)
			x = self.b_2(x, training=training)
			x = self.b_3(x, training=training)
			with tf.name_scope('Branch_1_1'):
				x_11 = self.b1_1(x)
			with tf.name_scope('Branch_1_2'):
				x_12 = self.b1_2(x, training=training)
			x = self.concat_1([x_11, x_12])
			with tf.name_scope('Branch_2_1'):
				x_21 = self.b2_1_1(x, training=training)
				x_21 = self.b2_1_2(x_21, training=training)
				x_21 = self.b2_1_3(x_21, training=training)
				x_21 = self.b2_1_4(x_21, training=training)
			with tf.name_scope('Branch_2_2'):
				x_22 = self.b2_2_1(x, training=training)
				x_22 = self.b2_2_2(x_22, training=training)
			x = self.concat_2([x_21, x_22])
			with tf.name_scope('Branch_3_1'):
				x_31 = self.b3_1(x, training=training)
			with tf.name_scope('Branch_3_2'):
				x_32 = self.b3_2(x)
			x = self.concat_3([x_31, x_32])
		return x

class reductionA(Layer):
	def __init__(self, name=''):
		super(reductionA, self).__init__()
		self.b_11 = MaxPooling2D(3, strides=2, padding='valid', name='red_maxpool_1')
		self.b_21 = conv2d(384, 3, 2, 'valid', True, name='x_red1_c1')
		self.b_31 = conv2d(256, 1, 1, 'same', True, name='x_red1_c2_1')
		self.b_32 = conv2d(256, 3, 1, 'same', True, name='x_red1_c2_2')
		self.b_33 = conv2d(384, 3, 2, 'valid', True, name='x_red1_c2_3')
		self.concat = Concatenate(axis=3, name='red_concat_1')

	@tf.function
	def call(self, x, training=None):
		with tf.name_scope('Reduction_A'):
			with tf.name_scope('Branch_1_1'):
				x_red_11 = self.b_11(x)
			with tf.name_scope('Branch_1_2'):
				x_red_12 = self.b_21(x, training=training)
			with tf.name_scope('Branch_1_3'):
				x_red_13 = self.b_31(x, training=training)
				x_red_13 = self.b_32(x_red_13, training=training)
				x_red_13 = self.b_33(x_red_13, training=training)
			branches = [x_red_11, x_red_12, x_red_13]
			x = self.concat(branches)
		return x

class reductionB(Layer):
	def __init__(self, name=''):
		super(reductionB, self).__init__()
		self.b_11 = MaxPooling2D(3, strides=2, padding='valid', name='red_maxpool_2')
		self.b_21 = conv2d(256, 1, 1, 'same', True, name='x_red2_c11')
		self.b_22 = conv2d(384, 3, 2, 'valid', True, name='x_red2_c12')
		self.b_31 = conv2d(256, 1, 1, 'same', True, name='x_red2_c21')
		self.b_32 = conv2d(288, 3, 2, 'valid', True, name='x_red2_c22')
		self.b_41 = conv2d(256, 1, 1, 'same', True, name='x_red2_c31')
		self.b_42 = conv2d(288, 3, 1, 'same', True, name='x_red2_c32')
		self.b_43 = conv2d(320, 3, 2, 'valid', True, name='x_red2_c33')
		self.concat = Concatenate(axis=3, name='red_concat_2')

	@tf.function
	def call(self, x, training=None):
		with tf.name_scope('Reduction_B'):
			with tf.name_scope('Branch_1_1'):
				x_red_21 = self.b_11(x)
			with tf.name_scope('Branch_1_1'):
				x_red_22 = self.b_21(x, training=training)
				x_red_22 = self.b_22(x_red_22, training=training)
			with tf.name_scope('Branch_1_3'):
				x_red_23 = self.b_31(x, training=training)
				x_red_23 = self.b_32(x_red_23, training=training)
			with tf.name_scope('Branch_1_4'):
				x_red_24 = self.b_41(x, training=training)
				x_red_24 = self.b_42(x_red_24, training=training)
				x_red_24 = self.b_43(x_red_24, training=training)
			branches = [x_red_21, x_red_22, x_red_23, x_red_24]
			x = self.concat(branches)
		return x


class Inception_Resnet_V2(Model):
	def __init__(self, embeddings_size):
		super(Inception_Resnet_V2, self).__init__()
		self.stemBlock = stem(name='Stem')
		self.blockA_1 = blockA(0.15, name='blockA_1')
		self.blockA_2 = blockA(0.15, name='blockA_2')
		self.blockA_3 = blockA(0.15, name='blockA_3')
		self.blockA_4 = blockA(0.15, name='blockA_4')
		self.blockA_5 = blockA(0.15, name='blockA_5')
		self.reduction1 = reductionA(name='reduction_A')
		self.blockB_1 = blockB(0.1, name='blockB_1')
		self.blockB_2 = blockB(0.1, name='blockB_2')
		self.blockB_3 = blockB(0.1, name='blockB_3')
		self.blockB_4 = blockB(0.1, name='blockB_4')
		self.blockB_5 = blockB(0.1, name='blockB_5')
		self.blockB_6 = blockB(0.1, name='blockB_6')
		self.blockB_7 = blockB(0.1, name='blockB_7')
		self.blockB_8 = blockB(0.1, name='blockB_8')
		self.blockB_9 = blockB(0.1, name='blockB_9')
		self.blockB_10 = blockB(0.1, name='blockB_10')
		self.reduction2 = reductionB(name='reduction_B')
		self.blockC_1 =  blockC(0.2, name='blockC_1')
		self.blockC_2 =  blockC(0.2, name='blockC_2')
		self.blockC_3 =  blockC(0.2, name='blockC_3')
		self.blockC_4 =  blockC(0.2, name='blockC_4')
		self.blockC_5 =  blockC(0.2, name='blockC_5')
		self.avgpool = GlobalAveragePooling2D(data_format='channels_last')
		self.dropout = Dropout(0.8, name='Dropout')
		self.flat = Flatten(name='Flatten')
		self.fc = Dense(embeddings_size, activation='softmax', name='Fully_connected')

	@tf.function
	def call(self, inputs, training=None):
		with tf.name_scope('Inception_Resnet_V2'):
			img_inputs = tf.reshape(inputs, [-1, 160, 160, 3], name='input')
			x = self.stemBlock(img_inputs, training=training)
			x = self.blockA_1(x, training=training)
			x = self.blockA_2(x, training=training)
			x = self.blockA_3(x, training=training)
			x = self.blockA_4(x, training=training)
			x = self.blockA_5(x, training=training)
			x = self.reduction1(x, training=training)
			x = self.blockB_1(x, training=training)
			x = self.blockB_2(x, training=training)
			x = self.blockB_3(x, training=training)
			x = self.blockB_4(x, training=training)
			x = self.blockB_5(x, training=training)
			x = self.blockB_6(x, training=training)
			x = self.blockB_7(x, training=training)
			x = self.blockB_8(x, training=training)
			x = self.blockB_9(x, training=training)
			x = self.blockB_10(x, training=training)
			x = self.reduction2(x, training=training)
			x = self.blockC_1(x, training=training)
			x = self.blockC_2(x, training=training)
			x = self.blockC_3(x, training=training)
			x = self.blockC_4(x, training=training)
			x = self.blockC_5(x, training=training)
			x = self.avgpool(x)
			x = self.dropout(x, training=training)
			x = self.flat(x)
			x = self.fc(x, training=training)
			x = tf.math.l2_normalize(x, 1, 1e-10, name='embeddings')
		return x

