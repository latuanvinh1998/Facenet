import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, DepthwiseConv2D, Reshape
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda, Layer
from tensorflow.keras.models import Model

class ConvBlock(Layer):
	def __init__(self, numfilt, strides, alpha=1.0, name=None):
		super(ConvBlock, self).__init__()
		self.zeropad = ZeroPadding2D(padding=((0, 1), (0, 1)), name=name+'conv2d'+'pad')
		self.convolution = Conv2D(filters=int(numfilt*alpha),
								kernel_size=3,
								strides=strides,
								padding='valid',
								name=name+'conv2d',
								data_format='channels_last',
								use_bias=True)
		self.bn = BatchNormalization(axis=3, scale=False, name=name+'conv2d'+'bn')
		self.relu = Activation('relu', name=name+'conv2d'+'relu')

	@tf.function
	def call(self, inputs, training=None):
		x = self.zeropad(inputs)
		x = self.convolution(x)
		x = self.bn(x, training=training)
		x = self.relu(x)
		return x

class DepthwiseConvBlock(Layer):
	def __init__(self, numfilt, alpha=1.0, depth_multiplier=1, strides=1, name=None):
		super(DepthwiseConvBlock, self).__init__()
		self.strides = strides
		self.zeropad = ZeroPadding2D(padding=((0, 1), (0, 1)), name=name+'convdw'+'pad')
		self.convdw = DepthwiseConv2D(kernel_size=3,
										strides=strides,
										padding='same' if strides == 1 else 'valid',
										depth_multiplier=depth_multiplier,
										data_format='channels_last',
										name=name+'convdw')
		self.bn1 = BatchNormalization(axis=3, scale=False, name=name+'convdw'+'bn1')
		self.relu1 = Activation('relu', name=name+'convdw'+'relu1')
		self.convolution = Conv2D(filters=int(numfilt*alpha),
								kernel_size=1,
								strides=1,
								padding='same',
								name=name+'convpw',
								data_format='channels_last',
								use_bias=True)
		self.bn2 = BatchNormalization(axis=3, scale=False, name=name+'convdw'+'bn2')
		self.relu2 = Activation('relu', name=name+'convdw'+'relu2')

	@tf.function
	def call(self, inputs, training=None):
		if self.strides == 1:
			x = inputs
		else:
			x = self.zeropad(inputs)
		x = self.convdw(x)
		x = self.bn1(x, training=training)
		x = self.relu1(x)
		x = self.convolution(x)
		x = self.bn2(x, training=training)
		x = self.relu2(x)
		return x

class MobileNetV1(Model):
	def __init__(self, embeddings_size, alpha=1.0, depth_multiplier=1):
		super(MobileNetV1, self).__init__()
		self.conv1 = ConvBlock(32, 2, alpha, 'Conv2D_1')
		self.convdw1 = DepthwiseConvBlock(64, alpha, depth_multiplier, name='ConvDw_1')
		self.convdw2 = DepthwiseConvBlock(128, alpha, depth_multiplier, strides=2, name='ConvDw_2')
		self.convdw3 = DepthwiseConvBlock(128, alpha, depth_multiplier, name='ConvDw_3')
		self.convdw4 = DepthwiseConvBlock(256, alpha, depth_multiplier, strides=2, name='ConvDw_4')
		self.convdw5 = DepthwiseConvBlock(256, alpha, depth_multiplier, name='ConvDw_5')
		self.convdw6 = DepthwiseConvBlock(512, alpha, depth_multiplier, strides=2, name='ConvDw_6')
		self.convdw7 = DepthwiseConvBlock(512, alpha, depth_multiplier, name='ConvDw_7')
		self.convdw8 = DepthwiseConvBlock(512, alpha, depth_multiplier, name='ConvDw_8')
		self.convdw9 = DepthwiseConvBlock(512, alpha, depth_multiplier, name='ConvDw_9')
		self.convdw10 = DepthwiseConvBlock(512, alpha, depth_multiplier, name='ConvDw_10')
		self.convdw11 = DepthwiseConvBlock(512, alpha, depth_multiplier, name='ConvDw_11')
		self.convdw12 = DepthwiseConvBlock(1024, alpha, depth_multiplier, strides=2, name='ConvDw_12')
		self.convdw13 = DepthwiseConvBlock(1024, alpha, depth_multiplier, strides=2, name='ConvDw_13')
		self.avgpool = GlobalAveragePooling2D(data_format='channels_last')
		self.dropout = Dropout(1e-3, name='Dropout')
		self.flat = Flatten(name='Flatten')
		self.fc = Dense(embeddings_size, activation='softmax', name='Fully_connected')


	@tf.function
	def call(self, inputs, training=None):
		img_inputs = tf.reshape(inputs, [-1, 160, 160, 3], name='input')
		x = self.conv1(img_inputs, training=training)
		x = self.convdw1(x, training=training)
		x = self.convdw2(x, training=training)
		x = self.convdw3(x, training=training)
		x = self.convdw4(x, training=training)
		x = self.convdw5(x, training=training)
		x = self.convdw6(x, training=training)
		x = self.convdw7(x, training=training)
		x = self.convdw8(x, training=training)
		x = self.convdw9(x, training=training)
		x = self.convdw10(x, training=training)
		x = self.convdw11(x, training=training)
		x = self.convdw12(x, training=training)
		x = self.convdw13(x, training=training)
		x = self.avgpool(x)
		x = self.dropout(x, training=training)
		x = self.flat(x)
		x = self.fc(x, training=training)
		x = tf.math.l2_normalize(x, 1, 1e-10, name='embeddings')
		return x