import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, DepthwiseConv2D, Reshape
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D, ReLU, Add
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda, Layer
from tensorflow.keras.models import Model

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBlock(Layer):
	def __init__(self, numfilt, filtsize, strides, pad, act=True, alpha=None, name=None):
		super(ConvBlock, self).__init__()
		self.act = act
		self.conv = Conv2D(filters=int(numfilt*alpha),
							kernel_size=filtsize,
							strides=strides,
							padding=pad,
							name=name+'conv2d',
							data_format='channels_last',
							use_bias=False)
		self.bn = BatchNormalization(axis=3, epsilon=1e-3, momentum=0.99, name='conv2d'+'bn')

	def call(self, inputs, training=None):
		with tf.name_scope('Convolution_Block'):
			x = self.conv(inputs)
			x = self.bn(x, training=training)
			if self.act:
				return tf.nn.relu6(x)
			else:
				return x

class ResBlock(Layer):
	def __init__(self, numfilt, strides, alpha, expansion, in_channel, block_id):
		super(ResBlock, self).__init__()
		prefix = 'block_{}_'.format(block_id)
		self.strides = strides
		self.in_channel = in_channel
		pw_filters = _make_divisible(int(numfilt*alpha), 8)
		self.pw_filters = pw_filters
		self.expand = ConvBlock(expansion*in_channel, 1, 1, 'same', True, alpha, prefix+'expand')
		self.dwconv = DepthwiseConv2D(kernel_size=3,
										strides=strides,
										activation=None,
										padding='same' if strides == 1 else 'valid',
										name=prefix+'depthwise',
										data_format='channels_last',
										use_bias=False)
		self.bn = BatchNormalization(axis=3, epsilon=1e-3, momentum=0.99, name=prefix+'depthwise_bn')
		self.relu = ReLU(6., name=prefix + 'depthwise_relu')
		self.pwconv = ConvBlock(pw_filters, 1, 1, 'same', False, alpha, prefix+'pointwise')
		self.add = Add(name=prefix + 'add')

	def call(self, inputs, training=None):
		with tf.name_scope('Residual_Block'):
			with tf.name_scope('Expand_Convolution'):
				x = self.expand(inputs, training=training)
			with tf.name_scope('Depthwise_Convolution'):
				x = self.dwconv(x)
				x = self.bn(x, training=training)
				x = self.relu(x)
			with tf.name_scope('Pointwise_Convolution'):
				x = self.pwconv(x, training=training)
			if self.pw_filters == self.in_channel and self.strides == 1:
				with tf.name_scope('Add'):
					x = self.add([x, inputs])
		return x

class MobileNetV2(Model):
	def __init__(self, embeddings_size, alpha=1.0):
		super(MobileNetV2, self).__init__()
		first_filters = _make_divisible(32 * alpha, 8)
		self.conv1 = ConvBlock(first_filters, 3, 2, 'valid', True, alpha, 'conv1')
		self.resblock1 = ResBlock(16, 1, alpha, 1, first_filters, block_id=1)
		self.resblock2 = ResBlock(24, 2, alpha, 6, _make_divisible(int(16*alpha),8), block_id=2)
		self.resblock3 = ResBlock(24, 1, alpha, 6, _make_divisible(int(24*alpha),8), block_id=3)
		self.resblock4 = ResBlock(32, 2, alpha, 6, _make_divisible(int(24*alpha),8), block_id=4)
		self.resblock5 = ResBlock(32, 1, alpha, 6, _make_divisible(int(32*alpha),8), block_id=5)
		self.resblock6 = ResBlock(32, 1, alpha, 6, _make_divisible(int(32*alpha),8), block_id=6)
		self.resblock7 = ResBlock(64, 2, alpha, 6, _make_divisible(int(32*alpha),8), block_id=7)
		self.resblock8 = ResBlock(64, 1, alpha, 6, _make_divisible(int(64*alpha),8), block_id=8)
		self.resblock9 = ResBlock(64, 1, alpha, 6, _make_divisible(int(64*alpha),8), block_id=9)
		self.resblock10 = ResBlock(64, 1, alpha, 6, _make_divisible(int(64*alpha),8), block_id=10)
		self.resblock11 = ResBlock(96, 1, alpha, 6, _make_divisible(int(64*alpha),8), block_id=11)
		self.resblock12 = ResBlock(96, 1, alpha, 6, _make_divisible(int(96*alpha),8), block_id=12)
		self.resblock13 = ResBlock(96, 1, alpha, 6, _make_divisible(int(96*alpha),8), block_id=13)
		self.resblock14 = ResBlock(160, 2, alpha, 6, _make_divisible(int(96*alpha),8), block_id=14)
		self.resblock15 = ResBlock(160, 1, alpha, 6, _make_divisible(int(160*alpha),8), block_id=15)
		self.resblock16 = ResBlock(160, 1, alpha, 6, _make_divisible(int(160*alpha),8), block_id=16)
		self.resblock17 = ResBlock(320, 1, alpha, 6, _make_divisible(int(160*alpha),8), block_id=17)
		if alpha > 1.0:
			last_filters = _make_divisible(1280 * alpha, 8)
		else:
			last_filters = 1280
		self.conv2 = ConvBlock(last_filters, 1, 1, 'same', True, alpha, 'conv2')
		self.avgpool = GlobalAveragePooling2D(data_format='channels_last', name='Global_Avg_Pooling')
		self.fc = Dense(embeddings_size, activation='softmax', use_bias=True, name='Fully_connected')

	@tf.function
	def call (self, inputs, training=None):
		img_inputs = tf.reshape(inputs, [-1, 160, 160, 3], name='input')
		with tf.name_scope('MobileNetV2'):
			x = self.conv1(img_inputs, training=training)
			x = self.resblock1(x, training=training)
			x = self.resblock2(x, training=training)
			x = self.resblock3(x, training=training)
			x = self.resblock4(x, training=training)
			x = self.resblock5(x, training=training)
			x = self.resblock6(x, training=training)
			x = self.resblock7(x, training=training)
			x = self.resblock8(x, training=training)
			x = self.resblock9(x, training=training)
			x = self.resblock10(x, training=training)
			x = self.resblock11(x, training=training)
			x = self.resblock12(x, training=training)
			x = self.resblock13(x, training=training)
			x = self.resblock14(x, training=training)
			x = self.resblock15(x, training=training)
			x = self.resblock16(x, training=training)
			x = self.resblock17(x, training=training)
			x = self.conv2(x, training=training)
			x = self.avgpool(x)
			x = self.fc(x, training=training)
			x = tf.math.l2_normalize(x, 1, 1e-10, name='embeddings')
		return x
