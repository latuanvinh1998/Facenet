from keras.models import Model
from keras.layers import Input, Activation, merge
from keras.layers import Flatten, Dropout, Concatenate, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import AveragePooling2D, GlobalAveragePooling2D
import tensorflow as tf 


class SqueezeNet(Model):
	def __init__(self, embedding_size):
		super(SqueezeNet, self).__init__()
		self.conv1 = Conv2D(96,7,2,activation='relu',data_format='channels_last',padding='same',name='conv1')
		self.maxpool1 = MaxPooling2D(3,strides=2,name='maxpool1')
		self.fire2_squeeze = Conv2D(16,1,1,activation='relu',data_format='channels_last',padding='same',name='fire2_squeeze')
		self.fire2_expand1 = Conv2D(64,1,1,activation='relu',data_format='channels_last',padding='same',name='fire2_expand1')
		self.fire2_expand2 = Conv2D(64,3,1,activation='relu',data_format='channels_last',padding='same',name='fire2_expand2')
		self.concat2 = Concatenate(axis=3, name='concat_2')
		self.fire3_squeeze = Conv2D(16,1,1,activation='relu',data_format='channels_last',padding='same',name='fire3_squeeze')
		self.fire3_expand1 = Conv2D(64,1,1,activation='relu',data_format='channels_last',padding='same',name='fire3_expand1')
		self.fire3_expand2 = Conv2D(64,3,1,activation='relu',data_format='channels_last',padding='same',name='fire3_expand2')
		self.concat3 = Concatenate(axis=3, name='concat_3')
		self.fire4_squeeze = Conv2D(32 ,1,1,activation='relu',data_format='channels_last',padding='same',name='fire4_squeeze')
		self.fire4_expand1 = Conv2D(128,1,1,activation='relu',data_format='channels_last',padding='same',name='fire4_expand1')
		self.fire4_expand2 = Conv2D(128,3,1,activation='relu',data_format='channels_last',padding='same',name='fire4_expand2')
		self.concat4 = Concatenate(axis=3, name='concat_4')
		self.maxpool4 = MaxPooling2D(3,strides=2,name='maxpool4')
		self.fire5_squeeze = Conv2D(32 ,1,1,activation='relu',data_format='channels_last',padding='same',name='fire5_squeeze')
		self.fire5_expand1 = Conv2D(128,1,1,activation='relu',data_format='channels_last',padding='same',name='fire5_expand1')
		self.fire5_expand2 = Conv2D(128,3,1,activation='relu',data_format='channels_last',padding='same',name='fire5_expand2')
		self.concat5 = Concatenate(axis=3, name='concat_5')
		self.fire6_squeeze = Conv2D(48 ,1,1,activation='relu',data_format='channels_last',padding='same',name='fire6_squeeze')
		self.fire6_expand1 = Conv2D(192,1,1,activation='relu',data_format='channels_last',padding='same',name='fire6_expand1')
		self.fire6_expand2 = Conv2D(192,3,1,activation='relu',data_format='channels_last',padding='same',name='fire6_expand2')
		self.concat6 = Concatenate(axis=3, name='concat_6')
		self.fire7_squeeze = Conv2D(48 ,1,1,activation='relu',data_format='channels_last',padding='same',name='fire7_squeeze')
		self.fire7_expand1 = Conv2D(192,1,1,activation='relu',data_format='channels_last',padding='same',name='fire7_expand1')
		self.fire7_expand2 = Conv2D(192,3,1,activation='relu',data_format='channels_last',padding='same',name='fire7_expand2')
		self.concat7 = Concatenate(axis=3, name='concat_7')
		self.fire8_squeeze = Conv2D(64 ,1,1,activation='relu',data_format='channels_last',padding='same',name='fire8_squeeze')
		self.fire8_expand1 = Conv2D(256,1,1,activation='relu',data_format='channels_last',padding='same',name='fire8_expand1')
		self.fire8_expand2 = Conv2D(256,3,1,activation='relu',data_format='channels_last',padding='same',name='fire8_expand2')
		self.concat8 = Concatenate(axis=3, name='concat_8')
		self.maxpool8 = MaxPooling2D(3,strides=2,name='maxpool8')
		self.fire9_squeeze = Conv2D(64 ,1,1,activation='relu',data_format='channels_last',padding='same',name='fire9_squeeze')
		self.fire9_expand1 = Conv2D(256,1,1,activation='relu',data_format='channels_last',padding='same',name='fire9_expand1')
		self.fire9_expand2 = Conv2D(256,3,1,activation='relu',data_format='channels_last',padding='same',name='fire9_expand2')
		self.concat9 = Concatenate(axis=3, name='concat_9')
		self.dropout = Dropout(0.5, name='dropout')
		self.conv10 = Conv2D(1000,1,1,padding='valid',name='conv10')
		self.avgpool = AveragePooling2D(pool_size=(9,9),data_format='channels_last')
		self.fc = Dense(embedding_size,activation='softmax')

	@tf.function
	def call(self, inputs, training=None):
		with tf.name_scope('Squeeze_Net'):
			img_inputs = tf.reshape(inputs, [-1, 160, 160, 3], name='input')
			with tf.name_scope('squeeze1'):
				x = self.conv1(img_inputs, training=training)
				x = self.maxpool1(x)
			with tf.name_scope('squeeze2'):
				x = self.fire2_squeeze(x, training=training)
				x21 = self.fire2_expand1(x, training=training)
				x22 = self.fire2_expand2(x, training=training)
				x = self.concat2([x21, x22])
			with tf.name_scope('squeeze3'):
				x = self.fire3_squeeze(x, training=training)
				x31 = self.fire3_expand1(x, training=training)
				x32 = self.fire3_expand2(x, training=training)
				x = self.concat3([x31, x32])
			with tf.name_scope('squeeze4'):
				x = self.fire4_squeeze(x, training=training)
				x41 = self.fire4_expand1(x, training=training)
				x42 = self.fire4_expand2(x, training=training)
				x = self.concat4([x41, x42])
				x = self.maxpool4(x)
			with tf.name_scope('squeeze5'):
				x = self.fire5_squeeze(x, training=training)
				x51 = self.fire5_expand1(x, training=training)
				x52 = self.fire5_expand2(x, training=training)
				x = self.concat5([x51, x52])
			with tf.name_scope('squeeze6'):
				x = self.fire6_squeeze(x, training=training)
				x61 = self.fire6_expand1(x, training=training)
				x62 = self.fire6_expand2(x, training=training)
				x = self.concat6([x61, x62])
			with tf.name_scope('squeeze7'):
				x = self.fire7_squeeze(x, training=training)
				x71 = self.fire7_expand1(x, training=training)
				x72 = self.fire7_expand2(x, training=training)
				x = self.concat7([x71, x72])
			with tf.name_scope('squeeze8'):
				x = self.fire8_squeeze(x, training=training)
				x81 = self.fire8_expand1(x, training=training)
				x82 = self.fire8_expand2(x, training=training)
				x = self.concat6([x81, x82])
				x = self.maxpool8(x)
			with tf.name_scope('squeeze9'):
				x = self.fire9_squeeze(x, training=training)
				x91 = self.fire9_expand1(x, training=training)
				x92 = self.fire9_expand2(x, training=training)
				x = self.concat9([x91, x92])
			with tf.name_scope('dropout'):
				x = self.dropout(x, training=training)
			with tf.name_scope('conv10'):
				x = self.conv10(x, training=training)
			with tf.name_scope('average_pool_10'):
				x = self.avgpool(x)
			with tf.name_scope('squeeze10'):
				x = tf.squeeze(x, [1, 2], name='logits')
			with tf.name_scope('bottleneck'):
				x = self.fc(x, training=training)
			x = tf.math.l2_normalize(x, 1, 1e-10, name='embeddings')
		return x