import tensorflow as tf
import numpy as np

def load_weights(weights):
	vgg_keras = tf.keras.applications.VGG19(
		include_top=False,
		weights=weights
	)
	weights = {}
	for val in vgg_keras.weights:
		weights[val.name] = val
	return weights


class VGG:
	def __init__(self, weights="imagenet"):
		self.weights = load_weights(weights)

	@tf.function
	def layers(self, x, layers):
		weights = self.weights
		out = []

		# LAYER 0
		x = tf.keras.applications.vgg19.preprocess_input(x*255.0)
		if 0 in layers:	out.append(x)

		# LAYER 1
		x = tf.nn.conv2d(x, weights["block1_conv1/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block1_conv1/bias:0"])
		x = tf.math.maximum(x, 0)
		if 1 in layers:	out.append(x)

		# LAYER 2
		x = tf.nn.conv2d(x, weights["block1_conv2/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block1_conv2/bias:0"])
		x = tf.math.maximum(x, 0)
		if 2 in layers:	out.append(x)

		# LAYER 3
		x = tf.nn.avg_pool(x, [2,2], [2,2], "VALID")
		if 3 in layers:	out.append(x)

		# LAYER 4
		x = tf.nn.conv2d(x, weights["block2_conv1/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block2_conv1/bias:0"])
		x = tf.math.maximum(x, 0)
		if 4 in layers:	out.append(x)

		# LAYER 5
		x = tf.nn.conv2d(x, weights["block2_conv2/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block2_conv2/bias:0"])
		x = tf.math.maximum(x, 0)
		if 5 in layers:	out.append(x)

		# LAYER 6
		x = tf.nn.avg_pool(x, [2,2], [2,2], "VALID")
		if 6 in layers:	out.append(x)

		# LAYER 7
		x = tf.nn.conv2d(x, weights["block3_conv1/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block3_conv1/bias:0"])
		x = tf.math.maximum(x, 0)
		if 7 in layers:	out.append(x)

		# LAYER 8
		x = tf.nn.conv2d(x, weights["block3_conv2/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block3_conv2/bias:0"])
		x = tf.math.maximum(x, 0)
		if 8 in layers:	out.append(x)

		# LAYER 9
		x = tf.nn.conv2d(x, weights["block3_conv3/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block3_conv3/bias:0"])
		x = tf.math.maximum(x, 0)
		if 9 in layers:	out.append(x)

		# LAYER 10
		x = tf.nn.conv2d(x, weights["block3_conv4/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block3_conv4/bias:0"])
		x = tf.math.maximum(x, 0)
		if 10 in layers:	out.append(x)

		# LAYER 11
		x = tf.nn.avg_pool(x, [2,2], [2,2], "VALID")
		if 11 in layers:	out.append(x)

		# LAYER 12
		x = tf.nn.conv2d(x, weights["block4_conv1/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block4_conv1/bias:0"])
		x = tf.math.maximum(x, 0)
		if 12 in layers:	out.append(x)

		# LAYER 13
		x = tf.nn.conv2d(x, weights["block4_conv2/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block4_conv2/bias:0"])
		x = tf.math.maximum(x, 0)
		if 13 in layers:	out.append(x)

		# LAYER 14
		x = tf.nn.conv2d(x, weights["block4_conv3/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block4_conv3/bias:0"])
		x = tf.math.maximum(x, 0)
		if 14 in layers:	out.append(x)

		# LAYER 15
		x = tf.nn.conv2d(x, weights["block4_conv4/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block4_conv4/bias:0"])
		x = tf.math.maximum(x, 0)
		if 15 in layers:	out.append(x)

		# LAYER 16
		x = tf.nn.avg_pool(x, [2,2], [2,2], "VALID")
		if 16 in layers:	out.append(x)

		# LAYER 17
		x = tf.nn.conv2d(x, weights["block5_conv1/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block5_conv1/bias:0"])
		x = tf.math.maximum(x, 0)
		if 17 in layers:	out.append(x)

		# LAYER 18
		x = tf.nn.conv2d(x, weights["block5_conv2/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block5_conv2/bias:0"])
		x = tf.math.maximum(x, 0)
		if 18 in layers:	out.append(x)

		# LAYER 19
		x = tf.nn.conv2d(x, weights["block5_conv3/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block5_conv3/bias:0"])
		x = tf.math.maximum(x, 0)
		if 19 in layers:	out.append(x)

		# LAYER 20
		x = tf.nn.conv2d(x, weights["block5_conv4/kernel:0"], [1,1], "SAME")
		x = tf.nn.bias_add(x, weights["block5_conv4/bias:0"])
		x = tf.math.maximum(x, 0)
		if 20 in layers:	out.append(x)

		# LAYER 21
		x = tf.nn.avg_pool(x, [2,2], [2,2], "VALID")
		if 21 in layers:	out.append(x)

		return out

	@tf.function
	def pack_gram(self, gram):
		size = tf.shape(gram)[-1]
		lin = tf.range(0, size)
		x, y = tf.meshgrid(lin,lin)
		mask = x + y < size
		gram = tf.boolean_mask(gram, mask, 1)
		return gram

	@tf.function
	def gram(self, x):
		shape = tf.cast(tf.shape(x), tf.float32)
		x = tf.reshape(x, [shape[0], -1, shape[3]])
		x = tf.matmul(tf.transpose(x, [0, 2, 1]), x)
		x = x / (shape[1]*shape[2])**2
		# x = x / shape[1]*shape[2]*shape[3]
		return tf.reshape(x, (shape[0], -1))
		# x = self.pack_gram(x)
		# return x

	@tf.function
	def style(self, x):
		x = tf.repeat(x[...,:1], 3, -1) * 255
		x = tf.keras.applications.vgg19.preprocess_input(x)
		ls = self.vgg(x)
		grams = [
			self.gram(ls[0]),
			self.gram(ls[1]),
			self.gram(ls[2])
		]
		return tf.concat(grams, -1)

	def octaves(self, x, octaves=4, step=0.5, weight=1.0):
		styles = []
		size0 = tf.shape(x)[1:-1]
		size0 = tf.cast(size0, tf.float32)
		for i in range(octaves):
			size = size0 * tf.pow(step, tf.cast(i, tf.float32))
			size = tf.cast(size, tf.int32)
			xs = tf.image.resize(x, size, antialias=True)
			s = self.style(xs) * tf.pow(weight, tf.cast(i, tf.float32))
			styles.append(s)
		return tf.concat(styles, -1)