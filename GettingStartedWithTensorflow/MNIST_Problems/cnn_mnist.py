
def cnn_model_fn(features, labels, mode):
	'''Model function for CNN'''
	# Input Layer
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

	# Convolution Layer
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5,5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2) 

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5,5],
		padding="same",
		activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)