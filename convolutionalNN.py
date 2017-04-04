import tensorflow as tf
import numpy as np
import random


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def CNN(features_train, labels_train, features_val, labels_val, features_test):

	# DEFINING THE PARAMETERS OF THE NETWORK
	print("CREATING NETWORK")	

	# First convolutional layer + relu
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	
	x = tf.placeholder(tf.float32, shape=[None, 65536])
	#x = np.reshape(features_train, (-1))
	#x_float = tf.cast(x, tf.float32)
	x_image = tf.reshape(x, [-1, 256, 256, 1])
	
	y_ = tf.placeholder(tf.float32, shape=[None, 8])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	# Pooling: the size of the images will be reduced to 128x128


	# Second convolutional layer
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	# Pooling: the size of the images will be reduced to 64x64


	# Fully connected layer
	# 64 filters of 64x64 and 1024 neurons
	W_fc1 = weight_variable([64*64*64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 64*64*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


	# Apply Dropout with probability keep_prob
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	
	# Readout Layer: Softmax regression for getting probabilities
	W_fc2 = weight_variable([1024, 8])
	b_fc2 = bias_variable([8])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


	print(len(features_train))
	print(len(labels_train))
	print(len(features_val))
	print(len(labels_val))
	print(len(features_test))



	# TRAIN AND EVALUATE THE MODEL
	print("TRAINING STARTED")

	group_train = list(zip(features_train, labels_train))
	batch_size = 3

	# Loss function
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	# Training step
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#Initialize variables
	init = tf.initialize_all_variables()

	sess = tf.InteractiveSession()

	sess.run(init)

	print("TRAIN")
	# Training: 10000 iterations
	for step in range(140):
		if step < 100:
			print("Step: " + str(step))
		#Select a new batch
		batch = random.sample(group_train, batch_size)
		batch_x, batch_y = list(zip(*batch))
		
		#Train on batch
		train_data = {x : batch_x, y_: batch_y, keep_prob: 0.5}
		#Run one more step of gradient descent
		train_step.run(feed_dict=train_data)
		if step % 100 == 10:
			train_accuracy = accuracy.eval(feed_dict={x: features_val, y_: labels_val, keep_prob: 1.0})
			print("step %d, validation accuracy %g"%(step, train_accuracy))
	

	print('Test accuracy: %g'%accuracy.eval(feed_dict={x: features_val, y_: labels_val, keep_prob: 1.0}))

	return sess.run(y_conv, feed_dict={x: features_test})
