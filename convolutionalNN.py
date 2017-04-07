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
	W_conv1 = weight_variable([5, 5, 1, 64])
	b_conv1 = bias_variable([64])
	
	x = tf.placeholder(tf.float32, shape=[None, 65536])
	#x = np.reshape(features_train, (-1))
	#x_float = tf.cast(x, tf.float32)
	x_image = tf.reshape(x, [-1, 256, 256, 1])
	
	y_ = tf.placeholder(tf.float32, shape=[None, 8])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	# Pooling: the size of the images will be reduced to 128*128


	# Second convolutional layer
	W_conv2 = weight_variable([5, 5, 64, 128])
	b_conv2 = bias_variable([128])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	# Pooling: the size of the images will be reduced to 64*64


	# Third convolutional layer
	W_conv3 = weight_variable([5, 5, 128, 256])
	b_conv3 = bias_variable([256])

	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
	h_pool3 = max_pool_2x2(h_conv3)
	# Pooling: the size of the images will be reduced to 32*32


	# Fourth convolutional layer
	W_conv4 = weight_variable([5, 5, 256, 256])
	b_conv4 = bias_variable([256])

	h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
	h_pool4 = max_pool_2x2(h_conv4)
	# Pooling: the size of the images will be reduced to 16*16


	# Fully connected layer
	# 256 filters of 16*16 and 1024 neurons
	W_fc1 = weight_variable([256*16*16, 1024])
	b_fc1 = bias_variable([1024])

	h_pool4_flat = tf.reshape(h_pool4, [-1, 256*16*16])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)


	# Apply Dropout with probability keep_prob
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	
	# Readout Layer: Softmax regression for getting probabilities
	W_fc2 = weight_variable([1024, 8])
	b_fc2 = bias_variable([8])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	train_prediction = tf.nn.softmax(y_conv)

	print(len(features_train))
	print(len(labels_train))
	print(len(features_val))
	print(len(labels_val))
	print(len(features_test))



	# TRAIN AND EVALUATE THE MODEL
	print("TRAINING STARTED")

	group_train = list(zip(features_train, labels_train))
	batch_size = 10

	# Loss function
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	# Training step with decay in learning rate
	learning_rate = tf.placeholder(tf.float32, shape=[]) 
	train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#Initialize variables
	init = tf.initialize_all_variables()

	sess = tf.InteractiveSession()

	sess.run(init)

	print("TRAIN")
	# Training: 10000 iterations
	for step in range(110):
		#if step < 100:
		#	print("Step: " + str(step))
		#Select a new batch
		batch = random.sample(group_train, batch_size)
		batch_x, batch_y = list(zip(*batch))

		if step % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
			print("step %d, training set accuracy %g"%(step, train_accuracy))
			print("Validation set accuracy %g"%accuracy.eval(feed_dict={x: features_val, y_: labels_val, keep_prob: 1.0}))

		# Update learning rate	
		if step != 0:
			learning_r = (float(1) / step)
		else:
			learning_r = 0.6
		#Train on batch
		train_data = {x : batch_x, y_: batch_y, keep_prob: 0.5, learning_rate: learning_r}
		#Run one more step of gradient descent
		train_step.run(feed_dict=train_data)
	

	print('Validation accuracy: %g'%accuracy.eval(feed_dict={x: features_val, y_: labels_val, keep_prob: 1.0}))

	# Predict on test data
	predictions = []
	for i in range(4051):
		feat = features_test[3*i:3*i+3]
		d = {x: feat, keep_prob: 1.0}
		pred = sess.run(train_prediction, feed_dict= d)
		predictions.append(pred[0])
		if i % 1000 == 0:
			print("step " + str(i))
	return predictions


