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
	return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')


def CNN(features_train, labels_train, features_val, labels_val):

	# DEFINING THE PARAMETERS OF THE NETWORK
	print("CREATING NETWORK")	

	# First convolutional layer + relu
	W_conv1 = weight_variable([5, 5, 3, 32])
	b_conv1 = bias_variable([32])
	
	x = tf.placeholder(tf.float32, shape=[None, 196608])
	#x = np.reshape(features_train, (-1))
	#x_float = tf.cast(x, tf.float32)
	x_image = tf.reshape(x, [-1, 256, 256, 3])
	
	y_ = tf.placeholder(tf.float32, shape=[None, 8])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	# Pooling: the size of the images will be reduced to 64x64


	# Second convolutional layer
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	# Pooling: the size of the images will be reduced to 16x16


	# Fully connected layer
	# 64 filters of 16*16 and 1024 neurons
	W_fc1 = weight_variable([64*16*16, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 64*16*16])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


	# Apply Dropout with probability keep_prob
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	
	# Readout Layer: Softmax regression for getting probabilities
	W_fc2 = weight_variable([1024, 8])
	b_fc2 = bias_variable([8])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	train_prediction = tf.nn.softmax(y_conv)

	print(len(features_train[0]))
	print(len(labels_train[0]))
	print(len(features_val))
	print(len(labels_val))


	# TRAIN AND EVALUATE THE MODEL
	print("TRAINING STARTED")

	group_train = list(zip(features_train, labels_train))
	batch_size = 10

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
	
	val_accuracies = []
	val_steps = []

	print("TRAIN")
	# Training: 10000 iterations
	for step in range(120):
		#Select a new batch
		batch = random.sample(group_train, batch_size)
		batch_x, batch_y = list(zip(*batch))
		if step % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
			print("step %d, training set accuracy %g"%(step, train_accuracy))
			val_accuracy = accuracy.eval(feed_dict={x: features_val, y_: labels_val, keep_prob: 1.0})
			print("Validation accuracy %g"%val_accuracy)
			val_accuracies.append(val_accuracy)
			val_steps.append(step)

		#Train on batch
		train_data = {x : batch_x, y_: batch_y, keep_prob: 0.5}
		#Run one more step of gradient descent
		train_step.run(feed_dict=train_data)
	

	print('Validation accuracy: %g'%accuracy.eval(feed_dict={x: features_val, y_: labels_val, keep_prob: 1.0}))


	'''
	# Predict on test data
	predictions = []
	for i in range(100):
		#feat1 = features_test[2*i]
		#feat1_f = map(lambda x: float(x), feat1)
		#feat2 = features_test[2*1+1]
		#feat2_f = map(lambda x: float(x), feat2)
		#feat = [feat1, feat2]
		
		#new_feat = np.reshape(feat, (-1))
		#feat_float = tf.cast(new_feat, tf.float32)
		feat = features_test[10*i: 10*i+10]
		d = {x: feat, keep_prob: 1.0}
		pred = sess.run(train_prediction, feed_dict=d)
		#print(len(pred))
		for j in range(len(pred)):
			predictions.append(pred[j])
		#print(i)

	print(predictions[0])
	return predictions
	'''

	return val_accuracies, val_steps


