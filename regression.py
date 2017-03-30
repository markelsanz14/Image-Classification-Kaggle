import tensorflow as tf
import numpy as np
import random

def CNN(features_train, labels_train, features_val, labels_val):
	
	print(len(features_train))
	print(len(features_val))
	print(len(labels_train))
	print(len(labels_val))
	
	sess = tf.InteractiveSession()

	group_train = list(zip(features_train, labels_train))
	batch_size = 100
	# Placeholders for our training features and labels
	x = tf.placeholder(tf.float32, shape=[None, 65536]) #3 for RGB
	# REAL LABELS
	num_labels = 8
	y_ = tf.placeholder(tf.float32, shape=[None, num_labels])
	#labels = (np.arange(num_labels) == labels[:]).astype(np.float32)
	
	# Weights and Biases for our learning algorithm
	W = tf.Variable(tf.zeros([65536, num_labels]))
	b = tf.Variable(tf.zeros([num_labels]))

	print("VARIABLES INITIALIZED")

	#Initialize variables
	sess.run(tf.global_variables_initializer())

	# Linear Regression function
	y = tf.matmul(x, W) + b
	# Loss function
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	# Training step (gradient descent)
	train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("TRAINING STARTED")
	# Training: 1000 iterations
	for step in range(10000):
		#Select a new batch
		batch = random.sample(group_train, batch_size)
		unzipped_batch = list(zip(*batch))
		batch_x = unzipped_batch[0]
		batch_y = unzipped_batch[1]
		#Train on batch
		train_data = {x : batch_x, y_: batch_y}
		#Run one more step of gradient descent
		train_step.run(feed_dict=train_data)
		if (step % 200 == 0):
			print("step " + str(step))
			#print('Loss at step %d: %f' % (step, cross_entropy.))
			print('Training accuracy: ' + str(accuracy.eval(feed_dict={x: features_train, y_: labels_train})))
			print('Test accuracy: ' + str(accuracy.eval(feed_dict={x: features_val, y_: labels_val})))
			#print(y[:3])
			print(sess.run(y, feed_dict={x: features_val})[:3])
			print(labels_val[:3])

	#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#print(accuracy.eval(feed_dict={x: features, y_: labels}))
	
