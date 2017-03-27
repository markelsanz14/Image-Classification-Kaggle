import tensorflow as tf

def CNN(features, labels):
	
	sess = tf.InteractiveSession()

	x = tf.placeholder(tf.float32, shape=[None, 65536])
	y_ = tf.placeholder(tf.float32, shape=[None, 8])
	
	W = tf.Variable(tf.zeros([65536,8]))
	b = tf.Variable(tf.zeros([8]))

	print("VARIABLES INITIALIZED")

	sess.run(tf.global_variables_initializer())

	y = tf.matmul(x,W) + b
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	print("TRAINING STARTED")
	for step in range(1000):
		batch = tf.train.batch([features, labels], batch_size=100)
		train_step.run(feed_dict={x: batch[0], y_: batch[1]})
		#if (step % 50 == 0):
            #print('Loss at step %d: %f' % (step, l))
            #print('Training accuracy: %.1f%%' % accuracy(predictions, tf_train_labels))

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(accuracy.eval(feed_dict={x: features, y_: labels}))
	

