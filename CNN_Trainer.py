import cv2
import tensorflow as tf
import numpy as np
import os
import pickle

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

def CNN_Model(numberofclasses, input_size):
	tf.reset_default_graph()
	#Input placeholder
	x = tf.placeholder(tf.float32, [None, input_size])
	#Define weight and bias
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	#reshape x
	x_image = tf.reshape(x, [-1, 300, 300, 1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	#Define weight and bias for second layer
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	#define weight and bias for final layer
	W_fc1 = weight_variable([28 * 28 * 32, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 28*28*32])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	#work out dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	#define output layer weight and bias
	W_fc2 = weight_variable([1024, numberofclasses])
	b_fc2 = bias_variable([numberofclasses])
	#work out loss
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	#define output layer
	y_ = tf.placeholder(tf.float32, [None, numberofclasses])
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	return train_step, accuracy, x, y_, keep_prob

def CNN_Trainer():
	TrainingDataDirectory = "../TrainingData/ImageData/"

	All_TrainingData_in = []
	All_TrainingData_out = []

	D_index = 0
	for Directory in os.listdir(TrainingDataDirectory):
		Classifier = np.zeros(len(os.listdir(TrainingDataDirectory)))
		Classifier[D_index] += 1
		ImageData_inD = pickle.load(open(TrainingDataDirectory + '/' + Directory + '/data.p', "rb"))
		for I_data in ImageData_inD:
			All_TrainingData_in.append(I_data)
			All_TrainingData_out.append(Classifier)

	TestingDataAmount = 0.001
	AmountOfTestingData =  int(round(len(All_TrainingData_in) * TestingDataAmount))

	TestingData_in = All_TrainingData_in[:AmountOfTestingData]
	TrainingData_in = All_TrainingData_in[AmountOfTestingData:]
	TestingData_out = All_TrainingData_out[:AmountOfTestingData]
	TrainingData_out = All_TrainingData_out[AmountOfTestingData:]

	train_step, accuracy, x, y_, keep_prob = CNN_Model(len(TrainingData_out[0]), len(TrainingData_in[0]))
	training_epochs = 50
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())

		test_input_store = []
		test_output_store = []
		print("Training CNN")
		for epoch in range(training_epochs):
			i = 0
			batch_size = 5
			while i < len(TrainingData_in):
				start = i
				end = i + batch_size
				batch_x = np.array(TrainingData_in[start:end], dtype=float)
				batch_y = np.array(TrainingData_out[start:end], dtype=float)

				sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
				i+=batch_size

			print("Epoch:",  epoch, "Accuracy: ", accuracy.eval(session=sess,feed_dict={x: TestingData_in, y_: TestingData_out, keep_prob: 1.0}))

	saver.save(sess, './TrainedModel/model.ckpt')
CNN_Trainer()
