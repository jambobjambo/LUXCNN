import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import pickle

filter_size1 = 50          # Convolution filters are 5 x 5 pixels.
num_filters1 = 8         # There are 16 of these filters.
filter_size2 = 50          # Convolution filters are 5 x 5 pixels.
num_filters2 = 16         # There are 36 of these filters.
fc_size = 128
img_size = 300
num_channels = 3
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

# Number of classes, one class for each of 10 digits.
SavedModelDIR = "./TrainedModel/"
TrainingDataDirectory = "./TrainingData/"

TrainingData = pickle.load(open(TrainingDataDirectory + '0.p', "rb"))
ClassifierDir = TrainingData[1]

num_classes = len(ClassifierDir[0])

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
	#equivalent to y intercept
	#constant value carried over across matrix math
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):
	# This format is determined by the TensorFlow API.
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	# Create new weights aka. filters with the given shape.
	weights = new_weights(shape=shape)
	# Create new biases, one for each filter.
	biases = new_biases(length=num_filters)
	layer = tf.nn.conv2d(input=input,filter=weights,strides=[1, 1, 1, 1],padding='SAME')
	# A bias-value is added to each filter-channel.
	layer += biases
	# Use pooling to down-sample the image resolution?
	if use_pooling:
		layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
	layer = tf.nn.relu(layer)
	return layer, weights


def flatten_layer(layer):
	# Get the shape of the input layer.
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat, num_features

def new_fc_layer(input,num_inputs,num_outputs,use_relu=True): # Use Rectified Linear Unit (ReLU)?
	# Create new weights and biases.
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)
	layer = tf.matmul(input, weights) + biases
	if use_relu:
	    layer = tf.nn.relu(layer)
	return layer

x = tf.placeholder(tf.float32, shape=[None, num_channels, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x_image,num_input_channels=num_channels,filter_size=filter_size1,num_filters=num_filters1,use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,num_input_channels=num_filters1,filter_size=filter_size2,num_filters=num_filters2,use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_size,use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,num_inputs=fc_size,num_outputs=num_classes,use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()

session.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

train_batch_size = 10
total_iterations = 0
def optimize(num_iterations):
	# Ensure we update the global variable rather than a local copy.
	global total_iterations

	# Start-time used for printing time-usage below.
	start_time = time.time()

	for i in range(total_iterations,total_iterations + num_iterations):
		for TrainingFile in os.listdir(TrainingDataDirectory):
			TrainingData = pickle.load(open(TrainingDataDirectory + TrainingFile, "rb"))
			All_TrainingData_in = TrainingData[0]
			All_TrainingData_out = TrainingData[1]
			Batch = 0
			while Batch < len(All_TrainingData_in):
				start = Batch
				end = Batch + train_batch_size

				x_batch = np.array(All_TrainingData_in[start:end], dtype=float)
				y_true_batch = np.array(All_TrainingData_out[start:end], dtype=float)

				feed_dict_train = {x: x_batch,y_true: y_true_batch}

				session.run(optimizer, feed_dict=feed_dict_train)
				if Batch == 0:
					acc = session.run(accuracy, feed_dict=feed_dict_train)
					msg = "File " + TrainingFile + " Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
					print(msg.format(i + 1, acc))

					total_iterations += num_iterations
					# Ending time.
					end_time = time.time()
					# Difference between start and end-times.
					time_dif = end_time - start_time
					print("Time for cycle: " + str(timedelta(seconds=int(round(time_dif)))))
				Batch += train_batch_size

	total_iterations += num_iterations
	# Ending time.
	end_time = time.time()
	# Difference between start and end-times.
	time_dif = end_time - start_time
	# Print the time-usage.
	if not os.path.exists(SavedModelDIR):
		os.makedirs(SavedModelDIR)
	saver.save(session, SavedModelDIR + "test/model.ckpt")

	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def Test():
	saver.restore(session, SavedModelDIR + "model.ckpt")
	DressExample = pickle.load(open(TrainingDataDirectory + '/Dress/data.p', "rb"))
	FeedDict = np.reshape(DressExample[0], (-1, len(DressExample[0])))
	NNPrediction = session.run(y_pred, feed_dict={x:FeedDict})
	Output = []
	for Predictions in NNPrediction[0]:
		Output.append(round(Predictions, 2))
	print(Output)

	TrouserExample = pickle.load(open(TrainingDataDirectory + '/Trousers/data.p', "rb"))
	for trouser in TrouserExample:
		FeedDict = np.reshape(trouser, (-1, len(TrouserExample[0])))
		NNPrediction = session.run(y_pred, feed_dict={x:FeedDict})
		Output = []
		for Predictions in NNPrediction[0]:
			Output.append(round(Predictions, 2))
		print(Output)

optimize(10)
#Test()
