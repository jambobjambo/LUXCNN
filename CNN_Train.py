import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os, zipfile
import pickle
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
import shutil

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

URLS_Download = ["https://www.dropbox.com/sh/jji569o0fhbxj1s/AADVF1WErPXWxaDqlgYqOs76a?dl=1", "https://www.dropbox.com/sh/bsaf1zk9hkkojr2/AAA2ESS1OF97VMef-a9wIEgBa?dl=1", "https://www.dropbox.com/sh/4jfauwpnq3kp1dp/AAB5xI-aXmmqzmxvtOAAzl4da?dl=1", "https://www.dropbox.com/sh/llllc6s6083q8p3/AADl-x70xxoIP6No6i8v99XMa?dl=1", "https://www.dropbox.com/sh/uf9lrsi6e204rnk/AABKOTxOnMSFlcG-npAMu94qa?dl=1", "https://www.dropbox.com/sh/imlx9kucm8wrj4s/AACe8yF5AZ16xZxydjwML_lxa?dl=1", "https://www.dropbox.com/sh/qnjfi52otw5ezs2/AABg79OLj8d4VJ-MP9F49IGha?dl=1", "https://www.dropbox.com/sh/qm1jsfnuryj6t6z/AAD7L9X0vOV4yOyOBsyWD2xNa?dl=1", "https://www.dropbox.com/sh/tni0nr8thhsyk3h/AACEVc2UXemcfiawhplX5rr_a?dl=1", "https://www.dropbox.com/sh/6317shewwpcl2qn/AACRLOtJ_g2idTKT7FzD73ipa?dl=1", "https://www.dropbox.com/sh/8ix02899jipy8b9/AABf-KgnDwYtZpwEy9uRTBj4a?dl=1", "https://www.dropbox.com/sh/wnqgvoy8b7s4n10/AADsHWL2GHmfSXmbmNrbUdDpa?dl=1", "https://www.dropbox.com/sh/3xcjnzobb9svprw/AAD3D7qlCJdwT4xYNKQwC8jma?dl=1", "https://www.dropbox.com/sh/halyj3kcy16g3kw/AACNbwd_VoqVAoZ6tx1ZaqfAa?dl=1", "https://www.dropbox.com/sh/0zburn3phsiy9k0/AAAya-fJmtQE-KgI8kl0IpULa?dl=1", "https://www.dropbox.com/sh/493hyh8v0idyvrb/AAAEqkY57eW8QDgBTDZ2cyAHa?dl=1", "https://www.dropbox.com/sh/x6hleu3ming4e2y/AADaNP0XQOQQcxvFqpemO3ota?dl=1", "https://www.dropbox.com/sh/89ybc9us8kh5mxt/AADbG466GfBEVeIHVnnie0KKa?dl=1", "https://www.dropbox.com/sh/2ry3qepnedjrd3r/AAByLARIGPt5nYigyBCNm5Uoa?dl=1", "https://www.dropbox.com/sh/mreqztyo4dfwoe9/AACLLPF0gDNORhjMAXDi6vq7a?dl=1", "https://www.dropbox.com/sh/kjljxemsfb7g7wh/AAAAA8QeQOSVz2Ts9CKjeqfua?dl=1", "https://www.dropbox.com/sh/6o3boq4w2udewu1/AAA9bSU0M2b-bq10ygH21yHRa?dl=1", "https://www.dropbox.com/sh/veq4z4xhj6kzjea/AADy6aextaMmbtQrl4_xBlk3a?dl=1", "https://www.dropbox.com/sh/h61of3ca8s9bnb3/AAAThONr34gFn-0f7hnjibsHa?dl=1", "https://www.dropbox.com/sh/4kztfx11jx66uzh/AAB-Zwh-EsK9UTZFVQqC9LpSa?dl=1", "https://www.dropbox.com/sh/vppgp08folzvhxo/AADxhq38KjLtloPtCULfz_-ja?dl=1", "https://www.dropbox.com/sh/t2w66sd5gbkkr4p/AACPN9uZS9ixeDiOX0RkZvMga?dl=1", "https://www.dropbox.com/sh/sg02ipq7v4wjfsd/AADNJzLhMoQPiCyDeWQ3Gb_Da?dl=1", "https://www.dropbox.com/sh/ct9xmoxfsmxuku5/AACfRwv-EDv3k_w6qVXWDIUta?dl=1", "https://www.dropbox.com/sh/qc2a9uad6ocxjcr/AAArIBsN266uQAZKQ_cRXNVRa?dl=1", "https://www.dropbox.com/sh/aplzgd8graf43cc/AABnduTl30kg7-dbMm8bq8nsa?dl=1", "https://www.dropbox.com/sh/14xf2zxhgzjyid0/AABOf2kXaJXMqx-I7ohXRbA3a?dl=1", "https://www.dropbox.com/sh/0kq7xjsdia3y4yt/AACQW2IHhwBRwDsIXG_9Emu6a?dl=1", "https://www.dropbox.com/sh/3g0c9jpxs1zpf87/AAAkTxFXRvxqhvSA0SHl7L2Ra?dl=1", "https://www.dropbox.com/sh/8z0im1rmfxngsjf/AADCMEXnrPVV1STdgwMNUyzMa?dl=1", "https://www.dropbox.com/sh/p6ld9xkonu7zei3/AADOC8Khm4XPkU0Fmjk4swWBa?dl=1", "https://www.dropbox.com/sh/w44nm1vchb82cd6/AABWiFfz_XchxoEkZHvOoK2Ta?dl=1"]

print("Downloading Training Data")
#total = 36
Limit = 1
Url_Complete = 1
for URL_DOWN in URLS_Download:
	if Url_Complete <= Limit:
		response = urlopen(URL_DOWN)
		zipcontent= response.read()
		with open("TrainingData.zip", 'wb') as f:
			f.write(zipcontent)

		if not os.path.exists(TrainingDataDirectory):
			os.makedirs(TrainingDataDirectory)

		z = zipfile.ZipFile('TrainingData.zip')
		for f in z.namelist():
			z.extract(f, TrainingDataDirectory)

		print("Downloaded and Extracted " + str(Url_Complete) + " out of " + str(len(URLS_Download)))
		Url_Complete += 1

print("Training Data Downloaded")


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

def zipdir(path, ziph):
	# ziph is zipfile handle
	for root, dirs, files in os.walk(path):
		for file in files:
			ziph.write(os.path.join(root, file))

def optimize(num_iterations):
	# Ensure we update the global variable rather than a local copy.
	global total_iterations

	# Start-time used for printing time-usage below.
	start_time = time.time()

	for i in range(total_iterations,total_iterations + num_iterations):
		for TrainingFile in os.listdir(TrainingDataDirectory):
			try:
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
			except KeyError:
				print("Error opening " + TrainingFile)

	total_iterations += num_iterations
	# Ending time.
	end_time = time.time()
	# Difference between start and end-times.
	time_dif = end_time - start_time
	# Print the time-usage.
	if not os.path.exists(SavedModelDIR):
		os.makedirs(SavedModelDIR)
	saver.save(session, SavedModelDIR + "model.ckpt")

	zipf = zipfile.ZipFile('Model.zip', 'w', zipfile.ZIP_DEFLATED)
	zipdir(SavedModelDIR, zipf)
	zipf.close()

	try:
		shutil.copy('./Model.zip', '/valohai/outputs')
	except ValueError:
		print("cant save to directory")

	'''with open(SavedModelDIR + "model.ckpt.index", 'rb') as f:
		print(f.read())'''

	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def Test():
	saver.restore(session, SavedModelDIR + "model.ckpt")
	DressExample = pickle.load(open(TrainingDataDirectory + '/0.p', "rb"))

	DressTest = 0
	for Dress in DressExample[0]:
		FeedDict = np.reshape(Dress, (-1, len(Dress), len(Dress[0])))
		NNPrediction = session.run(y_pred, feed_dict={x:FeedDict})
		Output = []
		for Predictions in NNPrediction[0]:
			Output.append(round(Predictions, 2))

		Indexs = []
		for Cycle in Output:
			Index_Output = 0
			HighestVal = 0
			HighestIndex = 0
			for Values in Output:
				if HighestVal == 0 and Index_Output not in Indexs:
					HighestVal = Values
					HighestIndex = Index_Output

				if Values > HighestVal and Index_Output not in Indexs:
					HighestVal = Values
					HighestIndex = Index_Output

				#print(Index_Output)
				#print(len(Output))
				if Index_Output == len(Output) - 1:
					Indexs.append(HighestIndex)

				Index_Output += 1

		#print(len(Indexs))
		Display = 5
		for DisplayVal in range(Display):
			print("Response " + str(Indexs[DisplayVal]) + " with confidence: " + str(Output[Indexs[DisplayVal]]))

		AccInd = 0
		ActualIndexs = []
		for ActualVal in DressExample[1][DressTest]:
			if ActualVal == 1:
				ActualIndexs.append(AccInd)
			AccInd += 1

		print("Actual Indexs: " + str(ActualIndexs))

		DressTest += 1

#download_data()
optimize(1)
#Test()
