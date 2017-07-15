import numpy as np
import os
import cv2
import pickle

ImageDirectory = "../TrainingData/Mask/"
DataDirectory = "../TrainingData/ImageData/"

def ConvertImages(ImageDirectory):
	im = cv2.imread(ImageDirectory)
	return(im)

D_index = 0
for Directory in os.listdir(ImageDirectory):
	Classifier = np.zeros(len(os.listdir(ImageDirectory)))
	Classifier[D_index] += 1
	ImageDataArray = []
	ClassifierData = []
	if Directory != ".DS_Store":
		for ImageFile in os.listdir(ImageDirectory + Directory):
			ImageData = ConvertImages(ImageDirectory + Directory + '/' + ImageFile)
			IM_Reshape = np.reshape(ImageData, (-1,len(ImageData) * len(ImageData[0])))
			#print(len(IM_Reshape[0]))
			ImageDataArray.append(IM_Reshape[0])
			ClassifierData.append(Classifier)

	if not os.path.exists(DataDirectory + Directory):
		os.makedirs(DataDirectory + Directory)

	DataSave = []
	DataSave.append(ImageDataArray)
	DataSave.append(ClassifierData)

	with open( DataDirectory + "/data.p", 'wb') as f:
		pickle.dump(DataSave, f, protocol=2)
		f.close()
		
	D_index += 1
