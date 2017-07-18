import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import os
from collections import Counter
lemmatizer = WordNetLemmatizer()
import json
import cv2

def ConvertImages(ImageDirectory):
	im = cv2.imread(ImageDirectory)
	return(im)

ProductIds = []
ProductNames = []
NamesCollection = []
for Products in os.listdir('./Products'):
	with open('./Products/' + Products) as data_file:
		try:
			data = json.load(data_file)
			ProductIds.append(str(data['id']))
			ProductNames.append(data['name'].lower())
			product_name = data['name'].split(' ')
			for workAdd in product_name:
				NamesCollection.append(workAdd.lower())

		except json.decoder.JSONDecodeError:
			print('error')
		except UnicodeDecodeError:
			print('error')

findalwords = []
word_counts = Counter(NamesCollection)
for word in word_counts:
	if word_counts[word] > 15 and word_counts[word] < 2000:
		findalwords.append(word)

VectorsNames = []
for Products in ProductNames:
	DefaultVec = np.zeros(len(findalwords))
	Word_in_Product = Products.split(' ')
	for SplWord in Word_in_Product:
		if SplWord in findalwords:
			DefaultVec[findalwords.index(SplWord)] += 1

	VectorsNames.append(DefaultVec)

ImageVecStore = []
NameVecStore = []
ImageNum = 0
FileNum = 0
for Image in os.listdir('./ImagesFormatted'):
	if len(NameVecStore) == 100:
		with open( './TrainingData/' + str(FileNum) + ".p", 'wb') as f:
			pickle.dump([ImageVecStore,NameVecStore], f, protocol=2)
			f.close()
		ImageVecStore = []
		NameVecStore = []
		FileNum += 1


	ImageData = ConvertImages('./ImagesFormatted' + '/' + Image)
	IM_Reshape = np.reshape(ImageData, (-1,len(ImageData) * len(ImageData[0])))

	ImageVecStore.append(IM_Reshape)

	IMG_split = Image.split('_')
	NameVecStore.append(VectorsNames[ProductIds.index(IMG_split[0])])
	print("Image " + str(ImageNum) + " out of " + str(len(os.listdir('./ImagesFormatted'))))
	ImageNum += 1
