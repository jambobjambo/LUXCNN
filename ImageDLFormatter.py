from PIL import Image
import json
from resizeimage import resizeimage
import os
from urllib import request
from urllib.error import HTTPError, URLError

ImageDirectoryIn = "./Products/"
ImageDirectoryOut = "./ImagesFormatted/"

def ReFormat(ImageDir, Name, Number):
	with open(ImageDir, 'r+b') as f:
		try:
			with Image.open(f) as image:
				cover = resizeimage.resize_contain(image, [300, 300])
				cover.save(ImageDirectoryOut + Name + '_' + Number + '.jpg', image.format)
		except OSError:
			print("Error")

IDsComplete = []
for Images in os.listdir(ImageDirectoryOut):
	ID = Images.split('_')
	if ID[0] not in IDsComplete:
		IDsComplete.append(ID[0])

DirectoryComplete = 1
for Directory in os.listdir(ImageDirectoryIn):
	with open(ImageDirectoryIn + Directory) as data_file:
		try:
			data = json.load(data_file)
			if str(data['id']) not in IDsComplete:
				f = open('./Temp/image.jpg', 'wb')
				try:
					ImageDL = request.urlopen(data['image']).read()
					f.write(ImageDL)
				except HTTPError:
					print("HTTP error")
				except URLError:
					print("HTTP error")

				f.close()
				ReFormat('./Temp/image.jpg', str(data['id']), '0')

				if 'altImages' in data:
					ImageNum = 1
					for AltImage in data['altImages']:
						f = open('./Temp/image.jpg', 'wb')
						try:
							f.write(request.urlopen(AltImage).read())
						except HTTPError:
							print("HTTP error")
						except URLError:
							print("HTTP error")
						f.close()
						ReFormat('./Temp/image.jpg', str(data['id']), str(ImageNum))
						ImageNum += 1
			else:
				print(str(data['id']) + " Already Downloaded")
		except json.decoder.JSONDecodeError:
			print('error')
		except UnicodeDecodeError:
			print('error')

	print("Directory " + str(DirectoryComplete) + " out of: " + str(len(os.listdir(ImageDirectoryIn))))
	DirectoryComplete += 1
