from PIL import Image

from resizeimage import resizeimage
import os

ImageDirectoryIn = "../RNNImages/"
ImageDirectoryOut = "../TrainingData/Mask/"

def ReFormat(ImageDir, Name):
    Filename = ImageDirectoryIn + ImageDir + '/' + Name
    with open(Filename, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_contain(image, [300, 300])
            if not os.path.exists(ImageDirectoryOut + ImageDir + '/'):
                os.makedirs(ImageDirectoryOut + ImageDir + '/')
            cover.save(ImageDirectoryOut + ImageDir + '/' + Name, image.format)

DirectoryComplete = 0
for Directory in os.listdir(ImageDirectoryIn):
    print("Directory " + str(DirectoryComplete) + " out of: " + str(len(os.listdir(ImageDirectoryIn))))
    DirectoryComplete += 1
    if Directory != ".DS_Store":
        for ImageFile in os.listdir(ImageDirectoryIn + Directory):
            ReFormat(Directory, ImageFile)
