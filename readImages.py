from PIL import Image
import os
import numpy as np

#Reads images from directory and creates the training set
def readImages(directory):
	features_train = []
	labels_train = []
	
	#Directory to read the images from
	for im in os.listdir(directory):
		image = Image.open(directory+im)
		#Read label from image (label = first character of image name)
		label = int(im[0])
		#Read image pixels and store them as an array
		image_pixels = np.asarray(image.getdata()).reshape((image.size[1]*image.size[1], 1))
		#Add pixels as features to feature list
		features_train.append(image_pixels)
		#Add label to label list
		labels_train.append(label)

	return features_train, labels_train
