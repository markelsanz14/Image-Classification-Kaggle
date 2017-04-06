from PIL import Image
import os
import numpy as np
import pandas as pd

#Reads images from directory and creates the training set
def readImages(directory):
	features_train = []
	labels_train = []
	ident = np.identity(8)
	a = True
	#Directory to read the images from
	for im in os.listdir(directory):
		image = Image.open(directory+im)
		#Read label from image (label = first character of image name)
		label = int(im[0])
		label_one_hot = ident[label]
		#Read image pixels and store them as an array
		image_pixels = np.asarray(image.getdata()).reshape((image.size[1]*image.size[1]*3))
		#Add pixels as features to feature list
		features_train.append(image_pixels)
		#Add label to label list
		labels_train.append(label_one_hot)
		if a is True:
			print(len(image_pixels))
		a = False

	features_val = features_train[3777:]
	features_train = features_train[:3777]
	labels_val = labels_train[3777:]
	labels_train = labels_train[:3777]
	return features_train, labels_train, features_val, labels_val
