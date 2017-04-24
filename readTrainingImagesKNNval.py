from PIL import Image
import os
import numpy as np
import pandas as pd

#Reads images from directory and creates the training set
def readImages(directory):
	features_train = []
	labels_train = []
	ident = np.identity(8)

	i = 0
	#Directory to read the images from
	for im in os.listdir(directory):
		image = Image.open(directory+im)
		#Read label from image (label = first character of image name)
		label = int(im[0])
		label_one_hot = ident[label]
		#Read image pixels and store them as an array
		image_pixels = np.asarray(image.getdata()).reshape((image.size[1]*image.size[1]))
		#Add pixels as features to feature list
		features_train.append(image_pixels)
		#Add label to label list
		labels_train.append(label)
		#if i > 3500:
		#	print(label)
		i += 1

	features_val = features_train[3300:]
	features_train = features_train[:3300]
	labels_val = labels_train[3300:]
	labels_train = labels_train[:3300]
	return features_train, labels_train, features_val, labels_val
