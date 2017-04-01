from PIL import Image
import os
import numpy as np
import pandas as pd

#Reads images from directory and creates the training set
def readImages(directory):
	features_test = []
	nameList = []

	#Directory to read the images from
	for im in os.listdir(directory):
		image = Image.open(directory+im)
		#Read image pixels and store them as an array
		image_pixels = np.asarray(image.getdata()).reshape((image.size[1]*image.size[1]))
		#Add pixels as features to feature list
		features_test.append(image_pixels)
		nameList.append(im)

	return features_test, nameList
