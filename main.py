import readTrainingImages as rtr
import readTestImages as rts
import convolutionalNN as cnn
import regression as reg
import writePrediction as wrpr
import matplotlib.pyplot as plt
import numpy as np

#
def main():
	path_to_train_images = '../NewImages/'
	features_train, labels_train, features_val, labels_val = rtr.readImages(path_to_train_images)
	print("TRAINING SET READ")
	path_to_test_images = '../NewImagesTest2/'
	#features_test, nameList = rts.readImages(path_to_test_images)
	# Deep Convolutional Neural Network
	accuracies, steps = cnn.CNN(features_train, labels_train, features_val, labels_val)

	plt.plot(steps, accuracies)
	plt.ylabel('Accuracy')
	plt.xlabel('Training Step Number')
	plt.savefig('../figure.png')
	#wrpr.writePrediction(predictions, nameList)


if __name__ == "__main__":
	main()
