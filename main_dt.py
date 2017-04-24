import readTrainingImages as rtr
import readTestImages as rts
import DecisionT as dt

import writePrediction as wrpr

#
def main():
	path_to_train_images = '../NewImages/'
	features_train, labels_train, features_val, labels_val = rtr.readImages(path_to_train_images)
	print("TRAINING SET READ")

	path_to_test_images = '../NewImagesTest2/'
	features_test, nameList = rts.readImages(path_to_test_images)
	# Uncomment to execute Regression
	#predictions = reg.CNN(features_train, labels_train, features_val, labels_val, features_test)
	# Uncomment to execute Deep Convolutional Neural Network
	predictions = dt.tree(features_train, labels_train, features_test)
	
	print(predictions)


if __name__ == "__main__":
	main()
