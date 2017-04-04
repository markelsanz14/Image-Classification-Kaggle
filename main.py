import readTrainingImages as rtr
import readTestImages as rts
import convolutionalNN as cnn
import regression as reg
import writePrediction as wrpr

#
def main():
	path_to_train_images = '../NewImages/'
	features_train, labels_train, features_val, labels_val = rtr.readImages(path_to_train_images)
	print("TRAINING SET READ")

	path_to_test_images = '../NewImagesTest/'
	features_test, nameList = rts.readImages(path_to_test_images)
	# Uncomment to execute Regression
	#predictions = reg.CNN(features_train, labels_train, features_val, labels_val, features_test)
	# Uncomment to execute Deep Convolutional Neural Network
	predictions = cnn.CNN(features_train, labels_train, features_val, labels_val, features_test)

	wrpr.writePrediction(predictions, nameList)


if __name__ == "__main__":
	main()
