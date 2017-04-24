import readTrainingImagesKNNval as rtr
import knn as knn

import writePrediction as wrpr

#
def main():
	path_to_train_images = '../NewImages/'
	features_train, labels_train, features_val, labels_val = rtr.readImages(path_to_train_images)
	print("TRAINING SET READ")


	# Uncomment to execute KNN
	predictions = knn.knn_val(features_train, labels_train, features_val, labels_val)
	
	print(predictions)


if __name__ == "__main__":
	main()
