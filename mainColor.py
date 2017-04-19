import readTrainingImagesColor as rtr
import readTestImagesColor as rts
import convolutionalNNColor as cnn
#import regression as reg
import writePredictionColor as wrpr
import matplotlib.pyplot as plt

#
def main():
	path_to_train_images = '../NewImagesColor/'
	features_train, labels_train, features_val, labels_val = rtr.readImages(path_to_train_images)
	print("TRAINING SET READ")

	path_to_test_images = '../NewImagesColorTest/'
	#features_test, nameList = rts.readImages(path_to_test_images)

	# Uncomment to execute Deep Convolutional Neural Network
	accuracies, steps = cnn.CNN(features_train, labels_train, features_val, labels_val)

	plt.plot(steps, accuracies)
	plt.ylabel('Accuracy RGB')
	plt.xlabel('Training Step Number')
	plt.savefig('../figure_rgb')


if __name__ == "__main__":
	main()
