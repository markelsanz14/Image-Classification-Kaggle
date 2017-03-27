import readImages as ri
import convolutionalNN as cnn

#
def main():
	path_to_images = '../NewImages/'
	features_train, labels_train = ri.readImages(path_to_images)
	print("TRAINING SET READ")
	cnn.CNN(features_train, labels_train)

if __name__ == "__main__":
	main()
