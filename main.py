import readImages as ri
import convolutionalNN as cnn
import regression as reg

#
def main():
	path_to_images = '../NewImages/'
	features_train, labels_train, features_val, labels_val = ri.readImages(path_to_images)
	print("TRAINING SET READ")
	reg.CNN(features_train, labels_train, features_val, labels_val)

if __name__ == "__main__":
	main()
