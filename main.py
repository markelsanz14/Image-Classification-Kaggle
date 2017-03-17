import readImages as ri
#
def main():
	path_to_images = '../NewImages/'
	features_train, labels_train = ri.readImages(path_to_images)
	print(len(labels_train))

if __name__ == "__main__":
	main()
