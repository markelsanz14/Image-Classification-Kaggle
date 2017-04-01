import pandas as pd

def writePrediction(prediction, nameList):
	print(prediction[:2])
	a = list(zip(*prediction))
	group = [nameList, list(a[0]), list(a[1]), list(a[2]), list(a[3]), list(a[4]), list(a[5]), list(a[6]), list(a[7])]
	output = pd.DataFrame(group)
	output = output.T
	output.to_csv('../output.csv')
	


