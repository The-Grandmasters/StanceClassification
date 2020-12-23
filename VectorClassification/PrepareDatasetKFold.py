import os
import torch
import numpy

def PrepareDataset():
	directory = r"../Data/output/" 	#directory containing the pose vectors
				#The ONLY files in this directory should be pose vectors	
		
	file_list = os.listdir(directory)

	X = numpy.zeros((len(file_list), 17, 3), dtype=float)
	Y = numpy.zeros((len(file_list)))

	#For each file in the directory, check which stance it is. 
	#If more training files need to be stored for that stance,
	#	store the vector in the training example array and put the correct label in the training label array.
	#If a sufficient number of training examples have been stored for that stance,
	#	store the vector and its label in the testing arrays
	i = 0
	for filename in file_list:
		X[i] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
		if "fighting" in filename:
			Y[i] = 0
		elif "front" in filename:
			Y[i] = 1
		elif "ready" in filename:
			Y[i] = 2
		elif "cat" in filename:
			Y[i] = 3
		elif "horse" in filename:
			Y[i] = 4
		elif "hicho" in filename:
			Y[i] = 5
		elif "seiza" in filename:
			Y[i] = 6
		i += 1
	
	return X, Y
