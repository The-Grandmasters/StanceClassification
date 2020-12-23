import os
import torch
import numpy
import random

directory = r"../Data/output/" 	#directory containing the pose vectors
			#The ONLY files in this directory should be pose vectors	
test_ratio = 0.2	#portion of dataset that should allocated for testing data
	
file_list = os.listdir(directory)
random.shuffle(file_list)
num_train = numpy.zeros((7), dtype=int)
num_test = numpy.zeros((7), dtype=int)

#count files of each stance
for filename in file_list:
	if "fighting" in filename:
		num_train[0] += 1 
	elif "front" in filename:
		num_train[1] += 1
	elif "ready" in filename:
		num_train[2] += 1
	elif "cat" in filename:
		num_train[3] += 1
	elif "horse" in filename:
		num_train[4] += 1
	elif "hicho" in filename:
		num_train[5] += 1
	elif "seiza" in filename:
		num_train[6] += 1

#set number of files of each stance to be allocated for training/testing
for i in range(7):
	num_test[i] = round(num_train[i] * test_ratio)
	num_train[i] -= num_test[i]

total_train = numpy.sum(num_train)
total_test = numpy.sum(num_test)

train_x = numpy.zeros((total_train, 17, 3), dtype=float)
train_y = numpy.zeros((total_train), dtype=int)
test_x = numpy.zeros((total_test, 17, 3), dtype=float)
test_y = numpy.zeros((total_test), dtype=int)

test_index = 0
train_index = 0
#For each file in the directory, check which stance it is. 
#If more training files need to be stored for that stance,
#	store the vector in the training example array and put the correct label in the training label array.
#If a sufficient number of training examples have been stored for that stance,
#	store the vector and its label in the testing arrays
for filename in file_list:
	if "fighting" in filename:
		if num_train[0] > 0:
			num_train[0] -= 1
			train_x[train_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			train_y[train_index] = 0
			train_index += 1
		else:
			test_x[test_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			test_y[test_index] = 0
			test_index += 1
	elif "front" in filename:
		if num_train[1] > 0:
			num_train[1] -= 1
			train_x[train_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			train_y[train_index] = 1
			train_index += 1
		else:
			test_x[test_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			test_y[test_index] = 1
			test_index += 1
	elif "ready" in filename:
		if num_train[2] > 0:
			num_train[2] -= 1
			train_x[train_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			train_y[train_index] = 2
			train_index += 1
		else:
			test_x[test_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			test_y[test_index] = 2
			test_index += 1
	elif "cat" in filename:
		if num_train[3] > 0:
			num_train[3] -= 1
			train_x[train_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			train_y[train_index] = 3
			train_index += 1
		else:
			test_x[test_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			test_y[test_index] = 3
			test_index += 1
	elif "horse" in filename:
		if num_train[4] > 0:
			num_train[4] -= 1
			train_x[train_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			train_y[train_index] = 4
			train_index += 1
		else:
			test_x[test_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			test_y[test_index] = 4
			test_index += 1
	elif "hicho" in filename:
		if num_train[5] > 0:
			num_train[5] -= 1
			train_x[train_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			train_y[train_index] = 5
			train_index += 1
		else:
			test_x[test_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			test_y[test_index] = 5
			test_index += 1
	elif "seiza" in filename:
		if num_train[6] > 0:
			num_train[6] -= 1
			train_x[train_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			train_y[train_index] = 6
			train_index += 1
		else:
			test_x[test_index] = torch.load(directory + filename, map_location=torch.device('cpu')).numpy()
			test_y[test_index] = 6
			test_index += 1
print("Test " + str(test_index))
print("Train " + str(train_index))

#write the files out to the current directory
torch.save(torch.from_numpy(train_x), "train_x.pt")
torch.save(torch.from_numpy(train_y), "train_y.pt")
torch.save(torch.from_numpy(test_x),  "test_x.pt")
torch.save(torch.from_numpy(test_y),  "test_y.pt")
