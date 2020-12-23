import torch
import numpy as np
import matplotlib.pyplot as plt

#number of points to generate between each connected joint
connection_points = 10

#	The parametric equation for a line between two points is a function of the parameter "t"
#	t=0 represents the first point, t=1 represents the second point
#	to find "n" evenly spaced points on the line, "t" must be calculated at intervals of (1 / (n + 1))  
step_size = 1.0 / (connection_points + 1)

#every pair of connected joints
joint_connections = np.array([[0,16], [16,1], [1,15], [15,14], [14,11], [11,12], [12,13], [14,8], [8,9], [9,10], [15,2], [2,3], [3,4], [15,5], [5,6], [6,7]])

#direction vector for calculating line in between joins
dir_vect = np.zeros(3)



#		*** CONVERT TRAINING DATA ***


train_x = torch.load("train_x.pt", map_location=torch.device('cpu')).numpy()

#new dataset array (num_examples, 17 joints + (n intermediate points * m connections), xyz coordinates) 
train_x_connected = np.zeros((np.shape(train_x)[0], (connection_points * len(joint_connections)) + 17, 3), dtype=np.float64)

#add the joint coordinates from the original dataset
train_x_connected[0:, 0:np.shape(train_x)[1], 0:] = train_x

for example in range(np.shape(train_x)[0]):
	for connection in range(len(joint_connections)):
		for dimension in range(3):
			joint1 = train_x[example][joint_connections[connection][0]]
			joint2 = train_x[example][joint_connections[connection][1]]
			dir_vect[dimension] = joint1[dimension] - joint2[dimension]
		for t in np.arange(step_size, 1, step_size):
			x_val = joint2[0] + (dir_vect[0] * t)
			y_val = joint2[1] + (dir_vect[1] * t)
			z_val = joint2[2] + (dir_vect[2] * t)
			train_x_connected[example][17 + (connection * connection_points) + int(round((t / step_size), 0)) - 1] = np.array([x_val, y_val, z_val])
			


torch.save(torch.from_numpy(train_x_connected), "train_x_connected.pt")



#		*** CONVERT TESTING DATA ***


test_x = torch.load("test_x.pt", map_location=torch.device('cpu')).numpy()

#new dataset array (num_examples, 17 joints + (n intermediate points * m connections), xyz coordinates) 
test_x_connected = np.zeros((np.shape(test_x)[0], (connection_points * len(joint_connections)) + 17, 3), dtype=np.float64)

#add the joint coordinates from the original dataset
test_x_connected[0:, 0:np.shape(test_x)[1], 0:] = test_x

for example in range(np.shape(test_x)[0]):
	for connection in range(len(joint_connections)):
		for dimension in range(3):
			joint1 = test_x[example][joint_connections[connection][0]]
			joint2 = test_x[example][joint_connections[connection][1]]
			dir_vect[dimension] = joint1[dimension] - joint2[dimension]
		for t in np.arange(step_size, 1, step_size):
			x_val = joint2[0] + (dir_vect[0] * t)
			y_val = joint2[1] + (dir_vect[1] * t)
			z_val = joint2[2] + (dir_vect[2] * t)
			test_x_connected[example][17 + (connection * connection_points) + (int(round((t / step_size), 0))) - 1] = np.array([x_val, y_val, z_val])

torch.save(torch.from_numpy(test_x_connected), "test_x_connected.pt")





