import torch
import numpy as np
import matplotlib.pyplot as plt

joint_connections = np.array([[0,16], [16,1], [1,15], [15,14], [14,11], [11,12], [12,13], [14,8], [8,9], [9,10], [15,2], [2,3], [3,4], [15,5], [5,6], [6,7]])

dir_vect = np.zeros(3)

train_x = torch.load("train_x.pt", map_location=torch.device('cpu')).numpy()

train_x = np.add(train_x, 1)
train_x = np.multiply(train_x, 100)
train_x =  np.round_(train_x, 0)
train_x = train_x.astype(int)

#Uncomment this section for sparse representation
#train_x_indicies = np.zeros((np.shape(train_x)[0] * len(joint_connections) * 11, 5))
#train_x_values = np.zeros((np.shape(train_x)[0] * len(joint_connections) * 11))
#train_x_shape = np.array([np.shape(train_x)[0], 200, 200, 200, 1])

#Comment this out for sparse representation
train_x_3D = np.zeros((np.shape(train_x)[0], 200, 200, 200, 1), dtype=np.ubyte)

for example in range(np.shape(train_x)[0]):
	for connection in range(len(joint_connections)):
		for dimension in range(3):
			joint1 = train_x[example][joint_connections[connection][0]]
			joint2 = train_x[example][joint_connections[connection][1]]
			dir_vect[dimension] = joint1[dimension] - joint2[dimension]
		for t in np.arange(0, 1.1, 0.1):
			x_val = int(round(joint2[0] + (dir_vect[0] * t)))
			y_val = int(round(joint2[1] + (dir_vect[1] * t)))
			z_val = int(round(joint2[2] + (dir_vect[2] * t)))
			
			#Comment this out for sparse representation
			train_x_3D[example][x_val][y_val][z_val][0] = 1
			
			#Uncomment this section for sparse representation
			#train_x_indicies[example * connection * int(t * 10)] = np.array([example, x_val, y_val, z_val, 0])
			#train_x_values[example * connection * int(t * 10)] = 1

#Comment this out for sparse representation
torch.save(torch.from_numpy(train_x_3D), "train_x_3D.pt")

#Uncomment this section for sparse representation
#torch.save(torch.from_numpy(train_x_indicies), "train_x_indicies.pt")
#torch.save(torch.from_numpy(train_x_values), "train_x_values.pt")
#torch.save(torch.from_numpy(train_x_shape), "train_x_shape.pt")

train_x_3d = None
train_x = None


test_x = torch.load("test_x.pt", map_location=torch.device('cpu')).numpy()

test_x = np.add(test_x, 1)
test_x = np.multiply(test_x, 100)
test_x =  np.round_(test_x, 0)
test_x = test_x.astype(int)

#Uncomment this section for sparse representation
#test_x_indicies = np.zeros((np.shape(test_x)[0] * len(joint_connections) * 11, 5))
#test_x_values = np.zeros((np.shape(test_x)[0] * len(joint_connections) * 11))
#test_x_shape = np.array([np.shape(test_x)[0], 200, 200, 200, 1])

#Comment this out for sparse representation
test_x_3D = np.zeros((np.shape(test_x)[0], 200, 200, 200, 1), dtype=np.ubyte)

for example in range(np.shape(test_x)[0]):
	for connection in range(len(joint_connections)):
		for dimension in range(3):
			joint1 = test_x[example][joint_connections[connection][0]]
			joint2 = test_x[example][joint_connections[connection][1]]
			dir_vect[dimension] = joint1[dimension] - joint2[dimension]
		for t in np.arange(0, 1.1, 0.1):
			x_val = int(round(joint2[0] + (dir_vect[0] * t)))
			y_val = int(round(joint2[1] + (dir_vect[1] * t)))
			z_val = int(round(joint2[2] + (dir_vect[2] * t)))
			
			#Comment this out for sparse representation
			test_x_3D[example][x_val][y_val][z_val][0] = 1

			#Uncomment this section for sparse representation		
			#test_x_indicies[example * connection * int(t * 10)] = np.array([example, x_val, y_val, z_val, 0])
			#test_x_values[example * connection * int(t * 10)] = 1

#Comment this out for sparse representation
torch.save(torch.from_numpy(test_x_3D), "test_x_3D.pt")

#Uncomment this section for sparse representation
#torch.save(torch.from_numpy(test_x_indicies), "test_x_indicies.pt")
#torch.save(torch.from_numpy(test_x_values), "test_x_values.pt")
#torch.save(torch.from_numpy(test_x_shape), "test_x_shape.pt")

