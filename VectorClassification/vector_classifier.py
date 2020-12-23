import tensorflow as tf
import torch
import numpy as np
import sys
import Performance as per
from sklearn.metrics import confusion_matrix 
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

#load dataset, training and testing, examples(x) and labels(y)
if sys.argv[1] == "1":
	train_x = torch.load("train_x_connected.pt", map_location=torch.device('cpu')).numpy()
	test_x = torch.load("test_x_connected.pt", map_location=torch.device('cpu')).numpy()
	print("\n**********\nUSING CONNECTED DATASET\n**********")
else:
	train_x = torch.load("train_x.pt", map_location=torch.device('cpu')).numpy()
	test_x = torch.load("test_x.pt", map_location=torch.device('cpu')).numpy()
	
train_y = torch.load("train_y.pt", map_location=torch.device('cpu')).numpy()
test_y = torch.load("test_y.pt", map_location=torch.device('cpu')).numpy()

#reshape input for Conv2D layer, expects each input to be 3D
# (num_examples, num_points, num_coordinates_per_point, extra_dimension)

shape1 = np.shape(train_x)
train_x = np.reshape(train_x, (shape1[0], shape1[1], shape1[2], 1))
shape2 = np.shape(test_x)
test_x = np.reshape(test_x, (shape2[0], shape2[1], shape2[2], 1))


model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(32, 2, strides=1, padding='valid', input_shape=(shape1[1], 3, 1)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(7, activation=tf.nn.softmax)])
	
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.summary()

model.fit(train_x, train_y, verbose=2, batch_size=2, epochs=70)

model.evaluate(test_x,  test_y, batch_size=2, verbose=2)
ypred =  np.argmax(model.predict(test_x, batch_size=2),axis=-1)

print(tf.math.confusion_matrix(test_y,ypred))
print("0: fighting\n 1: front\n 2: ready\n 3: cat\n 4: horse \n 5: hicho \n 6: seiza")

ACC, TPR, TNR, PPV, NPV, FPR = per.GetPerformanceMetrics(test_y, ypred, weighted=True)


print("Accuracy: ",ACC)
print("Recall: ",TPR)
print("Specificity: ",TNR)
print("Precision: ",PPV)
print("Negative Predictive Value: ",NPV)
print("FP rate(fall-out): ",FPR)
