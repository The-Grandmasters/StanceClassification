import tensorflow as tf
import torch
import numpy as np
import prope_iuncturam as ip
import debugLayer as dl
import Performance as per
from sklearn.metrics import confusion_matrix

#load dataset, training and testing, examples(x) and labels(y)
train_x = torch.load("train_x.pt", map_location=torch.device('cpu')).numpy()
train_y = torch.load("train_y.pt", map_location=torch.device('cpu')).numpy()
test_x = torch.load("test_x.pt", map_location=torch.device('cpu')).numpy()
test_y = torch.load("test_y.pt", map_location=torch.device('cpu')).numpy()

#reshape input for Conv2D layer, expects each input to be 3D
# (num_examples, num_points, num_coordinates_per_point, extra_dimension)
#training (222, 17, 3, 1)
#testing (56, 17, 3, 1)
shape1 = np.shape(train_x)
train_x = np.reshape(train_x, (shape1[0], shape1[1], shape1[2], 1))
shape2 = np.shape(test_x)
test_x = np.reshape(test_x, (shape2[0], shape2[1], shape2[2], 1))

layer = ip.PropeIuncturam()

model = tf.keras.models.Sequential([
	tf.keras.layers.Input(shape=(17,3,1)),
	layer,
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(7, activation=tf.nn.softmax)])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.summary()

model.fit(train_x, train_y, verbose=2, batch_size=2, epochs=50)

model.evaluate(test_x,  test_y, batch_size=2, verbose=2)
ypred =  np.argmax(model.predict(test_x, batch_size=2),axis=-1)

print("-------------------------------")
ACC, TPR, TNR, PPV, NPV, FPR = per.GetPerformanceMetrics(test_y, ypred, weighted=True)

print("Accuracy: ",ACC)
print("Recall: ",TPR)
print("Specificity: ",TNR)
print("Precision: ",PPV)
print("Negative Predictive Value: ",NPV)
print("FP rate(fall-out): ",FPR)
print( confusion_matrix(test_y, ypred))

print("-------------------------------")

print("0: fighting\n 1: front\n 2: ready\n 3: cat\n 4: horse \n 5: hicho \n 6: seiza")
