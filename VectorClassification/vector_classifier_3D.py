import tensorflow as tf
import torch
import numpy as np
from sklearn.metrics import confusion_matrix 

#load dataset, training and testing, examples(x) and labels(y)

#Uncomment this section for spare representation of the data
#train_x_indicies = torch.load("train_x_indicies.pt", map_location=torch.device('cpu')).numpy()
#train_x_values = torch.load("train_x_values.pt", map_location=torch.device('cpu')).numpy()
#train_x_shape = torch.load("train_x_shape.pt", map_location=torch.device('cpu')).numpy()
#train_x = tf.sparse.SparseTensor(train_x_indicies, train_x_values, train_x_shape)

#Comment this out for sparse representation
train_x = torch.load("train_x_3D.pt", map_location=torch.device('cpu')).numpy()
train_x_shape = np.shape(train_x)

#Uncomment this section for spare representation of the data
#test_x_indicies = torch.load("test_x_indicies.pt", map_location=torch.device('cpu')).numpy()
#test_x_values = torch.load("test_x_values.pt", map_location=torch.device('cpu')).numpy()
#test_x_shape = torch.load("test_x_shape.pt", map_location=torch.device('cpu')).numpy()
#test_x = tf.sparse.SparseTensor(test_x_indicies, test_x_values, test_x_shape)

#Comment this out for sparse representation
test_x = torch.load("test_x_3D.pt", map_location=torch.device('cpu')).numpy()

train_y = torch.load("train_y.pt", map_location=torch.device('cpu')).numpy()
test_y = torch.load("test_y.pt", map_location=torch.device('cpu')).numpy()



model = tf.keras.models.Sequential([
	#Uncomment this section for spare representation of the data
	#tf.keras.layers.InputLayer(input_shape=(train_x_shape[1], train_x_shape[2], train_x_shape[3], train_x_shape[4]), spare=True),
	tf.keras.layers.Conv3D(20, 2, strides=1, padding='valid', input_shape=(train_x_shape[1], train_x_shape[2], train_x_shape[3], train_x_shape[4])),
	tf.keras.layers.MaxPool3D(pool_size=(4, 4, 4), strides=None, padding='valid'),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(32, activation='relu'),
	tf.keras.layers.Dense(7, activation=tf.nn.softmax)])
	
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.summary()

model.fit(train_x, train_y, verbose=2, batch_size=1, epochs=70)

model.evaluate(test_x,  test_y, batch_size=2, verbose=2)
ypred = np.argmax(model.predict(test_x, batch_size=2), axis=-1)

#print(tf.math.confusion_matrix(test_y,ypred))
#print("0: fighting\n 1: front\n 2: ready\n 3: cat\n 4: horse \n 5: hicho \n 6: seiza")
