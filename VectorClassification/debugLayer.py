import tensorflow as tf
from tensorflow.python.framework import tensor_shape
import numpy

class DebugLayer(tf.keras.layers.Layer):

	def __init__(self):
		print("---DEBUG LAYER INITIALIZING---")
		super(DebugLayer, self).__init__()
	
	def build(self, input_shape):
		print("---DEBUG LAYER BUILDING---")
		input_shape=input_shape[1:]
		print("Shape:")
		print(input_shape)
		self.w = tf.constant(1,shape=input_shape, dtype="float32")
		self.built = True
	
	def call(self, input):
		print("---DEBUG LAYER RUNNING---")
		print(input)
		print(self.w)
		return tf.multiply(input, self.w)

	def compute_output_shape(self, input_shape):
		print("---DEBUG LAYER COMPUTING OUTPUT SHAPE---")
		return input_shape
