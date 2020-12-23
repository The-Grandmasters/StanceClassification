import tensorflow as tf
import joints
import numpy

c_joints = [
	[joints.HEAD_TOP, joints.HEAD],
	[joints.HEAD, joints.NECK, joints.SHOULDER_RIGHT, joints.SHOULDER_LEFT, joints.SPINE],
	[joints.NECK, joints.SHOULDER_RIGHT, joints.ELBOW_RIGHT],
	[joints.SHOULDER_RIGHT, joints.ELBOW_RIGHT, joints.WRIST_RIGHT],
	[joints.ELBOW_RIGHT, joints.WRIST_RIGHT],
	[joints.NECK, joints.SHOULDER_LEFT, joints.ELBOW_LEFT],
	[joints.SHOULDER_LEFT, joints.ELBOW_LEFT, joints.WRIST_LEFT],
	[joints.ELBOW_LEFT, joints.WRIST_LEFT],
	[joints.PELVIS, joints.HIP_RIGHT, joints.KNEE_RIGHT],
	[joints.HIP_RIGHT, joints.KNEE_RIGHT, joints.ANKLE_RIGHT],
	[joints.KNEE_RIGHT, joints.ANKLE_RIGHT],
	[joints.PELVIS, joints.HIP_LEFT, joints.KNEE_LEFT],
	[joints.HIP_LEFT, joints.KNEE_LEFT, joints.ANKLE_LEFT],
	[joints.KNEE_LEFT, joints.ANKLE_LEFT],
	[joints.SPINE, joints.PELVIS, joints.HIP_RIGHT, joints.HIP_LEFT],
	[joints.NECK, joints.SPINE, joints.PELVIS],
	[joints.HEAD_TOP, joints.HEAD, joints.NECK]
	]

c_num = [2, 5, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 4, 3, 3]

class PropeIuncturam(tf.keras.layers.Layer):

	def __init__(self):
		super(PropeIuncturam, self).__init__()

	def build(self, input_shape):
		self.w_ht = self.add_weight(shape=(2, 3), initializer = 'random_normal', trainable = True)
		self.w_nk = self.add_weight(shape=(5, 3), initializer = 'random_normal', trainable = True)
		self.w_rs = self.add_weight(shape=(3, 3), initializer = 'random_normal', trainable = True)
		self.w_re = self.add_weight(shape=(3, 3), initializer = 'random_normal', trainable = True)
		self.w_rw = self.add_weight(shape=(2, 3), initializer = 'random_normal', trainable = True)
		self.w_ls = self.add_weight(shape=(3, 3), initializer = 'random_normal', trainable = True)
		self.w_le = self.add_weight(shape=(3, 3), initializer = 'random_normal', trainable = True)
		self.w_lw = self.add_weight(shape=(2, 3), initializer = 'random_normal', trainable = True)
		self.w_rh = self.add_weight(shape=(3, 3), initializer = 'random_normal', trainable = True)
		self.w_rk = self.add_weight(shape=(3, 3), initializer = 'random_normal', trainable = True)
		self.w_ra = self.add_weight(shape=(2, 3), initializer = 'random_normal', trainable = True)
		self.w_lh = self.add_weight(shape=(3, 3), initializer = 'random_normal', trainable = True)
		self.w_lk = self.add_weight(shape=(3, 3), initializer = 'random_normal', trainable = True)
		self.w_la = self.add_weight(shape=(2, 3), initializer = 'random_normal', trainable = True)
		self.w_pv = self.add_weight(shape=(4, 3), initializer = 'random_normal', trainable = True)
		self.w_sp = self.add_weight(shape=(3, 3), initializer = 'random_normal', trainable = True)
		self.w_hd = self.add_weight(shape=(3, 3), initializer = 'random_normal', trainable = True)
		
		self.b_ht = self.add_weight(shape=(2, 3), initializer = 'zeros', trainable = True)
		self.b_nk = self.add_weight(shape=(5, 3), initializer = 'zeros', trainable = True)
		self.b_rs = self.add_weight(shape=(3, 3), initializer = 'zeros', trainable = True)
		self.b_re = self.add_weight(shape=(3, 3), initializer = 'zeros', trainable = True)
		self.b_rw = self.add_weight(shape=(2, 3), initializer = 'zeros', trainable = True)
		self.b_ls = self.add_weight(shape=(3, 3), initializer = 'zeros', trainable = True)
		self.b_le = self.add_weight(shape=(3, 3), initializer = 'zeros', trainable = True)
		self.b_lw = self.add_weight(shape=(2, 3), initializer = 'zeros', trainable = True)
		self.b_rh = self.add_weight(shape=(3, 3), initializer = 'zeros', trainable = True)
		self.b_rk = self.add_weight(shape=(3, 3), initializer = 'zeros', trainable = True)
		self.b_ra = self.add_weight(shape=(2, 3), initializer = 'zeros', trainable = True)
		self.b_lh = self.add_weight(shape=(3, 3), initializer = 'zeros', trainable = True)
		self.b_lk = self.add_weight(shape=(3, 3), initializer = 'zeros', trainable = True)
		self.b_la = self.add_weight(shape=(2, 3), initializer = 'zeros', trainable = True)
		self.b_pv = self.add_weight(shape=(4, 3), initializer = 'zeros', trainable = True)
		self.b_sp = self.add_weight(shape=(3, 3), initializer = 'zeros', trainable = True)
		self.b_hd = self.add_weight(shape=(3, 3), initializer = 'zeros', trainable = True)

		self.built = True

	def call(self, input):

		input = tf.reduce_sum(input, axis=3)
		
		a=1

		j_ht = tf.gather(input, [joints.HEAD_TOP, joints.HEAD], axis=a)
		j_nk = tf.gather(input, [joints.HEAD, joints.NECK, joints.SHOULDER_RIGHT, joints.SHOULDER_LEFT, joints.SPINE], axis=a)
		j_rs = tf.gather(input, [joints.NECK, joints.SHOULDER_RIGHT, joints.ELBOW_RIGHT], axis = a)
		j_re = tf.gather(input, [joints.SHOULDER_RIGHT, joints.ELBOW_RIGHT, joints.WRIST_RIGHT], axis = a)
		j_rw = tf.gather(input, [joints.ELBOW_RIGHT, joints.WRIST_RIGHT], axis = a)
		j_ls = tf.gather(input, [joints.NECK, joints.SHOULDER_LEFT, joints.ELBOW_LEFT], axis = a)
		j_le = tf.gather(input, [joints.SHOULDER_LEFT, joints.ELBOW_LEFT, joints.WRIST_LEFT], axis = a)
		j_lw = tf.gather(input, [joints.ELBOW_LEFT, joints.WRIST_LEFT], axis = a)
		j_rh = tf.gather(input, [joints.PELVIS, joints.HIP_RIGHT, joints.KNEE_RIGHT], axis = a)
		j_rk = tf.gather(input, [joints.HIP_RIGHT, joints.KNEE_RIGHT, joints.ANKLE_RIGHT], axis = a)
		j_ra = tf.gather(input, [joints.KNEE_RIGHT, joints.ANKLE_RIGHT], axis = a)
		j_lh = tf.gather(input, [joints.PELVIS, joints.HIP_LEFT, joints.KNEE_LEFT], axis = a)
		j_lk = tf.gather(input, [joints.HIP_LEFT, joints.KNEE_LEFT, joints.ANKLE_LEFT], axis = a)
		j_la = tf.gather(input, [joints.KNEE_LEFT, joints.ANKLE_LEFT], axis = a)
		j_pv = tf.gather(input, [joints.SPINE, joints.PELVIS, joints.HIP_RIGHT, joints.HIP_LEFT], axis = a)
		j_sp = tf.gather(input, [joints.NECK, joints.SPINE, joints.PELVIS], axis = a)
		j_hd = tf.gather(input, [joints.HEAD_TOP, joints.HEAD, joints.NECK], axis = a)
		
		a=1
		r_ht = tf.reduce_sum(tf.add(tf.multiply(self.w_ht, j_ht), self.b_ht), axis=a)
		r_nk = tf.reduce_sum(tf.add(tf.multiply(self.w_nk, j_nk), self.b_nk), axis=a)
		r_rs = tf.reduce_sum(tf.add(tf.multiply(self.w_rs, j_rs), self.b_rs), axis=a)
		r_re = tf.reduce_sum(tf.add(tf.multiply(self.w_re, j_re), self.b_re), axis=a)
		r_rw = tf.reduce_sum(tf.add(tf.multiply(self.w_rw, j_rw), self.b_rw), axis=a)
		r_ls = tf.reduce_sum(tf.add(tf.multiply(self.w_ls, j_ls), self.b_ls), axis=a)
		r_le = tf.reduce_sum(tf.add(tf.multiply(self.w_le, j_le), self.b_le), axis=a)
		r_lw = tf.reduce_sum(tf.add(tf.multiply(self.w_lw, j_lw), self.b_lw), axis=a)
		r_rh = tf.reduce_sum(tf.add(tf.multiply(self.w_rh, j_rh), self.b_rh), axis=a)
		r_rk = tf.reduce_sum(tf.add(tf.multiply(self.w_rk, j_rk), self.b_rk), axis=a)
		r_ra = tf.reduce_sum(tf.add(tf.multiply(self.w_ra, j_ra), self.b_ra), axis=a)
		r_lh = tf.reduce_sum(tf.add(tf.multiply(self.w_lh, j_lh), self.b_lh), axis=a)
		r_lk = tf.reduce_sum(tf.add(tf.multiply(self.w_lk, j_lk), self.b_lk), axis=a)
		r_la = tf.reduce_sum(tf.add(tf.multiply(self.w_la, j_la), self.b_la), axis=a)
		r_pv = tf.reduce_sum(tf.add(tf.multiply(self.w_pv, j_pv), self.b_pv), axis=a)
		r_sp = tf.reduce_sum(tf.add(tf.multiply(self.w_sp, j_sp), self.b_sp), axis=a)
		r_hd = tf.reduce_sum(tf.add(tf.multiply(self.w_hd, j_hd), self.b_hd), axis=a)
		
		return tf.concat([r_ht, r_nk, r_rs, r_re, r_rw, r_ls, r_le, r_lw, r_rh, r_rk, r_ra, r_lh, r_lk, r_la, r_pv, r_sp, r_hd], axis=1)

	def compute_output_shape(self, input_shape):
		return (51,)

