from abc import abstractmethod

import tensorflow as tf


class NN (object):
	@staticmethod
	def add_layer (assign_list, inputs, in_size, out_size, activation_function=None):
		w = tf.Variable (tf.random_normal ([in_size, out_size]))
		w_holder = tf.placeholder (tf.float32, [in_size, out_size])
		assign_list.append (tf.assign (w, w_holder))
		b = tf.Variable (tf.zeros ([1, out_size]) + 0.1)
		b_holder = tf.placeholder (tf.float32, [1, out_size])
		assign_list.append (tf.assign (b, b_holder))
		wx_plus_b = tf.matmul (inputs, w) + b
		if activation_function is None:
			outputs = wx_plus_b
		else:
			outputs = activation_function (wx_plus_b)
		return outputs

	def __init__ (self, _test_x, _test_y, _xs, _ys, _assign_list, _loss, _accuracy, _weights, _sess, _size, _path):
		self.test_x = _test_x
		self.test_y = _test_y
		self.xs = _xs
		self.ys = _ys
		self.assign_list = _assign_list
		self.loss = _loss
		self.accuracy = _accuracy
		self.weights = _weights
		self.sess = _sess
		self.size = _size
		self.path = _path
		self.batch_size = None
		self.batch_num = None
		self.batch = None
		self.train_step = None

	@abstractmethod
	def set_batch (self, batch_size, round_repeat, start_index, end_index):
		# Assign value to self.batch_size, self.batch_num and self.batch
		# Refer to nn_mnist.py and nn_cifar10.py
		pass

	@abstractmethod
	def set_train_step (self, lr):
		# Assign value to self.train_step
		# Refer to nn_mnist.py and nn_cifar10.py
		pass
