import tensorflow as tf
import numpy as np

from collections import namedtuple

NN_Individual = namedtuple("NN_Individual", ["nn", "opt_obj", "LR_constant", "reg_constant"])

# For Google Colab Keras 3: need to pass new optimizer with new model
def clone_optimizer(opt):
	# Shallow clone with the same config
	config = opt.get_config()
	return opt.__class__.from_config(config)


def randomizer(NN_object, normalized_amount, input_factor):
	# original: (0, 1e-3), (0, normalized_amount), (0, normalized amount)

	with tf.device('/device:GPU:0'):
		factor = input_factor

		# randomizing NN weights
		model_clone = tf.keras.models.clone_model(NN_object.nn)
		# model_clone.set_weights(np.array(NN_object.nn.get_weights()))
		model_clone.set_weights(NN_object.nn.get_weights())

		mu, sigma = 0, (1e-2) #1e-4 for sin
		# gNoise = (np.random.normal(mu, sigma))*(normalized_amount)
		gNoise = (np.random.normal(mu, sigma))*(normalized_amount)

		# weights = np.array((NN_object.nn.get_weights()))
		weights = (NN_object.nn.get_weights())
		randomized_weights = [w + gNoise for w in NN_object.nn.get_weights()]

		model_clone.set_weights(randomized_weights)

		# randomizing regularization rate
		mu, sigma = 0, (normalized_amount*factor) # 0.7, 1 #10 # 0.3
		randomization = 2**(np.random.normal(mu, sigma))
		new_reg_constant = (NN_object.reg_constant) * randomization

		# randomizing learning_rates
		mu, sigma = 0, (normalized_amount*factor) # 0.7, 1, 10,x 0.3
		randomization = 2**(np.random.normal(mu, sigma))
		new_LR_constant = (NN_object.LR_constant) * randomization
  
		new_optimizer = clone_optimizer(NN_object.opt_obj)

		new_NN_Individual = NN_Individual(model_clone, new_optimizer, new_LR_constant, new_reg_constant) # without randoimzed LR

	return new_NN_Individual