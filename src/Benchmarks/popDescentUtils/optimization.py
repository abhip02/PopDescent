import tensorflow as tf
import numpy as np

from tqdm import tqdm
from src import utils

# load dataset, loss function for optimization/evaluation for CIFAR100
train_images, train_labels, validation_images, validation_labels, lossfn = None, None, None, None, None
def load_data(dataset):
	global train_images
	global train_labels
	global validation_images
	global validation_labels
	global lossfn
 
	train_images, train_labels, validation_images, validation_labels = dataset.train_images, dataset.train_labels, dataset.validation_images, dataset.validation_labels
	lossfn = dataset.lossfn


# calls gradient steps to train model, returns NORMALIZED training and validation loss
def optimizer(NN_object, training_parameters):

	# classification_NN_compiler(NN_object.nn)
	batches = training_parameters.batches
	batch_size = training_parameters.batch_size
	epochs = training_parameters.epochs
	normalized_training_loss, normalized_validation_loss = [], []

	optimizer = NN_object.opt_obj

	indices = np.random.choice(len(train_images), size = (batch_size*batches, ), replace=False)
	vIndices = np.random.choice(len(validation_images), size = (batch_size*10, ), replace=False)

	# batching dataset
	random_batch_train_images, random_batch_train_labels = train_images[indices], train_labels[indices]
	random_batch_validation_images, random_batch_validation_labels = validation_images[vIndices], validation_labels[vIndices]

	model_loss = gradient_steps(lossfn, random_batch_train_images, random_batch_train_labels, batch_size, epochs, NN_object)

	validation_loss = lossfn(random_batch_validation_labels, NN_object.nn(random_batch_validation_images))
	tf.print("validation loss: %s" % validation_loss), print("")

	normalized_training_loss.append(2/(2+(model_loss)))
	normalized_training_loss = np.array(normalized_training_loss)

	normalized_validation_loss.append(2/(2+(validation_loss)))
	normalized_validation_loss = np.array(normalized_validation_loss)

	# print(""), print("normalized training loss: %s" % normalized_training_loss)
	# print("normalized validation loss: %s" % normalized_validation_loss)

	#print(model_loss)
	return normalized_training_loss, normalized_validation_loss

# function optimized to take gradient steps with tf variables
@tf.function
def gradient_steps(lossfn, training_set, labels, batch_size, epochs, NN_object):

	with tf.device('/device:GPU:0'):
		for e in range(epochs):
			for x_batch, y_batch in tqdm(tf.data.Dataset.from_tensor_slices((training_set, labels)).batch(batch_size)): # need this for tf.GradientTape to work like model.fit
				with tf.GradientTape() as tape:

					# make a prediction using the model and then calculate the loss
					model_loss = lossfn(y_batch, NN_object.nn(x_batch))
				
					# use regularization constant
					regularization_loss = NN_object.nn.losses
					if len(regularization_loss) == 0:
						reg_loss = 0
					else:
						reg_loss = regularization_loss[0]

					mreg_loss = reg_loss * NN_object.reg_constant
					total_training_loss = NN_object.LR_constant * (model_loss + mreg_loss) # LR + REG randomization

				# calculate the gradients using our tape and then update the model weights
				grads = tape.gradient(total_training_loss, NN_object.nn.trainable_variables) ## with LR randomization and regularization loss
				# tf.print(grads)
				# loop over gradients as a list, for each element do tf.absolutevalue and get tf.reduceMean
				NN_object.opt_obj.apply_gradients(zip(grads, NN_object.nn.trainable_variables))
    
	tf.print("training loss: %s" % model_loss) ## remove this --> put nothing (put at recombination)
	return model_loss