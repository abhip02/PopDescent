import tensorflow as tf
import numpy as np
from tqdm import tqdm

from src.Benchmarks.hpSUtils import observation

# DATA
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

def optimizer(population, training_parameters, reg_list, observer_history):
    with tf.device('/device:GPU:0'):
        # TRAINING
        for i in tqdm(range(training_parameters.iterations)):

            indices = np.random.choice(len(train_images), size = (training_parameters.batch_size*training_parameters.batches, ), replace=False)
            vIndices = np.random.choice(len(validation_images), size = (training_parameters.batch_size*1, ), replace=False)

            random_batch_train_images, random_batch_train_labels = train_images[indices], train_labels[indices]
            random_batch_validation_images, random_batch_validation_labels = validation_images[vIndices], validation_labels[vIndices]

            # indices
            tIndices = np.random.choice(len(validation_images), size = (training_parameters.batch_size*1, ), replace=False)
            population_training_losses = []

            for j in range(len(population)):

                print("model %s" % (j+1))
                population[j].fit(random_batch_train_images, random_batch_train_labels, validation_data = (random_batch_validation_images, random_batch_validation_labels), epochs=training_parameters.epochs, verbose=1, batch_size=training_parameters.batch_size)

                print("regularization_amount: %s" % reg_list[j])
                print("learning rate: %s" % population[j].optimizer.learning_rate)
                print("")

                # population_training_losses.append(training_loss)

                # observing optimization progress
                if (i%training_parameters.rr)==0:
                    if i!=(training_parameters.iterations-1):
                        individual_observer_loss = observation.observer(population[j], tIndices)
                        population_training_losses.append(individual_observer_loss)


            if (i%training_parameters.rr)==0:
                if population_training_losses:
                    population_training_losses = np.array(population_training_losses)
                    observer_history.append(np.min(population_training_losses))
                    population_training_losses = []
    
    return population, observer_history
