import tensorflow as tf
import numpy as np

# load dataset, loss function for optimization/evaluation for CIFAR100
# from src.Benchmarks.CIFAR100 import dataset
# dataset = dataset.preprocess_dataset()

from src import utils

# load dataset, loss function for optimization/evaluation for CIFAR100
validation_images, validation_labels, lossfn = None, None, None
def load_data(dataset):
	global validation_images
	global validation_labels
	global lossfn
 
	validation_images, validation_labels = dataset.validation_images, dataset.validation_labels
	lossfn = dataset.lossfn


# unnormalized
def observer(NN_object, tIndices):
    random_batch_validation_images, random_batch_validation_labels = validation_images[tIndices], validation_labels[tIndices]
    test_loss = lossfn(random_batch_validation_labels, NN_object.nn(random_batch_validation_images))

    return test_loss

def graph_history(history):
    integers = [i for i in range(1, (len(history))+1)]
    x = [j * rr for j in integers]
    y = history
    plt.scatter(x, history, s=20)

    plt.title("PD CIFAR10")
    plt.tight_layout()
    plt.show(block=True), plt.close()
    plt.close('all')