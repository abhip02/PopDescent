import tensorflow as tf
import numpy as np

from src import utils

# DATA
# load dataset, loss function for optimization/evaluation for CIFAR100
dataset = None
def load_data(dataset_input):
    global dataset
    global lossfn

    dataset = dataset_input
    lossfn = dataset.lossfn

# returns training and test loss (UNNORMALIZED) on data chosen with random seed
def evaluator(NN_object):
    random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = utils.random_eval_set_generator(dataset, batch_size=64)
    
    print(""), print(""), print("Evaluating models on test data after randomization")

    # evaluating on train, test images
    train_loss = lossfn(random_batch_train_labels, NN_object.nn(random_batch_train_images))
    test_loss = lossfn(random_batch_test_labels, NN_object.nn(random_batch_test_images))

    # NN_object.nn.evaluate()

    ntest_loss = 2/(2+test_loss)
    print("unnormalized train loss: %s" % train_loss)
    print("unnormalized test loss: %s" % test_loss)
    # print("normalized (1/1+loss) test loss: %s" % ntest_loss)

    return train_loss, test_loss # unnormalized