import tensorflow as tf
import numpy as np

from src import utils

# load dataset, loss function for optimization/evaluation for CIFAR100
from src.Benchmarks.CIFAR100 import dataset
dataset = dataset.preprocess_dataset()

train_images, train_labels, test_images, test_labels = dataset.train_images, dataset.train_labels, dataset.test_images, dataset.test_labels

from src import utils
lossfn = dataset.lossfn

def evaluator(model, training_parameters):
    random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = utils.random_eval_set_generator(dataset, training_parameters.batch_size)
    print(""), print(""), print("Evaluating models on test data after randomization")

    # evaluating on train, test images
    lossfn = dataset.lossfn
    # train_loss = lossfn(random_batch_train_labels, model(random_batch_train_images))
    # test_loss = lossfn(random_batch_test_labels, model(random_batch_test_images))

    train_loss = model.evaluate(random_batch_train_images, random_batch_train_labels)[0]
    test_loss = model.evaluate(random_batch_test_images, random_batch_test_labels)[0]

    print("unnormalized train loss: %s" % train_loss)
    print("unnormalized test loss: %s" % test_loss)
    # print("normalized (1/1+loss) test loss: %s" % ntest_loss)

    return train_loss, test_loss