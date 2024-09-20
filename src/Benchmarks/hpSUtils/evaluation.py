import tensorflow as tf
import numpy as np

from tqdm import tqdm
import statistics

from src import utils

# DATA
# load dataset, loss function for optimization/evaluation for CIFAR100
train_images, train_labels, test_images, test_labels, lossfn = None, None, None, None, None
def load_data(dataset_input):
    global dataset
    dataset = dataset_input
    # lossfn = dataset.lossfn


def evaluator(population, training_parameters, reg_list):
    with tf.device('/device:GPU:0'):
        # # Evaluating on test data
        random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = utils.random_eval_set_generator(dataset, training_parameters.batch_size)

        training_losses, evaluation_losses, evaluation_accuracies = [], [], []

        for h in range(len(population)):
            print("model %s" % (h+1))

            training_loss, training_accuracy = population[h].evaluate(random_batch_train_images, random_batch_train_labels, batch_size=training_parameters.batch_size)
            test_loss, test_acc = population[h].evaluate(random_batch_test_images, random_batch_test_labels, batch_size=training_parameters.batch_size)

            # ntest_loss = 1/(1+test_loss)
            # ntest_loss = np.array(ntest_loss)

            training_losses.append(training_loss)

            evaluation_losses.append(test_loss)
            evaluation_accuracies.append(test_acc)


        best_training_model_loss_unnormalized = np.min(training_losses)

        best_test_model_loss = np.min(evaluation_losses)
        best_index = evaluation_losses.index(best_test_model_loss)

        best_lr = (population[best_index]).optimizer.learning_rate
        best_reg_amount = reg_list[best_index]

        evaluation_losses = np.array(evaluation_losses)

        avg_test_loss = statistics.mean(evaluation_losses)
        test_acc_data = statistics.mean(evaluation_accuracies)

    return best_test_model_loss, avg_test_loss, best_training_model_loss_unnormalized, best_lr, best_reg_amount