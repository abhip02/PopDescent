import tensorflow as tf
from tensorflow import keras

from keras.callbacks import TensorBoard

import numpy as np
import random

import os
import csv


from hyperopt import fmin, tpe, hp

from models import sklHO
from src import utils
from src.Benchmarks.sklHOUtils import sklHO_parameters, space_definition, objective_function, evaluation

# DATA
import dataset
dataset = dataset.preprocess_dataset()

train_images, train_labels, validation_images, validation_labels, test_images, test_labels = dataset.train_images, dataset.train_labels, dataset.validation_images, dataset.validation_labels, dataset.test_images, dataset.test_labels


# GC
# from google.colab import auth
# from google.cloud import storage

# # Authenticate with Google Cloud
# auth.authenticate_user()

# project_id = 'schedulesktrsfmnist'
# bucket_name = 'schedulesktrsfmnist'

# client = storage.Client(project=project_id)
# bucket = client.get_bucket(bucket_name)



# grad_steps = 25 trials * 2 executions each trial * 782 batches per execution + (5 * 782) for final training = 43000 steps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_reg", action="store_true")
    args = parser.parse_args()
    with_reg = args.with_reg

    # get training parameters used
    training_parameters = sklHO_parameters.return_parameters()

    for i in range(len(training_parameters.SEED)):
        print(""), print("MAJOR ITERATION %s: " % (i+1)), print("")

        # set seed
        utils.Set_Seed(training_parameters.SEED[i])

        # define search space
        space = space_definition.define_space(training_parameters)

        # define objective function
        objective = objective_function.return_objective(sklHO, training_parameters)
        
        # start timer
        import time
        start_time = time.time()
        
        # Run Hyperopt to find the best hyperparameters
        max_trials = training_parameters.max_trials # 37
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_trials)
        
        # retrieve and build best model
        model = sklHO(with_reg, best)
    
        # Use early stopping
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=training_parameters.patience)
    
        # TRAIN Model
        print("")
        print("TRAINING")
        # hist = model.fit(train_images, train_labels, batch_size=training_parameters.batch_size, validation_data=(validation_images, validation_labels), epochs=training_parameters.epochs, callbacks=[callback])

        # end timer
        time_lapsed = time.time() - start_time

        # # for graphs
        # grad_steps = [i * 780 for i in hist.history['val_loss']]
        
        # evaluate model
        train_loss, test_loss = evaluation.evaluator(model, training_parameters)
        
        # writing data to excel file
        data = [[test_loss, train_loss, training_parameters.max_trials, time_lapsed, training_parameters.SEED[i]]]

        csv_file = '../sklHO_%s_Benchmark_reg=%s.csv' % (dataset.name, with_reg)
        with open(csv_file, 'a', newline = '') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        

if __name__ == "__main__":
    main()