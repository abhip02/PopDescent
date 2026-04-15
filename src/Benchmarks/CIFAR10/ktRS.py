import keras_tuner
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

import os
import csv


# grad_steps = 25 trials * 2 executions each trial * 782 batches per execution + (5 * 782) for final training = 43000 steps


# import models # for CIFAR100
from models import ktRSm
from src import utils
from src.Benchmarks.ktRSUtils import ktRS_parameters, define_tuner, train_model, evaluation, graph_history


# DATA
import dataset
dataset = dataset.preprocess_dataset()

train_images, train_labels, validation_images, validation_labels, test_images, test_labels = dataset.train_images, dataset.train_labels, dataset.validation_images, dataset.validation_labels, dataset.test_images, dataset.test_labels


# use argparse to parse if with regularization or not
def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--with_reg", action="store_true")
	parser.add_argument("--with_sch", action="store_true")
	args = parser.parse_args()
	with_reg = args.with_reg
	with_sch = args.with_sch
 
 	# get training parameters used
	training_parameters = ktRS_parameters.return_parameters()
 
 
	for i in range(len(training_parameters.SEED)):
		print(""), print("MAJOR ITERATION %s: " % (i+1)), print("")
    
		# set seed
		utils.Set_Seed(training_parameters.SEED[i])
  
  		# define tuner
		if with_sch:
			build_model = ktRSm
		build_model = ktRSm(training_parameters, with_reg, with_sch)
		print("random search")
		tuner = keras_tuner.RandomSearch(
			hypermodel=build_model,
			objective="val_accuracy",
			max_trials=training_parameters.max_trials,
			executions_per_trial=training_parameters.executions_per_trial,
			overwrite=True,
			project_name="%s: %s" % (dataset.name, training_parameters.SEED[i])
		)

		# doesn't work, "module cannot be called" error
		# tuner = define_tuner(build_model, training_parameters)

		# start timer
		import time
		start_time = time.time()  

		# search
		tuner.search(train_images, train_labels, validation_data=(validation_images, validation_labels), batch_size=training_parameters.batch_size)

		# retrieve and train best model
		best_hps = tuner.get_best_hyperparameters(5)
		model = build_model(best_hps[0])

		# Use early stopping
		callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=training_parameters.patience)

		# TRAIN Model
		train_model.train_KT_model(model, training_parameters, train_images, train_labels, validation_images, validation_labels, callback)
		
		# end timer
		time_lapsed = time.time() - start_time

		# evaluating model on test and train data
		# generating psuedo random eval set
		batch_size = training_parameters.batch_size
		random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = utils.random_eval_set_generator(dataset, batch_size)

		# evaluating on train, test images
		train_loss, test_loss = evaluation.evaluate_model(model, random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels)

		print("unnormalized train loss: %s" % train_loss)
		print("unnormalized test loss: %s" % test_loss)

		# writing data to excel file
		data = [[test_loss, train_loss, training_parameters.max_trials, time_lapsed, training_parameters.SEED[i]]]

		csv_file = '../ktRS_%s_Benchmark_reg=%s_sch=%s.csv' % (dataset.name, with_reg, with_sch)
		with open(csv_file, 'a', newline = '') as file:
			writer = csv.writer(file)
			writer.writerows(data)
  
if __name__ == "__main__":
	main()