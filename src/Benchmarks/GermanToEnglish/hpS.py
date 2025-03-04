# use Python 3.9
# python3.9 -m venv env
# source new3.9/bin/activate

import random
import math
import numpy as np

# Typing
import typing
from typing import TypeVar, Generic
from collections.abc import Callable

from tqdm import tqdm
import tensorflow as tf
import csv

# from populationDescent import populationDescent
import models
# PD functions
from src import utils
from src.Benchmarks.hpSUtils import optimization, observation, evaluation, hpS_parameters

# DATA
import dataset
dataset = dataset.preprocess_dataset()
optimization.load_data(dataset)
observation.load_data(dataset)
evaluation.load_data(dataset)

train_images, train_labels, validation_images, validation_labels, test_images, test_labels = dataset.train_images, dataset.train_labels, dataset.validation_images, dataset.validation_labels, dataset.test_images, dataset.test_labels



def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--with_reg", action="store_true")
	args = parser.parse_args()
	with_reg = args.with_reg

	# get training parameters used
	training_parameters = hpS_parameters.return_parameters()

	for i in range(len(training_parameters.SEED)):
		print(""), print("MAJOR ITERATION %s: " % (i+1)), print("")

		# set seed
		utils.Set_Seed(training_parameters.SEED[i])

		# observer history
		observer_history = []

		# start timer
		import time
		start_time = time.time()
	
		# load model
		population, reg_list = models.hpS(with_reg)

		# train model
		population, observer_history = optimization.optimizer(population, training_parameters, reg_list, observer_history)
		time_lapsed = time.time() - start_time

		# evaluate model
		best_test_model_loss, avg_test_loss, best_training_model_loss_unnormalized, best_lr, best_reg_amount = evaluation.evaluator(population, training_parameters, reg_list)

		# writing data to excel file
		data = [[best_test_model_loss, best_training_model_loss_unnormalized, best_lr, best_reg_amount, training_parameters.grad_steps, training_parameters.iterations, training_parameters.epochs, training_parameters.batches, training_parameters.batch_size, time_lapsed, training_parameters.SEED[i]]]

		csv_file = '../hpS_%s_Benchmark_reg=%s.csv' % (dataset.name, with_reg)
		with open(csv_file, 'a', newline = '') as file:
			writer = csv.writer(file)
			writer.writerows(data)

		# graphing data
		if training_parameters.graph:
			graph_history(observer_history, trial, model_string, training_loss_data_string, test_loss_data_string, best_lr_data, best_reg_amount_string)

		
if __name__ == "__main__":
	main()