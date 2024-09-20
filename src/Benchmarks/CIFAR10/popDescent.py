import csv
import os

import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

# Typing
import typing
from typing import TypeVar, Generic
from collections.abc import Callable

from tqdm import tqdm
from collections import namedtuple
import statistics
import dataclasses
from dataclasses import dataclass
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models
import tensorflow as tf



# from populationDescent import populationDescent
import models
# PD functions
from src import utils, populationDescent
from src.Benchmarks.popDescentUtils import ParametersClass, popDescent_parameters, optimization, randomization, observation, evaluation



# import warnings
# warnings.filterwarnings("ignore")

# NN_Individual = namedtuple("NN_Individual", ["nn", "opt_obj", "LR_constant", "reg_constant"])
tf.config.run_functions_eagerly(True)

# DATA
import dataset
dataset = dataset.preprocess_dataset()
optimization.load_data(dataset)








# use argparse to parse if with regularization or not
# python3 -m KT_RandomSearch_CIFAR100_Benchmark_with_regularization --with_reg

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--with_reg", action="store_true")
	args = parser.parse_args()
	with_reg = args.with_reg

	# get training parameters used
	training_parameters = popDescent_parameters.return_parameters()

	# main test code
	for i in range(len(training_parameters.SEED)):
		print(""), print("MAJOR ITERATION %s: " % (i+1)), print("")

		utils.Set_Seed(training_parameters.SEED[i])

		# create_Parameters_NN_object creates pdFunctionsClass object to call proper popDescent Functions
		# pass in with_reg arg to choose which model to build
		Parameters_object, model_num = ParametersClass.create_Parameters_NN_object(models, training_parameters, with_reg)

		#creating lists to store data
		loss_data, acc_data, total_test_loss, batch_test_loss, total_test_acc = [], [], [], [], []

		# measure time
		import time
		start_time = time.time()

		#RUNNING OPTIMIZATION
		optimized_population, lfitnesses, vfitnesses, history = populationDescent.populationDescent(Parameters=Parameters_object, training_parameters=training_parameters)

		#measuring how long optimization took
		time_lapsed = time.time() - start_time
		print(""), print(""), print("time:"), print("--- %s seconds ---" % time_lapsed), print(""), print("")

		# evaluate from outside
		total_hist, batch_hist = [], []

		# returns UNNORMALIZED training and test loss, data chosen with a random seed
		best_train_model_loss, best_test_model_loss = ParametersClass.Parameter_class_evaluator(optimized_population)

		# writing data to excel file
		data = [[training_parameters, time_lapsed, with_reg]]
		print(data)

		csv_file = '../pd_%s_Benchmark_reg=%s.csv' % (dataset.name, with_reg)
		with open(csv_file, 'a', newline = '') as file:
			writer = csv.writer(file)
			writer.writerows(data)

		# graph data
		if training_parameters.graph:
			graph_history(history)



if __name__ == "__main__":
	main()


