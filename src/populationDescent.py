import random
import warnings
import matplotlib.pyplot as plt
import scipy
import numpy as np
from tqdm import tqdm
import statistics

# warnings.filterwarnings("ignore", category=DeprecationWarning)

def populationDescent(Parameters, training_parameters):

	# optimizer: (individual -> scalar) -> (individual -> individual)
	# normalized_objective: individual -> (float0-1)
	# new_individual: () -> individual
	# randomizer: (individual, float0-1) -> individual
	# observer: () -> graph
	# pop_size: int (number of individuals)
	# fine_tuner: (individual -> scalar) -> (individual -> individual)

	for i in tqdm(range(training_parameters.iterations), desc = "Iterations"):

		# calling OPTIMIZER
		lFitnesses, vFitnesses = Parameters.optimizer(Parameters.population, training_parameters)

		if i%(Parameters.rr)==0:
			if Parameters.CV_selection==False:
				#sorting losses (based on training)
				sorted_ind = np.argsort(lFitnesses)
				lFitnesses = lFitnesses[sorted_ind] #worst to best
				Parameters.population = Parameters.population[sorted_ind] #worst to best

				# #choosing individuals from weighted distribution (training)
				chosen_indices = np.array((random.choices(np.arange(Parameters.population.shape[0]), weights = lFitnesses, k = training_parameters.number_of_replaced_individuals)))
				chosen_population = Parameters.population[chosen_indices]
				randomizer_strength = 1 - (lFitnesses[chosen_indices])
			if Parameters.CV_selection==True:
				#sorting losses (based on validation)
				sorted_ind = np.argsort(vFitnesses)
				vFitnesses = vFitnesses[sorted_ind] #worst to best
				Parameters.population = Parameters.population[sorted_ind] #worst to best

				#choosing individuals from weighted distribution (validation)
				chosen_indices = np.array((random.choices(np.arange(Parameters.population.shape[0]), weights = vFitnesses, k = training_parameters.number_of_replaced_individuals)))
				chosen_population = Parameters.population[chosen_indices]
				randomizer_strength = 1 - (vFitnesses[chosen_indices])

			# calling WEIGHTED RANDOMIZER
			if (Parameters.randomization)==True and i!=(training_parameters.iterations-1):
				Parameters.population[0:training_parameters.number_of_replaced_individuals] = Parameters.randomizer(chosen_population, randomizer_strength, training_parameters)

		# observing optimization progress
		# if i%(1)==0 and i!=(iterations-1):
		if i%(Parameters.rr)==0:
			Parameters.observer(Parameters.population, Parameters.history)


	# fine tune parameters
	Parameters.fine_tuner(Parameters.population, training_parameters)

	return Parameters.population, lFitnesses, vFitnesses, Parameters.history

