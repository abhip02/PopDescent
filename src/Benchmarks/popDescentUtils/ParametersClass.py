# from src.Benchmarks.popDescentUtils.pdFunctionsClass import return_pd_functions
from . import pdFunctionsClass
from src.Benchmarks.CIFAR100 import models

import typing
from typing import TypeVar, Generic
from collections.abc import Callable
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

NN_Individual = namedtuple("NN_Individual", ["nn", "opt_obj", "LR_constant", "reg_constant"])

# CLASSES
Individual = TypeVar('Individual')



@dataclass
class Parameters(Generic[Individual]):

    population: Callable[[int], np.array]
    randomizer: Callable[[np.array, float], np.array]
    optimizer: Callable[[np.array], np.array]
    observer: Callable[[np.array], np.array] # check this for typing
    randomization: bool
    CV_selection: bool
    rr: int
    history: [np.array]
    fine_tuner: Callable[[np.array], np.array]


def individual_to_params(
    pop_size: int,
    new_individual: Callable[[], Individual],
    individual_randomizer: Callable[[Individual, float], Individual],
    individual_optimizer: Callable[[Individual], Individual],
    observer: Callable[[Individual], float],
    randomization: bool,
    CV_selection: bool,
    rr: int, # randomization rate
    history: [float]
    ) -> Parameters[Individual]:

    def Parameter_new_population(pop_size: int) -> np.array(Individual):
        population = np.zeros(pop_size, dtype=object)
        for i in range(pop_size):
            population[i], model_num = new_individual()

        return population, model_num

    def Parameter_class_randomizer(population: np.array(Individual), normalized_amount: float, training_parameters) -> np.array(Individual):
        print(""), print("RANDOMIZING")
        randomized_population = np.zeros(len(population), dtype=object)
        for i in range(len(population)):
            new_object = individual_randomizer(population[i], normalized_amount[i], training_parameters.input_factor)
            randomized_population[i] = new_object

        return randomized_population

    def Parameter_class_optimizer(population: np.array(Individual), training_parameters) -> np.array(Individual):
        lFitnesses, vFitnesses = [], []
        for i in range(len(population)):
            print(""), print("model #%s" % (i+1))
            normalized_training_loss, normalized_validation_loss = individual_optimizer(NN_Individual(*population[i]), training_parameters)
            lFitnesses.append(normalized_training_loss)
            vFitnesses.append(normalized_validation_loss)

        lFitnesses = np.array(lFitnesses)
        lFitnesses = lFitnesses.reshape([len(lFitnesses), ])

        vFitnesses = np.array(vFitnesses)
        vFitnesses = vFitnesses.reshape([len(vFitnesses), ])

        return lFitnesses, vFitnesses

    # (during optimization)
    def Parameter_class_observer(population, history):

        batch_size = 64
        tIndices = np.random.choice(4999, size = (batch_size*10, ), replace=False)

        all_test_loss = []
        for i in range(len(population)):
            unnormalized_model_loss = observer(population[i], tIndices)
            all_test_loss.append(unnormalized_model_loss)

        avg_test_loss = np.mean(all_test_loss)
        best_test_model_loss = np.min(all_test_loss)

        history.append(best_test_model_loss) ## main action of observer (to graph optimization progress later)
        return

    def fine_tuner(population: np.array(Individual), training_parameters) -> np.array(Individual):
        for j in range(1):
            for i in range(len(population)):
                print(""), print("Fine-Tuning models"), print("model #%s" % (i+1)), print("")
                normalized_training_loss, normalized_validation_loss = individual_optimizer(NN_Individual(*population[i]), training_parameters)

        return

    Parameters_object = Parameters(Parameter_new_population, Parameter_class_randomizer, Parameter_class_optimizer, Parameter_class_observer, randomization, CV_selection, rr, history, fine_tuner)
    return Parameters_object


# import "evaluator" from evaluation:
from src.Benchmarks.popDescentUtils import evaluation
# External Evaluator
def Parameter_class_evaluator(population):
	pop_train_loss, pop_test_loss = [], []

	for i in range(len(population)):
		individual_train_loss, individual_test_loss = evaluation.evaluator(population[i])

		pop_train_loss.append(individual_train_loss)
		pop_test_loss.append(individual_test_loss)

	best_test_model_loss = np.min(pop_test_loss)
	best_train_model_loss = pop_train_loss[pop_test_loss.index(best_test_model_loss)]

	return best_train_model_loss, best_test_model_loss


# make model
def create_Parameters_NN_object(training_parameters, with_reg):
    history = []
    pd_functions = pdFunctionsClass.return_pd_functions()
    # creates Parameter object to pass into Population Descent
    object = individual_to_params(training_parameters.pop_size, lambda: models.PD(with_reg), pd_functions.randomizer, pd_functions.optimizer, pd_functions.observer, randomization=training_parameters.randomization, CV_selection=training_parameters.CV_selection, rr=training_parameters.rr, history=history)
    object.population, model_num = object.population(training_parameters.pop_size) # initiazling population

    return object, model_num
