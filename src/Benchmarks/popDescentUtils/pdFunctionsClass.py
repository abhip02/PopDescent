from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

# import file for functions
from . import optimization, randomization, evaluation, observation

@dataclass
class pdFunctionsClass:
    optimizer: Callable[[np.array], np.array]
    randomizer: Callable[[np.array, float], np.array]
    evaluator: Callable[[np.array], np.array]
    observer: Callable[[np.array], np.array]



def return_pd_functions():
    return pdFunctionsClass(optimization.optimizer, randomization.randomizer, evaluation.evaluator, observation.observer)
