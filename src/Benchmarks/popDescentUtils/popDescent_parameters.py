from dataclasses import dataclass

@dataclass
class popDescent_parameters:
	SEED: list

	# pd parameters
	iterations: int
	pop_size: int
	number_of_replaced_individuals: int
	randomization: bool
	CV_selection: bool

	# gradient descent parameters
	batch_size: int
	batches: int
	epochs: int
	lr: int
	grad_steps: int

	# randomization
	input_factor: int
	rr : int # leash
	graph: bool

def return_parameters():
	SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

	iterations = 50
	pop_size = 5
	number_of_replaced_individuals = 2
	randomization = True
	CV_selection = True
	
	# gradient descent parameters
	batch_size = 64
	batches = 128
	epochs = 1
	lr = 1e-3
	grad_steps = iterations * epochs * batches * pop_size

	# randomization amount
	input_factor = 15
	rr = 1 # leash for exploration (how many iterations of gradient descent to run before randomization)
	graph = False

	return popDescent_parameters(SEED, iterations, pop_size, number_of_replaced_individuals, randomization, CV_selection, batch_size, batches, epochs, lr, grad_steps, input_factor, rr, graph)
