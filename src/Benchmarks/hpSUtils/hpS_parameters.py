from dataclasses import dataclass

@dataclass
class hpS_parameters:
    SEED: list

    # pd parameters
    iterations: int

    # gradient descent parameters
    batch_size: int
    batches: int
    epochs: int
    grad_steps: int
    
    
    # observation parameters
    rr: int
    graph: bool

def return_parameters():
    SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

    iterations = 1

    # gradient descent parameters
    batch_size = 64
    batches = 780
    epochs = 1
    grad_steps = iterations * epochs * batches * 5

    # observation parameters
    rr = 1
    graph = False

    return hpS_parameters(SEED, iterations, batch_size, batches, epochs, grad_steps, rr, graph)
