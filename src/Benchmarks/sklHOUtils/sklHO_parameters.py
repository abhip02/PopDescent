from dataclasses import dataclass

@dataclass
class hpS_parameters:
    SEED: list

    # pd parameters
    max_trials: int

    # gradient descent parameters
    batch_size: int
    batches: int
    epochs: int
    patience: int
    grad_steps: int
    
    # model parameters
    lr_min_value: float
    lr_max_value: float
    reg_min_value: float
    reg_max_value: float
    
    
    # observation parameters
    # graph: bool

def return_parameters():
    SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

    max_trials = 37 # 37

    # gradient descent parameters
    batch_size = 64
    batches = 780
    epochs = 2 # 2
    patience = 2
    grad_steps = max_trials * epochs * batches
        
    # model parameters
    lr_min_value = -5
    lr_max_value = -0
    reg_min_value = -6
    reg_max_value = -1

    # observation parameters
    # graph = False

    return hpS_parameters(SEED, max_trials, batch_size, batches, epochs, patience, grad_steps, lr_min_value, lr_max_value, reg_min_value, reg_max_value,)
