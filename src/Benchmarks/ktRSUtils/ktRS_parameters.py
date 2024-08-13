from dataclasses import dataclass

@dataclass
class ktRS_parameters:
    SEED: list

    # ktRS training parameters:
    max_trials: int
    executions_per_trial: int

    # model parameters
    lr_min_value: float
    lr_max_value: float
    reg_min_value: float
    reg_max_value: float
    
    # # scheduling parameters
    # min_decay_rate: float
    # max_decay_rate: float
    # default_decay_rate: float
    
    # min_decay_steps: int
    # max_decay_steps: int
    # step: int
    # default_decay_rate: int

    # gradient descent parameters
    batch_size: int
    batches: int
    train_epochs: int
    patience: bool
    
    graph: bool


def return_parameters():
    SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

    max_trials = 25
    executions_per_trial = 2
    
    # model parameters
    lr_min_value = 1e-5
    lr_max_value = 1e-1
    reg_min_value = 1e-4
    reg_max_value = 1e-2
    
    # gradient descent parameters
    batch_size = 64
    batches = 780
    train_epochs = 20
    patience = 2
    grad_steps = max_trials * executions_per_trial * batch_size * batches

    graph = False
    # dataset_name = "CIFAR100"

    return ktRS_parameters(SEED, max_trials, executions_per_trial, lr_min_value, lr_max_value, reg_min_value, reg_max_value, batch_size, train_epochs, patience, grad_steps, graph)
