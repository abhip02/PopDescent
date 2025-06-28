from hyperopt import fmin, tpe, hp

# Define the search space
def define_space(training_parameters):
    # space = {
    #     'learning_rate': hp.loguniform('learning_rate', -5, -0),
    #     'l2_reg': hp.loguniform('l2_reg', -6, -1)
    # }
    space = {
        'learning_rate': hp.loguniform('learning_rate', training_parameters.lr_min_value, training_parameters.lr_max_value),
        'l2_reg': hp.loguniform('l2_reg', training_parameters.reg_min_value, training_parameters.reg_max_value)
    }
    
    return space