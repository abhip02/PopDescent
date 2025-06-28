def define_tuner(build_model, training_parameters):
    # define tuner
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=training_parameters.max_trials,
        executions_per_trial=training_parameters.executions_per_trial,
        overwrite=True,
        project_name="ktRS: %s" % SEED
    )
    return tuner