import tensorflow as tf
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

from collections import namedtuple
NN_Individual = namedtuple("NN_Individual", ["nn", "opt_obj", "LR_constant", "reg_constant"])

import dataset

# model
def ktRSm(training_parameters, with_reg, with_sch):
    GtoEParams = dataset.getModelParams()
    src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units = GtoEParams.src_vocab, GtoEParams.tar_vocab, GtoEParams.src_timesteps, GtoEParams.tar_timesteps, GtoEParams.n_units
    
    if with_sch:
        print("Scheduling Enabled")
    def build_model(hp):
        hp_reg = hp.Float("reg_term", min_value=training_parameters.reg_min_value, max_value=training_parameters.reg_max_value)
        regularizer = tf.keras.regularizers.l2(l=hp_reg) if with_reg else None

        model = Sequential()
        model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True, embeddings_regularizer=regularizer))
        model.add(tf.keras.layers.Bidirectional(LSTM(n_units, kernel_regularizer=regularizer)))
        model.add(RepeatVector(tar_timesteps))
        model.add(tf.keras.layers.Bidirectional(LSTM(n_units, return_sequences=True, kernel_regularizer=regularizer)))

        # model.add(tf.keras.layers.Attention())
        model.add(TimeDistributed(Dense(tar_vocab, activation='softmax', kernel_regularizer=regularizer)))

        # Tune the learning rate schedule
        if with_sch:
            lr_schedule = hp.Choice('learning_rate_schedule', values=['exponential', 'inverse_time', 'polynomial'])
        
            if lr_schedule == 'exponential':
                lr = hp.Float('initial_learning_rate', min_value=training_parameters.lr_min_value, max_value=training_parameters.lr_max_value, sampling='log', default=1e-3)
                decay_rate = hp.Float('decay_rate', min_value=0.8, max_value=0.99, default=0.9)
                decay_steps = hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000, default=5000)
                lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate)
                optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            elif lr_schedule == 'inverse_time':
                lr = hp.Float('initial_learning_rate', min_value=training_parameters.lr_min_value, max_value=training_parameters.lr_max_value, sampling='log', default=1e-3)
                decay_rate = hp.Float('decay_rate', min_value=0.1, max_value=0.9, default=0.5)
                decay_steps = hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000, default=5000)
                lr_schedule = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate)
                optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            elif lr_schedule == 'polynomial':
                lr = hp.Float('initial_learning_rate', min_value=training_parameters.lr_min_value, max_value=training_parameters.lr_max_value, sampling='log', default=1e-3)
                end_learning_rate = hp.Float('end_learning_rate', min_value=training_parameters.lr_min_value, max_value=training_parameters.lr_max_value, sampling='log', default=1e-6)
                decay_steps = hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000, default=5000)
                power = hp.Float('power', min_value=0.1, max_value=2.0, default=1.0)
                lr_schedule = keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=lr, decay_steps=decay_steps, end_learning_rate=end_learning_rate, power=power)
        
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            # Compile the model
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        else:
            hp_learning_rate = hp.Float("lr", min_value=training_parameters.lr_min_value, max_value=training_parameters.lr_max_value, sampling="log")

            model.compile(
                optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]
            )

        return model

    return build_model

# basic hyperparameter search
def hpS(with_reg):
    GtoEParams = dataset.getModelParams()
    src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units = GtoEParams.src_vocab, GtoEParams.tar_vocab, GtoEParams.src_timesteps, GtoEParams.tar_timesteps, GtoEParams.n_units
    
    regularization_amount = [0.01 / (10**i) for i in range(5)] if with_reg else [None] # [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    learning_rate = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

    population = []
    reg_list = []

    for r in range(len(regularization_amount)):
        for l in range(len(learning_rate)):
            model = Sequential()
            model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True, embeddings_regularizer=regularization_amount[r]))
            model.add(tf.keras.layers.Bidirectional(LSTM(n_units, kernel_regularizer=regularization_amount[r])))
            model.add(RepeatVector(tar_timesteps))
            model.add(tf.keras.layers.Bidirectional(LSTM(n_units, return_sequences=True, kernel_regularizer=regularization_amount[r])))

            # model.add(tf.keras.layers.Attention())
            model.add(TimeDistributed(Dense(tar_vocab, activation='softmax', kernel_regularizer=regularization_amount[r])))
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            
            population.append(model)
            reg_list.append(regularization_amount[r])
            
    population = np.array(population)
    print(model.summary())
    
    return population, reg_list


def sklHO(with_reg, params):
    GtoEParams = dataset.getModelParams()
    src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units = GtoEParams.src_vocab, GtoEParams.tar_vocab, GtoEParams.src_timesteps, GtoEParams.tar_timesteps, GtoEParams.n_units
    
    regularizer = tf.keras.regularizers.l2(params['l2_reg']) if with_reg else None

    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True, embeddings_regularizer=regularizer))
    model.add(tf.keras.layers.Bidirectional(LSTM(n_units, kernel_regularizer=regularizer)))
    model.add(RepeatVector(tar_timesteps))
    model.add(tf.keras.layers.Bidirectional(LSTM(n_units, return_sequences=True, kernel_regularizer=regularizer)))

    # model.add(tf.keras.layers.Attention())
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax', kernel_regularizer=regularizer)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Testing population descent
def PD(with_reg):
    GtoEParams = dataset.getModelParams()
    src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units = GtoEParams.src_vocab, GtoEParams.tar_vocab, GtoEParams.src_timesteps, GtoEParams.tar_timesteps, GtoEParams.n_units
    
    model_num = "model 4"
    regularizer = tf.keras.regularizers.l2(l=1e-3) if with_reg else None
    
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True, embeddings_regularizer=regularizer))
    model.add(tf.keras.layers.Bidirectional(LSTM(n_units, kernel_regularizer=regularizer)))
    model.add(RepeatVector(tar_timesteps))
    model.add(tf.keras.layers.Bidirectional(LSTM(n_units, return_sequences=True, kernel_regularizer=regularizer)))

    # model.add(tf.keras.layers.Attention())
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax', kernel_regularizer=regularizer)))

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3) # 1e-3 (for FMNIST)
    LR_constant = 10**(np.random.normal(-4, 2))
    reg_constant = 10**(np.random.normal(0, 2))

    # creating NN object with initialized parameters
    NN_object = NN_Individual(model, optimizer, LR_constant, reg_constant)

    return NN_object, model_num