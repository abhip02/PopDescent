import os
import random
import tensorflow as tf
import numpy as np

# SETTING SEED
def set_seeds(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=21):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

# use to set seed
def Set_Seed(seed):
    set_global_determinism(seed=seed)


# loss functions
def SparseCategoricalCrossentropy_loss_fn():
    lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return lossfn

def CategoricalCrossentropy_loss_fn():
    lossfn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    return lossfn

# generate and return random evaluation set given a dataset, and batch size input
def random_eval_set_generator(dataset, batch_size):    
    np.random.seed(0)
    eIndices = np.random.choice(len(dataset.test_images), size = (batch_size*25, ), replace=False)
    random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = dataset.train_images[eIndices], dataset.train_labels[eIndices], dataset.test_images[eIndices], dataset.test_labels[eIndices]

    return random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels



# def main(seed):
#     set_seed(seed)

# if __name__ == "__main__":
# 	main(seed)


