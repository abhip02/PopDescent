from pickle import load
# from numpy import array
# from keras.preprocessing.text import Tokenizer
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import Embedding
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
# from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import keras
import numpy as np

from dataclasses import dataclass

# keras model/dataset parameters
max_len = 200
vocab_size=10000
embed_dim=32
num_heads=2
ff_dim=32
num_transformer_blocks=2
mlp_units=[32]
dropout=0.1
mlp_dropout=0.1

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# preprocess data
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)



@dataclass
class IMBDMRDataset:
    name: str
    lossfn: tf.Tensor
    train_images: tf.Tensor
    train_labels: tf.Tensor
    validation_images: tf.Tensor
    validation_labels: tf.Tensor
    test_images: tf.Tensor
    test_labels: tf.Tensor

@dataclass
class IMBDMRParams:
    max_len: int
    vocab_size: int
    embed_dim: int
    num_heads: int
    ff_dim: int
    num_transformer_blocks: int
    mlp_units: [int]
    dropout: float
    mlp_dropout: float

def getModelParams():
    return IMBDMRParams(max_len, vocab_size, embed_dim, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout)

def preprocess_dataset():
    # totalSize = 12000
    trainSize = 20000
    # valSize = 5000?
    valX, valY = [0], [0]
    
    # download dataset
    (trainX, trainY), (testX, testY) = keras.datasets.imdb.load_data(num_words=10000)
    
    valX, valY = trainX[:5000], trainY[:5000]  # First 5000 for validation
    testX, testY = trainX[5000:], trainY[5000:]  # Remaining 20000 for testing
    
    # Reshape labels to match model output shape (convert (N,) to (N,1))
    trainY = np.reshape(trainY, (-1, 1))
    valY = np.reshape(valY, (-1, 1))
    testY = np.reshape(testY, (-1, 1))
    
    # preprocess data
    trainX = keras.preprocessing.sequence.pad_sequences(trainX, maxlen=max_len)
    valX = keras.preprocessing.sequence.pad_sequences(valX, maxlen=max_len)
    testX = keras.preprocessing.sequence.pad_sequences(testX, maxlen=max_len)

    
    print(trainX.shape, valX.shape, testX.shape)
    
    # lossfn
    lossfn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    print("IMBD Movie Reviews")
    
    return IMBDMRDataset("IMBDMovieReviews", lossfn, trainX, trainY, valX, valY, testX, testY)