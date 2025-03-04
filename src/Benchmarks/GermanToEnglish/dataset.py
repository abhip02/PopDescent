from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

import tensorflow as tf


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs



# # We are now ready to clean each sentence. The specific cleaning operations we will perform are as follows:

# Remove all non-printable characters.
# Remove all punctuation characters.
# Normalize all Unicode characters to ASCII (e.g. Latin characters).
# Normalize the case to lowercase.
# Remove any remaining tokens that are not alphabetic.

# We will perform these operations on each phrase for each pair in the loaded dataset.

# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)


# Finally, now that the data has been cleaned, we can save the list of phrase pairs to a file ready for use.
# The function save_clean_data() uses the pickle API to save the list of clean text to file.

import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs

# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
filename = 'deu.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'english-german.pkl')
# spot check

# print(len(clean_pairs))
# # for i in range(100):
# # 	print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))




from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle

def load_dataset(totalSize, trainSize):
    # load a clean dataset
    def load_clean_sentences(filename):
        return load(open(filename, 'rb'))

    # save a list of clean sentences to file
    def save_clean_data(sentences, filename):
        dump(sentences, open(filename, 'wb'))
        print('Saved: %s' % filename)

    # load dataset
    raw_dataset = load_clean_sentences('english-german.pkl')

    # reduce dataset size
    n_sentences = totalSize
    dataset = raw_dataset[:n_sentences, :]
    # random shuffle
    shuffle(dataset)
    # split into train/test
    train, test = dataset[:trainSize], dataset[trainSize:]
    # save
    save_clean_data(dataset, 'english-german-both.pkl')
    save_clean_data(train, 'english-german-train.pkl')
    save_clean_data(test, 'english-german-test.pkl')

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')


# from keras import Tokenizer
import keras
import tensorflow as tf

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = tf.keras.preprocessing.text.Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer



# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)


# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y



from dataclasses import dataclass


@dataclass
class EnglishToGermanDataset:
    name: str
    lossfn: tf.Tensor
    train_images: tf.Tensor
    train_labels: tf.Tensor
    validation_images: tf.Tensor
    validation_labels: tf.Tensor
    test_images: tf.Tensor
    test_labels: tf.Tensor

@dataclass
class GtoEParams:
    src_vocab: int
    tar_vocab: int
    src_timesteps: int
    tar_timesteps: int
    n_units: int

def getModelParams():
    # ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256
    return GtoEParams(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)

def preprocess_dataset():
    totalSize = 12000
    trainSize = 10000
    valSize = 1000
    
    load_dataset(totalSize, trainSize)
    
    # prepare training data
    trainX = encode_sequences(ger_tokenizer, ger_length, train[:trainSize - valSize, 1])
    trainY = encode_sequences(eng_tokenizer, eng_length, train[:trainSize - valSize, 0])
    trainY = encode_output(trainY, eng_vocab_size)
    
    # prepare validation data
    valX = encode_sequences(ger_tokenizer, ger_length, train[trainSize - valSize:, 1])
    valY = encode_sequences(eng_tokenizer, eng_length, train[trainSize - valSize:, 0])
    valY = encode_output(valY, eng_vocab_size)
    
    print(valX.shape)
    
    # prepare test data
    testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
    testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
    testY = encode_output(testY, eng_vocab_size)
    
    # print(trainX.shape, valX.shape, testX.shape)
    
    # lossfn
    lossfn = tf.keras.losses.CategoricalCrossentropy()
    
    # src_vocab, n_units, input_length=src_timesteps
    ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256
    
    print("GermanToEnglish")
    
    return EnglishToGermanDataset("GermanToEnglish", lossfn, trainX, trainY, valX, valY, testX, testY)