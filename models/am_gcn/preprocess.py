import pyarrow.parquet as pq
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


VOCAB_SIZE = 20000
MAX_LEN = 100

def preprocess(path):
    """takes in the dataset and returns the train_input, train_labels, test_input, test_labels in this order"""
    dataset = pq.ParquetDataset(path)
    
    data = dataset.read().to_pandas()
    inputs = data.iloc[:,0]
    labels = data.iloc[:,1]

    # print("inputs: ", inputs)
    # print("labels: ", labels)
    # print('First sample before preprocessing: \n', inputs[0], '\n')

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(inputs)
    
    inputs = tokenizer.texts_to_sequences(inputs)
    inputs = pad_sequences(inputs, maxlen=MAX_LEN, padding="post", value=0)

    # print('First sample after preprocessing: \n', inputs[0], labels[0], '\n')
    return inputs, labels
    
def shuffle(inputs, labels):
    indices = tf.random.shuffle(np.arange(len(inputs)))
    inputs = tf.gather(inputs, indices)
    labels = tf. gather(labels, indices)
    return inputs, labels


def train_test_split(X, test_size=0.2):
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    
    return X_train, X_test
