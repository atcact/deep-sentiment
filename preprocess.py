import pyarrow.parquet as pq
import tensorflow as tf
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Flatten, Conv1D, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from utils import loadWord2Vec, clean_str
import nltk 
from nltk.corpus import stopwords

VOCAB_SIZE = 20000
MAX_LEN = 100

def preprocess(path):
    """takes in the dataset and returns the inputs and labels both as tensors"""
    dataset = pq.ParquetDataset(path)
    
    data = dataset.read().to_pandas()
    inputs = data.iloc[:,0]
    labels = data.iloc[:,1]
    print(inputs)
    # removes all stopwords and very rare words
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    word_freq = {}
    for input in inputs:
        temp1 = clean_str(input)
        words1 = temp1.split()
        for word in words1:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    clean_docs = []

    for input in inputs:
        temp = clean_str(input)
        words = []
        temp = temp.split()
        for word in temp:
            if word not in stop_words and word_freq[word] >= 5:
                words.append(word)
        doc_str = ' '.join(words).strip()
        clean_docs.append(doc_str)
    inputs = clean_docs
    
    labels = labels.values.tolist()
    # print('inputs', inputs[1:10])
    # print('labels', labels[1:10])

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(inputs)
    
    inputs = tokenizer.texts_to_sequences(inputs)
    inputs = pad_sequences(inputs, maxlen=MAX_LEN, padding="post", value=0)
    
    print(tf.convert_to_tensor(inputs), tf.convert_to_tensor(labels))
    return tf.convert_to_tensor(inputs), tf.convert_to_tensor(labels)
    
def shuffle(inputs, labels):
    indices = tf.random.shuffle(np.arange(len(inputs)))
    inputs = tf.gather(inputs, indices)
    labels = tf. gather(labels, indices)
    return inputs, labels


def train_test_split(X, y, test_size=0.2):
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test


def merge(path1, path2, target_path):
    try:
        file1 = pq.read_table(path1)
        file2 = pq.read_table(path2)
        
        with pq.ParquetWriter(target_path,
                file1.schema,
                version='2.0',
                compression='gzip',
                use_dictionary=True,
                data_page_size=2097152, #2MB
                write_statistics=True) as writer:
            writer.write_table(file1)
            writer.write_table(file2)
    except Exception as e:
        print(e)

# merge('train-00000-of-00001.parquet','test-00000-of-00001.parquet', 'movie_review.parquet')
preprocess('data/movie_review.parquet')