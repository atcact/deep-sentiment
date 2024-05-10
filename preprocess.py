import pyarrow.parquet as pq
import tensorflow as tf
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Flatten, Conv1D, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import re

VOCAB_SIZE = 20000
MAX_LEN = 100

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Removing special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = text.split()
    
    # Removing stopwords (you can define your own list of stopwords)
    stopwords = {'is', 'the', 'and', 'that', 'this', 'to', 'of', 'a', 'an', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'it', 'as', 'has', 'have', 'was', 'were', 'be', 'been'}
    tokens = [word for word in tokens if word not in stopwords]
    
    # Join tokens back into string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def preprocess(path, input_length=MAX_LEN, process_text=False):
    """takes in the dataset and returns the train_input, train_labels, test_input, test_labels in this order"""
    dataset = pq.ParquetDataset(path)
    
    data = dataset.read().to_pandas()
    inputs = data.iloc[:,0]
    labels = data.iloc[:,1]
    
    if labels.max() > 1:
        labels = labels // labels.max()
    if process_text:
        inputs = inputs.apply(preprocess_text)

    # print("inputs: ", inputs)
    # print("labels: ", labels)
    # print('First sample before preprocessing: \n', inputs[0], '\n')

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(inputs)
    
    inputs = tokenizer.texts_to_sequences(inputs)
    inputs = pad_sequences(inputs, maxlen=input_length, padding="post", value=0)

    # print('First sample after preprocessing: \n', inputs[0], '\n')
    return inputs, labels

preprocess('data/sts_gold_tweet.parquet', process_text=True)
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
# preprocess('data/movie_review.parquet')
        


