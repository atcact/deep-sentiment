import pyarrow.parquet as pq
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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


def preprocess(path):
    """Takes in the dataset and returns the train_input, train_labels, test_input, test_labels in this order"""
    dataset = pq.ParquetDataset(path)
    
    data = dataset.read().to_pandas()
    inputs = data["tweet"]
    labels = data["polarity"]

    # Apply text cleaning
    inputs = inputs.apply(preprocess_text)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(inputs)
    
    inputs = tokenizer.texts_to_sequences(inputs)
    inputs = pad_sequences(inputs, maxlen=MAX_LEN, padding="post", value=0)

    return inputs, labels

#merge('tweet-00000.parquet','train-00000.parquet', 'sts_gold_tweet.parquet')
#preprocess('data/sts_gold_tweet.parquet')
