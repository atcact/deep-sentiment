import pyarrow.parquet as pq
import tensorflow as tf
import numpy as np

def preprocess(path1, path2):
    """takes in the dataset and returns the train_input, train_labels,test_input, test_labels in this order"""
    train = pq.ParquetDataset('test-00000-of-00001.parquet')
    test = pq.ParquetDataset('test-00000-of-00001.parquet')

    train_data= train.read().to_pandas()
    train_input = train_data.iloc[:,0]
    train_labels = train_data.iloc[:,1]

    print('train input: ',train_input)
    print('train labels: ', train_labels) # to see how the dataset looks like

    test_data= test.read().to_pandas()
    test_input = test_data.iloc[:,0]
    test_labels = test_data.iloc[:,1]

    # Shuffling
    indices = tf.random.shuffle(np.arange(len(train_input)))
    test_input = tf.gather(test_input, indices)
    test_labels = tf.gather(test_labels, indices)
    train_input = tf.gather(train_input, indices)
    train_labels = tf. gather(train_labels, indices)
    return train_input, train_labels, test_input, test_labels
preprocess('test-00000-of-00001.parquet','test-00000-of-00001.parquet')