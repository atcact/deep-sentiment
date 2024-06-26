import scipy.sparse as sp
import tensorflow as tf
from preprocess import *
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

#preprocess

def spatial_graph(path, window_size):
    '''builds a graph in the form of an adjacency matrix, takes in path to data, window size'''
    
    tokenized_inputs, _, _ = preprocess(path)    
    # build adjacency matrix. Words are already converted to a corresponding number in preprocess
    adj_matrix = np.zeros((VOCAB_SIZE,VOCAB_SIZE))
    print(adj_matrix.shape)
    progress = 0
    for input in tokenized_inputs:
        progress += 1
        print('progress:', progress, '/', len(tokenized_inputs))
        if progress == 1000:
            break
        for i in range(len(input) - window_size):
            cur_word = input[i]
            words_window = input[i+1:i+window_size+1]
            for word in words_window:
                adj_matrix[cur_word, word] += 1
                adj_matrix[word, cur_word] += 1

    print(adj_matrix.shape)
    print(adj_matrix)
    adj_matrix = np.asarray(adj_matrix)
    with open('spatial_matrix.pkl', 'wb') as f:
        pickle.dump(adj_matrix,f)

    # Convert the tensor to a string
    # adj_saved = tf.io.serialize_tensor(tf.Tensor(adj_matrix))

    # tf.io.write_file('graph.txt', adj_saved)
    

def feature_graph(path):
    '''builds a feature graph given a dataset'''
    _, inputs, _ = preprocess(path)  
    # print('generate')
    word2vec = gensim.models.Word2Vec(inputs, vector_size=100,window=20)
    # print('after')
    feature_matrix = cosine_similarity(word2vec.wv.vectors)
    
    np.fill_diagonal(feature_matrix, 0) # make sure self similarity is 0

    feature_matrix = np.asarray(feature_matrix)
    # print(feature_matrix.shape, feature_matrix)
    with open('feature_matrix.pkl', 'wb') as f:
        pickle.dump(feature_matrix, f)
    return feature_matrix

    
feature_graph('data/movie_review.parquet')
# spatial_graph('data/movie_review.parquet', 20)