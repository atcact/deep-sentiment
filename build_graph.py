import scipy.sparse as sp
import tensorflow as tf
from preprocess import *
#preprocess
def build_graph(path,window_size, embedding_dim ):
    '''builds a graph, takes in path to data, window size, wordembedding dim, 
    and returns adjacency matrix (vocabsize,vocabsize), word_vectors (vocab_size, embedding_dim), edge_features (
    not implemented yet)'''
    
    inputs, labels = preprocess(path)    
    #build adjacency matrix. Words are already converted to a corresponding number in preprocess
    
    adj_matrix = tf.zeros((VOCAB_SIZE,VOCAB_SIZE))
    for input in inputs:
        for i in range(len(input-window_size)):
            cur_word = input[i]
            words_window = input[i+1:i+window_size+1]
            for word in words_window:
                adj_matrix = tf.tensor_scatter_nd_add(adj_matrix, 
                                                   indices=[[cur_word, word], [word, cur_word]], 
                                                   updates=[1, 1])
    word_vectors = tf.random.uniform(-0.01, 0.01, (VOCAB_SIZE,embedding_dim))

    return tf.sparse.from_dense(adj_matrix),word_vectors


build_graph('data/movie_review.parquet', 20)