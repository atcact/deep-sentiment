import scipy.sparse as sp
import tensorflow as tf
from preprocess import *
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy
#preprocess
def build_graph(path,window_size):
    '''builds a graph in the form of an adjacency matrix, takes in path to data, window size'''
    
    tokenized_inputs, a, b = preprocess(path)    
    #build adjacency matrix. Words are already converted to a corresponding number in preprocess
    adj_matrix = np.zeros((VOCAB_SIZE,VOCAB_SIZE))
    print(adj_matrix.shape)
    # windows = []
    # for input in inputs[:100]:
    #     print('running')
    #     for i in range(len(input-window_size)):
    #         cur_word = input[i]
    #         windows.append(input[i+1:i+window_size+1])
    
    # for window in windows:
    #     print('run')
    #     for i in range(len(window)):
    #         for j in range(len(window)):
    #             cur_word = window[i]
    #             word = window[j]
    #             adj_matrix = tf.tensor_scatter_nd_add(adj_matrix, 
    #                                                 indices=[[cur_word, word], [word, cur_word]], 
    #                                                 updates=[1, 1])
    #             edge_feature = word_vectors[cur_word] - word_vectors[word]
    #             edge_features.append(edge_feature)
    # create adjacency matrix
    progress = 0
    for input in tokenized_inputs:
        progress+=1
        print('progress:', progress, '/', len(tokenized_inputs))
        for i in range(len(input-window_size)):
            cur_word = input[i]
            words_window = input[i+1:i+window_size+1]
            for word in words_window:
                adj_matrix[cur_word, word] += 1
                adj_matrix[word,cur_word] += 1
    print(adj_matrix.shape)
    print(adj_matrix)
    adj_matrix = np.asarray(feature_graph)
    with open('spatial_matrix.pkl', 'wb') as f:
        pickle.dump(adj_matrix,f)

    # Convert the tensor to a string
    # adj_saved = tf.io.serialize_tensor(tf.Tensor(adj_matrix))

    # tf.io.write_file('graph.txt', adj_saved)
def feature_graph(path):
    'builds a feature graph given a dataset'
    tokenized_inputs, inputs, labels = preprocess(path)  
    word2vec = gensim.models.Word2Vec(inputs, vector_size=100,window=20)
    similarity_matrix = cosine_similarity(word2vec)
    feature_matrix = np.fill_diagonal(similarity_matrix, 0)

    feature_matrix = np.asarray(feature_matrix)
    with open('feature_matrix.pkl', 'wb') as f:
        pickle.dump(feature_matrix,f)
    return np.array(feature_matrix)

    
feature_graph('data/movie_review.parquet')
build_graph('data/movie_review.parquet', 20)