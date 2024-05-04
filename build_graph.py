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
    
    adj_matrix = np.asarray(adj_matrix)
    with open('spatial_matrix.pkl', 'wb') as f:
        pickle.dump(adj_matrix,f)
    print(adj_matrix)

    # Convert the tensor to a string
    # adj_saved = tf.io.serialize_tensor(tf.Tensor(adj_matrix))

    # tf.io.write_file('graph.txt', adj_saved)
# creates a feature graph
def feature_graph(path):
    '''builds a feature graph given a dataset'''
    tokenized_inputs, inputs, _ = preprocess(path)  
    tokenized_inputs = tokenized_inputs.numpy().tolist()
    word2vec = gensim.models.Word2Vec(tokenized_inputs, vector_size=100,window=20,
                                      max_vocab_size=VOCAB_SIZE,min_count=1)
    word_vectors = np.asarray(word2vec.wv.vectors)
    with open('word_vectors.pkl', 'wb') as f:
        pickle.dump(word2vec.wv.vectors, f)

    feature_matrix = cosine_similarity(word2vec.wv.vectors)
    print(feature_matrix.shape)
    for i in range(min(feature_matrix.shape)):
        feature_matrix[i,i] = 0
    feature_matrix = np.asarray(feature_matrix)
    # print(feature_matrix.shape, feature_matrix)
    with open('feature_matrix.pkl', 'wb') as f:
        pickle.dump(feature_matrix, f)
    return feature_matrix

# feature_graph('data/movie_review.parquet')
# build_graph('data/movie_review.parquet', 20)

def knn_graph(path, nearest_num):
    with open(path, 'rb') as f:
        cosine_similarity = pickle.load(f)
    knn_graph = []
    i=0
    for row in cosine_similarity:
        i+=1
        print('progress',i,"/",len(cosine_similarity))
        tuples = [(row[i],i) for i in range(len(row))]

        sorted_list = sorted(tuples, key = lambda x: x[0])
        kth_largest= sorted_list[len(sorted_list)-nearest_num-1][0]
        count = 0
        for tuple in sorted_list:
            if tuple[0] >kth_largest and count <nearest_num:
                knn_graph.append([i,tuple[1]])
                count +=1
    print(knn_graph)
    with open('knn_graph'+str(nearest_num)+'.txt','w') as f:
        for row in knn_graph:
            f.write(str(row[0]) +' ' + str(row[1])+ '\n')

knn_graph('feature_matrix.pkl',5)
            
        
