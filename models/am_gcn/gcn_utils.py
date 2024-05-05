import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import networkx as nx
import pickle
import pprint
import json
from preprocess import preprocess, train_test_split

def common_loss(emb1, emb2):
    emb1 = emb1 - tf.reduce_mean(emb1, axis=0, keepdims=True)
    emb2 = emb2 - tf.reduce_mean(emb2, axis=0, keepdims=True)
    emb1 = tf.nn.l2_normalize(emb1, axis=1)
    emb2 = tf.nn.l2_normalize(emb2, axis=1)
    cov1 = tf.matmul(emb1, emb1, transpose_b=True)
    cov2 = tf.matmul(emb2, emb2, transpose_b=True)
    cost = tf.reduce_mean((cov1 - cov2)**2)
    return cost

def loss_dependence(emb1, emb2, dim):
    R = tf.eye(dim) - (1/dim) * tf.ones((dim, dim))
    K1 = tf.matmul(emb1, emb1, transpose_b=True)
    K2 = tf.matmul(emb2, emb2, transpose_b=True)
    RK1 = tf.matmul(R, K1)
    RK2 = tf.matmul(R, K2)
    HSIC = tf.linalg.trace(tf.matmul(RK1, RK2))
    return HSIC

def accuracy(output, labels):
    preds = tf.argmax(output, axis=1)
    correct = tf.equal(preds, labels)
    correct = tf.reduce_sum(tf.cast(correct, tf.float32))
    return correct / len(labels)

def sparse_mx_to_tf_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a TensorFlow sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = tf.convert_to_tensor(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64), dtype=tf.int64)
    indices = tf.transpose(indices)
    values = tf.convert_to_tensor(sparse_mx.data, dtype=tf.float32)
    shape = sparse_mx.shape
    print(indices.shape, values.shape, shape)
    return tf.SparseTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(config):
    f = np.loadtxt(config.feature_path, dtype=float)
    l = np.loadtxt(config.label_path, dtype=int)
    test = np.loadtxt(config.test_path, dtype=int)
    train = np.loadtxt(config.train_path, dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = tf.convert_to_tensor(np.array(features.todense()), dtype=tf.float32)

    idx_test = test.tolist()
    idx_train = train.tolist()

    idx_train = tf.convert_to_tensor(idx_train, dtype=tf.int64)
    idx_test = tf.convert_to_tensor(idx_test, dtype=tf.int64)
    
    labels = tf.convert_to_tensor(np.array(l), dtype=tf.int64)

    return features, labels, idx_train, idx_test

def my_load_data():
    features, labels = preprocess("data/movie_review.parquet")
    idx_train, idx_test = train_test_split(np.arange(15000), test_size=0.2)
    

    return features, labels, idx_train, idx_test

def load_graph(dataset, config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'

    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))
    print('finish loading graph', nfadj.shape, nsadj.shape)
    nsadj = sparse_mx_to_tf_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_tf_sparse_tensor(nfadj)
    return nsadj, nfadj

# def load_graph_to_tensor(path):
#     data = pickle.load(open(path, 'rb'))
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
#     tensor_data = tf.convert_to_tensor(data, dtype=tf.float32)
#     tensor_data = tf.sparse.from_dense(tensor_data)
#     return tensor_data
    # with open(filename, 'w') as f:
    #     pprint.pprint(data, stream=f, width=10000)
        # json.dump(data, f, indent=2)

def load_graph_to_tensor(config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'
        
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fedges = fedges - 1
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    
    sadj = pickle.load(open(config.structgraph_path, 'rb'))
    sadj = sp.csr_matrix(sadj, dtype=np.float32)
    # sadj = tf.convert_to_tensor(sadj, dtype=tf.float32)
    # sadj = tf.sparse.from_dense(sadj)
    # nsadj = normalize(sadj)
    nsadj = sadj
    
    # struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    # sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    # sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    # sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize(sadj + sp.eye(sadj.shape[0]))
    print('finish loading graph', nfadj.shape, nsadj.shape)
    nsadj = sparse_mx_to_tf_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_tf_sparse_tensor(nfadj)
    return nsadj, nfadj
        
# convert_to_txt('data/imdb/feature_matrix.pkl', 'data/imdb/feature.txt')
# convert_to_txt('data/imdb/spatial_matrix.pkl', 'data/imdb/spatial.txt')