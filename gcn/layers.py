import math

import tensorflow as tf


class GraphConvolution(tf.keras.layers.Layer):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self.add_weight(name='weight', shape=(in_features, out_features), initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1))
        if bias:
            self.bias = self.add_weight(name='bias', shape=(out_features,), initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1))
        else:
            self.bias = None

    def call(self, inputs, adj):
        # print("inputs: ", inputs.shape, "adj: ", adj.shape)
        support = tf.matmul(inputs, self.weight)
        # print("support: ", support.shape)
        output = tf.sparse.sparse_dense_matmul(adj, support)
        if self.bias is not None:
            return tf.add(output, self.bias)  # output + self.bias
        else:
            return output