import tensorflow as tf
from layers import GraphConvolution

class GCN(tf.keras.Model):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout
     
    def call(self, x, adj, training=False):
        x = tf.nn.relu(self.gc1(x, adj))
        x = tf.nn.dropout(x, self.dropout)
        x = self.gc2(x, adj)
        return x
    
class SimpleGCN(tf.keras.Model):
    def __init__(self, nfeat, nclass, nhid, dropout):
        super(SimpleGCN, self).__init__()

        self.SGCN = GCN(nfeat, nhid, nclass, dropout)
        self.dropout = dropout

    def call(self, input, training=False):
        x, adj = input
        output = self.SGCN(x, adj, training=training)
        output = tf.nn.softmax(output)
        return output
