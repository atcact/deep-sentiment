import tensorflow as tf
from .layers import GraphConvolution

'''
GCN and AMGCN models
'''
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
    
class Attention(tf.keras.Model):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='tanh'),
            tf.keras.layers.Dense(1, bias_initializer=tf.constant_initializer(0))
        ])

    def call(self, z):
        w = self.project(z)
        beta = tf.nn.softmax(w, axis=1)
        return tf.reduce_sum(beta * z, axis=1), beta
        
    
class AMGCN(tf.keras.Model):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(AMGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout) # Special GCN for semantic graph
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout) # Special GCN for feature graph
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout) # Common GCN

        self.dropout = dropout
        self.a = tf.Variable(tf.zeros((nhid2, 1)), trainable=True)
        self.attention = Attention(nhid2)
        self.tanh = tf.keras.layers.Activation('tanh')

        self.MLP = tf.keras.Sequential([
            tf.keras.layers.Dense(nclass),
            tf.keras.layers.Activation('softmax')
        ])

    def call(self, input, training=False):
        x, sadj, fadj = input
        emb1 = self.SGCN1(x, sadj, training=training) 
        com1 = self.CGCN(x, sadj, training=training) 
        com2 = self.CGCN(x, fadj, training=training) 
        emb2 = self.SGCN2(x, fadj, training=training) 
        Xcom = (com1 + com2) / 2
        emb = tf.stack([emb1, emb2, Xcom], axis=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        return output, att, emb1, com1, com2, emb2, emb
    
    
