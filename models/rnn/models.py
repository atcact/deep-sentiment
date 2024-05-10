import tensorflow as tf
import numpy as np

VOCAB_SIZE = 20000

class RNN(tf.keras.Model):
    def __init__(self, vocab_size=VOCAB_SIZE, input_length=100):
        super(RNN, self).__init__()
        self.rnn_layers = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=input_length),
            tf.keras.layers.LSTM(128, dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        

    def call(self, inputs):
        return self.rnn_layers(inputs)