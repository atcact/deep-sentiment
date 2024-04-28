import tensorflow as tf
import numpy as np
from keras.models import Model

def MC_CNN(length, vocab_size):
    # channel 1
        inputs1 = tf.keras.layers.Input(shape=(length,))
        embedding1 = tf.keras.layers.Embedding(vocab_size, 100)(inputs1)
        conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=4, activation='relu')(embedding1)
        drop1 = tf.keras.layers.Dropout(0.6)(conv1)
        pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1)
        flat1 = tf.keras.layers.Flatten()(pool1)
        # channel 2
        inputs2 = tf.keras.layers.Input(shape=(length,))
        embedding2 = tf.keras.layers.Embedding(vocab_size, 100)(inputs2)
        conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
        drop2 = tf.keras.layers.Dropout(0.7)(conv2)
        pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop2)
        flat2 = tf.keras.layers.Flatten()(pool2)
        # channel 3
        inputs3 = tf.keras.layers.Input(shape=(length,))
        embedding3 = tf.keras.layers.Embedding(vocab_size, 100)(inputs3)
        conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu')(embedding3)
        drop3 = tf.keras.layers.Dropout(0.7)(conv3)
        pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop3)
        flat3 = tf.keras.layers.Flatten()(pool3)
        # merge
        merged = tf.keras.layers.concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = tf.keras.layers.Dense(10, activation='relu')(merged)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

        return model
