import argparse
import math
import numpy as np
import os
import random
import tensorflow as tf

from preprocess import *
from models.rnn.models import RNN
from models.am_gcn.models import AMGCN
from models.mc_cnn.models import MC_CNN


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rnn")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    return args

def set_seed(seed=0):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def main(args):
    set_seed()
    inputs, labels = prepreprocess('data/movie_review.parquet')
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2)
    if args.model == "rnn":
        model = RNN()
    elif args.model == "amgcn":
        model = AMGCN(nfeat=300, nclass=2, nhid1=256, nhid2=128, n=2, dropout=0.5)
    # TODO: Add more models
    elif args.model == "mccnn":
        model = MC_CNN(length=100, vocab_size=20000)
    else:
        raise ValueError("Invalid model name")
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.save('model_' + args.model + '.h5')
    # Train & evaluate the model
    if args.model == "mccnn":
        model.fit([train_inputs, train_inputs, train_inputs], train_labels, batch_size=args.batch_size, epochs=args.epochs) # Update parameters
        loss, accuracy = model.evaluate([test_inputs, test_inputs, test_inputs], test_labels)
    else:
        model.fit(train_inputs, train_labels, batch_size=args.batch_size, epochs=args.epochs)
        loss, accuracy = model.evaluate(test_inputs, test_labels)
        
    print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')

    
if __name__ == "__main__":
    args = parseArguments()
    main(args)