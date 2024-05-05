import argparse
import math
import numpy as np
import os
import random
import tensorflow as tf

import pandas as pd

 
from preprocess1 import preprocess, train_test_split
from models.rnn.models import RNN
from models.am_gcn.models import AMGCN
from models.mc_cnn.models import MC_CNN

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rnn")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    return args

def main(args):
    # Preprocess the data
    inputs, labels = preprocess('data/sts_gold_tweet.parquet')
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2)
    
    # Define and compile the model
    if args.model == "rnn":
        model = RNN()
    elif args.model == "amgcn":
        model = AMGCN(nfeat=300, nclass=2, nhid1=256, nhid2=128, n=2, dropout=0.5)
    elif args.model == "mccnn":
        model = MC_CNN(length=100, vocab_size=20000)
    else:
        raise ValueError("Invalid model name")
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    if args.model == "mccnn":
        model.fit([train_inputs, train_inputs, train_inputs], train_labels, batch_size=8, epochs=3) 
        loss, accuracy = model.evaluate([test_inputs, test_inputs, test_inputs], test_labels)
    else:
        model.fit(train_inputs, train_labels, batch_size=8, epochs=3)
        loss, accuracy = model.evaluate(test_inputs, test_labels)
        
    print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')

if __name__ == "__main__":
    args = parseArguments()
    main(args)