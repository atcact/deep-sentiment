import argparse
import math
import numpy as np
import os
import random
import tensorflow as tf

import pandas as pd


from preprocess import preprocess, train_test_split, shuffle
from models.rnn.models import RNN
from models.am_gcn.models import AMGCN
from models.mc_cnn.models import MC_CNN


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rnn")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()
    return args


def main(args):
    input_length = 100
    if args.dataset == "imdb":
        inputs, labels = preprocess('data/movie_review.parquet', input_length=input_length)
    elif args.dataset == "sts_gold":
        input_length = 10
        inputs, labels = preprocess('data/sts_gold_tweet.parquet', input_length=input_length, process_text=True) 
    
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2)
    if args.model == "rnn":
        model = RNN(input_length=input_length)
    elif args.model == "amgcn":
        model = AMGCN(nfeat=300, nclass=2, nhid1=256, nhid2=128, n=2, dropout=args.dropout)
    elif args.model == "mccnn":
        model = MC_CNN(length=input_length, vocab_size=20000, dropout_rate=args.dropout)
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