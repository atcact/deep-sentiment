import argparse
import math
import numpy as np
import os
import random
import tensorflow as tf

from preprocess import preprocess, train_test_split, shuffle
from models.rnn.models import RNN
from models.am_gcn.models import AMGCN


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rnn")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    return args


def main(args):
    inputs, labels = preprocess('data/movie_review.parquet')
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2)
    if args.model == "rnn":
        model = RNN()
    elif args.model == "am_gcn":
        model = AMGCN()
    else:
        raise ValueError("Invalid model name")
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(train_inputs, train_labels, batch_size=32, epochs=10)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_inputs, test_labels)
    print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')
    # train(model, train_inputs, train_labels, args.batch_size, args.num_epochs)
    # evaluate(model, test_inputs, test_labels)
    
if __name__ == "__main__":
    args = parseArguments()
    main(args)