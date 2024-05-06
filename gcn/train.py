import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np
import os
from pathlib import Path
import argparse
from models import SimpleGCN
from gcn_utils import *
from config import Config

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, required = True)
    args = parse.parse_args()
    
    path = Path(__file__)
    ROOT_DIR = path.parent.absolute()
    config_file = "./config/" + args.dataset + "_" + str(args.labelrate) + ".ini"
    config_path = os.path.join(ROOT_DIR, config_file)
    config = Config(config_path)

    tf.random.set_seed(0)

    sadj, fadj = load_graph(args.labelrate, config)
    # sadj, fadj = load_graph_to_tensor("./data/imdb/spatial_matrix.pkl"), load_graph_to_tensor("./data/imdb/feature_matrix.pkl")
    features, labels, idx_train, idx_test = load_data(config)

    model = SimpleGCN(config.fdim, config.class_num, config.nhid1, config.dropout)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    def train(model):
        loss = 0
        acc_test = 0
        macro_f1 = 0
        emb_test = 0
        with tf.GradientTape() as tape:
            output = model((features, sadj), training=True)
            loss = loss_fn(tf.gather(labels, idx_train), tf.gather(output, idx_train)) 

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc_test, macro_f1 = main_test(model)
        print('e:{}'.format(epoch),
            'ltr: {:.4f}'.format(loss),
            'atr: {:.4f}'.format(acc_test),
            'ate: {:.4f}'.format(acc_test),
            'f1te:{:.4f}'.format(macro_f1))
        return loss, acc_test, macro_f1

    def main_test(model):
        output = model((features, sadj))
        
        acc_test = accuracy(tf.gather(output, idx_test), tf.gather(labels, idx_test))
        
        label_max = []
        for idx in idx_test:
            label_max.append(tf.argmax(output[idx]).numpy())
        labelcpu = tf.gather(labels, idx_test)
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        
        return acc_test, macro_f1

    acc_max = 0
    f1_max = 0
    epoch_max = 0

    for epoch in range(config.epochs):
        loss, acc_test, macro_f1 = train(model)  
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch

    print('epoch:{}'.format(epoch_max),
        'acc_max: {:.4f}'.format(acc_max),
        'f1_max: {:.4f}'.format(f1_max))