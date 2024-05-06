import tensorflow as tf
from sklearn.metrics import f1_score
import keras
import numpy as np
import os
from pathlib import Path
import argparse
from models import AMGCN
from gcn_utils import *
from config import Config
from preprocess import preprocess, train_test_split

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parse = argparse.ArgumentParser()
    # parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    # parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, required = True)
    args = parse.parse_args()
    
    path = Path(__file__)
    ROOT_DIR = path.parent.absolute()
    # config_file = "./config/" + args.dataset + "_" + str(args.labelrate) + ".ini"
    config_file = "./config/imdb_20.ini"
    config_path = os.path.join(ROOT_DIR, config_file)
    config = Config(config_path)

    tf.random.set_seed(0)

    sadj, fadj = load_graph_to_tensor(config)
    # sadj, fadj = load_graph_to_tensor("./data/imdb/spatial_matrix.pkl"), load_graph_to_tensor("./data/imdb/feature_matrix.pkl")
    
    print(sadj.shape, fadj.shape)
    # features, labels, idx_train, idx_test = load_data(config)
    features, labels = preprocess("data/movie_review.parquet")
    features = tf.cast(features, dtype=tf.float32)
    idx_train, idx_test = train_test_split(np.arange(15000), test_size=0.2)    

    amgcn_model = AMGCN(config.fdim, config.class_num, config.nhid1, config.nhid2, config.class_num, config.dropout)
    empty_input = tf.zeros((config.n, config.fdim))
    emb = tf.Variable(tf.zeros((config.n, config.nhid2)), trainable=True)
    # define model to classify reviews based on word embeddings generated by AMGCN
    inputs1 = tf.keras.Input(shape=(100,))
    inputs2 = tf.keras.Input(shape=(config.fdim,))
    output, att, emb1, com1, com2, emb2, emb = amgcn_model((inputs2, sadj, fadj))
    classifier = tf.keras.Sequential([
                            tf.keras.layers.Embedding(input_dim=emb.shape[0], output_dim=emb.shape[1], input_length=100, weights=[emb], trainable=True),
                            tf.keras.layers.GlobalAveragePooling1D(),
                            tf.keras.layers.Dense(1, activation='sigmoid')
                        ])
    final_out = classifier(inputs1)
    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=final_out)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, weight_decay=config.weight_decay)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'], run_eagerly=False)

    def train(model, epochs):
        loss = 0
        acc_test = 0
        macro_f1 = 0
        emb_test = 0
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # output, att, emb1, com1, com2, emb2, emb = model((features, sadj, fadj), training=True)
                output = model([features, empty_input], training=True)
                # print("output: ", output.shape, labels.shape, idx_test.shape)
                loss = loss_fn(tf.gather(labels, idx_train), tf.gather(output, idx_train)) 

                # loss_dep = (loss_dependence(emb1, com1, config.n) + loss_dependence(emb2, com2, config.n))/2
                # loss_com = common_loss(com1,com2)
                # loss = loss_class + config.beta * loss_dep + config.theta * loss_com
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            acc_test, macro_f1, emb_test = main_test(model)
            print('e:{}'.format(epoch),
                'ltr: {:.4f}'.format(loss),
                'atr: {:.4f}'.format(acc_test),
                'ate: {:.4f}'.format(acc_test),
                'f1te:{:.4f}'.format(macro_f1))
        return loss, acc_test, macro_f1, emb_test

    def main_test(model):
        # output, att, emb1, com1, com2, emb2, emb = model((features, sadj, fadj))
        # print(output)
        output = model([features, empty_input], training=False)
        acc_test = accuracy(tf.gather(output, idx_test), tf.gather(labels, idx_test))
        label_max = []
        for idx in idx_test:
            label_max.append(tf.argmax(output[idx]).numpy())
        labelcpu = tf.gather(labels, idx_test)
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1, emb

    acc_max = 0
    f1_max = 0
    epoch_max = 0
    for epoch in range(config.epochs):
        loss, acc_test, macro_f1, emb = train(model, epoch)
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch
    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max))