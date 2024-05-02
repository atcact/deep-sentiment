import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import f1_score
import numpy as np
import os
import argparse
from models.am_gcn.models import AMGCN
from models.am_gcn.utils import load_graph, load_data

nfeat=300
nclass=2
nhid1=256
nhid2=128
n=2
dropout=0.5
fdim = 300

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, required = True)
    args = parse.parse_args()

    tf.random.set_seed(0)

    sadj, fadj = load_graph(args.labelrate, config)
    features, labels, idx_train, idx_test = load_data(config)

    inputs = Input(shape=(fdim,))
    x = AMGCN(nfeat, nhid1, nhid2, nclass, n, dropout)(inputs)
    x = tf.keras.layers.Dense(nclass, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)

    optimizer = Adam(lr=0.001, decay=0.0005)
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)

    def train(model, epochs):
        model.train()
        with tf.GradientTape() as tape:
            output = model(features)
            loss_class = loss_fn(labels[idx_train], output[idx_train])
            loss_dep = (loss_dependence(emb1, com1, config.n) + loss_dependence(emb2, com2, config.n))/2
            loss_com = common_loss(com1,com2)
            loss = loss_class + config.beta * loss_dep + config.theta * loss_com
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc_test, macro_f1, emb_test = main_test(model)
        print('e:{}'.format(epochs),
              'ltr: {:.4f}'.format(loss.numpy()),
              'atr: {:.4f}'.format(acc_test.numpy()),
              'ate: {:.4f}'.format(acc_test.numpy()),
              'f1te:{:.4f}'.format(macro_f1.numpy()))
        return loss.numpy(), acc_test.numpy(), macro_f1.numpy(), emb_test

    def main_test(model):
        model.eval()
        output = model(features)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        label_max = []
        for idx in idx_test:
            label_max.append(tf.argmax(output[idx]).numpy())
        labelcpu = labels[idx_test].numpy()
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