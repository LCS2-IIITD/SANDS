import pandas as pd
import numpy as np
import pickle
import sys
from huggingface_utils import MODELS
from preprocessing import generate_sequence_data
from sklearn.utils import shuffle
from transformers import *
from ust import train_model

import argparse
import logging
import os
import random

from collections import defaultdict
from sklearn.utils import shuffle
from transformers import *
from sklearn.metrics import classification_report
import math
import models
import sampler
import tensorflow as tf
import tensorflow.keras as K

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, help="path of the task directory containing train, test and unlabeled data files")
parser.add_argument("--model_path", required=True, help="path of model")
parser.add_argument("--seq_len", required=True, type=int, help="sequence length")
parser.add_argument("--pt_teacher", nargs="?", default="TFBertModel",help="Pre-trained teacher model")
parser.add_argument("--pt_teacher_checkpoint", nargs="?", default="bert-base-uncased", help="teacher model checkpoint to load pre-trained weights")
parser.add_argument("--do_pairwise", action="store_true", default=False, help="whether to perform pairwise classification tasks like MNLI")
parser.add_argument("--hidden_dropout_prob", nargs="?", type=float, default=0.2, help="dropout probability for hidden layer of teacher model")
parser.add_argument("--attention_probs_dropout_prob", nargs="?", type=float, default=0.2, help="dropout probability for attention layer of teacher model")
parser.add_argument("--dense_dropout", nargs="?", type=float, default=0.5, help="dropout probability for final layers of teacher model")

args = vars(parser.parse_args())
task_name = args["task"]
max_seq_length = args["seq_len"]
model_path = args["model_path"]
pt_teacher = args["pt_teacher"]
pt_teacher_checkpoint = args["pt_teacher_checkpoint"]
do_pairwise = args["do_pairwise"]
dense_dropout = args["dense_dropout"]
attention_probs_dropout_prob = args["attention_probs_dropout_prob"]
hidden_dropout_prob = args["hidden_dropout_prob"]

for indx, model in enumerate(MODELS):
    if model[0].__name__ == pt_teacher:
        TFModel, Tokenizer, Config = MODELS[indx]

tokenizer = Tokenizer.from_pretrained(pt_teacher_checkpoint)

X_test, y_test = generate_sequence_data(max_seq_length,
	task_name+"/test.tsv", tokenizer, do_pairwise=do_pairwise)

labels = set(y_test)
if 0 not in labels:
    y_test -= 1
labels = set(y_test)

model = models.construct_teacher(TFModel, Config, pt_teacher_checkpoint,
                                 max_seq_length, len(labels), dense_dropout=dense_dropout,
                                 attention_probs_dropout_prob=attention_probs_dropout_prob,
                                 hidden_dropout_prob=hidden_dropout_prob)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

model.load_weights(model_path)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1).flatten()
print(classification_report(y_test, y_pred))