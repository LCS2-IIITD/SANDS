# -*- coding: utf-8 -*-
"""SiamNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O_lu8V1lt4y3boOt_smcjX7Ho0GQXU8s
"""

from time import time
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors
import os, copy, pickle, string, re, nltk
from scipy.stats import truncnorm
from collections import Counter
from sklearn.model_selection import train_test_split
from itertools import product
from tensorflow.python.keras.utils import losses_utils
from sklearn.utils import class_weight
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout
from sklearn.metrics import classification_report
import argparse

nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="path of the directory containing data files")
parser.add_argument("--glove_vector_file", required=True)
parser.add_argument("--task", required=True, help="USA or India")

args = vars(parser.parse_args())
data_dir = args["data"]
# TARGET = args["target"]
GLOVE_FILE = args["glove_vector_file"]
task_name = args["task"]

if task_name=='USA':
    TARGET = 'republican democratic political party usa'
elif task_name=='India':
    TARGET = 'congress bjp aap political party india'
else:
    sys.exit()

with open(data_dir + '/train', 'rb') as F:
    train = pickle.load(F)

with open(data_dir + '/test', 'rb') as F:
    test = pickle.load(F)

TweetInfoDF = pd.read_csv(data_dir + '/TweetInfoDF.csv')
TweetInfoDFText = list(TweetInfoDF['text'])
num_class = len(np.unique(train['Tag']))

def load_glove_matrix(vec_file):
    word2vec = {}
    with open(vec_file, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word2vec[word] = coefs
    print('Found %s word vectors.' % len(word2vec))
    return word2vec

word2vec = load_glove_matrix(GLOVE_FILE)

def remove_punctuations_and_numbers(String):
    L = []
    for s in tokenize(String):
        if(s not in string.punctuation)and not(s>="0" and s<="9")  and not(s=="???") and 'a'<=s and s<='z':
            L.append(s)
    return " ".join(L)

def tokenize(String):
    return nltk.word_tokenize(String)

def remove_stopwords(List):
    L = []
    for s in List:
        if(s not in stopwords.words("english") and (s not in hindi_stp) and (s not in hinglish_stp)):
            L.append(s)
    return L

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def generate_sample_weights(training_data, class_weight_dictionary): 
    sample_weights = [class_weight_dictionary[np.where(one_hot_row==1)[0][0]] for one_hot_row in training_data]
    return np.asarray(sample_weights)

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def save_glove_matrix(word2vec, word_index, output_file, mean,max_val,std_dev):
    MAX_NB_WORDS = 200000
    WORD_EMBEDDING_DIM = 200

    nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((nb_words, WORD_EMBEDDING_DIM))
    for word, i in tqdm(word_index.items()):
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = get_truncated_normal(mean=mean, sd=std_dev, upp=max_val).rvs(WORD_EMBEDDING_DIM)

    print('Vocabulary size: %d' % len(word_index))
    print('Valid word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) != 0))
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    print('saving glove matrix: %s ...' % output_file)
    np.save(output_file, embedding_matrix)
    print('saved.')

def encoding(max_length, TAG_TWEET, TAG):
    text_tokenizer = Tokenizer()

    text_tokenizer.fit_on_texts(TAG_TWEET)
    INDEXES = text_tokenizer.word_index
    
    ENCODED_TAG_TWEET = []
    
    for xyz in TAG_TWEET:
        tok = tokenize(xyz)
        LI = []
        for _ in tok:
            LI.append(INDEXES[_])
        ENCODED_TAG_TWEET.append(LI)
    print(len(ENCODED_TAG_TWEET))

    ENCODED_TAG_TWEET=pad_sequences(ENCODED_TAG_TWEET,maxlen=max_length,padding='post',value=0.0)

    return (text_tokenizer,pd.DataFrame(list(zip(TAG, ENCODED_TAG_TWEET)), columns =['Tag', 'Tweet']))

def mapping(TagList):
    NewTagList = []
    for tag in TagList:
        L = np.zeros((1,num_class))
        tag_list = [tag]
        for m in tag_list:
            L[0,m] = 1
        NewTagList.append(L)
    return NewTagList

cleaned_tweets = []
L = 0

for i,j in train.iterrows():
    tw = TweetInfoDFText[i].lower()
    tw = strip_links(strip_all_entities(tw))
    tw = remove_punctuations_and_numbers(tw)
    cleaned_tweets.append(tw)
    L_ = len(tokenize(tw))
    if L_>L:
        L=L_

train_cleaned_tweets = copy.deepcopy(cleaned_tweets)

cleaned_tweets = []

for i,j in test.iterrows():
    tw = TweetInfoDFText[i].lower()
    tw = strip_links(strip_all_entities(tw))
    tw = remove_punctuations_and_numbers(tw)
    cleaned_tweets.append(tw)
    L_ = len(tokenize(tw))
    if L_>L:
        L=L_

test_cleaned_tweets = copy.deepcopy(cleaned_tweets)

cleaned_tweets = copy.deepcopy(train_cleaned_tweets)
cleaned_tweets.extend(test_cleaned_tweets)
TagList = pd.concat([train['Tag'], test['Tag']])
text_tokenizer, encodedtexttag = encoding(L, cleaned_tweets, TagList)

X_train = copy.deepcopy(train)
X_test = copy.deepcopy(test)  
y_train = copy.deepcopy(train['Tag'])
y_test = copy.deepcopy(test['Tag'])
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

y_train = mapping(y_train)
y_train = np.asarray(y_train)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
print(y_train.shape)

y_test = mapping(y_test)
y_test = np.asarray(y_test)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])

sample_weights_train = generate_sample_weights(y_train, class_weights)

word_index = text_tokenizer.word_index

tok = tokenize(TARGET)
LI = []
for _ in tok:
    LI.append(word_index[_])
target_seq = pad_sequences([LI],maxlen=L,padding='post',value=0.0)
target_seq = target_seq.reshape(target_seq.shape[1],)

X_train['cleaned_tweet'] = train_cleaned_tweets
X_test['cleaned_tweet'] = test_cleaned_tweets
X_train['encoded_cleaned_tweet'] = list(encodedtexttag['Tweet'][:len(train_cleaned_tweets)])
X_test['encoded_cleaned_tweet'] = list(encodedtexttag['Tweet'][len(train_cleaned_tweets):])
X_train['target'] = [target_seq]*len(train_cleaned_tweets)
X_test['target'] = [target_seq]*len(test_cleaned_tweets)

del X_train['Text']
del X_train['Hashtag']
del X_test['Text']
del X_test['Hashtag']

W2VEC = []

for s in word2vec:
    if len(word2vec[s])==200:
        W2VEC.append(word2vec[s])

W2VEC = np.asarray(W2VEC)
W2VEC.shape

mean = np.mean(W2VEC)
max_val = np.max(W2VEC)
std_dev = np.std(W2VEC)

print(mean,max_val,std_dev)

save_glove_matrix(word2vec, word_index,
                  data_dir+"glove_matrix",
                  mean, max_val, std_dev)

embedding_matrix = np.load(open(data_dir+"glove_matrix.npy", 'rb'))

num_tokens = len(word_index) + 1
batch_size = 32
n_epoch = 10
hidden_dim = 265*2
embedding_dim = 200

class InverseExpManDist(Layer):
    def __init__(self, **kwargs):
        self.result = None
        super(InverseExpManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(InverseExpManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

class AttentionLayer(Layer):
    """
    Hierarchial Attention Layer as described by Hierarchical Attention Networks for Document Classification(2016)
    - Yang et. al.
    Source: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    Theano backend
    """
    def __init__(self,attention_dim=100,return_coefficients=False,**kwargs):
        # Initializer 
        self.return_coefficients = return_coefficients
        self.init = tf.keras.initializers.get('glorot_uniform') # initializes values with uniform distribution
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)),name='W')
        self.b = K.variable(self.init((self.attention_dim, )),name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)),name='u')
        self.trainable_weights = [self.W, self.b, self.u]

        super(AttentionLayer, self).build(input_shape)

    def call(self, hit):
        # Here, the actual calculation is done
        uit = K.bias_add(K.dot(hit, self.W),self.b)
        uit = K.tanh(uit)
        
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = hit * ait
        
        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), ait]
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]

class Model (tf.keras.models.Model):
  def __init__(self,
               embedding_dim=embedding_dim,
               vocab_size=num_tokens,
               hidden_dim=hidden_dim,
               num_classes=5,
               drop_rate=0.2,
               **kwargs):
    super(Model, self).__init__()
    self.embedding = embedding_layer = Embedding(num_tokens, embedding_dim,
                                                 embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                                 trainable=False,)
    self.bilstm = Bidirectional(tf.keras.layers.LSTM(hidden_dim,
                                                     recurrent_dropout=drop_rate,
                                                     return_sequences=True))
    self.attention = AttentionLayer(embedding_dim)
    self.malstm_distance = InverseExpManDist()
    self.dense = Dense(num_classes, activation='softmax')

  def forward_once(self, x):
        # Forward pass 
        output = self.embedding(x)
        output = self.bilstm(output)
        return output
  
  def call(self, tweet, target, training):
    outputs = self.malstm_distance([self.forward_once(tweet),
                                   self.forward_once(target)])
    outputs = self.dense(outputs)
    return outputs

def train_step(tweet_list, actual_v, target, sample_weights_list):
    tweet_list = tf.convert_to_tensor(tweet_list, dtype=tf.int32)
    target = tf.convert_to_tensor(target, dtype=tf.int32)

    with tf.GradientTape(persistent=True) as tape:
        prediction = model(tf.convert_to_tensor(tweet_list), target)
        loss = cce(prediction, actual_v, sample_weight=sample_weights_list)
        epoch_accuracy.update_state(actual_v, prediction)
        
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(
        model.trainable_variables, grads)]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss.update_state(loss)

model = Model(num_classes=num_class)
learning_rate = 2*1e-4
optimizer=keras.optimizers.Adam(learning_rate)
cce = tf.keras.losses.CategoricalCrossentropy()
epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
train_loss = tf.keras.metrics.Mean(name='train_loss')

batches = []

st = 0
while(st<X_train.shape[0]):
    if st+batch_size <X_train.shape[0]:
        batches.append([X_train[st:st+batch_size]['encoded_cleaned_tweet'],
                        y_train[st:st+batch_size],
                        X_train[st:st+batch_size]['target'],
                        sample_weights_train[st:st+batch_size]])
    else:
        batches.append([X_train[st:]['encoded_cleaned_tweet'],y_train[st:],
                        X_train[st:]['target'],
                        sample_weights_train[st:]])

    st = st+batch_size

pbar = tf.keras.utils.Progbar(target=n_epoch*len(batches), width=15, interval=0.005,
                              stateful_metrics=['train_loss', 'accuracy'])

training_start_time = time()
for epoch in range(0,n_epoch):
    
    for encoded_tweet_list, tag, target, sample_weights_list in batches:
          tag_ = np.array(tag)
          tag_ = tag_.reshape(tag_.shape[0],1,tag_.shape[1])
          train_step(list(encoded_tweet_list), tag_, list(target), sample_weights_list)
          pbar.add(1, values=[("train_loss", train_loss.result()),
                              ("accuracy", epoch_accuracy.result())])
    train_loss.reset_states()
    epoch_accuracy.reset_states()

training_end_time = time()
print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                        training_end_time - training_start_time))

Tar = [target_seq for _ in range(len(X_test))]
Tar = tf.convert_to_tensor(Tar)
pred_y = model(tf.convert_to_tensor(list(X_test['encoded_cleaned_tweet'])), Tar)
pred_y = tf.reshape(pred_y, [pred_y.shape[0], pred_y.shape[2]])
print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred_y, axis=1)))