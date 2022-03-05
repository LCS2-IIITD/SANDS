import sys
import numpy as np
import tensorflow as tf
import codecs, os
import copy, re
import pandas as pd
import pickle
import scipy as sp
from models import ClassifierModel1, ClassifierModel2
from utils import pretrain_classifier, compute_majority_class_with_weights, initialize_op, test_step, max_f1_over_time
from params import model_params, data_params, _common_location, _split_location
import os

data_name, split_size, num_classes = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.exists('LogFiles'):
    os.makedirs('LogFiles')
logname = 'LogFiles/'+'_'.join([data_name,
                                split_size,
                                str(model_params['pretrain_epochs']),
                                str(model_params['semisup_epochs']),
                                str(model_params['pretrain_batch']),
                                str(model_params['semisup_batch']),
                                str(model_params['pretrain_dropout']),
                                str(model_params['semisup_dropout']),
                                str(model_params['min_degree'])])
logfile = open(logname+'.txt', 'w')
data_params['num_classes'] = int(num_classes)


'''Data loading'''

with open(_common_location(data_name, data_params['vocab']), "rb") as F:
    VOCAB = pickle.load(F)
    
with open(_common_location(data_name, data_params['hashtag']), "rb") as F:
    HASHTAGS = pickle.load(F)

with open(_split_location(data_name, split_size, data_params['batch']), "rb") as F:
    train_data = pickle.load(F)
    
with open(_common_location(data_name, data_params['EW']), "rb") as F:
    EW = pickle.load(F)
    
with open(_common_location(data_name, data_params['userlist']), "rb") as F:
    USER_LIST = pickle.load(F)

first_tweets_text = np.load(_common_location(data_name, data_params['first_tweet_text']))
first_tweets_hashtag = np.load(_common_location(data_name, data_params['first_tweet_hashtag']))
embedding_matrix = np.load(_common_location(data_name, data_params['word_embedding']))

cls_tweet_list = np.load(_split_location(data_name, split_size, data_params['train_text']))
cls_hashtag_list = np.load(_split_location(data_name, split_size, data_params['train_hashtag']))
cls_label_list = np.load(_split_location(data_name, split_size, data_params['train_labels']))
traindata = [cls_tweet_list, cls_hashtag_list, cls_label_list]

test_tweets = np.load(_split_location(data_name, split_size, data_params['test_text']))
test_hashtags = np.load(_split_location(data_name, split_size, data_params['test_hashtag']))
test_labels = np.load(_split_location(data_name, split_size, data_params['test_labels']))
if len(test_labels.shape)>1:
    test_labels = np.argmax(test_labels, axis=-1)
testdata = tf.data.Dataset.from_tensor_slices((test_tweets, test_hashtags, test_labels)).batch(model_params['test_batch'])

"""# Training"""
classifier_model1 = ClassifierModel1(embedding_matrix.shape[0], 
                                     len(HASHTAGS)+1, 
                                     embedding_matrix,
                                     data_params['num_classes'])
classifier_model2 = ClassifierModel2(embedding_matrix.shape[0], 
                                     len(HASHTAGS)+1, 
                                     embedding_matrix,
                                     data_params['num_classes'])

optimizer_classifier1 = tf.keras.optimizers.Adam(model_params['lr'])
optimizer_classifier2 = tf.keras.optimizers.Adam(model_params['lr'])

ckpt = tf.train.Checkpoint(classifier_model1=classifier_model1,
                           classifier_model2=classifier_model2, 
                           optimizer_classifier1=optimizer_classifier1,
                           optimizer_classifier2=optimizer_classifier2)
checkpoint_path = 'Model_'+logname+'/'
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored; Model was trained for {} steps.'.format(ckpt.optimizer_classifier2.iterations.numpy()))
else:
    print('Training from scratch!')

cce_soft = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)
cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)

total_ = 0
for element in train_data:
    try:
        encoded_tweet_list, encoded_hashtag_list, user_list, indices, class_list, label_list, weight_list = element
    except ValueError:
        tid, encoded_tweet_list, encoded_hashtag_list, user_list, indices, class_list, label_list, weight_list = element    
    b_size = len(indices)
    start = 0
    while(b_size>model_params['semisup_batch']):
        start = start+model_params['semisup_batch']
        b_size = b_size-model_params['semisup_batch']
        total_ += 1
    if b_size>0:
        total_ = total_+1


def train_step(tweet_list, encoded_hashtag_list, indices, Labels, Weights):
    with tf.GradientTape(persistent=True) as tape:
        classifier_prediction1 = classifier_model1(tf.convert_to_tensor(tweet_list, dtype=tf.int32), 
                                                   tf.convert_to_tensor(encoded_hashtag_list, dtype=tf.int32),
                                                   drop_rate=model_params['semisup_dropout'])
        classifier_prediction2 = classifier_model2(tf.convert_to_tensor(tweet_list, dtype=tf.int32), 
                                                   tf.convert_to_tensor(encoded_hashtag_list, dtype=tf.int32),
                                                   drop_rate=model_params['semisup_dropout'])
        EW_batch = tf.convert_to_tensor(EW[indices].toarray(), dtype=tf.int32)
        labels1, weights1 = compute_majority_class_with_weights(EW_batch, Op2)
        labels2, weights2 = compute_majority_class_with_weights(EW_batch, Op1)
        mask = tf.cast(tf.greater_equal(tf.reduce_max(EW_batch, axis=-1), model_params['min_degree']), dtype=tf.float32)
        loss_a = cce_soft(labels1, classifier_prediction1, sample_weight=mask*weights1)
        loss_b = cce_soft(labels2, classifier_prediction2, sample_weight=mask*weights2)
        loss_c = cce(tf.convert_to_tensor(Labels), classifier_prediction1, sample_weight=Weights)
        loss_d = cce(tf.convert_to_tensor(Labels), classifier_prediction2, sample_weight=Weights)
        loss1 = loss_a + loss_c
        loss2 = loss_b + loss_d
    
    grads_classifier1 = tape.gradient(loss1, classifier_model1.trainable_variables)
    grads_classifier2 = tape.gradient(loss2, classifier_model2.trainable_variables)
    optimizer_classifier1.apply_gradients(zip(grads_classifier1, classifier_model1.trainable_variables))
    optimizer_classifier2.apply_gradients(zip(grads_classifier2, classifier_model2.trainable_variables))
    
    Op1[indices, :] = classifier_prediction1
    Op2[indices, :] = classifier_prediction2


for epoch in range(0,model_params['semisup_epochs']):
    classifier_model1, optimizer_classifier1 = pretrain_classifier(classifier_model1, 
                                                               optimizer_classifier1, 
                                                               traindata,
                                                               testdata,
                                                               data_params['num_classes'], 
                                                               logfile,
                                                               'Sup C1',
                                                               dropout = model_params['pretrain_dropout'], 
                                                               batch_size=model_params['pretrain_batch'],
                                                               epochs=model_params['pretrain_epochs'])

    classifier_model2, optimizer_classifier2 = pretrain_classifier(classifier_model2, 
                                                               optimizer_classifier2, 
                                                               traindata,
                                                               testdata,
                                                               data_params['num_classes'], 
                                                               logfile,
                                                               'Sup C2', 
                                                               dropout=model_params['pretrain_dropout'],
                                                               batch_size=model_params['pretrain_batch'],
                                                               epochs=model_params['pretrain_epochs'])

    Op1 = initialize_op(classifier_model1, first_tweets_text, first_tweets_hashtag, model_params['test_batch'])
    Op2 = initialize_op(classifier_model2, first_tweets_text, first_tweets_hashtag, model_params['test_batch'])
    pbar = tf.keras.utils.Progbar(target=total_,
                                  width=15, 
                                  interval=0.005)

    steps = 0
    c = 0
    for element in train_data:
        try:
            encoded_tweet_list, encoded_hashtag_list, user_list, indices, class_list, label_list, weight_list = element
        except ValueError:
            tid, encoded_tweet_list, encoded_hashtag_list, user_list, indices, class_list, label_list, weight_list = element
        b_size = len(indices)
        start = 0
        
        while(b_size>model_params['semisup_batch']):
            train_step(list(encoded_tweet_list)[start:start+model_params['semisup_batch']],
                        list(encoded_hashtag_list)[start:start+model_params['semisup_batch']],
                        list(indices)[start:start+model_params['semisup_batch']],
                        list(label_list)[start:start+model_params['semisup_batch']],
                        list(weight_list)[start:start+model_params['semisup_batch']])
            start = start+model_params['semisup_batch']
            b_size = b_size-model_params['semisup_batch']
            pbar.add(1)
            steps += 1
            if steps%200 == 0:
                test_step(classifier_model1, 
                          testdata, 
                          data_params['num_classes'], 
                          logfile, 'Semi-sup C1', 
                          optimizer_classifier1.iterations.numpy())
                test_step(classifier_model2, 
                          testdata, 
                          data_params['num_classes'], 
                          logfile, 'Semi-sup C2', 
                          optimizer_classifier2.iterations.numpy())
        if b_size>0:
            train_step(list(encoded_tweet_list)[start:start+model_params['semisup_batch']],
                        list(encoded_hashtag_list)[start:start+model_params['semisup_batch']],
                        list(indices)[start:start+model_params['semisup_batch']],
                        list(label_list)[start:start+model_params['semisup_batch']],
                        list(weight_list)[start:start+model_params['semisup_batch']])
            pbar.add(1)
            steps += 1
            if steps%200 == 0:
                test_step(classifier_model1, 
                          testdata, 
                          data_params['num_classes'], 
                          logfile, 'Semi-sup C1', 
                          optimizer_classifier1.iterations.numpy())
                test_step(classifier_model2, 
                          testdata, 
                          data_params['num_classes'], 
                          logfile, 'Semi-sup C2', 
                          optimizer_classifier2.iterations.numpy())
    ckpt_save_path = ckpt_manager.save()
    print ('\nSaving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
logfile.close()
max_f1_over_time(logname)
