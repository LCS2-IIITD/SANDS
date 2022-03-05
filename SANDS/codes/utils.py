import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from matplotlib import pyplot as plt

def compute_majority_class_with_weights(edge_weight, follower_opinion):
    s = tf.expand_dims(edge_weight, axis=-1)*follower_opinion
    p_classes = tf.cast(tf.argmax(s, axis=-1), dtype=tf.int32)
    num_classes = tf.shape(follower_opinion)[-1]
    p_class_counts = tf.math.bincount(p_classes, minlength=num_classes, weights=edge_weight, axis=-1)
    labels = tf.cast(tf.argmax(p_class_counts, axis=-1), dtype=tf.int32)
    class_counts = tf.cast(tf.math.bincount(labels, minlength=num_classes), dtype=tf.float32)
    class_weights = class_counts/tf.reduce_sum(class_counts)
    sample_weights = 1./(tf.gather(class_weights, labels)+tf.keras.backend.epsilon())
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes), sample_weights

def initialize_op(classifier_model, text, hashtag, batch_size):
    X = tf.data.Dataset.from_tensor_slices((text, hashtag)).batch(batch_size)
    Y = []
    for t, h in X:
        Y.append(classifier_model(t, h))
    return tf.concat(Y, axis=0).numpy().astype('float32')

def test_step(classifier_model, test_data, num_classes, logfile, header, steps):
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    test_f1 = tfa.metrics.F1Score(num_classes=num_classes, average='macro')
    per_class_f1 = tfa.metrics.F1Score(num_classes=num_classes, average=None)
    for tweets, hashtags, labels in test_data:
        pred = classifier_model(tweets, hashtags, training=False)
        label_categorical = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        test_accuracy.update_state(labels, pred)
        test_f1.update_state(label_categorical, pred)
        per_class_f1.update_state(label_categorical, pred)
    logfile.write('{} {} {:.4f} {:.4f}\n'.format(header, steps, test_accuracy.result(), test_f1.result()))
    logfile.write('{}\n'.format(per_class_f1.result()))
    test_accuracy.reset_states()
    test_f1.reset_states()
    per_class_f1.reset_states()

def pretrain_classifier(classifier_model, 
                        optimizer,
                        traindata,
                        testdata,
                        num_classes,
                        logfile,
                        header,
                        dropout=0.2, 
                        batch_size=64,
                        epochs=8):
    cls_tweet_list, cls_hashtag_list, cls_label_list = traindata
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)
    if len(cls_label_list.shape)==1:
        cls_label_list = tf.keras.utils.to_categorical(cls_label_list, num_classes=num_classes)
    sample_bias = np.histogram(np.argmax(cls_label_list, axis=1), 
                               bins=np.arange(np.amax(np.argmax(cls_label_list, axis=1))+1))[1]
    cls_weights = np.log(cls_label_list.shape[0]/(1.+sample_bias))
    weights = np.array([cls_weights[i] for i in np.argmax(cls_label_list, axis=1)])
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)

    def train_step_classifier(tweet_list, hashtag_list, Labels, weight):
        with tf.GradientTape(persistent=True) as tape:
            classifier_prediction = classifier_model(tf.convert_to_tensor(tweet_list, dtype=tf.int32), 
                                                     tf.convert_to_tensor(hashtag_list, dtype=tf.int32),
                                                     drop_rate=dropout)
            loss = cce(tf.convert_to_tensor(Labels), classifier_prediction, sample_weight=weight)
            
        grads_classifier = tape.gradient(loss, classifier_model.trainable_variables)
        optimizer.apply_gradients(zip(grads_classifier, classifier_model.trainable_variables))

    train_dataset = tf.data.Dataset.from_tensor_slices((cls_tweet_list,
                                                        cls_hashtag_list,
                                                        cls_label_list,
                                                        weights)).batch(batch_size)
    for epoch in range(epochs):
        for tweet, hashtag, label, weight in train_dataset:
            train_step_classifier(tweet, hashtag, label, weight)
        test_step(classifier_model, testdata, num_classes, logfile, header, optimizer.iterations.numpy())
    return classifier_model, optimizer


def max_f1_over_time(filename):
    resultfile = open(filename+'.txt','r')
    savefile = filename+'.png'
    C1, C2 = [], []
    C1_steps, C2_steps = [], []
    max_f1 = 0.
    max_f1_step = 0
    for line in resultfile.readlines():
        if 'C' in line:
            f1 = float(line.split()[-1])
            step = int(line.split()[2])
            if 'C1' in line:
                C1.append(f1)
                C1_steps.append(step)
            elif 'C2' in line:
                C2.append(f1)
                C2_steps.append(step)
            if f1 > max_f1:
                max_f1 = f1
                max_f1_step = step
    resultfile.close()
    print('Maximum macro-f1 achieved: {} in {} steps'.format(max_f1, max_f1_step))
