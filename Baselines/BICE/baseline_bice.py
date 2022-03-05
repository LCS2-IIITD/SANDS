from gensim.models import word2vec, Phrases
import logging
from readwrite import reader, writer
from stancedetection.preprocess import tokenise_tweets, transform_tweet
from stancedetection.word2vec_training import trainWord2VecModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
import os, pickle, copy
from sklearn.metrics import classification_report
import argparse
import nltk
from tensorflow.python.ops import variable_scope as vs
from tensorflow.keras.layers import *
import time
import numpy.ma as ma

nltk.download('stopwords')

np.random.seed(1337)
tf.set_random_seed(1337)

print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)


parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="path of the directory containing data files")
parser.add_argument("--task", required=True, help="USA or India")

args = vars(parser.parse_args())
data_dir = args["data"]
task_name = args["task"]

if task_name=='USA':
    TARGET = 'American Political Parties'
    LABEL_MAPPING_INV = {4:'NONE', 0:'AGAINST DEMOCRATIC PARTY', 1:'FOR DEMOCRATIC PARTY',
                        2:'AGAINST REPUBLICAN PARTY', 3:'FOR REPUBLICAN PARTY'}

elif task_name=='India':
    TARGET = 'Indian Political Parties'
    LABEL_MAPPING_INV = {6:'NONE', 0:'AGAINST BJP', 1:'FOR BJP', 2:'AGAINST CONGRESS', 3:'FOR CONGRESS', 4:'AGAINST AAP', 5:'FOR AAP'}

else:
    sys.exit()


KEYWORDS = {'clinton': ['hillary', 'clinton'],
            'trump': ['donald trump', 'trump', 'donald'],
            'climate': ['climate'],
            'feminism': ['feminism', 'feminist'],
            'abortion': ['abortion', 'aborting'],
            'atheism': ['atheism', 'atheist'],
            'indianpolitics': ['bjp', 'aap', 'congress'],
            'americanpolitics': ['democratic', 'repulican'],
            }

TOPICS_LONG = {'clinton': 'Hillary Clinton',
               'trump': 'Donald Trump',
               'climate': 'Climate Change is a Real Concern',
               'feminism': 'Feminist Movement',
               'abortion': 'Legalization of Abortion',
               'atheism': 'Atheism',
               'indianpolitics': 'Indian Political Parties',
               'americanpolitics': 'American Political Parties'
               }

hidden_size = 265
max_epochs = 10
modeltype = "bicond" # this is default
word2vecmodel = "big"
stopwords = "most"
tanhOrSoftmax = "tanh"
dropout = "true"
pretrain = "pre_cont" # this is default
testsetting = "weaklySup"
testid = "test1"
learning_rate = 1e-3
batch_size = 32
input_size = 100

outfile = "results_quicktest_" + testsetting + "_" + modeltype + "_" + str(hidden_size) + "_" + dropout + "_" + tanhOrSoftmax + "_" + str(max_epochs) + "_" + testid + ".txt"

LOSS_TRACE_TAG = "Loss"
ACCURACY_TRACE_TAG = "Accuracy"

def transform_targets(targets):
    ret = []
    for target in targets:
        if target == "Atheism":
            ret.append("#atheism")
        elif target == "Climate Change is a Real Concern":
            ret.append("#climatechange")
        elif target == "Feminist Movement":
            ret.append("#feminism")
        elif target == "Hillary Clinton":
            ret.append("#hillaryclinton")
        elif target == "Legalization of Abortion":
            ret.append("#prochoice")
        elif target == "Donald Trump":
            ret.append("#donaldtrump")
        elif target == "Indian Political Parties":
            ret.append("#indianpoliticalparties")
        elif target == "American Political Parties":
            ret.append("#americanpoliticalparties")
    return ret


def transform_labels(labels, dim=3):
    labels_t = []
    for lab in labels:
        v = np.zeros(dim)
        if dim == 7:
            if lab == 'NONE':
                ix = 0
            elif lab == 'AGAINST BJP':
                ix = 1
            elif lab == 'FOR BJP':
                ix = 2
            elif lab == 'AGAINST AAP':
                ix = 3
            elif lab == 'FOR AAP':
                ix = 4
            elif lab == 'AGAINST CONGRESS':
                ix = 5
            elif lab == 'FOR CONGRESS':
                ix = 6
        if dim == 5:
            if lab == 'NONE':
                ix = 4
            elif lab == 'AGAINST DEMOCRATIC PARTY':
                ix = 0
            elif lab == 'FOR DEMOCRATIC PARTY':
                ix = 1
            elif lab == 'AGAINST REPUBLICAN PARTY':
                ix = 2
            elif lab == 'FOR REPUBLICAN PARTY':
                ix = 3
        if dim == 3:
            if lab == 'NONE':
                ix = 0
            elif lab == 'AGAINST':
                ix = 1
            elif lab == 'FAVOR':
                ix = 2
        else:
            if lab == 'AGAINST':
                ix = 0
            elif lab == 'FAVOR':
                ix = 1
        v[ix] = 1
        labels_t.append(v)
    return labels_t

def istargetInTweet(devdata, target_list):
    """
    Check if target is contained in tweet
    :param devdata: development data as a dictionary (keys: targets, values: tweets)
    :param target_short: short version of target, e.g. 'trump', 'clinton'
    :param id: tweet number
    :return: true if target contained in tweet, false if not
    """
    cntr = 0
    ret_dict = {}
    for id in devdata.keys():

        tweet = devdata.get(id)
        target_keywords = KEYWORDS.get(TOPICS_LONG_REVERSE.get(target_list[0]))
        target_in_tweet = False
        for key in target_keywords:
            if key.lower() in tweet.lower():
                target_in_tweet = True
                break
        ret_dict[id] = target_in_tweet
        cntr += 1
    return ret_dict


def istargetInTweetSing(devdata, target_short):
    """
    Check if target is contained in tweet
    :param devdata: development data as a dictionary (keys: targets, values: tweets)
    :param target_short: short version of target, e.g. 'trump', 'clinton'
    :param id: tweet number
    :return: true if target contained in tweet, false if not
    """
    ret_dict = {}
    for id in devdata.keys():

        tweet = devdata.get(id)
        target_keywords = KEYWORDS.get(target_short)
        target_in_tweet = False
        for key in target_keywords:
            if key.lower() in tweet.lower():
                target_in_tweet = True
                break
        ret_dict[id] = target_in_tweet
    return ret_dict


TOPICS_LONG_REVERSE = dict(zip(TOPICS_LONG.values(), TOPICS_LONG.keys()))


with open(data_dir + '/train', 'rb') as F:
    X_train = pickle.load(F)

with open(data_dir + '/test', 'rb') as F:
    X_val = pickle.load(F)

TweetInfoDF = pd.read_csv(data_dir + '/TweetInfoDF.csv', engine='python')

num_classes = len(np.unique(X_train['Tag']))
print(num_classes, "--------")

with open(data_dir + '/EncodedDataFrameWithLabel', 'rb') as F:
    EncodedTweetDFLabel = pickle.load(F)

EncodedTweetDFLabel['Tweet'] = list(TweetInfoDF['text'])
EncodedTweetDFLabel_w_label = EncodedTweetDFLabel[EncodedTweetDFLabel['Tag']!=-1]
EncodedTweetDFLabel_w_label_INFO = copy.deepcopy(EncodedTweetDFLabel_w_label)

TrainTweetDF=pd.DataFrame()
TestTweetDF=pd.DataFrame()
TweetDF=pd.DataFrame()

Train_Tweet_list = []
Train_Target_list = []
Train_Stance_list = []
Train_Id = []

Test_Tweet_list = []
Test_Target_list = []
Test_Stance_list = []
Test_Id = []

Tweet_list = []
Target_list = []
Stance_list = []
Id = []

for i,j in X_train.iterrows():
    try:
        tw = j['Tweet']
    except:
        tw = EncodedTweetDFLabel_w_label['Tweet'][i]
    tag = j['Tag']
    stance_ = LABEL_MAPPING_INV[tag]
    Train_Tweet_list.append(tw)
    Train_Target_list.append(TARGET)
    Train_Stance_list.append(stance_)
    Train_Id.append(i)
    Tweet_list.append(tw)
    Target_list.append(TARGET)
    Stance_list.append(stance_)
    Id.append(i)
TrainTweetDF = pd.DataFrame(list(zip(Train_Id, Train_Tweet_list, Train_Target_list, Train_Stance_list)), columns=['ID', 'Tweet', 'Target', 'Stance'])

for i,j in X_val.iterrows():
    try:
        tw = j['Tweet']
    except:
        tw = EncodedTweetDFLabel_w_label['Tweet'][i]
    tag = j['Tag']
    stance_ = LABEL_MAPPING_INV[tag]
    Test_Tweet_list.append(tw)
    Test_Target_list.append(TARGET)
    Test_Stance_list.append(stance_)
    Test_Id.append(i)
    Tweet_list.append(tw)
    Target_list.append(TARGET)
    Stance_list.append(stance_)
    Id.append(i)

TestTweetDF = pd.DataFrame(list(zip(Test_Id, Test_Tweet_list, Test_Target_list, Test_Stance_list)), columns=['ID', 'Tweet', 'Target', 'Stance'])
TweetDF = pd.DataFrame(list(zip(Id, Tweet_list, Target_list, Stance_list)), columns=['ID', 'Tweet', 'Target', 'Stance'])

print(list(TestTweetDF), TestTweetDF.shape)
print(list(TrainTweetDF), TrainTweetDF.shape)
print(list(TweetDF), TweetDF.shape)

tweets, targets, labels, ids = TweetDF['Tweet'], TweetDF['Target'], TweetDF['Stance'], TweetDF['ID']
tweet_tokens = tokenise_tweets(tweets, stopwords="all")

unk_tokens = [["unk"], ["unk"], ["unk"], ["unk"], ["unk"], ["unk"], ["unk"], ["unk"], ["unk"], ["unk"]]

trainWord2VecModel(unk_tokens+tweet_tokens, "Word2VecModel")
w2vmodel = word2vec.Word2Vec.load("Word2VecModel")

tweets, targets, labels, ids = TrainTweetDF['Tweet'], TrainTweetDF['Target'], TrainTweetDF['Stance'], TrainTweetDF['ID']

tweets_test, targets_test, labels_test, ids_test = TestTweetDF['Tweet'], TestTweetDF['Target'], TestTweetDF['Stance'], TestTweetDF['ID']

stopwords='all'
tweet_tokens = tokenise_tweets(tweets, stopwords)
target_tokens = tokenise_tweets(transform_targets(targets), stopwords)
transformed_tweets = [transform_tweet(w2vmodel.wv, senttoks) for senttoks in tweet_tokens]
transformed_targets = [transform_tweet(w2vmodel.wv, senttoks) for senttoks in target_tokens]
transformed_labels = transform_labels(labels, num_classes)

tweet_tokens_test = tokenise_tweets(tweets_test, stopwords)
target_tokens_test = tokenise_tweets(transform_targets(targets_test), stopwords)

transformed_tweets_test = [transform_tweet(w2vmodel.wv, senttoks) for senttoks in tweet_tokens_test]
transformed_targets_test = [transform_tweet(w2vmodel.wv, senttoks) for senttoks in target_tokens_test]
transformed_labels_test = transform_labels(labels_test, num_classes)

targetInTweet = {}
id_tweet_dict = dict(zip(ids_test, tweets_test))
targetInTweet = istargetInTweet(id_tweet_dict, list(targets_test))


class Encoder(tf.keras.models.Model):
    def __init__(self, cell_factory, input_size, hidden_size, input_dropout=None, output_dropout=None):
        super(Encoder, self).__init__()
        self.cell_factory = cell_factory
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = self.cell_factory(units=self.hidden_size, return_state=True,
                                      dropout=input_dropout,
                                      recurrent_dropout=output_dropout,
                                      input_shape=(input_size,1))
    
    def __call__(self, inputs, start_state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        Args:
          inputs: list of 2D Tensors with shape [batch_size x self.input_size].
          start_state: 2D Tensor with shape [batch_size x self.state_size].
          scope: VariableScope for the created subgraph; defaults to class name.
        Returns:
          A pair containing:
          - Outputs: list of 2D Tensors with shape [batch_size x self.output_size]
          - States: list of 2D Tensors with shape [batch_size x self.state_size].
        """
        with vs.variable_scope(scope or "Encoder"):
            inputs_ = tf.reshape(inputs, (inputs.shape[0],1,inputs.shape[1]))
            encoder_outputs, h, c = self.cell(inputs=inputs_, initial_state=start_state)
            h = tf.cast(h, tf.float32)
            c = tf.cast(c, tf.float32)
            return encoder_outputs, [h, c]

class Hook(object):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, sess, epoch, iteration, model, loss):
        raise NotImplementedError

class TraceHook(object):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer

    def __call__(self, sess, epoch, iteration, model, loss):
        raise NotImplementedError

    def update_summary(self, sess, current_step, title, value):
        cur_summary = tf.summary.scalar(title, value)
        merged_summary_op = tf.summary.merge([cur_summary])  # if you are using some summaries, merge them
        summary_str = sess.run(merged_summary_op)
        self.summary_writer.add_summary(summary_str, current_step)

class AccuracyHook(TraceHook):
    def __init__(self, summary_writer, batcher, placeholders, at_every_epoch):
        super().__init__(summary_writer)
        self.batcher = batcher
        self.placeholders = placeholders
        self.at_every_epoch = at_every_epoch

    def __call__(self, sess, epoch, iteration, model, loss):
        if iteration == 0 and epoch % self.at_every_epoch == 0:
            total = 0
            correct = 0
            for values in self.batcher:
                total += len(values[-1])
                feed_dict = {}
                for i in range(0, len(self.placeholders)):
                    # print(values[i].shape)
                    if(len(values[i].shape)==1):
                        values[i] = values[i].reshape(values[i].shape[0],1)
                    feed_dict[self.placeholders[i]] = values[i]
                truth = np.argmax(values[-1], 1)
                predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)
                correct += sum(truth == predicted)
            acc = float(correct) / total
            self.update_summary(sess, iteration, ACCURACY_TRACE_TAG, acc)
            print("Epoch " + str(epoch) +
                  "\tAcc " + str(acc) +
                  "\tCorrect " + str(correct) + "\tTotal " + str(total))

class SaveModelHookDev(Hook):
    def __init__(self, path, at_every_epoch=5):
        self.path = path
        self.at_every_epoch = at_every_epoch
        self.saver = tf.train.Saver(tf.trainable_variables())

    def __call__(self, sess, epoch, iteration, model, loss):
        if epoch%self.at_every_epoch == 0:
            #print("Saving model...")
            SaveModelHookDev.save_model_dev(self.saver, sess, self.path + "_ep" + str(epoch) + "/", "model.tf")

    def save_model_dev(saver, sess, path, modelname):
        if not os.path.exists(path):
            os.makedirs(path)
        saver.save(sess, os.path.join(path, modelname))

class BatchBucketSampler:
    def __init__(self, data, batch_size=1, buckets=None):
        """
        :param data: a list of higher order tensors where the first dimension
        corresponds to the number of examples which needs to be the same for
        all tensors
        :param batch_size: desired batch size
        :param buckets: a list of bucket boundaries
        :return:
        """
        self.data = data
        self.num_examples = len(self.data[0])
        self.batch_size = batch_size
        self.buckets = buckets
        self.to_sample = list(range(0, self.num_examples))
        np.random.shuffle(self.to_sample)
        self.counter = 0

    def __reset(self):
        self.to_sample = list(range(0, self.num_examples))
        np.random.shuffle(self.to_sample)
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_examples - self.counter <= self.batch_size:
            self.__reset()
            raise StopIteration
        return self.get_batch(self.batch_size)

    def get_batch(self, batch_size):
        if self.num_examples == self.counter:
            self.__reset()
            return self.get_batch(batch_size)
        else:
            num_to_sample = batch_size
            batch_indices = []
            if len(self.to_sample) < num_to_sample:
                batch_indices += self.to_sample
                num_to_sample -= len(self.to_sample)
                self.__reset()
            self.counter += batch_size
            batch_indices += self.to_sample[0:num_to_sample]
            self.to_sample = self.to_sample[num_to_sample:]
            return [x[batch_indices] for x in self.data]

class Trainer(object):
    """
    Object representing a TensorFlow trainer.
    """

    def __init__(self, optimizer, max_epochs, hooks):
        self.loss = None
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.hooks = hooks

    def __call__(self, batcher, placeholders, loss, acc_thresh, pretrain, embedd, sep=False, model=None, session=None):
        self.loss = loss
        minimization_op = self.optimizer.minimize(loss)
        close_session_after_training = False
        if session is None:
            session = tf.Session()
            close_session_after_training = True  # no session existed before, we provide a temporary session

        init = tf.initialize_all_variables()

        if (pretrain == "pre" or pretrain == "pre_cont") and sep == False: # hack if we want to use pre-trained embeddings
            vars = tf.all_variables()
            emb_var = vars[0]
            session.run(emb_var.assign(embedd))
        elif (pretrain == "pre" or pretrain == "pre_cont") and sep == True:
            vars = tf.all_variables()
            emb_var = vars[0]
            emb_var2 = vars[1]
            session.run(emb_var.assign(embedd))
            session.run(emb_var2.assign(embedd))

        session.run(init)
        epoch = 0
        pbar = tf.keras.utils.Progbar(target=self.max_epochs*batcher.num_examples/batcher.batch_size,
                                      width=15, interval=0.005)
        print(self.max_epochs, batcher.batch_size)
        while epoch < self.max_epochs:
            print(epoch)
            iteration = 1
            for values in batcher:
                iteration += 1
                feed_dict = {}
                for i in range(0, len(placeholders)):
                    if(len(values[i].shape)==1):
                        values[i] = values[i].reshape(values[i].shape[0],1)
                    feed_dict[placeholders[i]] = values[i]
                # print(loss.shape)
                _, current_loss = session.run([minimization_op, loss], feed_dict=feed_dict)
                current_loss = sum(current_loss)
                for hook in self.hooks:
                    hook(session, epoch, iteration, model, current_loss)
                pbar.add(1)
            # calling post-epoch hooks
            for hook in self.hooks:
                hook(session, epoch, 0, model, 0)
            epoch += 1

        if close_session_after_training:
            session.close()

        return self.max_epochs-1


def load_model_dev(sess, path, modelname):
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, os.path.join(path, modelname))


def get_model_bidirectional_conditioning(batch_size, max_seq_length, input_size, hidden_size, target_size,
                                         vocab_size, pretrain, tanhOrSoftmax, dropout):
    """
    Bidirectional conditioning model
    :param pretrain:  "pre": use pretrained word embeddings, "pre-cont": use pre-trained embeddings and continue training them, otherwise: random initialisation
    """

    inputs = tf.placeholder(tf.int64, [batch_size, max_seq_length])
    inputs_cond = tf.placeholder(tf.int64, [batch_size, max_seq_length])

    cont_train = True
    if pretrain == "pre":  # continue training embeddings or not. Currently works better to continue training them.
        cont_train = False
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, input_size], -0.1, 0.1),  # input_size is embeddings size
                               name="embedding_matrix", trainable=cont_train)

    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)
    embedded_inputs_cond = tf.nn.embedding_lookup(embedding_matrix, inputs_cond)
    embedded_inputs_cond = tf.cast(embedded_inputs_cond, tf.float32)
    max_seq_length = np.int32(max_seq_length)


    inputs_list = [tf.squeeze(x) for x in
               tf.split(embedded_inputs, max_seq_length, 1)]
    inputs_cond_list = [tf.squeeze(x) for x in
                    tf.split(embedded_inputs_cond, max_seq_length, 1)]

    drop_prob = None
    if dropout:
        drop_prob = 0.1
    lstm_encoder = Encoder(LSTM, input_size, hidden_size, drop_prob, drop_prob)

    start_state=None

    ### FORWARD
    print(embedded_inputs.shape, 'embedded_inputs', inputs_list[0].shape, 'inputs_list', len(inputs_list))

    # [h_i], [h_i, c_i] <-- LSTM
    # [h_i], [h_i] <-- RNN
    fw_outputs = []
    fw_states = []
    for inp in inputs_list:
        fw_output, fw_state = lstm_encoder(inp, start_state, "LSTM")
        fw_outputs.append(fw_output)
        fw_states.append(fw_state)

    # running a second LSTM conditioned on the last state of the first
    fw_outputs_cond = []
    fw_states_cond = []
    for inp in inputs_cond_list:
        fw_output_cond, fw_state_cond = lstm_encoder(inp, fw_states[-1],
                                                     "LSTMcond")
        fw_outputs_cond.append(fw_output_cond)
        fw_states_cond.append(fw_state_cond)

    fw_outputs_fin = fw_outputs_cond[-1]

    ### BACKWARD
    bw_outputs = []
    bw_states = []
    for inp in inputs_list[::-1]:
        bw_output, bw_state = lstm_encoder(inp, start_state, "LSTM_bw")
        bw_outputs.append(bw_output)
        bw_states.append(bw_state)
    
    bw_outputs_cond = []
    bw_states_cond = []
    for inp in inputs_cond_list[::-1]:
        bw_output_cond, bw_state_cond = lstm_encoder(inp, bw_states[-1],
                                                     "LSTMcond_bw")
        bw_outputs_cond.append(bw_output_cond)
        bw_states_cond.append(bw_state_cond)
        

    bw_outputs_fin = bw_outputs_cond[-1]
    print("fw_outputs_fin",  fw_outputs_fin.shape)
    print("bw_outputs_fin",  bw_outputs_fin.shape)
    outputs_fin = tf.concat([fw_outputs_fin, bw_outputs_fin], 1)


    if tanhOrSoftmax == "tanh":
        model = Dense(target_size, activation="tanh")(outputs_fin)
    else:
        model = Dense(target_size, activation="softmax")(outputs_fin)

    return model, [inputs, inputs_cond]


def train(testsetting, w2vmodel, tweets, targets, labels, ids,
          tweets_test, targets_test, labels_test, ids_test, hidden_size,
          max_epochs, tanhOrSoftmax, dropout, modeltype="conditional",
          targetInTweet={}, testid = "test-1", pretrain = "pre_cont", acc_thresh=0.9, sep = False):
    
    target_size = num_classes
    max_seq_length = len(tweets[0])
    print(max_seq_length, 'max_seq_length')

    data = [np.asarray(tweets), np.asarray(targets), np.asarray(ids), np.asarray(labels)]
    X = w2vmodel.wv.syn0
    vocab_size = len(w2vmodel.wv.vocab)

    outfolder = "_".join([testid, modeltype, testsetting, "hidden-" + str(hidden_size), tanhOrSoftmax])
    # outfolder = "/content/drive/My Drive/IP data/Baseline/BiCE/" + outfolder

    model, placeholders = get_model_bidirectional_conditioning(batch_size, max_seq_length, input_size, hidden_size, target_size,
                                                               vocab_size, pretrain, tanhOrSoftmax, dropout)
    
    ids = tf.placeholder(tf.float32, [batch_size, 1], "ids")  #ids are so that the dev/test samples can be recovered later since we shuffle
    targets = tf.placeholder(tf.float32, [batch_size, target_size], "targets")


    loss = tf.nn.softmax_cross_entropy_with_logits(logits= model, labels= targets)   # targets: labels (e.g. pos/neg/neutral)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    batcher = BatchBucketSampler(data, batch_size)
    acc_batcher = BatchBucketSampler(data, batch_size)

    placeholders += [ids]
    placeholders += [targets]

    pad_nr = batch_size - (len(labels_test) % batch_size) + 1 

    tweets_test = np.asarray(tweets_test)
    targets_test = np.asarray(targets_test)
    ids_test = np.asarray(ids_test)
    labels_test = np.asarray(labels_test)

    # print(tweets_test.shape)

    data_test = [np.lib.pad(tweets_test, ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                np.lib.pad(targets_test, ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                np.lib.pad(ids_test.reshape(ids_test.shape[0],1), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                np.lib.pad(labels_test, ((0, pad_nr), (0, 0)), 'constant', constant_values=(0))]

    corpus_test_batch = BatchBucketSampler(data_test, batch_size)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter("./out/save", graph_def=sess.graph_def)
        print('--------',outfolder,'--------')
        hooks = [
            SaveModelHookDev(path=outfolder, at_every_epoch=1),
            AccuracyHook(summary_writer, acc_batcher, placeholders, 2),
        ]

        trainer = Trainer(optimizer, max_epochs, hooks)
        epoch = trainer(batcher=batcher, acc_thresh=acc_thresh, pretrain=pretrain, embedd=X, placeholders=placeholders,
                        loss=loss, model=model, sep=sep)

        print("Applying to test data, getting predictions")
        epoch = max_epochs-1
        predictions_detailed_all = []
        predictions_all = []
        ids_all = []
        print('--------',outfolder,'--------')
        load_model_dev(sess, outfolder + "_ep" + str(epoch), "model.tf")

        total = 0
        correct = 0
        actual_test_label = []
        pred_test_label = []
        for values in corpus_test_batch:
            total += len(values[-1])
            feed_dict = {}
            for i in range(0, len(placeholders)):
                feed_dict[placeholders[i]] = values[i]
            truth = np.argmax(values[-1], 1)  # values[2] is a 3-length one-hot vector containing the labels
            if pretrain == "pre" and sep == True:  # this is a bit hacky. To do: improve
                vars = tf.all_variables()
                emb_var = vars[0]
                emb_var2 = vars[1]
                sess.run(emb_var.assign(X))
                sess.run(emb_var2.assign(X))
            if pretrain == "pre":  # this is a bit hacky. To do: improve
                vars = tf.all_variables()
                emb_var = vars[0]
                sess.run(emb_var.assign(X))
            predictions = sess.run(tf.nn.softmax(model), feed_dict=feed_dict)
            predictions_detailed_all.extend(predictions)
            ids_all.extend(values[-2])
            predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                     feed_dict=feed_dict)
            predictions_all.extend(predicted)
            correct += sum(truth == predicted)
            actual_test_label.extend(truth)
            pred_test_label.extend(predicted)
        
        print(classification_report(actual_test_label, pred_test_label))

    return predictions_all, predictions_detailed_all, ids_all

tf.compat.v1.disable_eager_execution()

predictions_all, predictions_detailed_all, ids_all = train(testsetting, w2vmodel, transformed_tweets,
                                                           transformed_targets, transformed_labels,
                                                           ids, transformed_tweets_test,
                                                           transformed_targets_test, transformed_labels_test, ids_test,
                                                           hidden_size, max_epochs,
                                                           tanhOrSoftmax, dropout, modeltype, targetInTweet,
                                                           testid, pretrain=pretrain)
