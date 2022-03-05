import numpy as np
import tensorflow as tf

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

"""# Classifier Models"""

'''Text convolution-based stance classifier model'''

class ClassifierModel1(tf.keras.models.Model):
    def __init__(self, 
                 vocab_size, 
                 num_hashtags, 
                 embedding_matrix,
                 num_classes,
                 emb_trainable=True, 
                 **kwargs):
        super(ClassifierModel1, self).__init__()

        self.hashtag_embedding = tf.keras.layers.Embedding(num_hashtags, 128, mask_zero=True)
        self.query_dense = tf.keras.layers.Dense(units=128, activation='relu', use_bias=False)
        self.key_dense = tf.keras.layers.Dense(units=128, activation='relu', use_bias=False)
        self.value_dense = tf.keras.layers.Dense(units=128, activation='relu', use_bias=False)
        self.attention = tf.keras.layers.Attention(use_scale=True)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()

        self.vocab_embedding = tf.keras.layers.Embedding(vocab_size, 
                                                200, 
                                                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                                mask_zero=True,
                                                trainable=False)

        self.conv5_1 = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')
        self.pool5_1 = tf.keras.layers.MaxPool1D(5, padding='same')
        self.conv5_2 = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')
        self.pool5_2 = tf.keras.layers.MaxPool1D(5, padding='same')
        self.conv5_3 = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')
        self.pool5_3 = tf.keras.layers.MaxPool1D(5, padding='same')
        self.flat5 = tf.keras.layers.Flatten()

        self.conv3_1 = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')
        self.pool3_1 = tf.keras.layers.MaxPool1D(3, padding='same')
        self.conv3_2 = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')
        self.pool3_2 = tf.keras.layers.MaxPool1D(3, padding='same')
        self.conv3_3 = tf.keras.layers.Conv1D(32, 4, padding='same', activation='relu')
        self.pool3_3 = tf.keras.layers.MaxPool1D(13, padding='same')
        self.flat3 = tf.keras.layers.Flatten()

        self.conv1_1 = tf.keras.layers.Conv1D(128, 1, padding='same', activation='relu')
        self.pool1_1 = tf.keras.layers.MaxPool1D(1, padding='same')
        self.conv1_2 = tf.keras.layers.Conv1D(64, 1, padding='same', activation='relu')
        self.pool1_2 = tf.keras.layers.MaxPool1D(1, padding='same')
        self.conv1_3 = tf.keras.layers.Conv1D(32, 1, padding='same', activation='relu')
        self.pool1_3 = tf.keras.layers.MaxPool1D(100, padding='same')
        self.flat1 = tf.keras.layers.Flatten()

        self.concatenate1 = tf.keras.layers.Concatenate()
        self.normalize = tf.keras.layers.LayerNormalization()
        self.dense_last = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, tweet_list, hashtag_list, training=True, drop_rate=0.1):
        hashtag_out = self.hashtag_embedding(hashtag_list)
        if training:
            hashtag_out = tf.nn.dropout(hashtag_out, rate=drop_rate)
        query = self.query_dense(hashtag_out)
        if training:
            query = tf.nn.dropout(query, rate=drop_rate)
        key = self.key_dense(hashtag_out)
        if training:
            key = tf.nn.dropout(key, rate=drop_rate)
        value = self.value_dense(hashtag_out)
        if training:
            value = tf.nn.dropout(value, rate=drop_rate)
        mask = tf.not_equal(hashtag_list, 0)
        output = self.attention([query, value, key], [mask, mask])
        output = tf.reduce_max(output, axis=1)

        vocab_out = self.vocab_embedding(tweet_list)
        if training:
            vocab_out = tf.nn.dropout(vocab_out, rate=drop_rate)
        
        vocab5_out = self.conv5_1(vocab_out)
        vocab5_out = self.pool5_1(vocab5_out)
        vocab5_out = self.conv5_2(vocab5_out)
        vocab5_out = self.pool5_2(vocab5_out)
        vocab5_out = self.conv5_3(vocab5_out)
        vocab5_out = self.pool5_3(vocab5_out)
        vocab5_out = self.flat5(vocab5_out)

        vocab3_out = self.conv3_1(vocab_out)
        vocab3_out = self.pool3_1(vocab3_out)
        vocab3_out = self.conv3_2(vocab3_out)
        vocab3_out = self.pool3_2(vocab3_out)
        vocab3_out = self.conv3_3(vocab3_out)
        vocab3_out = self.pool3_3(vocab3_out)
        vocab3_out = self.flat3(vocab3_out)

        vocab1_out = self.conv1_1(vocab_out)
        vocab1_out = self.pool1_1(vocab1_out)
        vocab1_out = self.conv1_2(vocab1_out)
        vocab1_out = self.pool1_2(vocab1_out)
        vocab1_out = self.conv1_3(vocab1_out)
        vocab1_out = self.pool1_3(vocab1_out)
        vocab1_out = self.flat1(vocab1_out)

        out = self.concatenate1([vocab5_out, vocab3_out, vocab1_out, output])
        if training:
            out = tf.nn.dropout(out, rate=drop_rate)
        
        o = self.normalize(out)
        o = self.dense_last(o)
        return o

'''Bi-LSTM-based stance classifier model'''

class ClassifierModel2(tf.keras.models.Model):
    def __init__(self,
                 vocab_size, 
                 num_hashtags, 
                 embedding_matrix,
                 num_classes, 
                 **kwargs):
        super(ClassifierModel2, self).__init__()

        self.hashtag_embedding = tf.keras.layers.Embedding(num_hashtags, 128, mask_zero=True)
        self.query_dense = tf.keras.layers.Dense(units=128, activation='relu', use_bias=False)
        self.key_dense = tf.keras.layers.Dense(units=128, activation='relu', use_bias=False)
        self.value_dense = tf.keras.layers.Dense(units=128, activation='relu', use_bias=False)
        self.attention = tf.keras.layers.Attention(use_scale=True)

        self.vocab_embedding = tf.keras.layers.Embedding(vocab_size, 
                                                embedding_matrix.shape[1], 
                                                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                                mask_zero=True)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, return_sequences=True))
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.normalize = tf.keras.layers.LayerNormalization()
        self.dense_last = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, tweet_list, hashtag_list, training=True, drop_rate=0.1):
        hashtag_out = self.hashtag_embedding(hashtag_list)
        if training:
            hashtag_out = tf.nn.dropout(hashtag_out, rate=drop_rate)
        query = self.query_dense(hashtag_out)
        if training:
            query = tf.nn.dropout(query, rate=drop_rate)
        key = self.key_dense(hashtag_out)
        if training:
            key = tf.nn.dropout(key, rate=drop_rate)
        value = self.value_dense(hashtag_out)
        if training:
            value = tf.nn.dropout(value, rate=drop_rate)
        mask = tf.not_equal(hashtag_list, 0)
        hashtag_out = self.attention([query, value, key], [mask, mask])
        hashtag_out = tf.reduce_max(hashtag_out, axis=1)
        if training:
            hashtag_out = tf.nn.dropout(hashtag_out, rate=drop_rate)

        emb_tweet = self.vocab_embedding(tweet_list)
        if training:
            emb_tweet = tf.nn.dropout(emb_tweet, rate=drop_rate)
        lstm_tweet = self.bilstm(emb_tweet)
        tweet_out = tf.reduce_max(lstm_tweet, axis=1)
        if training:
            tweet_out = tf.nn.dropout(tweet_out, rate=drop_rate)

        out = self.concat([tweet_out, hashtag_out])
        out = self.normalize(out)
        out = self.dense_last(out)
        return out
