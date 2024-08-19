import tensorflow as tf
import numpy as np

class AttentionProcessor:
    def __init__(self, X_train, X_test, channel_attention):
        self.X_train = X_train
        self.X_test = X_test
        self.channel_attention = channel_attention

    def return_attention(self):
        # Determine the most important dimension based on the channel attention
        important_dim = tf.argsort(tf.reduce_mean(self.channel_attention, axis=0))[-1]

        # Sort the training data based on the most important dimension
        X_train_idx = tf.argsort(self.X_train[:, important_dim])
        X_train_idx = tf.cast(X_train_idx, dtype=tf.int32)
        X_train_sorted = tf.gather(self.X_train, X_train_idx, axis=0)

        # Find the corresponding positions of the test data in the sorted training data
        idx = tf.searchsorted(X_train_sorted[:, important_dim], self.X_test[:, important_dim])

        attention_X_test = []
        n_train = X_train_sorted.shape[0]
        n_test = self.X_test.shape[0]
        dis = np.zeros((n_test, 2))

        for i, j in enumerate(idx):
            if j == n_train:
                former = self.channel_attention[j-2]
                later = self.channel_attention[j-1]
                dis_former = tf.norm(self.X_test[i] - X_train_sorted[j-2])
                dis_later = tf.norm(self.X_test[i] - X_train_sorted[j-1])
            else:
                former = self.channel_attention[j-1]
                later = self.channel_attention[j]
                dis_former = tf.norm(self.X_test[i] - X_train_sorted[j-1])
                dis_later = tf.norm(self.X_test[i] - X_train_sorted[j])

            part_1 = (dis_later / (dis_later + dis_former)) * former
            part_2 = (dis_former / (dis_later + dis_former)) * later
            dis[i, 0] = dis_later / (dis_later + dis_former)
            dis[i, 1] = dis_former / (dis_later + dis_former)
            attention_X_test.append(tf.reshape((part_1 + part_2), [1, -1]))

        attention_test = tf.concat(attention_X_test, axis=0)
        return attention_test, important_dim



