import tensorflow as tf
from Github.RP_CATE.model.Cyclic_Sliding_Window import CyclicSlidingWindowsProcessor

class ChannelAttentionModule(tf.keras.layers.Layer):
    def __init__(self, channel_num,windows,seed,expansion_ratio=3):
        super(ChannelAttentionModule, self).__init__()
        self.windows = windows
        self.channel_num = channel_num
        self.expansion_ratio = expansion_ratio
        self.mlp_1_max = tf.keras.layers.Dense(channel_num * expansion_ratio, activation='relu',
                                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))
        self.mlp_2_max = tf.keras.layers.Dense(channel_num, activation='relu',
                                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))
        self.mlp_1_avg = tf.keras.layers.Dense(channel_num * expansion_ratio, activation='relu',
                                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))
        self.mlp_2_avg = tf.keras.layers.Dense(channel_num, activation='relu',
                                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))

        self.CSW = CyclicSlidingWindowsProcessor(self.windows)


    def call(self, input):
        input = self.CSW.process(input)
        MaxPool_channel = tf.reduce_max(tf.reduce_max(input, 1, keepdims=True), 2, keepdims=True)
        AvgPool_channel = tf.reduce_mean(tf.reduce_mean(input, 1, keepdims=True), 2, keepdims=True)
        MaxPool_flatten = tf.keras.layers.Flatten()(MaxPool_channel)
        AvgPool_flatten = tf.keras.layers.Flatten()(AvgPool_channel)
        mlp_1_max = self.mlp_1_max(MaxPool_flatten)
        mlp_2_max = self.mlp_2_max(mlp_1_max)
        mlp_2_max = tf.reshape(mlp_2_max, [-1, 1, 1, self.channel_num])
        mlp_1_avg = self.mlp_1_avg(AvgPool_flatten)
        mlp_2_avg = self.mlp_2_avg(mlp_1_avg)
        mlp_2_avg = tf.reshape(mlp_2_avg, [-1, 1, 1, self.channel_num])

        channel_attention = mlp_2_max + mlp_2_avg
        channel_attention = tf.sigmoid(channel_attention)
        channel_attention = tf.nn.softmax(channel_attention)

        batch_size = len(input)
        channel_attention = tf.reshape(channel_attention, [batch_size, self.channel_num])  # (160,3)
        new_attention = []
        for i in range(batch_size):
              if i < self.windows:
                    part_1 = channel_attention[:(i + 1)]
                    part_2 = channel_attention[-(self.windows - i):]
                    whole = tf.concat([part_1, part_2], axis=0)
                    new_attention.append(tf.reduce_mean(whole, axis=0, keepdims=True))
              else:
                    new_attention.append(tf.reduce_mean(channel_attention[(i - self.windows):i], axis=0, keepdims=True))

        new_attention = tf.concat(new_attention, axis=0)  # (160,3)

        return new_attention

