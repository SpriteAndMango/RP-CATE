import tensorflow as tf
from RP_CATE.model.Channel_Attention_Module import ChannelAttentionModule



class RP_CATE(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, dense_dim, output_dim, windows,seed, U, V, W, Wd, N, expansion, Wd1, Wd2, Wd3):
        super(RP_CATE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim
        self.output_dim = output_dim
        self.windows = windows
        self.expansion = expansion

        self.U = U
        self.W = W
        self.V = V

        self.Wd1 = Wd1
        self.Wd2 = Wd2
        self.Wd3 = Wd3
        self.Wd = Wd

        self.N = N

        self.attention_modules = [ChannelAttentionModule(input_dim,windows,seed,expansion) for _ in range(N)]

    def call(self, X):
        input = X
        m = X.get_shape().as_list()[0]
        X_ = []
        attentions = []
        data = []
        means_1 = []
        means_2 = []
        variances_1 = []
        variances_2 = []
        for j in range(self.N):
            if j > 0:
                X = X + input

            hidden_values = []
            hidden_values.append(tf.zeros((1, self.hidden_dim)))
            for i in range(m):
                X_i = tf.reshape(X[i], [1, -1])
                h_keep = tf.sigmoid(tf.matmul(X_i, self.U) + tf.matmul(hidden_values[i], self.W))
                y_d = tf.sigmoid(tf.matmul(h_keep, self.V))

                X_.append(tf.reshape(y_d, [1, -1]))
                hidden_values.append(h_keep)
            X_ = tf.concat(X_, axis=0)
            X = X_
            X_ = []


            data.append(X)

            attention_module = self.attention_modules[j]
            channel_attention = attention_module(X)
            attentions.append(channel_attention)
            X = X * channel_attention
            X = tf.sigmoid(tf.matmul(X, self.Wd1))
            X = tf.sigmoid(tf.matmul(X, self.Wd2))
            X = tf.sigmoid(tf.matmul(X, self.Wd3))


        y_hat = tf.matmul(X, self.Wd)

        return y_hat, attentions, data, means_1, variances_1, means_2, variances_2
