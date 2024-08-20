import tensorflow as tf
from RP_CATE.tools.return_test_attention import AttentionProcessor


class RPCATEForward:
    def __init__(self, U, W, V, Wd, Wd1, Wd2, Wd3, N, hidden_dim):
        self.U = U
        self.W = W
        self.V = V
        self.Wd = Wd
        self.Wd1 = Wd1
        self.Wd2 = Wd2
        self.Wd3 = Wd3
        self.N = N
        self.hidden_dim = hidden_dim

    def forward_pass(self, X, attentions, data):
        m = X.get_shape().as_list()[0]
        X_ = []
        data_test = []
        input = X

        for j in range(self.N):
            if j > 0:
                X = X + input

            hidden_values = [tf.zeros((1, self.hidden_dim))]
            for i in range(m):
                X_i = tf.reshape(X[i], [1, -1])
                h_keep = tf.sigmoid(tf.matmul(X_i, self.U) + tf.matmul(hidden_values[i], self.W))
                y_d = tf.sigmoid(tf.matmul(h_keep, self.V))

                X_.append(tf.reshape(y_d, [1, -1]))
                hidden_values.append(h_keep)

            X_ = tf.concat(X_, axis=0)
            X = X_
            X_ = []

            data_test.append(X)
            data_j = data[j]

            # Assume Return_attention is a method in this class or another class
            processor = AttentionProcessor(data_j,data_test[j],attentions[j])

            channel_attention, important_dim = processor.return_attention()

            X = X * channel_attention  # Element-wise multiplication

            X = tf.sigmoid(tf.matmul(X, self.Wd1))
            X = tf.sigmoid(tf.matmul(X, self.Wd2))
            X = tf.sigmoid(tf.matmul(X, self.Wd3))

        y_hat = tf.matmul(X, self.Wd)

        return y_hat


