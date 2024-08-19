import numpy as np
import tensorflow as tf

class CyclicSlidingWindowsProcessor:
    def __init__(self, windows):
        self.windows = windows
        self.step = int(np.sqrt(windows))

    def process(self, X):
        n = len(X)
        dataset = []
        for i in range(n):
            if i + self.windows - 1 < n:
                dataset.append(X[i:(i + self.windows)])
            else:
                res_num = i + self.windows - 1 - n
                part_1 = X[i:]
                part_2 = X[:(res_num + 1)]
                res = tf.concat([part_1, part_2], axis=0)
                dataset.append(res)

        New_dataset = tf.stack(dataset)
        New_dataset = tf.reshape(New_dataset, [n, self.step, self.step, 3])

        return New_dataset



