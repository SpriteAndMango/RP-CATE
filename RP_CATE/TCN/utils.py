import torch
import numpy as np
import random



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




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
                res = torch.cat((part_1, part_2), dim=0)

                dataset.append(res)

        New_dataset = torch.stack(dataset)

        return New_dataset
