import numpy as np
import pandas as pd


class DatasetProcessor:
    def __init__(self, path, size_rate=0.3, seed=0):
        self.path = path
        self.size_rate = size_rate
        self.seed = seed

        self.data = None

        self.PSD = None
        self.y = None

        self.y_phy = None
        self.y_Aspen = None

        self.PSD_train = None
        self.PSD_test = None

        self.y_train = None
        self.y_test = None

        self.y_phy_train = None
        self.y_Aspen_train = None
        self.y_phy_test = None
        self.y_Aspen_test = None

    def load_data(self):
        self.data = pd.read_csv(self.path, sep='\t')

    def generate_PSD(self):
        X1 = self.data['Tbr'].values.reshape(-1, 1)
        X2 = self.data['Kw'].values.reshape(-1, 1)
        X3 = self.data['Pc'].values.reshape(-1, 1)
        y_Aspen = self.data['Acentric factor-Aspen'].values.reshape(-1, 1)
        y_phy = self.data['Acentric factor-LK'].values.reshape(-1, 1)
        X = np.c_[X1, X2, X3, y_Aspen, y_phy]


        index_sort = np.argsort(X[:, 0])
        PSD = X[index_sort]

        self.y_phy = PSD[:, -1].reshape(-1, 1)
        self.y_Aspen = PSD[:, -2].reshape(-1, 1)
        self.y = self.y_Aspen - self.y_phy
        self.PSD = PSD[:, :3].reshape(-1, 3)



    def split_data(self):
        size = np.int8(np.round(self.size_rate * len(self.PSD)))
        np.random.seed(self.seed)
        random_index = np.sort(np.random.choice(len(self.PSD), size=size, replace=False))

        self.PSD_test = self.PSD[random_index, :]
        self.y_test = self.y[random_index, :]
        self.y_phy_test = self.y_phy[random_index, :]
        self.y_Aspen_test = self.y_Aspen[random_index, :]

        self.PSD_train = np.delete(self.PSD, random_index, axis=0)
        self.y_train = np.delete(self.y, random_index, axis=0)
        self.y_phy_train = np.delete(self.y_phy, random_index, axis=0)
        self.y_Aspen_train = np.delete(self.y_Aspen, random_index, axis=0)



    def process(self):
        self.load_data()
        self.generate_PSD()
        self.split_data()
        return {
            'PSD_train': self.PSD_train,
            'PSD_test': self.PSD_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'y_phy_train': self.y_phy_train,
            'y_Aspen_train': self.y_Aspen_train,
            'y_phy_test': self.y_phy_test,
            'y_Aspen_test': self.y_Aspen_test
        }



