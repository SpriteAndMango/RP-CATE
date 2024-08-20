import tensorflow as tf
from Github.RP_CATE.dataset.dataset import DatasetProcessor




path = '../Dataset.txt'
processor = DatasetProcessor(path)
data_splits = processor.process()

# Access training and testing data
PSD_train = data_splits['PSD_train'].reshape(1,-1,3)
PSD_test = data_splits['PSD_test'].reshape(1,-1,3)
y_train = data_splits['y_train'].reshape(1,-1)
y_test = data_splits['y_test'].reshape(1,-1)
y_phy_train = data_splits['y_phy_train']
y_Aspen_train = data_splits['y_Aspen_train']
y_phy_test = data_splits['y_phy_test']
y_Aspen_test = data_splits['y_Aspen_test']


model = tf.keras.Sequential([
    tf.keras.layers.LSTM(30,activation='sigmoid',return_sequences=True),
    tf.keras.layers.Dense(20,activation='sigmoid'),
    tf.keras.layers.Dense(10,activation='sigmoid'),
    tf.keras.layers.Dense(1,activation=None)
])


model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=5e-3),loss='mse')


with tf.device('/CPU:0'):
    model.fit(PSD_train,y_train,epochs=2000)


##  train
y_hat_train = model.predict(PSD_train).reshape(-1,1)
y_predict_train = y_phy_train+y_hat_train

## test
y_hat_test = model.predict(PSD_test).reshape(-1,1)
y_predict_test = (y_phy_test+y_hat_test)

