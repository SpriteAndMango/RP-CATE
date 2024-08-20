import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from Github.RP_CATE.dataset.dataset import DatasetProcessor
from training.training import Training_former
from tools.forward import RPCATEForward



if __name__ == '__main__':

      path = 'Dataset.txt'
      processor = DatasetProcessor(path)
      data_splits = processor.process()

      # Access training and testing data
      PSD_train = data_splits['PSD_train']
      PSD_test = data_splits['PSD_test']
      y_train = data_splits['y_train']
      y_test = data_splits['y_test']
      y_phy_train = data_splits['y_phy_train']
      y_Aspen_train = data_splits['y_Aspen_train']
      y_phy_test = data_splits['y_phy_test']
      y_Aspen_test = data_splits['y_Aspen_test']


      PSD_train = tf.convert_to_tensor(PSD_train,dtype = tf.float32)
      y_train = tf.convert_to_tensor(y_train)
      PSD_test = tf.convert_to_tensor(PSD_test,dtype = tf.float32)           #(40,3)
      y_test = tf.convert_to_tensor(y_test)



# RP-CATE
      size_rate = 0.3
      random_seed = 0
      expansion = 3
      windows = 25

      input_dim = 3
      hidden_dim = 15
      dense_dim = 3
      output_dim = 1

      dim_Wd1 = 15
      dim_Wd2 = 9
      dim_Wd3 = dense_dim


      seed = 42
      N = 1
      ite = 3000

      lr = 1e-3

      start_time = time.time()
      with tf.device('/CPU:0'):
           prediction,U, W, V, Wd, attentions,data,means_1,variances_1,means_2,variances_2,Wd1,Wd2,Wd3 = Training_former(PSD_train,y_train,windows,seed,expansion,ite,lr,N,input_dim,hidden_dim,dense_dim,output_dim,dim_Wd1,dim_Wd2,dim_Wd3=3)
      cost_time = time.time() - start_time
      print(tf.reduce_mean(attentions[0], 0))
      print('The whole training process cost {}min{}s '.format(int(cost_time / 60), int(cost_time % 60)))




### forward_former

      forward_model = RPCATEForward(U, W, V, Wd, Wd1, Wd2, Wd3, N, hidden_dim)

      # Perform the forward pass
      y_predict_test = y_phy_test + forward_model.forward_pass(PSD_test,attentions,data)
      print('The test loss is :', tf.keras.losses.MeanSquaredError()(y_predict_test, y_Aspen_test).numpy())



      mae = np.sum(np.abs(y_predict_test - y_Aspen_test)) / len(y_Aspen_test)
      rmse = np.sqrt(np.sum(np.square(y_predict_test - y_Aspen_test)) / len(y_Aspen_test))
      are = np.sum(np.abs(y_predict_test - y_Aspen_test) / y_Aspen_test) / len(y_Aspen_test) * 100
      are_metric = np.abs(y_predict_test - y_Aspen_test) / y_Aspen_test
      num_1 = np.sum([1 if i < 0.01 else 0 for i in are_metric])
      num_5 = np.sum([1 if i > 0.05 else 0 for i in are_metric])
      improvement = (0.5 * (1 - (np.sum(np.abs(y_predict_test - y_Aspen_test)) / np.sum(y_test))) + 0.5 * (
            (num_1 - 17) / (len(y_Aspen_test) - 17))) * 100


      print('Window:',windows)
      print('N:',N)
      print('MAE:', mae)
      print('RMSE:', rmse)
      print('ARE:', are, '%')
      print('IMPRO:', improvement, '%')
      print(num_1)
      print(num_5)

      print('The attentions is :' ,attentions)


###  绘图




      fig2= plt.figure(figsize=(12,8))
      plt.scatter(PSD_test[:,0],y_Aspen_test[:,0],c='w',edgecolors = 'r',label='Aspen Plus',s=100)
      # plt.scatter(X_test[:,1],y_phy_test[:,0],c='b',marker='*',label='LK Results',s=100)
      plt.scatter(PSD_test[:,0],y_predict_test,c='g',marker='*',label='RPCATE-Based Hybrid Model',s=100)
      # plt.xlabel('Retrace Feature',fontsize=25)
      # plt.ylabel('Acentric Factor',fontsize=25)
      plt.xticks(fontsize=25)
      plt.yticks(fontsize=25)
      # plt.legend(fontsize=25)

      plt.show()