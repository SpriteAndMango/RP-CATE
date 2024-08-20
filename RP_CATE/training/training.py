import tensorflow as tf
from Github.RP_CATE.model.RPCATE import RP_CATE



def Training_former(X,y,windows,seed,expansion,ite,lr,N,input_dim,hidden_dim,dense_dim,output_dim,dim_Wd1,dim_Wd2,dim_Wd3=3):
      tf.random.set_seed(seed)
      U = tf.Variable(tf.random.normal((input_dim, hidden_dim)) * tf.sqrt(2 / (input_dim)),
                      trainable=True)  # (3,20)
      W = tf.Variable(tf.random.normal((hidden_dim, hidden_dim)) * tf.sqrt(2 / (hidden_dim)),
                      trainable=True)  # (20,20)
      V = tf.Variable(tf.random.normal((hidden_dim, dense_dim)) * tf.sqrt(2 / (hidden_dim)),
                      trainable=True)  # (20,3)

      Wd1 = tf.Variable(tf.random.normal((dense_dim,dim_Wd1)) * tf.sqrt(2 / (dense_dim)),
                       trainable=True)  # (3,20)
      Wd2 = tf.Variable(tf.random.normal((dim_Wd1, dim_Wd2)) * tf.sqrt(2 / (dim_Wd1)),
                       trainable=True)  # (20,10)
      Wd3 = tf.Variable(tf.random.normal((dim_Wd2, dim_Wd3)) * tf.sqrt(2 / (dim_Wd2)),
                       trainable=True)  # (10,3)
      Wd = tf.Variable(tf.random.normal((dim_Wd3, output_dim)) * tf.sqrt(2 / (dim_Wd3)),
                       trainable=True)  # (3,1)



      opt = tf.keras.optimizers.legacy.Adam(lr)
      loss_func = tf.keras.losses.MeanSquaredError()

      irp_layer = RP_CATE(input_dim, hidden_dim, dense_dim, output_dim, windows,seed, U, V, W, Wd, N, expansion,Wd1,Wd2,Wd3)

      for i in range(ite):
            with tf.GradientTape() as tape:

                  prediction,attentions,data,means_1,variances_1,means_2,variances_2 = irp_layer(X)
                  loss = loss_func(prediction,y)
            # trainable_vars = [U,W,V,Wd,Wd1,Wd2,Wd3]+irp_layer.trainable_variables
            trainable_vars = [U,W,V,Wd,Wd1,Wd2,Wd3]+irp_layer.trainable_variables
            gard = tape.gradient(loss,trainable_vars)
            opt.apply_gradients(zip(gard,trainable_vars))

            if i % 10 == 0:
                  print('loss after iteration {} is {}'.format(i, loss.numpy()))

      # y_predict_train = prediction+y_phy_train

      return prediction,U,W,V,Wd,attentions,data,means_1,variances_1,means_2,variances_2,Wd1,Wd2,Wd3
