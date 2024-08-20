from utils import *
from RP_CATE.dataset.dataset import DatasetProcessor





def BaseLine_RBF(X,y,learning_rate,ite,output_dim ,hidden_dim ,input_dim,seed):
    np.random.seed(seed)
    weight_h_to_out = np.random.uniform(0, 1, (output_dim, hidden_dim))  # (1,10)
    centroids = np.random.uniform(0, 1, (hidden_dim, input_dim))  # (10,1)
    delta = np.random.uniform(0, 1, (hidden_dim, input_dim))  # (10,1)

    costs = []
    for i in range(ite):
        a,b,c,d,e = feedforward_and_backforward(X,y,centroids,delta,weight_h_to_out,learning_rate)
        weight_h_to_out,delta,centroids = update(weight_h_to_out,delta,centroids,a,b,c)
        J = cost(d)
        costs.append(J)
        if i%100 == 0:
            print('Cost after iteration {} is {}'.format(i,J))


    return weight_h_to_out,centroids,delta


def feedforward(X,centroids,delta,weight_hidden_to_output):
    dist = np.zeros((len(X),len(centroids)))
    hidden_output = np.zeros((len(X),len(centroids)))

    for i in range(len(X)):
        for j in range(len(centroids)):
            dist[i][j] = distance(X[i,0],centroids[j,0])
            hidden_output[i,j] = Radial_basis_fuction(dist[i,j],delta[j,0])
    output = hidden_output@weight_hidden_to_output.T


    return output


if __name__=='__main__':

    path = '../Dataset.txt'
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


    input_dim_HM = 3
    hidden_dim_HM = 30
    output_dim_HM = 1
    learning_rate_HM = 1e-3
    ite_HM = 3000
    seed = 0


    weight_h_to_out,centroids,delta = BaseLine_RBF(PSD_train,y_train,learning_rate_HM,ite_HM,output_dim_HM,hidden_dim_HM,input_dim_HM,seed)

    output = feedforward(PSD_test,centroids,delta,weight_h_to_out)
    y_predict_test = output+y_phy_test
    loss = cost(y_predict_test-y_Aspen_test)
