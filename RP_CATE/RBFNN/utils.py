import numpy as np



def distance(x,y):
    return np.linalg.norm(x-y)


def Radial_basis_fuction(dist,delta):
    return np.exp(-dist**2/(2*delta**2))

def cost(error):
    return (error.T@error)[0,0]/(2*len(error))

def feedforward_and_backforward(X,y,centroids,delta,weight_hidden_to_output,learning_rate):
    dist = np.zeros((len(X),len(centroids)))
    hidden_output = np.zeros((len(X),len(centroids)))
    error = np.zeros((len(X)))
    output = np.zeros((len(X)))
    for i in range(len(X)):
        for j in range(len(centroids)):
            dist[i][j] = distance(X[i,0],centroids[j,0])
            hidden_output[i,j] = Radial_basis_fuction(dist[i,j],delta[j,0])
    output = hidden_output@weight_hidden_to_output.T
    error = y-output

    ### back_forward
    delta_W = (error.T@hidden_output)*learning_rate/(len(X))

    delta_sigma = np.zeros(len(centroids))
    for i in range(len(delta_sigma)):
        first = learning_rate*weight_hidden_to_output.T[i,0]/delta[i,0]**3
        second = 0
        for j in range(len(X)):
            second_j = error[j]*hidden_output[j,i]*dist[j,i]**2
            second += second_j
        delta_sigma[i] = first*second/len(X)

    delta_centroids = np.zeros(centroids.shape)
    for i in range(len(delta_centroids)):
        first = learning_rate*weight_hidden_to_output.T[i,0]/(2*delta[i,0]**2)
        second = 0
        for j in range(len(X)):
            second_j = error[j,0]*hidden_output[j,i]*(X[j]-centroids[i])/dist[j,i]
            second += second_j
        delta_centroids[i] = first*second/len(X)


    return delta_W,delta_sigma.reshape(-1,1),delta_centroids,error,output


def update(weight_h_to_out,delta,centroids,delta_W,delta_sigma,delta_centroids):
    weight_h_to_out = weight_h_to_out+ delta_W
    delta = delta +delta_sigma
    centroids = centroids +delta_centroids
    return weight_h_to_out,delta,centroids
