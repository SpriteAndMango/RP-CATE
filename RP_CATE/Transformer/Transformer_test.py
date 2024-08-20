import torch.nn as nn
from utils import *
from Github.RP_CATE.dataset.dataset import DatasetProcessor



class SingleHeadAttention(nn.Module):
    def __init__(self,dx,dq,dk,dv):
        super(SingleHeadAttention,self).__init__()
        self.dx = dx
        self.dq = dq
        self.dk = dk
        self.dv = dv

        self.W_Q = nn.Linear(dx,dq)
        self.W_K = nn.Linear(dx,dk)
        self.W_V = nn.Linear(dx,dv)

        self.fc = nn.Linear(dv,dx)

    def forward(self,X):            #(140,25,3)
        Q = self.W_Q(X)             #(140,25,dq)
        K = self.W_K(X)             #(140,25,dk)
        V = self.W_V(X)             #(140,25,dv)

        A = torch.einsum('nqc,nkc->nqk',[Q,K])      #(140,25,25)
        A_prime = torch.softmax(A/self.dk**0.5,-1)
        out = torch.einsum('nqk,ncv->nqv',[A_prime,V])  #(140,25,dv)

        out = self.fc(out)

        return out      #(140,25,3)



class SingleHead_former(nn.Module):
    def __init__(self,dx,dq,dk,dv,expansion=5):
        super(SingleHead_former,self).__init__()


        self.single_head_attention = SingleHeadAttention(dx,dq,dk,dv)
        self.norm1 = nn.BatchNorm1d(dx)
        self.norm2 = nn.BatchNorm1d(dx)

        self.feedforward = nn.Sequential(
            nn.Linear(dx,expansion*dx),
            nn.Sigmoid(),
            nn.Linear(expansion*dx,(expansion-2)*dx),
            nn.Sigmoid(),
            nn.Linear((expansion-2)*dx,dx),
            nn.Sigmoid()
        )

    def forward(self,X):            #(140,25,3)
        attentions = self.single_head_attention(X)         #(140,25,3)
        X1 = self.norm1((X+attentions).transpose(1,2)).transpose(1,2)       #(140,25,3)
        X2 = self.norm2((self.feedforward(X1)+X1).transpose(1,2)).transpose(1,2)        #(140,25,3)

        return X2                   #(140,25,3)



class My_Model(nn.Module):
    def __init__(self, dx, dq, dk, dv, expansion, N, dout=1):
        super(My_Model, self).__init__()

        layers =[]
        for i in range(N):
            layers.append(SingleHead_former(dx, dq, dk, dv, expansion))
        layers.append(nn.Linear(dx, dout))

        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)  # (140,25,1)



def training(train_y,dq,dk,dv,dx,dout,N,expansion,epochs,lr,device):

    mymodel = My_Model(dx,dq,dk,dv,expansion,N,dout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mymodel.parameters(),lr)

    train_X.to(device)
    train_y.to(device)
    losses = []
    for i in range(epochs):
        optimizer.zero_grad()
        predictions = mymodel(train_X)
        loss = criterion(predictions,train_y)

        loss.backward()
        optimizer.step()

        if i %100 ==0:
            print(f'Epoch [{i}|{epochs}], Loss:{loss.item()}')

        losses.append(loss.item())

    predictions = predictions.squeeze(-1)[:, 0]
    prediction_final = predictions.cpu().detach().numpy().reshape(-1, 1) + y_phy_train


    return mymodel, prediction_final, losses




if __name__ == '__main__':
    seed = 19
    set_seed(seed)

    path = '../Dataset.txt'
    processor = DatasetProcessor(path)
    data_splits = processor.process()

    # Access training and testing data
    PSD_train = torch.tensor(data_splits['PSD_train'], dtype=torch.float32)
    PSD_test = torch.tensor(data_splits['PSD_test'], dtype=torch.float32)
    y_train = torch.tensor(data_splits['y_train'], dtype=torch.float32)
    y_test = torch.tensor(data_splits['y_test'], dtype=torch.float32)
    y_phy_train = data_splits['y_phy_train']
    y_Aspen_train = data_splits['y_Aspen_train']
    y_phy_test = data_splits['y_phy_test']
    y_Aspen_test = data_splits['y_Aspen_test']



    dx,dq,dk,dv,dout = 3,15,15,24,1
    expansion = 5
    N = 1
    epochs = 5000
    lr = 1e-3
    device = 'mps'
    windows = 25


    CSW = CyclicSlidingWindowsProcessor(windows)

    train_X = CSW.process(PSD_train)
    train_y = CSW.process(y_train)

    test_X = CSW.process(PSD_test)
    test_y = CSW.process(y_test)


    single_head_transformer,prediction,losses = training(train_y,dq,dk,dv,dx,dout,N,expansion,epochs,lr,device)


    predict_test = single_head_transformer(test_X)
    predictions_test = predict_test.squeeze(-1)[:, 0]
    prediction_final_test = predictions_test.cpu().detach().numpy().reshape(-1, 1) + y_phy_test












