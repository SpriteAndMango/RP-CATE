import torch
from Github.RP_CATE.dataset.dataset import DatasetProcessor
from utils import set_seed,CyclicSlidingWindowsProcessor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        y1 = y1.transpose(1,2)
        return self.linear(y1)



def train(epochs):
    global lr
    model.train()
    total_loss = 0
    for i in range(epochs):
        x, y = X_train, Y_train
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss = loss.item()

        if i % 10 == 0:
            cur_loss = total_loss
            print('Train Epoch: [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
             i,epochs, 100.*i/epochs, lr, cur_loss))

    return output


def evaluate():
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        test_loss = F.mse_loss(output, Y_test)
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))

        return test_loss.item(),output



if __name__ == '__main__':

    seed=42
    set_seed(seed)

    path = '../Dataset.txt'
    processor = DatasetProcessor(path)
    data_splits = processor.process()

    # Access training and testing data
    PSD_train = torch.tensor(data_splits['PSD_train'],dtype=torch.float32)
    PSD_test = torch.tensor(data_splits['PSD_test'],dtype=torch.float32)
    y_train = torch.tensor(data_splits['y_train'],dtype=torch.float32)
    y_test = torch.tensor(data_splits['y_test'],dtype=torch.float32)
    y_phy_train = data_splits['y_phy_train']
    y_Aspen_train = data_splits['y_Aspen_train']
    y_phy_test = data_splits['y_phy_test']
    y_Aspen_test = data_splits['y_Aspen_test']



    windows = 25
    CSW = CyclicSlidingWindowsProcessor(windows)

    train_X= CSW.process(PSD_train).transpose(1,2)
    train_y = CSW.process(y_train)

    test_X= CSW.process(PSD_test).transpose(1,2)
    test_y = CSW.process(y_test)



    input_channels = 3
    n_classes = 1
    batch_size = test_X.shape[0]
    seq_length = windows
    epochs = 3000
    n_hid = 15
    levels = 3
    ksize = 3
    log_interval = 2
    clip = float(-1)


    channel_sizes = [n_hid]*levels
    kernel_size = ksize
    dropout = 0.1
    model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)


    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if torch.backends.mps.is_available():
        model.to(device)
        X_train = train_X.to(device)
        Y_train = train_y.to(device)
        X_test = test_X.to(device)
        Y_test = test_y.to(device)

    lr = 1e-3
    optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=lr)


    output_train = train(epochs)
    tloss,output = evaluate()

    y_predict_train_ = output_train.squeeze(2)[:,0].unsqueeze(1)
    y_predict_train = y_predict_train_.cpu().detach().numpy() + y_phy_train

    y_predict = output.squeeze(2)[:,0].unsqueeze(1)
    y_predict_test = y_predict.cpu().detach().numpy() + y_phy_test

