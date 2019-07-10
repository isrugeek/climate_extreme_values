import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ENDELSTM(nn.Module):
    def __init__(self, config):
        super(ENDELSTM, self).__init__()
        self.hidden_size = 64
        self.bi = 1
        self.lstm = nn.LSTM(config.get('features'), self.hidden_size,
                            1, dropout=0.1, bidirectional=self.bi-1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size //
                             4, 1, dropout=1, bidirectional=self.bi-1, batch_first=True)
        self.dense = nn.Linear(self.hidden_size // 4,
                               config.get('output_dim'))
        self.loss_fn = nn.MSELoss()

    def forward(self, x, batch_size=1):
        hidden = self.init_hidden(batch_size)
        output, _ = self.lstm(x, hidden)
        output = F.dropout(output, p=0.5, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=0.5, training=True)
        output = self.dense(output[:, -1, :])

        return output

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size))
        
        #return h0, c0
        return [t.cuda() for t in (h0, c0)]

    def init_hidden2(self, batch_size):
        h0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size//4))
        c0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size//4))
        #return h0, c0
        return [t.cuda() for t in (h0, c0)]

    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)


class GULSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.f = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.g = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.f(x, (h0, c0))
        out = self.g(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

    def loss(self, mu_hat, x_t):
        l1 = x_t - mu_hat
        return (l1 + torch.exp(-l1))
        #this loss function is MSE and the other loss fucntion is in the trainer
        #return self.loss_fn(pred, truth)
