import torch.nn as nn
import torch

class rnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(rnn, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMCell, self).__init__()

        self.hidden_size = hidden_size

        self.i2f = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2i = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2c = nn.Linear(input_size + hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)

        forget_gate = torch.sigmoid(self.i2f(combined))
        input_gate = torch.sigmoid(self.i2i(combined))
        output_gate = torch.sigmoid(self.i2o(combined))
        cell_candidate = torch.tanh(self.i2c(combined))

        cell = forget_gate * cell + input_gate * cell_candidate
        hidden = output_gate * torch.tanh(cell)

        output = self.out(hidden)
        output = self.softmax(output)

        return output, hidden, cell

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def initCell(self):
        return torch.zeros(1, self.hidden_size)
