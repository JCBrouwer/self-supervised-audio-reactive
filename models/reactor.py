from torch import nn


class LSTMReactor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def foward(self, x):
        return self.lstm(x)
