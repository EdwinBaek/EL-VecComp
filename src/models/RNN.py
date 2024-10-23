import torch
import torch.nn as nn


class RNNBase(nn.Module):
    def __init__(self, config):
        super(RNNBase, self).__init__()
        model_name = config['model_name']
        self.input_size = config[model_name]['rnn_input_size']
        self.hidden_size = config[model_name]['rnn_hidden_size']
        self.num_layers = config[model_name]['rnn_num_layers']
        self.output_size = config[model_name]['subset_model_output']
        self.dropout = config[model_name]['dropout'] if self.num_layers > 1 else 0
        self.bidirectional = True if 'Bi' in config[model_name]['seq_model'] else False

        if 'RNN' in config[model_name]['seq_model']:
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, batch_first=True,
                              dropout=self.dropout, bidirectional=self.bidirectional)
        elif 'GRU' in config[model_name]['seq_model']:
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, batch_first=True,
                              dropout=self.dropout, bidirectional=self.bidirectional)
        elif 'LSTM' in config[model_name]['seq_model']:
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, batch_first=True,
                               dropout=self.dropout, bidirectional=self.bidirectional)

        if self.bidirectional:
            self.output_layer = nn.Linear(self.hidden_size * 2, self.output_size)
        else:
            self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.elu = nn.ELU()

    def forward(self, x):
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
            output, (hidden, cell) = self.rnn(x, (h0, c0))
        else:
            output, hidden = self.rnn(x, h0)
        if self.bidirectional:
            # Concatenate the last hidden state from forward and backward directions
            last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # Use the last hidden state
            last_hidden = hidden[-1]

        output = self.elu(self.output_layer(last_hidden))    # Shape: [batch_size, output_size]

        return output


class RNN(RNNBase):
    def __init__(self, config):
        super(RNN, self).__init__(config)


class GRU(RNNBase):
    def __init__(self, config):
        super(GRU, self).__init__(config)


class LSTM(RNNBase):
    def __init__(self, config):
        super(LSTM, self).__init__(config)