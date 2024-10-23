import torch
import torch.nn as nn

class MLPSubsetModel(nn.Module):
    def __init__(self, config):
        super(MLPSubsetModel, self).__init__()
        model_name = config['model_name']
        input_size = config[model_name]['max_seq_length']
        hidden_sizes = config[model_name]['hidden_sizes']
        output_size = config[model_name]['subset_model_output']
        dropout_rate = config[model_name]['dropout']

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(p=dropout_rate))  # Dropout 추가
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.ELU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, 1]
        # Reshape to [batch_size, sequence_length]
        x = x.view(x.size(0), -1)
        return self.model(x)


class MLPMetaModel(nn.Module):
    def __init__(self, config):
        super(MLPMetaModel, self).__init__()
        model_name = config['model_name']
        input_size = config[model_name]['subset_model_output'] * 6
        hidden_sizes = config[model_name]['hidden_sizes']
        output_size = config[model_name]['output_size']

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ELU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.LogSoftmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)