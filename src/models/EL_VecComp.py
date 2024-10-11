import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args, seq_model, nonseq_model, meta_model):
        super(Model, self).__init__()
        self.seq_model = seq_model(args)
        self.nonseq_model = nonseq_model(args)
        self.meta_classifier = meta_model(args)

    def forward(self, input_data):
        seq_features = ['api_calls', 'file_system', 'registry', 'opcodes']
        nonseq_features = ['strings', 'import_table']

        seq_input = torch.cat([input_data[feature] for feature in seq_features], dim=-1)
        nonseq_input = torch.cat([input_data[feature] for feature in nonseq_features], dim=-1)

        seq_out = self.seq_model(seq_input)
        nonseq_out = self.nonseq_model(nonseq_input)
        combined = torch.cat((seq_out, nonseq_out), dim=1)
        output = self.meta_classifier(combined)
        return output