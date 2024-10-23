import torch
import torch.nn as nn
from ..models.RNN import RNN, GRU, LSTM
from ..models import MLP, Transformer, Autoformer, Informer, DLinear, LightTS

class Model(nn.Module):
    def __init__(self, config, seq_model_name, nonseq_model_name):
        super(Model, self).__init__()
        self.model_name = config['model_name']
        self.dataset_name = config['dataset_name']
        sequential_model_dict = {
            'RNN': RNN,
            'Bi-RNN': RNN,
            'GRU': GRU,
            'Bi-GRU': GRU,
            'LSTM': LSTM,
            'Bi-LSTM': LSTM
        }
        non_sequential_model_dict = {
            'Transformer': Transformer.TransformerEncoder,
            'MLP': MLP.MLPSubsetModel
        }
        meta_model_dict = {
            'MLP': MLP.MLPMetaModel
        }

        self.seq_features = ['api_calls', 'file_system', 'registry', 'opcodes']
        self.nonseq_features = ['strings', 'import_table']

        # Create a separate sequential model for each sequential feature
        self.seq_models = nn.ModuleDict({
            feature: sequential_model_dict[seq_model_name](config)
            for feature in self.seq_features
        })

        # Create a separate non-sequential model for each non-sequential feature
        self.nonseq_models = nn.ModuleDict({
            feature: non_sequential_model_dict[nonseq_model_name](config)
            for feature in self.nonseq_features
        })

        # Single meta classifier
        self.meta_classifier = meta_model_dict[config[self.model_name]['meta_model']](config)

    def forward(self, input_data):
        # Process each sequential feature separately
        subset_outputs = []
        for feature in self.seq_features:
            # print(f"input_data[{feature}] vector shape : {input_data[feature].shape}")
            seq_out = self.seq_models[feature](input_data[feature])    # [batch, subset_model_output]
            # print(f"{feature} vector shape : {seq_out.shape}")
            # print('=' * 100)
            subset_outputs.append(seq_out)

        # Process each non-sequential feature separately
        for feature in self.nonseq_features:
            # print(f"input_data[{feature}] vector shape : {input_data[feature].shape}")
            nonseq_out = self.nonseq_models[feature](input_data[feature])    # [batch, subset_model_output]
            # print(f"{feature} vector shape : {nonseq_out.shape}")
            # print('=' * 100)
            subset_outputs.append(nonseq_out)

        # Concatenate sequential and  outputs
        combined = torch.cat(subset_outputs, dim=1)

        # Meta classification
        output = self.meta_classifier(combined)

        return output