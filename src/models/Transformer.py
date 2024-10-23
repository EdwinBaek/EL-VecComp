import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        model_name = config['model_name']
        input_dim = config[model_name]['input_dim']
        embed_dim = config[model_name]['embed_dim']
        n_blocks = config[model_name]['n_blocks']
        n_heads = config[model_name]['n_heads']
        ff_hid_dim = config[model_name]['ff_hid_dim']
        dropout = config[model_name]['dropout']

        self.encoder = Encoder(input_dim, embed_dim, n_blocks, n_heads, ff_hid_dim, dropout)
        self.output_proj = nn.Linear(embed_dim, config[model_name]['subset_model_output'])

    def forward(self, src):
        src_mask = self.create_mask(src)
        encoded = self.encoder(src, src_mask)
        # Use the output of the last token for classification
        output = self.output_proj(encoded[:, -1, :])
        return output

    def create_mask(self, src):
        src_mask = (src.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        return src_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.head_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.scale = embed_dim ** 0.5

        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)
        self.values = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        N = q.size(0)
        Q = self.queries(q)
        K = self.keys(k)
        V = self.values(v)

        Q = Q.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = (Q @ K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e20)

        attention = energy.softmax(-1)
        x = self.dropout(attention) @ V
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(N, -1, self.embed_dim)
        x = self.proj(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_hid_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hid_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src, mask):
        attention = self.attention(src, src, src, mask)
        x = self.norm1(attention + self.dropout(src))
        out = self.mlp(x)
        out = self.norm2(out + self.dropout(x))
        return out


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, n_blocks, n_heads, ff_hid_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList([EncoderLayer(embed_dim, n_heads, ff_hid_dim, dropout)] * n_blocks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask):
        out = self.input_proj(src)
        out = self.dropout(out)

        for block in self.blocks:
            out = block(out, mask)

        return out

# Deprecated
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_hid_dim, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.joint_attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hid_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        trg_attention = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout(trg_attention))
        joint_attention = self.joint_attention(trg, src, src, src_mask)
        trg = self.norm2(trg + self.dropout(joint_attention))
        out = self.mlp(trg)
        out = self.norm3(trg + self.dropout(out))
        return out

# Deprecated
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, n_blocks, n_heads, ff_hid_dim, max_seq_length, dropout):
        super().__init__()
        self.input_proj = nn.Linear(output_dim, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderLayer(embed_dim, n_heads, ff_hid_dim, dropout)] * n_blocks)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, trg, src, trg_mask, src_mask):
        N, trg_len = trg.shape
        positions = torch.arange(0, trg_len).expand(N, trg_len)
        pos_embeddings = self.pos_embedding(positions)
        trg = self.input_proj(trg.unsqueeze(-1))
        trg = self.dropout(pos_embeddings + trg)

        for block in self.blocks:
            trg = block(trg, src, trg_mask, src_mask)

        output = self.fc(trg)
        return output