import torch
import numpy as np

from math import sqrt
from torch import nn


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def init_he_normal_mask(rows, cols):
    std_dev = np.sqrt(2 / cols)
    return torch.from_numpy(np.random.randn(rows, cols) * std_dev)


class MaskLayer(nn.Module):
    def __init__(self, masks_amount, n_row, n_col, mask_dropout=0.0):
        super(MaskLayer, self).__init__()
        masks = [init_he_normal_mask(n_row, n_col) for _ in range(masks_amount)]
        self.mask = nn.Parameter(torch.stack(masks).float())

    def forward(self, attention):
        # pos-mul: shift attention to be non-negative, then multiply by mask
        attention = attention - attention.min()
        return attention * self.mask


class AbsAct(nn.Module):
    """Signed absolute-sum normalization replacing softmax.

    Standard attention uses 1/sqrt(d_k) to keep attention weights from becoming
    too large before softmax, which can make the distribution too sharp and reduce
    gradients. In AbsAct, each row is normalized by the sum of its absolute values
    instead of using softmax. Since multiplying all attention weights in a row by
    1/sqrt(d_k) affects both the numerator and denominator equally, the scaling
    cancels out and is not needed.
    """
    def __init__(self):
        super(AbsAct, self).__init__()

    def forward(self, scores):
        scores = scores + 0.0001
        denom = torch.sum(torch.abs(scores), dim=-1, keepdim=True)
        return scores / (denom + 1e-8)


class AttentionLayer(nn.Module):
    def __init__(self, configs, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_model = configs.d_model
        n_heads = configs.n_heads
        self.n_heads = n_heads

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_features = configs.n_features
        self.n_sequence = configs.n_sequence
        self.include_decop = configs.include_decop

        # Projections
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)

        if self.include_decop:
            # Compressed mode: value projection transposes input
            self.value_projection = nn.Linear(
                self.n_sequence * self.n_features, configs.k
            )
            # Learnable relations parameter
            self.relations = nn.Parameter(
                torch.empty(self.n_sequence * self.n_features, n_heads, configs.k)
            )
            nn.init.kaiming_uniform_(self.relations)
            mask_layer = MaskLayer(
                n_heads,
                n_row=self.n_sequence * self.n_features,
                n_col=configs.k,
            )
            self.k = configs.k
        else:
            # Standard application mode
            self.value_projection = nn.Linear(d_model, d_values * n_heads)
            self.mask_layer = MaskLayer(
                n_heads,
                n_row=self.n_sequence * self.n_features,
                n_col=self.n_sequence * self.n_features,
            )

        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.attention_activation = AbsAct()
        self.attention_dropout = nn.Dropout(configs.attn_dropout)

    def forward(self, i_queries, i_keys, i_values, tau=None, delta=None):
        B, L = i_queries.shape[:2]
        S = i_keys.shape[1]
        H = self.n_heads

        queries = self.query_projection(i_queries).view(B, L, H, -1)
        keys = self.key_projection(i_keys).view(B, S, H, -1)

        if self.include_decop:
            # Compressed attention with transpose value projection
            R = self.k
            values = self.value_projection(i_values.transpose(-1, -2)).view(B, R, H, -1)

            # Compressed score computation
            kt_r = torch.einsum("bshe,shr->bher", keys, self.relations)
            scores = torch.einsum("blhe,bher->bhlr", queries, kt_r)

            A = self.attention_dropout(self.attention_activation(scores))

            # Output: compressed uses same einsum as application
            out = torch.einsum("bhls,bshd->blhd", A, values)
        else:
            # Standard application attention
            scores = torch.einsum("blhe,bshe->bhls", queries, keys)
            scores = self.mask_layer(scores)

            values = self.value_projection(i_values).view(B, S, H, -1)
            A = self.attention_dropout(self.attention_activation(scores))

            out = torch.einsum("bhls,bshd->blhd", A, values)

        out = out.reshape(B, L, -1)
        out = self.out_projection(out)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, configs):
        super(TransformerLayer, self).__init__()

        self.attention = AttentionLayer(configs)
        self.feed_forward = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.fc_dropout),
            nn.Linear(configs.d_ff, configs.d_model),
        )
        self.dropout = nn.Dropout(configs.fc_dropout)

        # BatchNorm with transpose trick
        self.norm1 = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)
        )
        self.norm2 = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)
        )

    def forward(self, x_q, x_k, x_v):
        output = self.attention(x_q, x_k, x_v)
        x = x_q + self.dropout(output)
        y = self.norm1(x)
        y = self.feed_forward(y)
        output = self.norm2(x + y)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, configs):
        super(TransformerEncoder, self).__init__()
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(configs) for _ in range(configs.e_layers)]
        )

    def forward(self, x):
        batch_size, variate_amount, sequence_amount, d_model = x.shape

        # Reshape: (B, V, S, D) -> (B, S*V, D) for cross-variate-temporal attention
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, -1, d_model)

        for layer in self.transformer_layers:
            x = layer(x, x, x)

        # Reshape back: (B, S*V, D) -> (B, V, S, D)
        z = x.reshape(batch_size, sequence_amount, variate_amount, d_model)
        z = z.permute(0, 2, 1, 3)
        return z
