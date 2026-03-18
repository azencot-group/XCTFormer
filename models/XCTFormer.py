import torch
import torch.nn as nn

from layers.XCTFormer_backbone import TransformerEncoder


def positional_encoding(q_len, d_model):
    W_pos = torch.empty((q_len, d_model))
    nn.init.uniform_(W_pos, -0.02, 0.02)
    return nn.Parameter(W_pos)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.task_name = configs.task_name

        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model

        # Calculate patch count
        self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        self.n_features = configs.enc_in

        # Patch embedding
        self.W_P = nn.Linear(self.patch_len, self.d_model)
        self.W_pos = positional_encoding(self.patch_num, self.d_model)

        self.dropout_layer = nn.Dropout(configs.dropout)

        # Encoder configs
        configs.n_features = self.n_features
        configs.n_sequence = self.patch_num
        self.encoder = TransformerEncoder(configs)

        # Prediction head
        output_dim = self.d_model * self.patch_num
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.head = nn.Sequential(
                nn.Linear(output_dim, self.pred_len),
                nn.Dropout(configs.head_dropout),
            )
        elif self.task_name in ('imputation', 'anomaly_detection'):
            self.head = nn.Sequential(
                nn.Linear(output_dim, self.seq_len),
                nn.Dropout(configs.head_dropout),
            )

    def _patch_and_embed(self, x):
        B, S, V = x.shape
        x = x.permute(0, 2, 1)  # [B, V, S]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, V, patch_num, patch_len]
        x = self.W_P(x)  # [B, V, patch_num, d_model]

        x = x.reshape(B * V, self.patch_num, self.d_model)
        x = self.dropout_layer(x + self.W_pos)
        x = x.reshape(B, V, self.patch_num, self.d_model)
        return x

    def forecast(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, S, V = x.shape

        # Instance normalization
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - means) / stdev

        x = self._patch_and_embed(x)
        z = self.encoder(x)
        z = z.reshape(B, V, -1)
        z = self.head(z)
        z = z.reshape(B, V, -1)

        # Denormalize
        z = z.permute(0, 2, 1)
        z = (z * stdev) + means
        return z

    def imputation(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, S, V = x.shape

        # Instance normalization with mask
        means = (torch.sum(x, dim=1) / torch.sum(mask == 1, dim=1)).unsqueeze(1).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x * x, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        ).unsqueeze(1).detach()
        x /= stdev

        x = self._patch_and_embed(x)
        z = self.encoder(x)
        z = z.reshape(B, V, -1)
        z = self.head(z)
        z = z.reshape(B, V, -1)

        z = z.permute(0, 2, 1)
        z = (z * stdev) + means
        return z

    def anomaly_detection(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, S, V = x.shape

        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - means) / stdev

        x = self._patch_and_embed(x)
        z = self.encoder(x)
        z = z.reshape(B, V, -1)
        z = self.head(z)
        z = z.reshape(B, V, -1)

        z = z.permute(0, 2, 1)
        z = (z * stdev) + means
        return z

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return None
