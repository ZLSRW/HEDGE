import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepASDPred(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_networks: int=1, dropout: float=0.3, mode: str='cnn'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_networks = num_networks
        self.dropout = dropout
        if mode not in ('cnn', 'cnn_lstm'):
            raise ValueError(f"mode must be 'cnn' or 'cnn_lstm', got {mode}")
        self.mode = mode
        out_channels = 64
        kernel_size = 3
        self.cnn_conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cnn_bn = nn.BatchNorm1d(out_channels)
        pooled_len = in_dim // 2
        if pooled_len <= 0:
            pooled_len = in_dim
        cnn_flat_dim = out_channels * pooled_len
        self.cnn_fc = nn.Linear(cnn_flat_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.att_fc = nn.Linear(hidden_dim, 1)
        if mode == 'cnn':
            fusion_in_dim = hidden_dim
        else:
            fusion_in_dim = hidden_dim * 2
        self.fusion = nn.Sequential(nn.Linear(fusion_in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.cls = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_indices_list=None, edge_weights_list=None):
        x_cnn = x.unsqueeze(1)
        h_cnn = self.cnn_conv(x_cnn)
        h_cnn = self.cnn_bn(h_cnn)
        h_cnn = F.relu(h_cnn)
        h_cnn = F.max_pool1d(h_cnn, kernel_size=2)
        h_cnn = torch.flatten(h_cnn, start_dim=1)
        h_cnn = self.cnn_fc(h_cnn)
        h_cnn = F.relu(h_cnn)
        h_cnn = F.dropout(h_cnn, p=self.dropout, training=self.training)
        if self.mode == 'cnn':
            fused = self.fusion(h_cnn)
        else:
            x_lstm = x.unsqueeze(-1)
            h_seq, _ = self.lstm(x_lstm)
            att_scores = self.att_fc(torch.tanh(h_seq)).squeeze(-1)
            att_weights = torch.softmax(att_scores, dim=1)
            context = torch.sum(att_weights.unsqueeze(-1) * h_seq, dim=1)
            context = F.dropout(context, p=self.dropout, training=self.training)
            fused_input = torch.cat([h_cnn, context], dim=-1)
            fused = self.fusion(fused_input)
        logits = self.cls(fused).squeeze(-1)
        aux = {'h_shared': fused}
        return (logits, aux)
