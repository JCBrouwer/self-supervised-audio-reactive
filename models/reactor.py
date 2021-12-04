import haste_pytorch as haste
import torch
from torch import nn

from models.stylegan2 import PixelNorm


class LSTMReactor(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=4, n_styles=18, dropout=0, zoneout=0):
        super().__init__()
        self.input_size, self.hidden_size, self.n_styles = input_size, hidden_size, n_styles

        self.lstms = nn.ModuleList(
            [
                haste.LayerNormLSTM(
                    input_size if layer == 0 else hidden_size,
                    hidden_size,
                    return_state_sequence=True,
                    batch_first=True,
                    dropout=dropout,
                    zoneout=zoneout,
                )
                for layer in range(num_layers)
            ]
        )

        self.T = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            PixelNorm(),
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
        )
        self.As = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(512),
                )
                for layer in range(n_styles)
            ]
        )

    def forward(self, x, m):
        B, S, _ = x.shape

        l = x
        state = (m[None, :, None], m[None, :, None])
        inter_l, inter_h = [], []
        for lstm_layer in self.lstms:
            l, state = lstm_layer(l, state=(state[0][:, :, -1], state[0][:, :, -1]))
            inter_l.append(l)
            inter_h.append(state[1])

        l = l.reshape(B * S, self.hidden_size)
        l = self.T(l)
        l_w = []
        for A in self.As:
            l_w.append(A(l))
        l_w = torch.stack(l_w)  # N, B * S, 512
        l_w = l_w.reshape(self.n_styles, B, S, 512).permute(1, 2, 0, 3)  # B, S, N, 512
        return l_w, torch.stack(inter_l), torch.cat(inter_h)
