from typing import List, Tuple

import haste_pytorch as haste
import torch
from torch import Tensor, nn

from models.stylegan2 import PixelNorm


class Hidden2Style(torch.jit.ScriptModule):
    def __init__(self, hidden_size: int, n_styles: int):
        super().__init__()
        self.hidden_size, self.n_styles = hidden_size, n_styles
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

    @torch.jit.script_method
    def forward(self, l):
        B, S, _ = l.shape
        l = l.reshape(B * S, self.hidden_size)
        l = self.T(l)
        l_w = []
        for A in self.As:
            l_w.append(A(l))
        l_w = torch.stack(l_w)  # N, B * S, 512
        l_w = l_w.reshape(self.n_styles, B, S, 512).permute(1, 2, 0, 3)  # B, S, N, 512
        return l_w


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

        self.hidden2style = Hidden2Style(hidden_size, n_styles)

    def forward(self, x, m):
        l = x
        state = (m[None, :, None], m[None, :, None])
        inter_l, inter_h = [], []
        for lstm_layer in self.lstms:
            l, state = lstm_layer(l, state=(state[0][:, :, -1], state[1][:, :, -1]))
            inter_l.append(l)
            inter_h.append(state[1].squeeze(0))
        return self.hidden2style(l), torch.stack(inter_l), torch.stack(inter_h)


class LSTMCell(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.randn(4 * hidden_size))

    @torch.jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayerStates(torch.jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(LSTMLayerStates, self).__init__()
        self.cell = cell(*cell_args)

    @torch.jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        states = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
            states += [state[1]]
        return torch.stack(outputs), torch.stack(states), state
