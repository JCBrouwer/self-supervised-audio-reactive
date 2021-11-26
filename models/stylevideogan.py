import torch
from torch import nn

from stylegan2 import PixelNorm


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers).cuda()
        self.gru.flatten_parameters()

    def forward(self, s: torch.Tensor, h: torch.Tensor):
        return self.gru(s, h)


class StyleVideoGenerator(torch.jit.ScriptModule):
    def __init__(self, n_latents: int):
        super().__init__()

        self.n_latents = n_latents

        self.H = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 96),
            nn.LeakyReLU(),
            nn.Linear(96, 96),
            nn.LeakyReLU(),
            nn.BatchNorm1d(96, affine=False),
        )

        self.P = GRU(input_size=32, hidden_size=32, num_layers=4)

        self.T = nn.Sequential(
            nn.BatchNorm1d(32),
            PixelNorm(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
        )
        self.Ts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(512),
                )
                for layer in range(n_latents)
            ]
        )

    @torch.jit.script_method
    def forward(self, s: torch.Tensor):
        s = s.permute(1, 0, 2)  # L, N, 32
        i = s[[0]]
        s = s[1:]

        h_123 = self.H(i.squeeze())
        h_123 = torch.stack(torch.split(h_123, 32, dim=1))
        h = torch.cat((h_123, i))

        l, _ = self.P(s, h)
        l = torch.cat((i, l))
        L, N, H = l.shape
        l = l.reshape(L * N, H)
        l = self.T(l)

        l_w = []
        for layer in self.Ts:
            l_w.append(layer(l))
        l_w = torch.stack(l_w)  # n_latents, L * N, 512

        l_w = l_w.permute(1, 0, 2)  # L * N, n_latents, 512
        l_w = l_w.reshape(L, N, self.n_latents, 512)
        l_w = l_w.permute(1, 0, 2, 3)  # N, L, n_latents, 512

        return l_w


class StyleVideoDiscriminator(torch.jit.ScriptModule):
    def __init__(self, seq_len: int, n_latents: int):
        super().__init__()

        self.n_latents = n_latents

        self.E = nn.Sequential(
            nn.Linear(n_latents * 512, n_latents * 256),
            nn.LeakyReLU(),
            nn.Linear(n_latents * 256, n_latents * 128),
            nn.LeakyReLU(),
            nn.Linear(n_latents * 128, n_latents * 64),
            nn.LeakyReLU(),
            nn.Linear(n_latents * 64, n_latents * 32),
            nn.LeakyReLU(),
            nn.Linear(n_latents * 32, n_latents * 16),
            nn.LeakyReLU(),
            nn.Linear(n_latents * 16, 32),
            nn.LeakyReLU(),
        )

        self.D = nn.Sequential(
            nn.Conv1d(32, 64, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(int(128 * seq_len / 4), 1),
        )

    @torch.jit.script_method
    def forward(self, l_w: torch.Tensor):
        N, L, _, _ = l_w.shape
        l_w = l_w.reshape(N * L, self.n_latents * 512)
        e_l_w = self.E(l_w)
        e_l_w = e_l_w.reshape(N, L, 32)
        e_l_w = e_l_w.permute(0, 2, 1)  # N, 32, L
        pred = self.D(e_l_w)
        return pred


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FPS = 24
    DUR = 1

    seq_len = DUR * FPS
    batch_size = 8

    G = StyleVideoGenerator(n_latents=16).to(device)
    D = StyleVideoDiscriminator(seq_len=seq_len, n_latents=16).to(device)

    s = torch.randn((batch_size, seq_len, 32), device=device)
    out = G(s)
    assert tuple(out.shape) == (batch_size, seq_len, 16, 512)

    pred = D(out)
    assert tuple(pred.shape) == (batch_size, 1)
