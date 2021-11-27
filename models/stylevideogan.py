import torch
from torch import nn

from models.stylegan2 import PixelNorm
from torch_ema import ExponentialMovingAverage


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers).cuda()
        self.gru.flatten_parameters()

    def forward(self, s: torch.Tensor, h: torch.Tensor):
        return self.gru(s, h)


class StyleVideoGenerator(torch.jit.ScriptModule):
    def __init__(self, n_styles: int, latent_dim: int):
        super().__init__()

        self.n_styles = n_styles
        self.latent_dim = latent_dim

        self.H = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 96),
            nn.LeakyReLU(),
            nn.Linear(96, 96),
            nn.LeakyReLU(),
            nn.BatchNorm1d(96, affine=False),
        )

        self.P = GRU(input_size=latent_dim, hidden_size=latent_dim, num_layers=4)

        self.T = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            PixelNorm(),
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
        )
        self.As = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(512),
                )
                for layer in range(n_styles)
            ]
        )

        self.register_buffer("l_mu", ExponentialMovingAverage([torch.zeros(latent_dim)], 0.995))
        self.register_buffer("l_sig", ExponentialMovingAverage([torch.ones(latent_dim)], 0.995))

    @torch.jit.script_method
    def forward(self, s: torch.Tensor):
        s = s.permute(1, 0, 2)  # L, N, latent_dim
        i = s[[0]]
        s = s[1:]

        h_123 = self.H(i.squeeze())
        h_123 = torch.stack(torch.split(h_123, self.latent_dim, dim=1))
        h = torch.cat((h_123, i))

        l, _ = self.P(s, h)
        self.l = torch.cat((i, l))
        L, N, H = l.shape
        l = self.l.reshape(L * N, H)
        l = self.T(l)

        l_w = []
        for A in self.As:
            l_w.append(A(l))
        l_w = torch.stack(l_w)  # n_styles, L * N, 512

        l_w = l_w.permute(1, 0, 2)  # L * N, n_styles, 512
        l_w = l_w.reshape(L, N, self.n_styles, 512)
        l_w = l_w.permute(1, 0, 2, 3)  # N, L, n_styles, 512

        return l_w


class StyleVideoDiscriminator(torch.jit.ScriptModule):
    def __init__(self, seq_len: int, n_styles: int, latent_dim: int):
        super().__init__()

        self.n_styles = n_styles
        self.latent_dim = latent_dim

        self.E = nn.Sequential(
            nn.Linear(n_styles * 512, n_styles * 256),
            nn.LeakyReLU(),
            nn.Linear(n_styles * 256, n_styles * 128),
            nn.LeakyReLU(),
            nn.Linear(n_styles * 128, n_styles * 64),
            nn.LeakyReLU(),
            nn.Linear(n_styles * 64, n_styles * 32),
            nn.LeakyReLU(),
            nn.Linear(n_styles * 32, n_styles * 16),
            nn.LeakyReLU(),
            nn.Linear(n_styles * 16, latent_dim),
            nn.LeakyReLU(),
        )

        self.D = nn.Sequential(
            nn.Conv1d(latent_dim, 64, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(int(128 * seq_len / 4), 1),
        )

    @torch.jit.script_method
    def forward(self, l_w: torch.Tensor):
        N, L, _, _ = l_w.shape
        l_w = l_w.reshape(N * L, self.n_styles * 512)
        e_l_w = self.E(l_w)
        e_l_w = e_l_w.reshape(N, L, self.latent_dim)
        e_l_w = e_l_w.permute(0, 2, 1)  # N, latent_dim, L
        pred = self.D(e_l_w)
        return pred


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FPS = 24
    DUR = 1

    seq_len = DUR * FPS
    batch_size = 8
    n_styles = 18
    latent_dim = 32

    G = StyleVideoGenerator(n_styles=n_styles, latent_dim=latent_dim).to(device)
    D = StyleVideoDiscriminator(seq_len=seq_len, n_styles=n_styles, latent_dim=latent_dim).to(device)

    s = torch.randn((batch_size, seq_len, latent_dim), device=device)
    out = G(s)
    assert tuple(out.shape) == (batch_size, seq_len, n_styles, 512)

    pred = D(out)
    assert tuple(pred.shape) == (batch_size, 1)
