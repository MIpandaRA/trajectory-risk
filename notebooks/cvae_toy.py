"""Why predict a distribution? A two-minute CPU experiment.

Every training example has the same history (walking up the y axis) and then
turns left or right with equal probability. A regressor trained with MSE
learns the conditional MEAN, which goes straight ahead: a path no agent
ever takes. A conditional VAE learns p(future | history) and its samples
cover both branches.

    python notebooks/cvae_toy.py      # writes docs/figures/cvae_toy.png

The CVAE in ~40 lines:
  encoder  q(z | x, y)  (training only: it sees the true future y)
  prior    p(z | x)     (inference: no future available)
  decoder  p(y | x, z)
  loss = reconstruction + KL(q || p)       (the negative ELBO)
  z = mu + sigma * eps                      (reparameterisation: lets gradients reach mu, sigma)
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
H = 6  # future steps
HIST = torch.tensor([[0.0, -3.0], [0.0, -2.0], [0.0, -1.0], [0.0, 0.0]]).flatten()


def make_data(n):
    side = torch.randint(0, 2, (n,)) * 2 - 1  # -1 left, +1 right
    t = torch.arange(1, H + 1).float()
    x = side[:, None] * 0.18 * t ** 2
    y = t[None, :] * 0.9
    fut = torch.stack([x, y.expand_as(x)], -1) + 0.05 * torch.randn(n, H, 2)
    return HIST.expand(n, -1), fut.flatten(1)


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, o))


class CVAE(nn.Module):
    def __init__(self, zd=2):
        super().__init__()
        self.q, self.p, self.dec, self.zd = mlp(8 + 2 * H, 2 * zd), mlp(8, 2 * zd), mlp(8 + zd, 2 * H), zd

    def forward(self, x, y):
        qm, qlv = self.q(torch.cat([x, y], 1)).chunk(2, 1)
        pm, plv = self.p(x).chunk(2, 1)
        z = qm + torch.randn_like(qm) * (0.5 * qlv).exp()          # reparameterisation trick
        recon = ((self.dec(torch.cat([x, z], 1)) - y) ** 2).sum(1)
        kl = 0.5 * (plv - qlv + (qlv.exp() + (qm - pm) ** 2) / plv.exp() - 1).sum(1)
        return (recon + kl).mean(), recon.mean(), kl.mean()

    @torch.no_grad()
    def sample(self, x, n):
        pm, plv = self.p(x[:1]).chunk(2, 1)
        z = pm + torch.randn(n, self.zd) * (0.5 * plv).exp()
        return self.dec(torch.cat([x[:1].expand(n, -1), z], 1)).view(n, H, 2)


def main():
    x, y = make_data(4000)
    reg = mlp(8, 2 * H)
    cvae = CVAE()
    o1, o2 = torch.optim.Adam(reg.parameters(), 3e-3), torch.optim.Adam(cvae.parameters(), 3e-3)
    for step in range(1500):
        i = torch.randint(0, len(x), (256,))
        l1 = ((reg(x[i]) - y[i]) ** 2).mean()
        o1.zero_grad(); l1.backward(); o1.step()
        l2, rec, kl = cvae(x[i], y[i])
        o2.zero_grad(); l2.backward(); o2.step()
        if step % 500 == 0:
            print(f"step {step}: mse {l1.item():.3f}  cvae recon {rec.item():.3f} kl {kl.item():.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "figures", "cvae_toy.png")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    hist = HIST.view(-1, 2).numpy()
    gt = y[:200].view(-1, H, 2).numpy()
    mean_path = reg(x[:1]).detach().view(H, 2).numpy()
    samples = cvae.sample(x, 30).numpy()
    for ax, title in zip(axes, ("Regression (MSE): predicts the mean", "CVAE: 30 samples from p(future | history)")):
        for g in gt[:60]:
            ax.plot(g[:, 0], g[:, 1], color="#c9c8c2", lw=1, zorder=1)
        ax.plot(hist[:, 0], hist[:, 1], "-o", color="#0b0b0b", ms=4, lw=2, zorder=3, label="history")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_aspect("equal")
        ax.grid(color="#ebeae5", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].plot(*np.vstack([hist[-1], mean_path]).T, color="#eb6834", lw=2.5, zorder=4, label="MSE prediction")
    for i, s in enumerate(samples):
        axes[1].plot(*np.vstack([hist[-1], s]).T, color="#2a78d6", lw=1.2, alpha=0.8, zorder=2,
                     label="CVAE samples" if i == 0 else None)
    axes[0].plot([], [], color="#c9c8c2", label="true futures")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
