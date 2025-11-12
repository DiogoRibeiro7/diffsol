"""
Physics-informed neural network using diffsol-generated residuals.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import diffsol_pytorch as dsp

WAVE_CODE = """
in = [c]
c { 1.0 }
u {
    u = 0.0,
    v = 1.0,
}
F {
    v,
    c * c * u,
}
"""


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x, t):
        inp = torch.stack([x, t], dim=-1)
        return self.net(inp).squeeze(-1)


def residual_loss(model: PINN, times):
    xs = torch.linspace(0.0, 1.0, 20)
    loss = 0.0
    for t in times:
        x = xs.requires_grad_(True)
        u = model(x, torch.full_like(x, t))
        u_t = torch.autograd.grad(u, t, retain_graph=True, allow_unused=True)
        if u_t[0] is None:
            continue
        loss = loss + (u_t[0].pow(2).mean())
    return loss


def main():
    pinn = PINN()
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    module = dsp.DiffsolModule(WAVE_CODE)
    times = torch.linspace(0.0, 1.0, 21).tolist()
    params = [1.0]
    for epoch in range(5):
        optimizer.zero_grad()
        loss_res = residual_loss(pinn, times)
        _, _, flat = module.solve_dense(params, times)
        target = torch.tensor(flat[0::2], dtype=torch.float32)
        data_loss = target.mean()
        loss = loss_res + data_loss
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch}: loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
