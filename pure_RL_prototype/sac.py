"""SAC for the residual policy on top of pi0 (image-based, DrQ-style), CPU learner.

Observation: img (6, 64, 64) uint8 = wrist + agentview, low (LOW_DIM,) = pi0 state + pi0's chunk actions.
Action: u in [-1, 1]^6 (tanh-squashed Gaussian); the env executes eps * u on top of pi0's actions.

  Encoder   4 conv layers (stride 2, 2, 2, 1) -> linear -> LayerNorm -> tanh. Owned by the critic; the actor
            uses its features detached (gradients into the encoder come from the critic only, as in DrQ / SAC-AE)
  Actor     MLP on [features, low] -> mean, log_std. The mean layer starts at zero, so the policy starts as
            pi0 plus zero-mean exploration noise; the deterministic policy (tanh(mean) = 0) is exactly pi0
  Critic    twin Q MLPs on [features, low, u]; target critic by Polyak averaging
  Update    random-shift augmentation (pad 4), entropy temperature tuned to target_entropy

Reward shaping is done here, not in the env, so it can change without recollecting data:
  r_train = r - act_penalty * mean(u^2)       (pulls the residual toward 0 where the critic is indifferent)

Python 3.8 / torch 1.11 (the LIBERO venv). Neither venv's torch has kernels for the RTX 5090 (sm_120), so the
learner runs on CPU, like pixel-Q.
"""

import numpy as np


def _torch():
    import torch

    return torch


# ------------------------------------------------------------------ networks -----------------------

def build(low_dim, act_dim=6, img_ch=6, img=64, feat_dim=128, hidden=256, init_log_std=0.0):
    """Returns (actor, critic, critic_target) torch modules."""
    torch = _torch()
    nn = torch.nn

    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(img_ch, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),     # 32
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),         # 16
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),         # 8
                nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True))         # 8
            n = 64 * (img // 8) * (img // 8)
            self.fc = nn.Sequential(nn.Linear(n, feat_dim), nn.LayerNorm(feat_dim), nn.Tanh())

        def forward(self, img_u8):
            x = img_u8.float() / 255.0 - 0.5
            return self.fc(self.conv(x).flatten(1))

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                             nn.ReLU(inplace=True), nn.Linear(hidden, o))

    class Critic(nn.Module):
        def __init__(self):
            super(Critic, self).__init__()
            self.encoder = Encoder()
            self.q1, self.q2 = mlp(feat_dim + low_dim + act_dim, 1), mlp(feat_dim + low_dim + act_dim, 1)

        def forward(self, feat, low, u):
            x = torch.cat([feat, low, u], dim=-1)
            return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    class Actor(nn.Module):
        def __init__(self):
            super(Actor, self).__init__()
            self.trunk = mlp(feat_dim + low_dim, 2 * act_dim)
            last = self.trunk[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            with torch.no_grad():
                last.bias[act_dim:] = init_log_std     # mean 0, std exp(init_log_std) at init

        def forward(self, feat, low):
            mu, log_std = self.trunk(torch.cat([feat, low], dim=-1)).chunk(2, dim=-1)
            return mu, log_std.clamp(-5.0, 1.0)

        def sample(self, feat, low):
            """Reparameterized tanh-Gaussian sample and its log-prob."""
            mu, log_std = self(feat, low)
            std = log_std.exp()
            x = mu + std * torch.randn_like(mu)
            u = torch.tanh(x)
            logp = (-0.5 * ((x - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
            logp = logp - (2 * (np.log(2.0) - x - torch.nn.functional.softplus(-2 * x))).sum(-1)
            return u, logp

    critic, target = Critic(), Critic()
    target.load_state_dict(critic.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    return Actor(), critic, target


# ------------------------------------------------------------------ acting (actor processes) --------

class Policy(object):
    """Actor + encoder on CPU for rollouts. Weights come as numpy state dicts from the learner."""

    def __init__(self, low_dim, act_dim=6, **kw):
        torch = _torch()
        torch.set_num_threads(1)
        self.actor, self.critic, _ = build(low_dim, act_dim, **kw)
        self.actor.eval()
        self.critic.eval()

    def load(self, weights):
        torch = _torch()
        self.actor.load_state_dict({k: torch.from_numpy(v) for k, v in weights["actor"].items()})
        self.critic.encoder.load_state_dict({k: torch.from_numpy(v) for k, v in weights["encoder"].items()})

    def act(self, pobs, deterministic=False):
        torch = _torch()
        with torch.no_grad():
            img = torch.from_numpy(pobs["img"][None])
            low = torch.from_numpy(pobs["low"][None])
            feat = self.critic.encoder(img)
            if deterministic:
                return torch.tanh(self.actor(feat, low)[0])[0].numpy()
            return self.actor.sample(feat, low)[0][0].numpy()


def policy_weights(actor, critic):
    """What the actor processes need: actor + encoder, as numpy (pickled into every job)."""
    return {"actor": {k: v.detach().cpu().numpy() for k, v in actor.state_dict().items()},
            "encoder": {k: v.detach().cpu().numpy() for k, v in critic.encoder.state_dict().items()}}


# ------------------------------------------------------------------ replay -------------------------

class Replay(object):
    """Preallocated ring buffer of decisions (next observations stored explicitly)."""

    def __init__(self, capacity, low_dim, act_dim=6, img_shape=(6, 64, 64)):
        self.cap, self.n, self.i = int(capacity), 0, 0
        self.img = np.zeros((self.cap,) + tuple(img_shape), np.uint8)
        self.img2 = np.zeros_like(self.img)
        self.low = np.zeros((self.cap, low_dim), np.float32)
        self.low2 = np.zeros_like(self.low)
        self.u = np.zeros((self.cap, act_dim), np.float32)
        self.r = np.zeros(self.cap, np.float32)
        self.done = np.zeros(self.cap, np.float32)

    def add(self, t):
        i = self.i
        self.img[i], self.img2[i], self.low[i], self.low2[i] = t["img"], t["img2"], t["low"], t["low2"]
        self.u[i], self.r[i], self.done[i] = t["u"], t["r"], float(t["done"])
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, size=batch)
        return {k: getattr(self, k)[idx] for k in ("img", "img2", "low", "low2", "u", "r", "done")}


# ------------------------------------------------------------------ learner ------------------------

def random_shift(torch, img, pad=4):
    """DrQ augmentation: replicate-pad by `pad` and crop back at a random offset per sample (uint8 in, out)."""
    n, c, h, w = img.shape
    x = torch.nn.functional.pad(img.float(), (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (n,))
    oy = torch.randint(0, 2 * pad + 1, (n,))
    rows = (oy[:, None] + torch.arange(h)[None, :])                     # (n, h)
    cols = (ox[:, None] + torch.arange(w)[None, :])                     # (n, w)
    x = x[torch.arange(n)[:, None, None, None], torch.arange(c)[None, :, None, None],
          rows[:, None, :, None], cols[:, None, None, :]]
    return x


class Learner(object):
    def __init__(self, low_dim, act_dim=6, lr=3e-4, gamma=0.97, tau=0.01, init_alpha=0.1, target_entropy=None,
                 act_penalty=0.005, threads=8, **net_kw):
        torch = _torch()
        torch.set_num_threads(threads)
        self.torch = torch
        self.actor, self.critic, self.target = build(low_dim, act_dim, **net_kw)
        self.log_alpha = torch.tensor(float(np.log(init_alpha)), requires_grad=True)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self.gamma, self.tau, self.act_penalty = gamma, tau, act_penalty
        self.target_entropy = -float(act_dim) if target_entropy is None else float(target_entropy)
        self.updates = 0

    def weights(self):
        return policy_weights(self.actor, self.critic)

    def update(self, b):
        torch = self.torch
        F = torch.nn.functional
        img, img2 = random_shift(torch, torch.from_numpy(b["img"])), random_shift(torch, torch.from_numpy(b["img2"]))
        low, low2 = torch.from_numpy(b["low"]), torch.from_numpy(b["low2"])
        u, done = torch.from_numpy(b["u"]), torch.from_numpy(b["done"])
        r = torch.from_numpy(b["r"]) - self.act_penalty * (u ** 2).mean(-1)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            f2 = self.critic.encoder(img2)
            u2, logp2 = self.actor.sample(f2, low2)
            q1t, q2t = self.target(self.target.encoder(img2), low2, u2)
            y = r + self.gamma * (1.0 - done) * (torch.min(q1t, q2t) - alpha * logp2)
        feat = self.critic.encoder(img)
        q1, q2 = self.critic(feat, low, u)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        fd = feat.detach()
        ua, logp = self.actor.sample(fd, low)
        qa1, qa2 = self.critic(fd, low, ua)
        actor_loss = (alpha * logp - torch.min(qa1, qa2)).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.target.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        self.updates += 1
        return {"critic_loss": float(critic_loss), "actor_loss": float(actor_loss), "alpha": float(alpha),
                "q": float(q1.mean()), "target": float(y.mean()), "entropy": float(-logp.mean()),
                "abs_u_mean": float(ua.abs().mean())}

    def state(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "target": self.target.state_dict(), "log_alpha": self.log_alpha.detach().clone(),
                "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
                "opt_alpha": self.opt_alpha.state_dict(), "updates": self.updates}

    def load(self, s):
        self.actor.load_state_dict(s["actor"])
        self.critic.load_state_dict(s["critic"])
        self.target.load_state_dict(s["target"])
        with self.torch.no_grad():
            self.log_alpha.copy_(s["log_alpha"])
        self.opt_actor.load_state_dict(s["opt_actor"])
        self.opt_critic.load_state_dict(s["opt_critic"])
        self.opt_alpha.load_state_dict(s["opt_alpha"])
        self.updates = int(s["updates"])
