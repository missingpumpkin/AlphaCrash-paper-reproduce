"""
QR-DQN Agent with Prioritized Experience Replay.

Algorithm 1 + Eqs. 8-12 from the paper. Hyperparameters from Table A1:
    network: 3 hidden layers x 64 neurons
    replay buffer: 600,000   PER alpha=0.6 beta=0.4
    batch size: 32           learning rate: 1e-4 (Adam)
    discount gamma: 0.9      soft target update tau: 0.01
    epsilon-greedy: 0.9 -> 0.1 over 15,000 steps
    quantiles N=20           Huber threshold kappa=1.0

Standard QR-DQN per Dabney et al. 2018: per-quantile target with pairwise
asymmetric Huber loss. The paper's Algorithm 1 line 14 has a sum-over-i
that would collapse the target to a scalar; we use the standard formulation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


# =====================================================================
# Prioritized Experience Replay (Eqs. 10-12)
# =====================================================================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(left + 1, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]


class PrioritizedReplayBuffer:
    """Table A1: alpha=0.6, beta=0.4, buffer=600,000."""

    def __init__(self, capacity=600000, alpha=0.6, beta=0.4, beta_inc=1e-5, eps=1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.eps = eps
        self.max_priority = 1.0

    def push(self, transition):
        """transition is a tuple (state, action, reward, next_state, done, n_eff)."""
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        batch, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_inc)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, pri, data = self.tree.get(s)
            if data is None:
                idx, pri, data = self.tree.get(random.uniform(0, self.tree.total()))
            if data is not None:
                batch.append(data)
                indices.append(idx)
                priorities.append(pri)

        if not batch:
            return None, None, None

        p = np.array(priorities) + self.eps
        probs = p / (self.tree.total() + self.eps)
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= weights.max()
        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, errors):
        for idx, e in zip(indices, errors):
            p = (abs(e) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, p)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.size


# =====================================================================
# QR-DQN Network (Table A1: 3 hidden layers x 64 neurons)
# =====================================================================
class QRDQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles=20, hidden_dim=64):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.action_dim = action_dim

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.quantile_head = nn.Linear(hidden_dim, action_dim * n_quantiles)

        # Small init on the output head so untrained Q-values are near zero
        # and roughly uniform across actions. Without this, the default init
        # produces a strongly preferred argmax action that the agent commits
        # to before any learning, polluting the early replay buffer.
        nn.init.uniform_(self.quantile_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.quantile_head.bias)

    def forward(self, x):
        """Returns quantile values: (batch, action_dim, n_quantiles)."""
        return self.quantile_head(self.feature(x)).view(-1, self.action_dim, self.n_quantiles)

    def q_values(self, x):
        """Mean over quantiles -> Q-values for action selection."""
        return self.forward(x).mean(dim=2)


# =====================================================================
# QR-DQN Agent
# =====================================================================
class QRDQNAgent:
    """Table A1 hyperparameters; per-step epsilon-greedy per Algorithm 1 line 9."""

    def __init__(self, state_dim, action_dim, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        self.n_quantiles = 20

        self.policy_net = QRDQNNetwork(state_dim, action_dim, n_quantiles=self.n_quantiles).to(device)
        self.target_net = QRDQNNetwork(state_dim, action_dim, n_quantiles=self.n_quantiles).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Table A1
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.buffer = PrioritizedReplayBuffer(capacity=600000, alpha=0.6, beta=0.4)
        self.gamma = 0.9
        self.tau = 0.01
        self.batch_size = 32
        self.kappa = 1.0

        # epsilon-greedy: 0.9 -> 0.1 linearly over 15,000 environment steps
        self.eps_start = 0.9
        self.eps_end = 0.1
        self.eps_decay_steps = 15000
        self.steps = 0
        self.episodes = 0

        # Quantile midpoints tau_i = (2i+1)/(2N)  (== (2j-1)/(2N) for j=1..N)
        self.taus = torch.FloatTensor(
            [(2 * i + 1) / (2 * self.n_quantiles) for i in range(self.n_quantiles)]
        ).to(device)

    def epsilon(self):
        """Linear decay over eps_decay_steps environment steps."""
        progress = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + progress * (self.eps_end - self.eps_start)

    def end_episode(self):
        self.episodes += 1

    def select_action(self, state, evaluate=False, action_mask=None):
        """Algorithm 1 line 9: epsilon-greedy on mean of quantiles.

        evaluate=True forces greedy regardless of epsilon (used during eval).
        action_mask is accepted for API compatibility; paper-faithful runs
        pass an all-True mask.
        """
        self.steps += 1

        if not evaluate and random.random() < self.epsilon():
            if action_mask is not None and not action_mask.all():
                valid = np.where(action_mask)[0]
                return int(np.random.choice(valid))
            return random.randrange(self.action_dim)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s).mean(dim=2)
            if action_mask is not None and not action_mask.all():
                mask_t = torch.from_numpy(action_mask.astype(np.float32)).to(self.device)
                q = q.masked_fill(mask_t.unsqueeze(0) == 0, float('-inf'))
            return q.argmax(1).item()

    def store(self, state, action, reward, next_state, done, n_eff=1):
        """Store an n-step transition.

        n_eff=1 is plain 1-step. For n-step returns:
            reward = sum_{k=0}^{n_eff-1} gamma^k * r_{t+k}
            next_state = s_{t+n_eff}  (or terminal state)
            done = whether terminal hit within the n-step window
            n_eff = effective n (less than n at episode tails)
        train_step uses gamma^n_eff for bootstrap, so terminal +/-1 reaches
        states up to n_eff steps earlier in the trajectory.
        """
        self.buffer.push((state, action, reward, next_state, float(done), int(n_eff)))

    def train_step(self):
        """Sample from PER, compute QR loss, update policy + soft target."""
        if len(self.buffer) < self.batch_size * 4:
            return 0.0

        batch, indices, weights = self.buffer.sample(self.batch_size)
        if batch is None:
            return 0.0
        weights = weights.to(self.device)

        states, actions, rewards, next_states, dones, n_effs = zip(*batch)
        s = torch.FloatTensor(np.array(states)).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        s2 = torch.FloatTensor(np.array(next_states)).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)
        gamma_n = torch.FloatTensor([self.gamma ** n for n in n_effs]).to(self.device)
        B = len(batch)

        # Current quantiles for chosen actions: (B, N)
        cur_q = self.policy_net(s)[torch.arange(B), a]

        with torch.no_grad():
            # Standard QR-DQN per-quantile target (Dabney 2018) with n-step:
            #   a* = argmax_a mean_i Q(s', a; theta)[tau_i]   (s' = s_{t+n_eff})
            #   y[i] = R_n + gamma^n_eff * Q_target(s', a*)[tau_i]
            # where R_n = sum_k gamma^k r_{t+k} is the n-step return.
            next_a = self.policy_net.q_values(s2).argmax(1)
            next_q = self.target_net(s2)[torch.arange(B), next_a]  # (B, N)
            target = r.unsqueeze(1) + gamma_n.unsqueeze(1) * (1 - d.unsqueeze(1)) * next_q  # (B, N)

        # Pairwise quantile-Huber loss (Eqs. 8-9):
        #   td[b, i, j] = target[b, i] - cur_q[b, j]
        td = target.unsqueeze(2) - cur_q.unsqueeze(1)  # (B, N_target, N_current)
        huber = torch.where(td.abs() <= self.kappa,
                            0.5 * td.pow(2),
                            self.kappa * (td.abs() - 0.5 * self.kappa))
        # Asymmetric weight |tau_j - 1{td<0}| over CURRENT quantile dim
        taus = self.taus.view(1, 1, -1)
        qr_loss = (taus - (td < 0).float()).abs() * huber
        # Sum over target quantiles, mean over current quantiles (Dabney 2018)
        loss_per_sample = qr_loss.sum(dim=1).mean(dim=1)  # (B,)
        loss = (weights * loss_per_sample).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # PER priority update (Eq. 11)
        self.buffer.update_priorities(indices, loss_per_sample.detach().cpu().numpy())

        # Table A1 soft target update
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.mul_(1 - self.tau).add_(pp.data, alpha=self.tau)

        return loss.item()

    def save(self, path):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.target_net.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.steps = ckpt['steps']
        self.episodes = ckpt.get('episodes', 0)
