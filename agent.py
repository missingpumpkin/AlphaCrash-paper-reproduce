"""
QR-DQN Agent with Prioritized Experience Replay.

Algorithm 1 from paper + Eqs. 8-12 for QR loss and PER.
All hyperparameters from Table A1.

Key design choice: QR-DQN (not standard DQN) is specifically needed
because it learns the FULL return distribution. This lets the agent
distinguish "95% win + 5% self-crash" from "50/50" — enabling aggressive
near-ground play that a mean-based agent would avoid.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


# ============================================================
# Prioritized Experience Replay (Eqs. 10-12)
# ============================================================
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
    """Table A1: α=0.6, β=0.4, buffer=600,000."""

    def __init__(self, capacity=600000, alpha=0.6, beta=0.4, beta_inc=1e-5, eps=1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.eps = eps
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

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


# ============================================================
# QR-DQN Network (Table A1: 3 layers × 64 neurons)
# ============================================================
class QRDQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles=20, hidden_dim=64):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.action_dim = action_dim

        # Table A1: 3 hidden layers, 64 neurons each
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.quantile_head = nn.Linear(hidden_dim, action_dim * n_quantiles)

        # Initialize quantile head with small weights so untrained Q-values
        # are near-zero and roughly uniform across actions. This prevents
        # the "init bias" problem where random initialization produces
        # strongly preferred (often suicidal) actions before any learning.
        # Diagnostic showed untrained network picks one action 85% of the time
        # in greedy mode — directly caused by default init magnitude.
        # Small uniform init makes greedy ≈ random until learning updates Q.
        nn.init.uniform_(self.quantile_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.quantile_head.bias)

    def forward(self, x):
        """Returns quantile values: (batch, action_dim, n_quantiles)."""
        return self.quantile_head(self.feature(x)).view(-1, self.action_dim, self.n_quantiles)

    def q_values(self, x):
        """Mean over quantiles → Q-values for action selection."""
        return self.forward(x).mean(dim=2)


# ============================================================
# QR-DQN Agent
# ============================================================
class QRDQNAgent:
    """
    Table A1 hyperparameters:
      lr=1e-4, γ=0.9, τ=0.01, batch=32
      buffer=600k, PER α=0.6 β=0.4
      ε: 0.9 → 0.1, LINEAR decay over 15,000 steps
      N=20 quantiles, κ=1.0
    """

    def __init__(self, state_dim, action_dim, device='cpu'):
        self.device = device
        self.action_dim = action_dim

        # Networks
        self.policy_net = QRDQNNetwork(state_dim, action_dim, n_quantiles=20).to(device)
        self.target_net = QRDQNNetwork(state_dim, action_dim, n_quantiles=20).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Table A1
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.buffer = PrioritizedReplayBuffer(capacity=600000, alpha=0.6, beta=0.4)
        self.gamma = 0.9
        self.tau = 0.01
        self.batch_size = 32
        self.kappa = 1.0
        self.n_quantiles = 20

        # Soft updates per Table A1 (hard updates tested empirically — caused loss explosion)
        self.use_hard_update = False
        self.hard_update_C = 500
        self.train_step_counter = 0

        # Exploration mode selector
        # 'per_step_egreedy': PAPER-FAITHFUL — at each step, sample uniform [0,1].
        #     If < ε, take random action. Otherwise greedy on mean of quantiles.
        #     Matches Algorithm 1 line 9 exactly.
        # 'per_episode_egreedy': sample one ε per episode; episode is fully greedy
        #     or fully random. Preserves multi-step strategy coherence.
        # 'bootstrapped_thompson': sample one quantile head per episode, use throughout.
        self.exploration_mode = 'per_step_egreedy'  # paper-faithful default

        # ε-greedy parameters (paper Table A1: 0.9 → 0.1 over 15,000 STEPS)
        # We use environment steps (not training updates) — most natural reading
        # With ~50 steps/episode, 15000 steps ≈ 300 episodes for ε to reach 0.1
        self.eps_start = 0.9
        self.eps_end = 0.1
        self.eps_decay_steps = 15000  # environment steps (per paper)
        self.episodes = 0
        self.steps = 0
        # Warmup phase: pure random actions for first N steps before any greedy.
        # Fills replay buffer with diverse experiences before init-biased policy
        # commits to single action. ~50-100 episodes worth of warmup at 100ms
        # decision step + ~50 steps/episode pre-warmup behavior.
        self.warmup_steps = 5000

        # Bootstrapped Thompson sampling state (used if exploration_mode = 'bootstrapped_thompson')
        # Each episode samples one quantile index to use for action selection
        # This produces COHERENT exploration: same "head" used throughout episode
        # Different heads sampled across episodes give diverse exploration
        self.episode_quantile_idx = random.randrange(self.n_quantiles)

        # Per-episode ε-greedy state (used if exploration_mode = 'per_episode_egreedy')
        # Each episode samples one ε; episode is fully random or fully greedy
        # Preserves multi-step strategy coherence vs per-step ε-greedy
        self.episode_is_random = False  # set per episode

        # Quantile midpoints: τ_i = (2i+1)/(2N)
        self.taus = torch.FloatTensor(
            [(2 * i + 1) / (2 * self.n_quantiles) for i in range(self.n_quantiles)]
        ).to(device)

    def epsilon(self):
        """Linear decay of ε from eps_start to eps_end over eps_decay_steps environment steps."""
        if self.exploration_mode == 'bootstrapped_thompson':
            return 0.0  # No epsilon used in BT mode
        progress = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + progress * (self.eps_end - self.eps_start)

    def end_episode(self):
        """Call at end of each episode.

        For 'bootstrapped_thompson' mode: resample quantile head for next episode.
        For 'per_episode_egreedy' mode: sample whether next episode is random or greedy.
        For 'per_step_egreedy' mode: nothing to do (decision made per-step).
        """
        self.episodes += 1

        if self.exploration_mode == 'bootstrapped_thompson':
            self.episode_quantile_idx = random.randrange(self.n_quantiles)
        elif self.exploration_mode == 'per_episode_egreedy':
            eps = self.epsilon()
            self.episode_is_random = (random.random() < eps)
        # per_step_egreedy: no per-episode state to update

    def select_action(self, state, evaluate=False, action_mask=None):
        """Action selection per current exploration mode.

        Modes:
            bootstrapped_thompson: use sampled quantile head this episode
            per_episode_egreedy: episode is fully random or fully greedy

        Evaluation (evaluate=True):
            Always uses mean of quantiles (paper's Algorithm 1 line 9)

        Args:
            state: current state vector
            evaluate: if True, use mean (deterministic eval)
            action_mask: optional bool array (True = valid action)
        """
        self.steps += 1

        # Eval mode: always greedy on mean of quantiles
        if evaluate:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                quantiles = self.policy_net(s)
                q = quantiles.mean(dim=2)
                if action_mask is not None:
                    mask_t = torch.from_numpy(action_mask.astype(np.float32)).to(self.device)
                    q = q.masked_fill(mask_t.unsqueeze(0) == 0, float('-inf'))
                return q.argmax(1).item()

        # Warmup phase: first N steps are pure random regardless of exploration mode.
        # Reason: untrained network has biased argmax (diagnostic showed 85%
        # concentration on one suicidal action). Without warmup, agent commits
        # to one action immediately, fills replay buffer with crash episodes only,
        # never gets diverse experiences. Warmup fills replay with diverse data
        # before any learning, so when greedy phase begins, network has had
        # exposure to varied trajectories.
        # 5000 steps ≈ 50-100 episodes worth of pure random exploration.
        if self.steps <= self.warmup_steps:
            if action_mask is not None:
                valid = np.where(action_mask)[0]
                return int(np.random.choice(valid))
            return random.randrange(self.action_dim)

        # Training mode: depends on exploration mode
        if self.exploration_mode == 'per_step_egreedy':
            # Paper-faithful: at each step, take random action with prob ε
            eps = self.epsilon()
            if random.random() < eps:
                if action_mask is not None:
                    valid = np.where(action_mask)[0]
                    return int(np.random.choice(valid))
                return random.randrange(self.action_dim)
            # Otherwise greedy on mean of quantiles (Algorithm 1 line 9)

        elif self.exploration_mode == 'per_episode_egreedy' and self.episode_is_random:
            # Random action (episode-coherent: entire episode is random)
            if action_mask is not None:
                valid = np.where(action_mask)[0]
                return int(np.random.choice(valid))
            return random.randrange(self.action_dim)

        # Greedy action selection (also reached for per_step when not random)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            quantiles = self.policy_net(s)  # (1, action_dim, N)

            if self.exploration_mode == 'bootstrapped_thompson':
                # Use the sampled quantile head for this episode
                q = quantiles[:, :, self.episode_quantile_idx]  # (1, action_dim)
            else:
                # per_step_egreedy or per_episode_egreedy (greedy episode): mean of quantiles
                q = quantiles.mean(dim=2)  # (1, action_dim)

            if action_mask is not None:
                mask_t = torch.from_numpy(action_mask.astype(np.float32)).to(self.device)
                q = q.masked_fill(mask_t.unsqueeze(0) == 0, float('-inf'))
            return q.argmax(1).item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, float(done))

    def train_step(self):
        """One training step: sample from PER, compute QR loss, update."""
        if len(self.buffer) < self.batch_size * 4:
            return 0.0

        batch, indices, weights = self.buffer.sample(self.batch_size)
        if batch is None:
            return 0.0
        weights = weights.to(self.device)

        states, actions, rewards, next_states, dones = zip(*batch)
        s = torch.FloatTensor(np.array(states)).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        s2 = torch.FloatTensor(np.array(next_states)).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)
        B = len(batch)

        # Current quantiles for chosen actions: (B, N)
        cur_q = self.policy_net(s)[torch.arange(B), a]

        with torch.no_grad():
            # STANDARD QR-DQN (Dabney et al. 2018) target:
            # Line 13 a* = argmax_a [mean_i Q(s', a)[τᵢ]]   -- mean for action selection
            # Line 14 y_j[i] = r_j + γ * Q⁻(s', a*)[τᵢ]    -- PER-QUANTILE target
            #
            # Previously had a bug: target was scalar (mean of next quantiles)
            # broadcast to all current quantiles. This reduced QR-DQN to
            # scalar DQN with redundant heads, losing distributional learning.
            # Fixed: target is now per-quantile vector (N values per sample).
            next_a = self.policy_net.q_values(s2).argmax(1)
            next_q = self.target_net(s2)[torch.arange(B), next_a]  # (B, N) per-quantile target
            # Per-quantile target: y[i] = r + γ * Q_target[i] for each i
            r_expanded = r.unsqueeze(1)  # (B, 1)
            d_expanded = d.unsqueeze(1)  # (B, 1)
            target = r_expanded + self.gamma * (1 - d_expanded) * next_q  # (B, N)

        # Standard QR-DQN loss (Dabney 2018):
        # For each sample, compute pairwise TD errors between target quantiles
        # and current quantiles, then asymmetric Huber weighted by τᵢ.
        # target: (B, N) - target quantile values
        # cur_q:  (B, N) - current quantile values
        # We need pairwise differences: td[b, i, j] = target[b, i] - cur_q[b, j]
        target_expanded = target.unsqueeze(2)  # (B, N, 1)
        cur_q_expanded = cur_q.unsqueeze(1)    # (B, 1, N)
        td = target_expanded - cur_q_expanded  # (B, N_target, N_current)

        huber = torch.where(td.abs() <= self.kappa, 0.5 * td.pow(2),
                            self.kappa * (td.abs() - 0.5 * self.kappa))

        # Asymmetric weighting: |τⱼ - 1{td<0}| where τⱼ is over CURRENT quantile dim
        # taus has shape (N,), corresponds to current quantile dimension (last dim of td)
        taus = self.taus.view(1, 1, -1)  # (1, 1, N_current)
        qr_loss = (taus - (td < 0).float()).abs() * huber  # (B, N_target, N_current)

        # Sum over target quantiles, mean over current quantiles
        # (Standard formulation per Dabney 2018)
        loss_per_sample = qr_loss.sum(dim=1).mean(dim=1)  # (B,)

        # PER-weighted loss
        loss = (weights * loss_per_sample).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update PER priorities
        self.buffer.update_priorities(indices, loss_per_sample.detach().cpu().numpy())

        # Target network update: hard (every C steps) or soft (every step)
        self.train_step_counter += 1
        if self.use_hard_update:
            # H2 test: Hard update every C steps (matches Algorithm 1)
            if self.train_step_counter % self.hard_update_C == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            # Soft update (Table A1: τ=0.01)
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
