"""
AlphaCrash training script with periodic clean evaluation.

Logs every 200 episodes; runs `eval_games` deterministic-policy games every
`eval_every` episodes against the paper distribution and saves the best
checkpoint.
"""
import os
import time
from collections import deque
from datetime import timedelta
import numpy as np
import torch
from environment import WVRCombatEnv
from agent import QRDQNAgent


# --- Undocumented patch #4: n-step returns ---
# With gamma=0.9 and 1200-step episodes, terminal +/-1 decays to ~0 by 25
# steps and is invisible at episode start. n-step returns propagate the
# terminal reward back n steps per training update, which is essential for
# learning the final crash-induction maneuver from the win signal.
N_STEP = 10


class NStepBuffer:
    """Rolling window for computing n-step returns on-the-fly."""

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.queue = deque()

    def push(self, transition):
        """Push (s, a, r, s', done). Returns ready n-step transitions to be
        forwarded to the PER buffer (zero or one per call)."""
        self.queue.append(transition)
        if len(self.queue) >= self.n:
            ready = self._compute(self.n)
            self.queue.popleft()
            return [ready]
        return []

    def _compute(self, n_eff):
        s_t, a_t = self.queue[0][0], self.queue[0][1]
        R = 0.0
        for k in range(n_eff):
            R += (self.gamma ** k) * self.queue[k][2]
        s_n = self.queue[n_eff - 1][3]
        done_n = self.queue[n_eff - 1][4]
        return (s_t, a_t, R, s_n, done_n, n_eff)

    def flush(self):
        """At episode end, drain remaining transitions as partial n-step."""
        results = []
        while self.queue:
            results.append(self._compute(len(self.queue)))
            self.queue.popleft()
        return results


def evaluate(agent, env, n_games=500):
    """Run n_games with epsilon=0 (deterministic policy)."""
    results = []
    for _ in range(n_games):
        state = env.reset()
        while True:
            mask = env.get_action_mask()
            action = agent.select_action(state, evaluate=True, action_mask=mask)
            state, _, done, info = env.step(action)
            if done:
                results.append(info)
                break

    wc = sum(1 for r in results if r == 'win_crash')
    wb = sum(1 for r in results if r == 'win_blood' or r == 'win_timeout')
    lc = sum(1 for r in results if r == 'lose_crash')
    lb = sum(1 for r in results if r == 'lose_blood' or r == 'lose_timeout')
    dr = sum(1 for r in results if 'draw' in r)

    wins = wc + wb
    wr = wins / n_games
    se = np.sqrt(wr * (1 - wr) / n_games)
    return {
        'win_rate': wr * 100,
        'ci_low': max(0.0, wr - 1.96 * se) * 100,
        'ci_high': min(1.0, wr + 1.96 * se) * 100,
        'wc': wc, 'wb': wb, 'lc': lc, 'lb': lb, 'draws': dr,
        'total': n_games,
    }


def train(num_episodes=100000, seed=42, eval_every=5000, eval_games=500):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = WVRCombatEnv()
    eval_env = WVRCombatEnv()
    agent = QRDQNAgent(env.state_dim, env.action_dim, device=device)

    os.makedirs('checkpoints', exist_ok=True)

    print(f"Device: {device}")
    print(f"State: {env.state_dim}D | Actions: {env.action_dim} (9x9x9)")
    print(f"Network: 3x64 | QR-DQN N=20 kappa={agent.kappa} | PER alpha=0.6 beta=0.4 | n-step={N_STEP}")
    print(f"Exploration: per-step epsilon-greedy {agent.eps_start} -> {agent.eps_end} "
          f"over {agent.eps_decay_steps} env steps")
    print(f"Undocumented patches: soft ceiling above {env.SOFT_CEILING:.0f}m "
          f"(slope {env.CEILING_PENALTY}/km), reward scale x{env.REWARD_SCALE}, "
          f"enemy 10x10ms substeps")
    print(f"Dynamics: paper Eq. 1 gravity-coupled, {env.red.N_SUBSTEPS} substeps/decision")
    print(f"WEZ: {env.WEZ_RANGE:.0f}m / {np.degrees(env.WEZ_ANGLE):.0f}deg / {env.HEALTH_DAMAGE_RATE} HP/s")
    print(f"Enemy: {env._n_enemy_actions} actions, gravity-coupled 1-step prediction")
    print(f"Eval: {eval_games} games every {eval_every} episodes (epsilon=0)")
    print(f"Total episodes: {num_episodes}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = []
    losses = []
    best_eval_wr = 0.0
    eval_history = []
    end_data = {'ra': [], 'ba': [], 'rh': [], 'bh': []}

    train_start_time = time.time()
    last_log_time = train_start_time

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        n_buf = NStepBuffer(n=N_STEP, gamma=agent.gamma)

        while True:
            mask = env.get_action_mask()
            action = agent.select_action(state, action_mask=mask)
            next_state, reward, done, info = env.step(action)
            for tr in n_buf.push((state, action, reward, next_state, float(done))):
                agent.store(*tr)
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)
            state = next_state
            if done:
                for tr in n_buf.flush():
                    agent.store(*tr)
                break

        results.append(info)
        end_data['ra'].append(env.red.z)
        end_data['ba'].append(env.blue.z)
        end_data['rh'].append(env.red.health)
        end_data['bh'].append(env.blue.health)
        agent.end_episode()

        if ep % 200 == 0:
            rc = results[-200:]
            wc = sum(1 for r in rc if r == 'win_crash')
            wb = sum(1 for r in rc if r == 'win_blood' or r == 'win_timeout')
            lc = sum(1 for r in rc if r == 'lose_crash')
            lb = sum(1 for r in rc if r == 'lose_blood' or r == 'lose_timeout')
            dr = sum(1 for r in rc if 'draw' in r)
            to = sum(1 for r in rc if 'timeout' in r)
            wr = (wc + wb) / len(rc) * 100
            al = np.mean(losses[-2000:]) if losses else 0
            ra = np.mean(end_data['ra'][-200:])
            ba = np.mean(end_data['ba'][-200:])
            rh = np.mean(end_data['rh'][-200:])
            bh = np.mean(end_data['bh'][-200:])

            now = time.time()
            block_time = now - last_log_time
            total_elapsed = now - train_start_time
            eps_per_sec = 200 / block_time if block_time > 0 else 0
            avg_eps_per_sec = ep / total_elapsed if total_elapsed > 0 else eps_per_sec
            eta_seconds = (num_episodes - ep) / avg_eps_per_sec if avg_eps_per_sec > 0 else 0

            print(f"Ep {ep:6d} | "
                  f"WC:{wc:3d} WB:{wb:2d} LC:{lc:3d} LB:{lb:2d} TO:{to:3d} D:{dr:2d} | "
                  f"WR:{wr:5.1f}% | Loss:{al:.4f} | Eps:{agent.epsilon():.2f} | "
                  f"RA:{ra:6.0f} BA:{ba:6.0f} | RH:{rh:5.1f} BH:{bh:5.1f} | "
                  f"Time: {timedelta(seconds=int(total_elapsed))} ({eps_per_sec:.1f} ep/s) | "
                  f"ETA: {timedelta(seconds=int(eta_seconds))}")
            last_log_time = now

        if ep % eval_every == 0:
            eval_start = time.time()
            print(f"\n[Eval @ Ep {ep}] Running {eval_games} games (epsilon=0)...")
            eval_result = evaluate(agent, eval_env, n_games=eval_games)
            eval_duration = time.time() - eval_start
            print(f"[Eval @ Ep {ep}] WR: {eval_result['win_rate']:.1f}% "
                  f"[95% CI: {eval_result['ci_low']:.1f}-{eval_result['ci_high']:.1f}%] | "
                  f"WC:{eval_result['wc']} WB:{eval_result['wb']} "
                  f"LC:{eval_result['lc']} LB:{eval_result['lb']} D:{eval_result['draws']} | "
                  f"Eval time: {timedelta(seconds=int(eval_duration))}")

            eval_history.append((ep, eval_result['win_rate'],
                                 eval_result['ci_low'], eval_result['ci_high']))

            if eval_result['win_rate'] > best_eval_wr:
                best_eval_wr = eval_result['win_rate']
                agent.save('checkpoints/best_eval.pt')
                print(f"[Eval @ Ep {ep}] -> NEW BEST EVAL: {eval_result['win_rate']:.1f}%")
            print()

            agent.save(f'checkpoints/ep{ep}.pt')
            last_log_time = time.time()

    agent.save('checkpoints/final.pt')
    total_time = time.time() - train_start_time
    print(f"\nDone. Total training time: {timedelta(seconds=int(total_time))}")
    print(f"Best evaluation WR: {best_eval_wr:.1f}%")
    print("\nEvaluation history:")
    for ep, wr, lo, hi in eval_history:
        print(f"  Ep {ep:6d}: {wr:.1f}% [{lo:.1f}-{hi:.1f}]")


if __name__ == '__main__':
    train()
