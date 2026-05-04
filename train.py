"""
AlphaCrash Training Script with Periodic Evaluation.

Runs training with clean evaluation every 5000 episodes:
  - 500 evaluation games with epsilon=0 (deterministic policy)
  - Random initial conditions (same as training)
  - Reports: mean win rate + 95% binomial confidence interval
"""
import os
import time
from datetime import timedelta
import numpy as np
import torch
from environment import WVRCombatEnv
from agent import QRDQNAgent


def evaluate(agent, env, n_games=500, eval_episode=None):
    """Run n_games with epsilon=0 for clean policy evaluation.

    Args:
        agent: the agent to evaluate
        env: the environment
        n_games: number of evaluation games
        eval_episode: if None, uses paper distribution (Stage 3, no curriculum).
                      Otherwise, uses the curriculum stage corresponding to that episode.
                      Use this to test in-distribution performance at current training stage.

    Returns dict with win_rate, 95% CI bounds, and outcome breakdown.
    """
    results = []
    for _ in range(n_games):
        if eval_episode is None:
            state = env.reset()  # paper distribution
        else:
            state = env.reset(episode=eval_episode)  # stage-matched
        while True:
            mask = env.get_action_mask()
            action = agent.select_action(state, evaluate=True, action_mask=mask)
            next_state, reward, done, info = env.step(action)
            state = next_state
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

    # 95% binomial confidence interval (normal approximation)
    se = np.sqrt(wr * (1 - wr) / n_games)
    ci_low = max(0.0, wr - 1.96 * se)
    ci_high = min(1.0, wr + 1.96 * se)

    return {
        'win_rate': wr * 100,
        'ci_low': ci_low * 100,
        'ci_high': ci_high * 100,
        'wc': wc, 'wb': wb, 'lc': lc, 'lb': lb, 'draws': dr,
        'total': n_games,
    }


def train(num_episodes=100000, seed=42, eval_every=5000, eval_games=500):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = WVRCombatEnv()
    eval_env = WVRCombatEnv()  # separate env for evaluation
    agent = QRDQNAgent(env.state_dim, env.action_dim, device=device)

    os.makedirs('checkpoints', exist_ok=True)

    print(f"Device: {device}")
    print(f"State: {env.state_dim}D ({env._single_frame_dim}D x {env.N_FRAMES} frames)")
    print(f"Actions: {env.action_dim} (9x9x9)")
    print(f"Network: 3x64 | QR-DQN N=20 | PER alpha=0.6 beta=0.4")
    if agent.exploration_mode == 'per_step_egreedy':
        print(f"Exploration: per-step ε-greedy [paper-faithful] ({agent.eps_start} → {agent.eps_end} over {agent.eps_decay_steps} env steps), warmup: {agent.warmup_steps} pure random steps")
    elif agent.exploration_mode == 'per_episode_egreedy':
        print(f"Exploration: per-episode ε-greedy ({agent.eps_start} → {agent.eps_end} over {agent.eps_decay_steps} env steps)")
    else:
        print(f"Exploration: Bootstrapped Thompson via quantile sampling (NO epsilon-greedy)")
    print(f"Action masking: DISABLED (paper-faithful, no action filtering)")
    if env.CURRICULUM_STAGE_1_END == 0 and env.CURRICULUM_STAGE_2_END == 0:
        print(f"Curriculum: DISABLED (paper's uniform random distribution throughout, Section 4.1.1)")
    else:
        print(f"Curriculum: Stage1<{env.CURRICULUM_STAGE_1_END} (low-alt engaged), Stage2<{env.CURRICULUM_STAGE_2_END} (mixed), Stage3+ (paper)")
    print(f"Dynamics: paper Eq. 1 gravity-coupled (n_z, μ via inner-loop controller, n_z clamped to [-3, +9]G per Table B1), {env.red.N_SUBSTEPS} substeps/decision")
    print(f"WEZ: {env.WEZ_RANGE}m/{np.degrees(env.WEZ_ANGLE):.0f}deg/{env.HEALTH_DAMAGE_RATE}HP/s")
    print(f"Enemy: {env._n_enemy_actions} actions, gravity-coupled Eq. 1 prediction (1 decision step, 100 substeps, paper-specified)")
    print(f"Eval: {eval_games} games every {eval_every} episodes (epsilon=0)")
    print(f"Training started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total episodes: {num_episodes}")
    print()

    results = []
    losses = []
    best_eval_wr = 0.0
    eval_history = []
    end_data = {'ra': [], 'ba': [], 'rh': [], 'bh': []}

    # Timing
    train_start_time = time.time()
    last_log_time = train_start_time

    for ep in range(1, num_episodes + 1):
        state = env.reset(episode=ep)

        while True:
            mask = env.get_action_mask()
            action = agent.select_action(state, action_mask=mask)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)
            state = next_state
            if done:
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
            wins = wc + wb
            wr = wins / len(rc) * 100
            al = np.mean(losses[-2000:]) if losses else 0
            ra = np.mean(end_data['ra'][-200:])
            ba = np.mean(end_data['ba'][-200:])
            rh = np.mean(end_data['rh'][-200:])
            bh = np.mean(end_data['bh'][-200:])

            # Timing
            now = time.time()
            block_time = now - last_log_time
            total_elapsed = now - train_start_time
            eps_per_sec = 200 / block_time if block_time > 0 else 0
            # ETA based on average rate so far
            avg_eps_per_sec = ep / total_elapsed if total_elapsed > 0 else eps_per_sec
            remaining_eps = num_episodes - ep
            eta_seconds = remaining_eps / avg_eps_per_sec if avg_eps_per_sec > 0 else 0

            print(f"Ep {ep:6d} | "
                  f"WC:{wc:2d} WB:{wb:2d} LC:{lc:3d} LB:{lb:2d} TO:{to:3d} D:{dr:2d} | "
                  f"WR:{wr:5.1f}% | Loss:{al:.4f} | Eps:{agent.epsilon():.2f} | "
                  f"RA:{ra:6.0f} BA:{ba:6.0f} | RH:{rh:5.1f} BH:{bh:5.1f} | "
                  f"Time: {timedelta(seconds=int(total_elapsed))} "
                  f"({eps_per_sec:.1f} ep/s) | "
                  f"ETA: {timedelta(seconds=int(eta_seconds))}")
            last_log_time = now

        # Periodic clean evaluation
        if ep % eval_every == 0:
            eval_start = time.time()
            print(f"\n[Eval @ Ep {ep}] Running {eval_games} games at PAPER distribution (epsilon=0)...")
            eval_result = evaluate(agent, eval_env, n_games=eval_games, eval_episode=None)
            eval_duration = time.time() - eval_start
            print(f"[Eval @ Ep {ep}] PAPER: WR: {eval_result['win_rate']:.1f}% "
                  f"[95% CI: {eval_result['ci_low']:.1f}-{eval_result['ci_high']:.1f}%] | "
                  f"WC:{eval_result['wc']} WB:{eval_result['wb']} "
                  f"LC:{eval_result['lc']} LB:{eval_result['lb']} D:{eval_result['draws']} | "
                  f"Eval time: {timedelta(seconds=int(eval_duration))}")

            # Diagnostic: evaluate at conditions matching agent's CURRENT training stage.
            # Reveals whether agent works in-distribution vs only at paper distribution.
            # During Stage 1: tests low-alt with ceiling.
            # During Stage 2: tests mixed altitude (50/50 low + paper).
            # During Stage 3: same as paper, so we skip.
            current_stage = (1 if ep < env.CURRICULUM_STAGE_1_END else
                             2 if ep < env.CURRICULUM_STAGE_2_END else 3)
            if current_stage < 3:
                stage_label = f"STAGE-{current_stage}"
                print(f"[Eval @ Ep {ep}] Running {eval_games // 2} games at {stage_label} distribution diagnostic (epsilon=0)...")
                diag_start = time.time()
                diag_result = evaluate(agent, eval_env, n_games=eval_games // 2, eval_episode=ep)
                diag_duration = time.time() - diag_start
                print(f"[Eval @ Ep {ep}] {stage_label}: WR: {diag_result['win_rate']:.1f}% "
                      f"[95% CI: {diag_result['ci_low']:.1f}-{diag_result['ci_high']:.1f}%] | "
                      f"WC:{diag_result['wc']} WB:{diag_result['wb']} "
                      f"LC:{diag_result['lc']} LB:{diag_result['lb']} D:{diag_result['draws']} | "
                      f"Eval time: {timedelta(seconds=int(diag_duration))}")

            eval_history.append((ep, eval_result['win_rate'], eval_result['ci_low'], eval_result['ci_high']))

            if eval_result['win_rate'] > best_eval_wr:
                best_eval_wr = eval_result['win_rate']
                agent.save('checkpoints/best_eval.pt')
                print(f"[Eval @ Ep {ep}] -> NEW BEST EVAL: {eval_result['win_rate']:.1f}%")
            print()

            agent.save(f'checkpoints/ep{ep}.pt')
            # Reset block timer after eval (eval time shouldn't count against next block)
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
