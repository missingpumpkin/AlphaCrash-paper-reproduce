"""
WVR Air Combat Environment - paper-faithful implementation.

Section/equation references:
  - Section 2.1 / Eq. 1: aircraft dynamics  (delegated to aircraft.py)
  - Section 2.3:        WEZ + health damage rules
  - Section 3.5 / Table 1: 37-D state representation
  - Section 3.6:        9x9x9 action space (paper's exact values)
  - Section 3.7 / Eqs. 13-18: terminal + non-terminal rewards
  - Section 3.8 / Eqs. 20-22 / Fig. 5: high-level enemy policy
  - Section 4.1.1:      uniform-random initial conditions
  - Table B1:           120 s episode, 100 ms decision step
"""
import numpy as np
from aircraft import Aircraft


# Paper's exact action values (Section 3.6)
ACC_VALUES = np.array([-30, -20, -10, 0, 10, 20, 40, 60, 90], dtype=np.float64)
YAW_VALUES = np.radians(np.array([-30, -21, -12, -5, 0, 5, 12, 21, 30], dtype=np.float64))
PITCH_VALUES = np.radians(np.array([-60, -34, -18, -8, 0, 8, 18, 34, 60], dtype=np.float64))
N_ACTIONS = 9 * 9 * 9  # 729


def decode_action(idx):
    """Flat action index [0, 728] -> (acc, yaw_rate, pitch_rate)."""
    p_i = idx % 9
    y_i = (idx // 9) % 9
    a_i = idx // 81
    return ACC_VALUES[a_i], YAW_VALUES[y_i], PITCH_VALUES[p_i]


class WVRCombatEnv:
    """One-on-one WVR air combat environment."""

    # Reward parameters (Section 4.1.3)
    D_IDEAL = 10000.0    # 10 km
    H_WARNING = 5000.0   # 5 km warning altitude
    D_CLOSE = 3000.0     # close-range threshold
    D_ENGAGE = 3000.0    # engagement range (enemy policy)

    # Combat rules (Section 2.3)
    WEZ_RANGE = 3000.0
    WEZ_ANGLE = np.radians(3.0)
    HEALTH_DAMAGE_RATE = 20.0     # HP per second
    COLLISION_DIST = 20.0

    # Episode limits (Table B1)
    MAX_TIME = 120.0
    DT = 0.1

    # --- Constrained-RL formulation (undocumented in paper) ---
    # Crash induction is fundamentally an offensive task: disengagement is
    # mission failure, not a neutral outcome. Combined with the hard service
    # ceiling enforced in Aircraft.step (z<=10km, eta clamped to <=0 there),
    # this confines exploration to a combat-relevant volume that's ~10x
    # smaller than unbounded space, giving 10x denser sample coverage.
    DISENGAGE_DIST = 50000.0          # >50km apart -> -1 (lose_disengaged)
    NO_ENGAGE_HEALTH = 99.0           # both >99 HP at timeout -> -1
    REWARD_SCALE = 0.1                # bound per-step rewards (Rb_enemy 1/h spike)

    # Engineering safety: NaN guard and absolute-position blow-up
    MAX_POSITION_RANGE = 200000.0     # 200km from origin (numerical only)

    def __init__(self):
        self.red = Aircraft()
        self.blue = Aircraft()
        self.t = 0.0

        # Enemy policy: pre-build the 729-action grid for vectorized search
        aa, yy, pp = np.meshgrid(ACC_VALUES, YAW_VALUES, PITCH_VALUES, indexing='ij')
        self._e_acc = aa.ravel().astype(np.float64)
        self._e_yaw = yy.ravel().astype(np.float64)
        self._e_pitch = pp.ravel().astype(np.float64)
        self._n_enemy_actions = len(self._e_acc)

        # Table 1: 14 features per aircraft x 2 + 9 relative = 37
        self.state_dim = 37
        self.action_dim = N_ACTIONS

    def reset(self, episode=None):
        """Section 4.1.1: uniform-random initial conditions.

        episode is accepted but ignored (no curriculum) -- kept for API
        compatibility with training loops that pass it.
        """
        del episode
        self.t = 0.0

        dist = np.random.uniform(500, 10000)
        r_alt = np.random.uniform(7000, 10000)
        b_alt = np.random.uniform(7000, 10000)
        r_heading = np.random.uniform(0, 2 * np.pi)
        b_heading = np.random.uniform(0, 2 * np.pi)

        angle = np.random.uniform(0, 2 * np.pi)
        self.red.reset(0, 0, r_alt, 300, 0, r_heading)
        self.blue.reset(dist * np.cos(angle), dist * np.sin(angle), b_alt, 300, 0, b_heading)
        return self._get_state()

    def get_action_mask(self):
        """No action masking (paper-faithful). Kept for API compatibility."""
        return np.ones(N_ACTIONS, dtype=bool)

    def step(self, action):
        """One decision step (Table B1: 100 ms)."""
        acc, yaw, pitch = decode_action(action)
        self.red.step(acc, yaw, pitch, self.DT)

        b_acc, b_yaw, b_pitch = self._enemy_policy()
        self.blue.step(b_acc, b_yaw, b_pitch, self.DT)

        self.t += self.DT

        # Engineering safety: numerical blow-up
        if not (np.isfinite(self.red.x) and np.isfinite(self.red.y) and np.isfinite(self.red.z)
                and np.isfinite(self.blue.x) and np.isfinite(self.blue.y) and np.isfinite(self.blue.z)):
            return self._get_state(), 0.0, True, 'draw_numerical_error'

        # Engineering safety: aircraft drifted absurdly far from origin
        if (max(abs(self.red.x), abs(self.red.y)) > self.MAX_POSITION_RANGE or
                max(abs(self.blue.x), abs(self.blue.y)) > self.MAX_POSITION_RANGE):
            return self._get_state(), 0.0, True, 'draw_out_of_bounds'

        self._apply_wez()
        reward, done, info = self._evaluate()

        if not np.isfinite(reward):
            return self._get_state(), 0.0, True, 'draw_numerical_error'
        return self._get_state(), float(reward), done, info

    def _get_state(self):
        """Table 1: 37-D state, scaled to [-1, 1] via tanh.

        Per aircraft (14 each, 28 total):
            position (3), direction angles yaw/roll/pitch (3),
            world heading vector (3), speed (1), accel (1),
            yaw_rate (1), pitch_rate (1), health (1)
        Relative (9):
            relative position (3), distance (1),
            sigma on XY/YZ/XZ planes (3), phi for each aircraft (2)
        """
        rp, bp = self.red.pos(), self.blue.pos()
        delta = bp - rp
        dist = max(np.linalg.norm(delta), 1.0)
        rw = self.red.heading()
        bw = self.blue.heading()

        phi_red = np.arccos(np.clip(np.dot(rw, delta) / (np.linalg.norm(rw) * dist), -1, 1))
        phi_blue = np.arccos(np.clip(np.dot(bw, -delta) / (np.linalg.norm(bw) * dist), -1, 1))

        sigma_xy = np.arctan2(delta[1], delta[0]) if (abs(delta[0]) + abs(delta[1])) > 1e-6 else 0.0
        sigma_yz = np.arctan2(delta[2], delta[1]) if (abs(delta[1]) + abs(delta[2])) > 1e-6 else 0.0
        sigma_xz = np.arctan2(delta[2], delta[0]) if (abs(delta[0]) + abs(delta[2])) > 1e-6 else 0.0

        state = np.array([
            self.red.x / 10000, self.red.y / 10000, self.red.z / 10000,
            self.red.psi / np.pi, self.red.mu / (np.pi / 2), self.red.eta / (np.pi / 2),
            rw[0], rw[1], rw[2],
            self.red.v / Aircraft.V_MAX,
            self.red.acc / 90,
            self.red.yaw_rate / Aircraft.MAX_YAW_RATE,
            self.red.pitch_rate / Aircraft.MAX_PITCH_RATE,
            self.red.health / 100,

            self.blue.x / 10000, self.blue.y / 10000, self.blue.z / 10000,
            self.blue.psi / np.pi, self.blue.mu / (np.pi / 2), self.blue.eta / (np.pi / 2),
            bw[0], bw[1], bw[2],
            self.blue.v / Aircraft.V_MAX,
            self.blue.acc / 90,
            self.blue.yaw_rate / Aircraft.MAX_YAW_RATE,
            self.blue.pitch_rate / Aircraft.MAX_PITCH_RATE,
            self.blue.health / 100,

            delta[0] / 10000, delta[1] / 10000, delta[2] / 10000,
            dist / 10000,
            sigma_xy / np.pi, sigma_yz / np.pi, sigma_xz / np.pi,
            phi_red / np.pi, phi_blue / np.pi,
        ], dtype=np.float32)

        # tanh keeps long-range information instead of clipping
        return np.tanh(state)

    def _apply_wez(self):
        """Section 2.3: Weapon Engagement Zone damage."""
        delta = self.blue.pos() - self.red.pos()
        dist = np.linalg.norm(delta)
        if dist < 1:
            return

        rw = self.red.heading()
        ata_r = np.arccos(np.clip(np.dot(rw, delta) / (np.linalg.norm(rw) * dist), -1, 1))
        if dist <= self.WEZ_RANGE and ata_r <= self.WEZ_ANGLE:
            self.blue.health -= self.HEALTH_DAMAGE_RATE * self.DT

        bw = self.blue.heading()
        ata_b = np.arccos(np.clip(np.dot(bw, -delta) / (np.linalg.norm(bw) * dist), -1, 1))
        if dist <= self.WEZ_RANGE and ata_b <= self.WEZ_ANGLE:
            self.red.health -= self.HEALTH_DAMAGE_RATE * self.DT

    def _evaluate(self):
        """Compute reward (Eqs. 13-18) and check terminal conditions."""
        delta = self.blue.pos() - self.red.pos()
        dist = max(np.linalg.norm(delta), 1.0)
        rw = self.red.heading()
        bw = self.blue.heading()

        # ATA: red's nose to blue (0 = red points at blue)
        ata = np.arccos(np.clip(np.dot(rw, delta) / (np.linalg.norm(rw) * dist), -1, 1))
        # AA: blue's heading to direction red->blue (0 = red on blue's 6 o'clock)
        aa = np.arccos(np.clip(np.dot(bw, delta) / (np.linalg.norm(bw) * dist), -1, 1))

        # === Eq. 13 (extended) -- terminal conditions ===
        # Terminal rewards stay at +/-1 (NOT scaled by REWARD_SCALE) so the
        # win signal stays large relative to per-step shaping. n-step returns
        # in train.py propagate the terminal back through the trajectory.

        # Constrained-RL: disengagement is mission failure (not a draw).
        # If aircraft drifted >50km apart, the agent has failed at the
        # offensive task regardless of who "caused" it.
        if dist > self.DISENGAGE_DIST:
            return -1.0, True, 'lose_disengaged'

        if dist < self.COLLISION_DIST:
            return 0.0, True, 'draw_collision'

        rc, bc = self.red.crashed, self.blue.crashed
        if rc and bc:
            return 0.0, True, 'draw_crash'
        if bc and not rc:
            return 1.0, True, 'win_crash'
        if rc and not bc:
            return -1.0, True, 'lose_crash'

        # Legitimate combat draw (mutual destruction) preserved
        if self.blue.health <= 0 and self.red.health <= 0:
            return 0.0, True, 'draw_blood'
        if self.blue.health <= 0:
            return 1.0, True, 'win_blood'
        if self.red.health <= 0:
            return -1.0, True, 'lose_blood'

        if self.t >= self.MAX_TIME:
            # Constrained-RL: if both aircraft survived ~unscathed, there
            # was no real engagement -> mission failure
            if self.red.health > self.NO_ENGAGE_HEALTH and self.blue.health > self.NO_ENGAGE_HEALTH:
                return -1.0, True, 'lose_timeout_no_engagement'
            if self.red.health > self.blue.health:
                return 1.0, True, 'win_timeout'
            elif self.red.health < self.blue.health:
                return -1.0, True, 'lose_timeout'
            return 0.0, True, 'draw_timeout'

        # === Eqs. 14-18: Non-terminal shaping reward ===
        # Eq. 14 distance reward (paper-verified by rendering page 7):
        #   d > d_ideal:  Rd = 1 - d/d_ideal             (negative, decreases as d grows)
        #   d <= d_ideal: Rd = e^(-(d/d_ideal - 1)) - 1  (positive, peaks at d=0 with +1.72)
        # Matches paper text: "providing incentives based on how close they are".
        if dist > self.D_IDEAL:
            Rd = 1.0 - dist / self.D_IDEAL
        else:
            Rd = np.exp(1.0 - dist / self.D_IDEAL) - 1.0

        # Eq. 15: angle reward (arccos already returns non-negative)
        Ra = 1.0 - (np.degrees(ata) + np.degrees(aa)) / 360.0

        # Eq. 16: speed reward
        vd = self.red.v - self.blue.v
        Rv = vd / 700.0 if vd <= 0 else vd / 300.0

        # Eq. 17: boundary rewards (zero above warning altitude)
        if self.red.z < self.H_WARNING:
            Rb_self = self.red.z / self.H_WARNING - 1.0
        else:
            Rb_self = 0.0

        if self.blue.z < self.H_WARNING:
            Rb_enemy = self.H_WARNING / max(self.blue.z, 1.0) - 1.0
        else:
            Rb_enemy = 0.0

        # Eq. 18: adaptive weighting on distance
        if dist <= self.D_CLOSE:
            R = 0.6 * Rd + 0.3 * Ra + 0.1 * Rv + 0.5 * (Rb_self + Rb_enemy)
        else:
            R = 0.7 * Ra + 0.2 * Rv + 0.5 * (Rb_self + Rb_enemy)

        # Reward scaling: Rb_enemy can spike to +49 at h=100m (Eq. 17 has a
        # 1/h singularity). With typical step rewards O(1), QR-DQN's 20
        # quantiles can't span that range cleanly. Scaling brings step
        # rewards into ~[-1, +1]. Terminal +/-1 rewards are NOT scaled.
        R *= self.REWARD_SCALE

        return R, False, 'ongoing'

    def _enemy_policy(self):
        """Eqs. 20-22 / Section 3.8 / Fig. 5: high-level pilot-experience policy.

        For each of 729 candidate actions, integrate Blue forward by one
        decision step (100 ms = 100 x 1 ms substeps) using the same
        gravity-coupled inner-loop dynamics as Aircraft.step. Then evaluate
        Eqs. 20-22 advantage at the predicted next state and pick argmin.
        """
        G = Aircraft.G
        N = self._n_enemy_actions
        red_pos = self.red.pos()
        red_w = self.red.heading()
        red_w_n = np.linalg.norm(red_w)

        px = np.full(N, self.blue.x, dtype=np.float64)
        py = np.full(N, self.blue.y, dtype=np.float64)
        pz = np.full(N, self.blue.z, dtype=np.float64)
        pv = np.full(N, self.blue.v, dtype=np.float64)
        pe = np.full(N, self.blue.eta, dtype=np.float64)
        pp = np.full(N, self.blue.psi, dtype=np.float64)

        dy = np.clip(self._e_yaw, -Aircraft.MAX_YAW_RATE, Aircraft.MAX_YAW_RATE)
        dp = np.clip(self._e_pitch, -Aircraft.MAX_PITCH_RATE, Aircraft.MAX_PITCH_RATE)

        # --- Undocumented patch #3: speed up enemy policy ---
        # The actual aircraft uses 100 x 1ms substeps for high-G dynamics
        # stability. The enemy's forward prediction over a single 100ms
        # decision step doesn't need that resolution; 10 x 10ms is plenty
        # accurate and ~10x faster (this is the wall-clock bottleneck).
        N_PRED_SUBSTEPS = 10
        h = self.DT / N_PRED_SUBSTEPS

        for _ in range(N_PRED_SUBSTEPS):
            cos_pe = np.cos(pe)

            A = pv * dp / G + cos_pe
            B = np.where(np.abs(cos_pe) > 1e-8, pv * cos_pe * dy / G, 0.0)
            nz_magnitude = np.sqrt(A * A + B * B)
            nz_required = np.where(A < 0, -nz_magnitude, nz_magnitude)
            mu = np.arctan2(B, np.abs(A))
            nz = np.clip(nz_required, Aircraft.NZ_MIN, Aircraft.NZ_MAX)

            actual_eta_dot = (G / pv) * (nz * np.cos(mu) - cos_pe)
            actual_psi_dot = np.where(np.abs(cos_pe) > 1e-8,
                                      G * nz * np.sin(mu) / (pv * cos_pe),
                                      0.0)

            px += pv * cos_pe * np.sin(pp) * h
            py += pv * cos_pe * np.cos(pp) * h
            pz += pv * np.sin(pe) * h
            pv = np.clip(pv + self._e_acc * h, Aircraft.V_MIN, Aircraft.V_MAX)
            pe = np.clip(pe + actual_eta_dot * h, -np.pi / 2, np.pi / 2)
            pp += actual_psi_dot * h

            # Constrained-RL: hard ceiling (matches Aircraft.step exactly so
            # the predicted next state is what blue will actually experience)
            at_ceiling = pz >= Aircraft.CEILING_ALT
            pz = np.where(at_ceiling, Aircraft.CEILING_ALT, pz)
            pe = np.where(at_ceiling & (pe > 0), 0.0, pe)

        ce_final = np.cos(pe)
        bw = np.column_stack([ce_final * np.sin(pp),
                              ce_final * np.cos(pp),
                              np.sin(pe)])

        d2r = red_pos[None, :] - np.column_stack([px, py, pz])
        d = np.maximum(np.linalg.norm(d2r, axis=1), 1.0)

        # Eq. 20: Ta from blue's perspective (smaller is better for blue)
        bn = np.linalg.norm(bw, axis=1)
        cos_b_ata = np.einsum('ij,ij->i', bw, d2r) / (bn * d)
        b_ata = np.arccos(np.clip(cos_b_ata, -1, 1))
        cos_r_aa = np.einsum('j,ij->i', red_w, d2r) / (red_w_n * d)
        r_aa = np.arccos(np.clip(cos_r_aa, -1, 1))
        Ta = (np.degrees(b_ata) + np.degrees(r_aa)) / 360.0

        # Eq. 21: distance advantage (4 cases on Ta vs 0.5 and d vs d_engage)
        Td = np.where(
            (Ta < 0.5) & (d <= self.D_ENGAGE), 1.0 - (self.D_ENGAGE - d) / self.D_ENGAGE,
            np.where(
                (Ta < 0.5) & (d > self.D_ENGAGE), 1.0 - (self.D_ENGAGE - d) / d,
                np.where(
                    (Ta >= 0.5) & (d > self.D_ENGAGE), 3.0 + (self.D_ENGAGE - d) / d,
                    3.0 + (self.D_ENGAGE - d) / self.D_ENGAGE,
                )))

        # Eq. 22: T = Ta * Td; pick argmin (smallest T = best for blue)
        T = Ta * Td
        idx = np.argmin(T)
        return self._e_acc[idx], self._e_yaw[idx], self._e_pitch[idx]
