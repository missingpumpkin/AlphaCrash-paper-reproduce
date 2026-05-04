"""
WVR Air Combat Environment - Complete paper implementation.

Components:
  - Full state representation (Table 1, ~30D per frame)
  - Frame stacking (Eq. 2): past τ frames for temporal info
  - WEZ + Health system (Section 2.3)
  - Terminal reward with health AND crash (Eq. 13)
  - Non-terminal reward: distance, angle, speed, boundary (Eqs. 14-18)
  - Enemy policy: 729-action search, multi-step gravity prediction (Eqs. 20-22)
  - Action space: 9×9×9 = 729 (Section 3.6, paper's exact values)
"""
import numpy as np
from aircraft import Aircraft


# Paper's EXACT action values (Section 3.6)
ACC_VALUES = np.array([-30, -20, -10, 0, 10, 20, 40, 60, 90], dtype=np.float64)
YAW_VALUES = np.radians(np.array([-30, -21, -12, -5, 0, 5, 12, 21, 30], dtype=np.float64))
PITCH_VALUES = np.radians(np.array([-60, -34, -18, -8, 0, 8, 18, 34, 60], dtype=np.float64))
N_ACTIONS = 9 * 9 * 9  # 729


def decode_action(idx):
    """Decode flat action index [0, 728] → (acc, yaw_rate, pitch_rate)."""
    p_i = idx % 9
    y_i = (idx // 9) % 9
    a_i = idx // 81
    return ACC_VALUES[a_i], YAW_VALUES[y_i], PITCH_VALUES[p_i]


class WVRCombatEnv:
    """One-on-one WVR air combat environment."""

    # Reward parameters (Section 4.1.3)
    D_IDEAL = 10000.0    # 10 km
    H_WARNING = 5000.0   # 5 km warning altitude
    D_CLOSE = 3000.0     # close range threshold
    D_ENGAGE = 3000.0    # engagement range

    # Combat rules (Section 2.3)
    WEZ_RANGE = 3000.0
    WEZ_ANGLE = np.radians(3.0)  # 3° cone
    HEALTH_DAMAGE_RATE = 20.0     # HP per second
    COLLISION_DIST = 20.0

    # Episode limits (Table B1)
    MAX_TIME = 120.0  # seconds
    DT = 0.1          # 100ms decision step

    # Frame stacking (Eq. 2)
    N_FRAMES = 1       # H1 test: single-frame (paper doesn't specify τ; remove stacking)

    def __init__(self):
        self.red = Aircraft()
        self.blue = Aircraft()
        self.t = 0.0

        # Pre-build enemy action grid (729 actions)
        aa, yy, pp = np.meshgrid(ACC_VALUES, YAW_VALUES, PITCH_VALUES, indexing='ij')
        self._e_acc = aa.ravel().astype(np.float64)
        self._e_yaw = yy.ravel().astype(np.float64)
        self._e_pitch = pp.ravel().astype(np.float64)
        self._n_enemy_actions = len(self._e_acc)

        # Single-frame state dimension
        self._single_frame_dim = 37  # Paper Table 1: 14 per aircraft × 2 + 9 relative
        # Total state dimension with frame stacking
        self.state_dim = self._single_frame_dim * self.N_FRAMES
        self.action_dim = N_ACTIONS

        # Frame buffer for stacking
        self._frame_buffer = []

    # Curriculum learning configuration
    # DISABLED: Paper does not use curriculum (Section 4.1.1 specifies uniform random
    # initial conditions throughout training). Setting both stage endpoints to 0
    # makes all episodes use Stage 3 (paper distribution).
    # Previous curriculum (Stage 1: ceiling 3000m + low alt + close, Stage 2: mixed)
    # introduced transfer problem at Stage 1→2 transition. Removed to test paper's
    # claim of emergent crash induction strategy without curriculum.
    # All non-paper reward shaping (gentle bridge, descent incentive) has been
    # removed in favor of paper-faithful Eq. 17 (Rb=0 above warning altitude).
    # The hard 10km altitude cap (Aircraft.CEILING_ALT) bounds the no-penalty zone.
    CURRICULUM_STAGE_1_END = 0      # disabled (was 3000)
    CURRICULUM_STAGE_2_END = 0      # disabled (was 8000)
    STAGE_1_ALTITUDE_CEILING = 3000.0  # legacy (no longer enforced)
    STAGE_2_DESCENT_REWARD_THRESHOLD = 5000.0  # legacy

    def reset(self, episode=None):
        """Reset environment with curriculum-aware initialization.

        Args:
            episode: current episode number for curriculum scheduling.
                     If None, uses paper's full random distribution (Stage 3 / no curriculum).
        """
        self.t = 0.0

        # Determine curriculum stage
        if episode is None:
            stage = 3  # default to paper distribution
        elif episode < self.CURRICULUM_STAGE_1_END:
            stage = 1
        elif episode < self.CURRICULUM_STAGE_2_END:
            stage = 2
        else:
            stage = 3

        if stage == 1:
            # Stage 1: pre-established engagement at low altitude
            # Forces immediate exposure to critical states (low alt, close engagement)
            dist = np.random.uniform(500, 1500)            # close range
            r_alt = np.random.uniform(800, 2500)            # low altitude (above warning issue)
            b_alt = np.random.uniform(r_alt + 500, r_alt + 2000)  # blue above red
            r_heading = np.random.uniform(0, 2 * np.pi)
            # Blue heading mostly toward red (chasing)
            angle_to_red = np.random.uniform(-np.pi / 6, np.pi / 6)  # within ±30° of facing red
            b_heading = r_heading + np.pi + angle_to_red  # opposite of red ± noise
        elif stage == 2:
            # Stage 2: mix of stage 1 and paper conditions (50/50)
            if np.random.random() < 0.5:
                # Stage 1 style
                dist = np.random.uniform(500, 1500)
                r_alt = np.random.uniform(800, 2500)
                b_alt = np.random.uniform(r_alt + 500, r_alt + 2000)
                r_heading = np.random.uniform(0, 2 * np.pi)
                angle_to_red = np.random.uniform(-np.pi / 6, np.pi / 6)
                b_heading = r_heading + np.pi + angle_to_red
            else:
                # Paper style
                dist = np.random.uniform(500, 10000)
                r_alt = np.random.uniform(7000, 10000)
                b_alt = np.random.uniform(7000, 10000)
                r_heading = np.random.uniform(0, 2 * np.pi)
                b_heading = np.random.uniform(0, 2 * np.pi)
        else:
            # Stage 3: paper's full random distribution (Section 4.1.1)
            dist = np.random.uniform(500, 10000)
            r_alt = np.random.uniform(7000, 10000)
            b_alt = np.random.uniform(7000, 10000)
            r_heading = np.random.uniform(0, 2 * np.pi)
            b_heading = np.random.uniform(0, 2 * np.pi)

        angle = np.random.uniform(0, 2 * np.pi)
        self.red.reset(0, 0, r_alt, 300, 0, r_heading)
        self.blue.reset(dist * np.cos(angle), dist * np.sin(angle), b_alt, 300, 0, b_heading)

        # Track initial altitudes for shaping reward calculations
        self.red_init_alt = r_alt
        self.blue_init_alt = b_alt

        # Initialize frame buffer with copies of first frame
        first_frame = self._get_single_frame()
        self._frame_buffer = [first_frame.copy() for _ in range(self.N_FRAMES)]

        # Track current stage for logging
        self.curriculum_stage = stage

        return self._get_stacked_state()

    # Defensive bounds for edge case enforcement
    MAX_POSITION_RANGE = 50000.0  # 50km from origin; episode ends if exceeded
    MAX_ALT = 10000.0             # 10 km hard altitude cap (paper's operational ceiling)
    MAX_PER_STEP_REWARD = 100.0   # clip extreme reward magnitudes (Rb_enemy can spike)

    # Action masking thresholds
    # SAFETY_FLOOR: paper-aligned level-out altitude with engineering margin.
    # Section 4.3.1 describes Red as "levels out approximately 100 m from the limit"
    # where the limit = CRASH_ALT (100m). We use interpretation (a):
    # "100m from the limit" = 100m above 100m crash altitude = 200m absolute altitude.
    # Engineering rationale: Decision step 100ms with max pitch rate 60°/s and combat
    # speeds 300-500 m/s can produce 5-15m altitude loss per step. With 10-substep
    # forward prediction error, 100m buffer above crash altitude is engineering-safe.
    # Tactical rationale: matches paper's emergent optimal behavior (Fig 13b).
    # Earlier test of 100m floor combined with aggressive shaping caused agent to
    # collapse to 600-800m altitude with 50%+ self-crash rate (reward hacking).
    # PASSIVE_DISTANCE: set to D_IDEAL=10km, the paper's "ideal" distance beyond which
    #   distance reward becomes negative (Eq. 14). Forcing maintain-or-close beyond this
    #   aligns with the paper's reward structure.
    SAFETY_FLOOR = 200.0          # paper interp (a): 100m above crash limit, engineering margin
    PASSIVE_DISTANCE = 10000.0    # = D_IDEAL (paper-defined)
    LURE_ALTITUDE = 1000.0        # legacy parameter (currently unused)

    def get_action_mask(self):
        """
        Action masking is NOT in paper - removed for paper-faithful methodology.

        Paper has zero mention of action filtering. Agent has full 729-action
        space at every step. Altitude management and engagement come entirely
        from reward signal (Rb_self penalty for low altitude, Rd penalty for
        far distance, etc.).

        Returns: bool array of all True (no actions masked).

        Previous implementation had two engineering constraints:
        1. Anti-suicide: predicted z < SAFETY_FLOOR (200m) masked
        2. Anti-passivity: actions increasing distance when d > 10000m masked

        Removed both. Agent must learn altitude management and engagement
        purely from reward signal, as paper specifies.

        Trade-off: slower learning, more catastrophic episodes during
        exploration. But cleaner paper-faithful methodology.
        """
        return np.ones(N_ACTIONS, dtype=bool)

    def step(self, action):
        """Execute one decision step."""
        # Decode and apply red (agent) action
        acc, yaw, pitch = decode_action(action)
        self.red.step(acc, yaw, pitch, self.DT)

        # Compute and apply blue (enemy) action
        b_acc, b_yaw, b_pitch = self._enemy_policy()
        self.blue.step(b_acc, b_yaw, b_pitch, self.DT)

        self.t += self.DT

        # Stage 1 hard altitude ceiling: clamp aircraft altitudes during Stage 1
        # This prevents the agent from escaping to high altitude where crash induction
        # is impossible, forcing it to learn at low altitude.
        if self.curriculum_stage == 1:
            if self.red.z > self.STAGE_1_ALTITUDE_CEILING:
                self.red.z = self.STAGE_1_ALTITUDE_CEILING
                # Force level flight at ceiling
                if self.red.eta > 0:
                    self.red.eta = 0.0
            if self.blue.z > self.STAGE_1_ALTITUDE_CEILING:
                self.blue.z = self.STAGE_1_ALTITUDE_CEILING
                if self.blue.eta > 0:
                    self.blue.eta = 0.0

        # Edge case: detect NaN/Inf in aircraft state (numerical instability)
        if not (np.isfinite(self.red.x) and np.isfinite(self.red.y) and np.isfinite(self.red.z)
                and np.isfinite(self.blue.x) and np.isfinite(self.blue.y) and np.isfinite(self.blue.z)):
            # Numerical blow-up: terminate as draw with no signal
            return self._get_stacked_state(), 0.0, True, 'draw_numerical_error'

        # Edge case: aircraft flew too far from origin (state representation breaks down)
        red_range = max(abs(self.red.x), abs(self.red.y))
        blue_range = max(abs(self.blue.x), abs(self.blue.y))
        if red_range > self.MAX_POSITION_RANGE or blue_range > self.MAX_POSITION_RANGE:
            # Out-of-bounds: terminate as draw (neither side wins through evasion)
            return self._get_stacked_state(), 0.0, True, 'draw_out_of_bounds'

        # Altitude management is now via soft reward penalty (in _evaluate),
        # not hard episode termination. Hermes diagnostic showed hard ceiling
        # caused 75-85% of episodes to terminate as zero-reward draws when
        # aircraft drifted up — destroying learning signal. Paper has no
        # explicit ceiling. Soft penalty above 8km (computed in _evaluate)
        # creates gradient pulling agent down without killing episodes.
        # The intra-substep ceiling clamp in Aircraft.step is also disabled.

        # Apply WEZ damage
        self._apply_wez()

        # Update frame buffer
        new_frame = self._get_single_frame()
        self._frame_buffer.pop(0)
        self._frame_buffer.append(new_frame)

        # Evaluate
        reward, done, info = self._evaluate()

        # Edge case: clip reward magnitude to prevent extreme outliers
        # (Rb_enemy can reach +49 when blue.z near 100m; cumulative could explode)
        reward = float(np.clip(reward, -self.MAX_PER_STEP_REWARD, self.MAX_PER_STEP_REWARD))

        # Edge case: detect NaN reward
        if not np.isfinite(reward):
            return self._get_stacked_state(), 0.0, True, 'draw_numerical_error'

        return self._get_stacked_state(), reward, done, info

    def _get_single_frame(self):
        """Build single-frame state vector per paper Table 1, normalized to [-1, 1].

        Total: 37 features (matching paper specification)
            Per aircraft (14 each, 28 total):
                Position n_i (3D): x, y, z
                Direction angles (3D): yaw psi, roll mu, pitch eta
                World direction vector w_i (3D): heading in 3D
                Speed v_i (1D)
                Acceleration a_i (1D)
                Yaw rate (1D)
                Pitch rate (1D)
                Health H_i (1D)
            Relative (9):
                Relative position vector Delta_n (3D)
                Relative distance d (1D)
                Relative angles sigma on XY, YZ, XZ planes (3D)
                Angular difference phi for each aircraft (2 scalars)
        """
        rp, bp = self.red.pos(), self.blue.pos()
        delta = bp - rp
        dist = max(np.linalg.norm(delta), 1.0)
        rw = self.red.heading()
        bw = self.blue.heading()

        # Angular difference φ for each aircraft (Table 1 definition):
        # φ_i = arccos(w_i · Δn / |w_i| |Δn|)
        # For Red: angle between Red's heading and direction to Blue
        # For Blue: angle between Blue's heading and direction to Red (negated delta)
        phi_red = np.arccos(np.clip(np.dot(rw, delta) / (np.linalg.norm(rw) * dist), -1, 1))
        phi_blue = np.arccos(np.clip(np.dot(bw, -delta) / (np.linalg.norm(bw) * dist), -1, 1))

        # Relative angles σ: projections of Δn onto XY, YZ, XZ planes
        # σ_xy: angle in XY plane (azimuth from +X axis)
        # σ_yz: angle in YZ plane (elevation from +Y axis viewed from X)
        # σ_xz: angle in XZ plane (elevation from +X axis viewed from Y)
        sigma_xy = np.arctan2(delta[1], delta[0]) if (abs(delta[0]) + abs(delta[1])) > 1e-6 else 0.0
        sigma_yz = np.arctan2(delta[2], delta[1]) if (abs(delta[1]) + abs(delta[2])) > 1e-6 else 0.0
        sigma_xz = np.arctan2(delta[2], delta[0]) if (abs(delta[0]) + abs(delta[2])) > 1e-6 else 0.0

        state = np.array([
            # Red aircraft (14 features)
            self.red.x / 10000, self.red.y / 10000, self.red.z / 10000,         # position (3)
            self.red.psi / np.pi, self.red.mu / (np.pi / 2), self.red.eta / (np.pi / 2),  # angles yaw,roll,pitch (3)
            rw[0], rw[1], rw[2],                                                  # world heading (3)
            self.red.v / Aircraft.V_MAX,                                          # speed (1)
            self.red.acc / 90,                                                    # acceleration (1)
            self.red.yaw_rate / Aircraft.MAX_YAW_RATE,                            # yaw rate (1)
            self.red.pitch_rate / Aircraft.MAX_PITCH_RATE,                        # pitch rate (1)
            self.red.health / 100,                                                # health (1)
            # Blue aircraft (14 features)
            self.blue.x / 10000, self.blue.y / 10000, self.blue.z / 10000,        # position (3)
            self.blue.psi / np.pi, self.blue.mu / (np.pi / 2), self.blue.eta / (np.pi / 2),  # angles (3)
            bw[0], bw[1], bw[2],                                                  # world heading (3)
            self.blue.v / Aircraft.V_MAX,                                         # speed (1)
            self.blue.acc / 90,                                                   # acceleration (1)
            self.blue.yaw_rate / Aircraft.MAX_YAW_RATE,                           # yaw rate (1)
            self.blue.pitch_rate / Aircraft.MAX_PITCH_RATE,                       # pitch rate (1)
            self.blue.health / 100,                                               # health (1)
            # Relative features (9)
            delta[0] / 10000, delta[1] / 10000, delta[2] / 10000,                 # relative position (3)
            dist / 10000,                                                          # distance (1)
            sigma_xy / np.pi, sigma_yz / np.pi, sigma_xz / np.pi,                 # plane angles (3)
            phi_red / np.pi, phi_blue / np.pi,                                    # angular diff (2)
        ], dtype=np.float32)

        # Use tanh instead of clip to preserve long-range information.
        # Hermes diagnostic: clip(state, -1, 1) made "enemy 10km away" look
        # identical to "enemy 30km away" because both x/10000 and dist/10000
        # exceed 1.0 routinely (aircraft fly far in 120s episodes).
        # tanh(x) is smooth, monotonic, preserves order, asymptotes to ±1
        # but never loses information. Paper says "scaled to [-1,1]" not
        # "clipped to [-1,1]" — tanh is the standard interpretation.
        return np.tanh(state)

    def _get_stacked_state(self):
        """Eq. 2: stack past N_FRAMES frames for temporal information."""
        return np.concatenate(self._frame_buffer)

    def _apply_wez(self):
        """Section 2.3: Weapon Engagement Zone damage."""
        delta = self.blue.pos() - self.red.pos()
        dist = np.linalg.norm(delta)
        if dist < 1:
            return

        # Red shooting at Blue
        rw = self.red.heading()
        ata_r = np.arccos(np.clip(np.dot(rw, delta) / (np.linalg.norm(rw) * dist), -1, 1))
        if dist <= self.WEZ_RANGE and ata_r <= self.WEZ_ANGLE:
            self.blue.health -= self.HEALTH_DAMAGE_RATE * self.DT

        # Blue shooting at Red
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
        # ATA (Antenna Train Angle): angle from red's nose to blue.
        #   ATA=0 means red points at blue (good for tracking/shooting)
        #   ATA=180 means blue is behind red (bad)
        ata = np.arccos(np.clip(np.dot(rw, delta) / (np.linalg.norm(rw) * dist), -1, 1))
        # AA (Aspect Angle): standard aerospace convention is angle from target's
        # tail extended to the line-of-sight from attacker to target.
        # Equivalent: angle between blue's heading direction and the direction
        # from red to blue (delta vector). This gives:
        #   AA=0: red is behind blue (red on blue's 6 o'clock)
        #   AA=180: red is in front of blue (head-on)
        # Per paper Eq. 20: Ta=0 means "directly behind enemy" requiring AA=0
        # in this position. Previously had AA computed as angle from blue's
        # NOSE to red (using -delta), which made Ra peak at head-on instead
        # of tail position. Verified against paper definition.
        aa = np.arccos(np.clip(np.dot(bw, delta) / (np.linalg.norm(bw) * dist), -1, 1))

        # === Eq. 13: Terminal conditions ===
        if dist < self.COLLISION_DIST:
            return 0.0, True, 'draw_collision'

        rc, bc = self.red.crashed, self.blue.crashed
        if rc and bc:
            return 0.0, True, 'draw_crash'
        if bc and not rc:
            return 1.0, True, 'win_crash'
        if rc and not bc:
            return -1.0, True, 'lose_crash'

        if self.blue.health <= 0 and self.red.health <= 0:
            return 0.0, True, 'draw_blood'
        if self.blue.health <= 0:
            return 1.0, True, 'win_blood'
        if self.red.health <= 0:
            return -1.0, True, 'lose_blood'

        if self.t >= self.MAX_TIME:
            if self.red.health > self.blue.health:
                return 1.0, True, 'win_timeout'
            elif self.red.health < self.blue.health:
                return -1.0, True, 'lose_timeout'
            return 0.0, True, 'draw_timeout'

        # === Eqs. 14-18: Non-terminal shaping reward ===
        # Eq. 14: Distance reward (paper-verified by rendering PDF page 7)
        # When d > d_ideal:  Rd = 1 - d/d_ideal             (negative, decreases as d grows)
        # When d ≤ d_ideal:  Rd = e^(-(d/d_ideal - 1)) - 1  (positive, peaks at d=0)
        # Shape: monotonically decreases as d grows.
        # At d=0: Rd ≈ +1.72 (max reward for being right on top of enemy).
        # At d=d_ideal: Rd = 0.
        # At d=2*d_ideal: Rd = -1.
        # Matches paper text: "providing incentives based on how close they are"
        # and "encouraging the enemy to follow us closely".
        if dist > self.D_IDEAL:
            Rd = 1.0 - dist / self.D_IDEAL
        else:
            Rd = np.exp(1.0 - dist / self.D_IDEAL) - 1.0

        # Eq. 15: Angle reward
        Ra = 1.0 - (np.degrees(ata) + np.degrees(aa)) / 360.0

        # Eq. 16: Speed reward (paper uses ≤, not <)
        vd = self.red.v - self.blue.v
        Rv = vd / 700.0 if vd <= 0 else vd / 300.0

        # Eq. 17: Boundary rewards (paper-faithful)
        # Below warning: Rb_self = z/h_warning - 1 (negative, max -0.96 at 200m)
        # Below warning: Rb_enemy = h_warning/z - 1 (positive, +49 at 100m)
        # Above warning: both = 0 (paper's exact formulation)
        if self.red.z < self.H_WARNING:
            Rb_self = (self.red.z / self.H_WARNING - 1.0)
        else:
            Rb_self = 0.0

        if self.blue.z < self.H_WARNING:
            Rb_enemy = (self.H_WARNING / max(self.blue.z, 1.0) - 1.0)
        else:
            Rb_enemy = 0.0

        # Eq. 18: Adaptive weighting (paper)
        # Paper: d ≤ d_close uses 0.6 Rd + 0.3 Ra + 0.1 Rv + 0.5(Rb_self + Rb_enemy)
        # Paper: d > d_close uses 0.7 Ra + 0.2 Rv + 0.5(Rb_self + Rb_enemy)
        if dist <= self.D_CLOSE:
            R = 0.6 * Rd + 0.3 * Ra + 0.1 * Rv + 0.5 * (Rb_self + Rb_enemy)
        else:
            R = 0.7 * Ra + 0.2 * Rv + 0.5 * (Rb_self + Rb_enemy)

        return R, False, 'ongoing'

    def _enemy_policy(self):
        """
        Eqs. 20-22: High-level enemy policy (Section 3.8, Figure 5).

        Per paper Figure 5 pipeline:
          Air combat info -> Action space search (9x9x9)
            -> Runge-Kutta transition -> Advantage assessment -> Best action

        Implementation: For each of 729 candidate actions, integrate Blue
        forward by 1 decision step (100ms = 100 substeps × 1ms each) using
        the same gravity-coupled inner-loop dynamics as Aircraft.step().
        Then evaluate Eqs. 20-22 advantage at the predicted next state.

        Horizon: 1 decision step (paper says "next-phase scenarios" without
        specifying further; we interpret as next decision step).

        Substeps: 100 × 1ms = 100ms decision step (matches Table B1:
        State update 1ms, Decision-making 100ms). Identical physics
        integration to Aircraft.step() — the enemy "feels" exactly the
        same dynamics it will encounter when the action is executed.
        """
        G = Aircraft.G
        N = self._n_enemy_actions
        red_pos = self.red.pos()
        red_w = self.red.heading()
        red_w_n = np.linalg.norm(red_w)

        # Vectorized parallel integration of N=729 candidate next states
        px = np.full(N, self.blue.x, dtype=np.float64)
        py = np.full(N, self.blue.y, dtype=np.float64)
        pz = np.full(N, self.blue.z, dtype=np.float64)
        pv = np.full(N, self.blue.v, dtype=np.float64)
        pe = np.full(N, self.blue.eta, dtype=np.float64)
        pp = np.full(N, self.blue.psi, dtype=np.float64)

        # Pre-clip desired rates for each candidate action to kinematic limits
        dy = np.clip(self._e_yaw, -Aircraft.MAX_YAW_RATE, Aircraft.MAX_YAW_RATE)
        dp = np.clip(self._e_pitch, -Aircraft.MAX_PITCH_RATE, Aircraft.MAX_PITCH_RATE)

        # 1 decision step = 100 substeps × 1ms each (Table B1)
        # Use paper Eq. 1 gravity-coupled dynamics with inner-loop n_z, μ
        # (matches Aircraft.step exactly so blue "feels" what it'll execute)
        N_PRED_SUBSTEPS = Aircraft.N_SUBSTEPS  # 100
        h = self.DT / N_PRED_SUBSTEPS  # 1ms each

        for _ in range(N_PRED_SUBSTEPS):
            cos_pe = np.cos(pe)

            # Invert Eq. 1 for each of N candidate actions
            A = pv * dp / G + cos_pe
            B = np.where(np.abs(cos_pe) > 1e-8,
                         pv * cos_pe * dy / G,
                         0.0)

            nz_magnitude = np.sqrt(A * A + B * B)
            nz_required = np.where(A < 0, -nz_magnitude, nz_magnitude)
            mu = np.arctan2(B, np.abs(A))

            # Clamp n_z to physical limits (Table B1)
            nz = np.clip(nz_required, Aircraft.NZ_MIN, Aircraft.NZ_MAX)

            # Apply Eq. 1 forward with clamped n_z
            actual_eta_dot = (G / pv) * (nz * np.cos(mu) - cos_pe)
            actual_psi_dot = np.where(np.abs(cos_pe) > 1e-8,
                                      G * nz * np.sin(mu) / (pv * cos_pe),
                                      0.0)

            # Integrate position
            px += pv * cos_pe * np.sin(pp) * h
            py += pv * cos_pe * np.cos(pp) * h
            pz += pv * np.sin(pe) * h

            # Integrate state
            pv = np.clip(pv + self._e_acc * h, Aircraft.V_MIN, Aircraft.V_MAX)
            pe = np.clip(pe + actual_eta_dot * h, -np.pi / 2, np.pi / 2)
            pp += actual_psi_dot * h

        # Predicted next state for each action: position (px, py, pz), heading (pe, pp)
        ce_final = np.cos(pe)
        bw = np.column_stack([
            ce_final * np.sin(pp),
            ce_final * np.cos(pp),
            np.sin(pe)
        ])

        # Relative position from predicted blue to current red
        d2r = red_pos[None, :] - np.column_stack([px, py, pz])
        d = np.maximum(np.linalg.norm(d2r, axis=1), 1.0)

        # Eq. 20: Angular advantage Ta (from blue's perspective)
        # ATA: blue's nose to red (direction blue->red = +d2r)
        # AA: red's tail extended to blue = angle from rw to d2r
        #     (NOT -d2r; corrected for paper's standard aerospace AA convention)
        # When Ta=0: blue is behind red AND blue points at red (best for blue)
        # When Ta=1: red is behind blue AND red points at blue (worst for blue)
        bn = np.linalg.norm(bw, axis=1)
        cos_b_ata = np.einsum('ij,ij->i', bw, d2r) / (bn * d)
        b_ata = np.arccos(np.clip(cos_b_ata, -1, 1))
        cos_r_aa = np.einsum('j,ij->i', red_w, d2r) / (red_w_n * d)
        r_aa = np.arccos(np.clip(cos_r_aa, -1, 1))
        Ta = (np.degrees(b_ata) + np.degrees(r_aa)) / 360.0

        # Eq. 21: Distance advantage Td (boundaries match paper exactly: ≤ vs >)
        Td = np.where(
            (Ta < 0.5) & (d <= self.D_ENGAGE), 1.0 - (self.D_ENGAGE - d) / self.D_ENGAGE,
            np.where(
                (Ta < 0.5) & (d > self.D_ENGAGE), 1.0 - (self.D_ENGAGE - d) / d,
                np.where(
                    (Ta >= 0.5) & (d > self.D_ENGAGE), 3.0 + (self.D_ENGAGE - d) / d,
                    3.0 + (self.D_ENGAGE - d) / self.D_ENGAGE
                )))

        # Eq. 22: Overall advantage T = Ta × Td
        T = Ta * Td
        idx = np.argmin(T)
        return self._e_acc[idx], self._e_yaw[idx], self._e_pitch[idx]
