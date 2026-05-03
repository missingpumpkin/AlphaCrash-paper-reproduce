"""
Aircraft dynamics — Paper Eq. 1: 3D gravity-coupled coordinated turn model.

Paper Eq. 1 (verified from paper image):
    ẋ = v cos η sin ψ
    ẏ = v cos η cos ψ
    ż = v sin η
    η̇ = (g/v)(n_z cos μ - cos η)         <-- gravity-coupled flight path
    ψ̇ = (g n_z sin μ) / (v cos η)         <-- gravity-coupled heading

Where:
    η = flight path angle (pitch, "track angle")
    ψ = heading angle (yaw)
    μ = bank angle (roll)
    n_z = normal load factor (G-force)

Action space (Section 3.6): commanded yaw_rate, pitch_rate, accel
    Inner-loop controller: Convert (desired_pitch_rate, desired_yaw_rate)
    to required (n_z, μ) via Eq. 1 inversion.
    Clamp n_z to [NZ_MIN, NZ_MAX] = [-3, 9] (Table B1).
    Apply Eq. 1 with clamped n_z to get ACTUAL η̇, ψ̇.

Why gravity-coupled (paper Eq. 1) over direct-rate Dubins:
    1. Eq. 1 explicitly shows gravity coupling (-cos η term)
    2. n_z and μ are the actual control variables in Eq. 1
    3. Crash induction physics REQUIRES gravity coupling:
       - Hard turns deplete G-budget
       - Aircraft sinks naturally during high-G maneuvers
       - At low altitude, can't recover -> crash
    4. Table B1's G-limits (9G/-3G) are MEANINGFUL in this model
    5. Paper's described "spiral descent" requires energy loss in turns

Integration: 100 substeps × 1ms = 100ms decision step (Table B1).
"""
import numpy as np


class Aircraft:
    G = 9.81
    NZ_MAX = 9.0   # Max normal load factor (positive = pull-up)
    NZ_MIN = -3.0  # Min normal load factor (negative = push-over)
    V_MAX = 800.0
    V_MIN = 100.0
    MAX_YAW_RATE = np.radians(30)    # 30°/s commanded
    MAX_PITCH_RATE = np.radians(60)  # 60°/s commanded
    CRASH_ALT = 100.0
    CEILING_ALT = 10000.0  # Hard altitude cap (paper's operational ceiling)
    N_SUBSTEPS = 100  # 1ms substeps per 100ms decision (Table B1)

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.v = 300.0
        self.eta = 0.0
        self.psi = 0.0
        self.mu = 0.0
        self.health = 100.0
        self.crashed = False
        self.out_of_bounds = False  # True if aircraft exceeded 10km ceiling
        self.acc = 0.0
        self.yaw_rate = 0.0
        self.pitch_rate = 0.0

    def reset(self, x, y, z, v, eta, psi):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.v, self.eta, self.psi = float(v), float(eta), float(psi)
        self.mu = 0.0
        self.health = 100.0
        self.crashed = False
        self.out_of_bounds = False
        self.acc = self.yaw_rate = self.pitch_rate = 0.0

    def step(self, acc, desired_yaw_rate, desired_pitch_rate, dt=0.1):
        """
        Advance state by dt seconds using paper Eq. 1 gravity-coupled dynamics.

        Inner-loop: invert Eq. 1 to get required (n_z, μ), clamp n_z, then
        apply Eq. 1 forward with clamped n_z. Saturated G produces less rate
        than commanded, generating altitude/energy loss for unsustainable
        maneuvers (paper's crash induction physics).
        """
        desired_yaw = np.clip(desired_yaw_rate, -self.MAX_YAW_RATE, self.MAX_YAW_RATE)
        desired_pitch = np.clip(desired_pitch_rate, -self.MAX_PITCH_RATE, self.MAX_PITCH_RATE)
        self.acc = acc

        h = dt / self.N_SUBSTEPS
        actual_eta_dot = 0.0
        actual_psi_dot = 0.0
        last_mu = 0.0

        for _ in range(self.N_SUBSTEPS):
            cos_eta = np.cos(self.eta)

            # Inner-loop controller: invert Eq. 1 to find (n_z, μ) for commanded rates.
            # From Eq. 1:
            #   η̇ = (g/v)(n_z cos μ - cos η)  =>  n_z cos μ = (v η̇/g) + cos η  ≡ A
            #   ψ̇ = g n_z sin μ / (v cos η)   =>  n_z sin μ = v cos η ψ̇ / g     ≡ B
            # Therefore:
            #   n_z = sign(A) * √(A² + B²)
            #   μ = arctan2(B, |A|)
            #
            # When n_z saturates (|n_z| > 9 or n_z < -3), we keep μ as computed
            # but use clamped n_z. This means BOTH rates reduce proportionally
            # when G-budget is exhausted - consistent with coordinated maneuver
            # physics. The pilot's stick choice "mixes" pitch and turn; when
            # saturated, the mix is preserved but magnitudes reduced.
            #
            # This produces sensible behavior:
            #   - Pure pitch command: μ=0, n_z varies with command
            #   - Pure yaw command: μ depends on yaw magnitude, n_z increases
            #     to support coordinated turn (saturates if turn too aggressive)
            #   - Combined: G-budget split between pitch and turn

            A = self.v * desired_pitch / self.G + cos_eta
            B = self.v * cos_eta * desired_yaw / self.G if abs(cos_eta) > 1e-8 else 0.0

            nz_magnitude = np.sqrt(A * A + B * B)
            # Sign of n_z determined by sign of A (cos μ direction)
            nz_required = -nz_magnitude if A < 0 else nz_magnitude
            mu = np.arctan2(B, abs(A))

            # Clamp n_z to physical limits (Table B1: [-3, +9] G)
            nz = np.clip(nz_required, self.NZ_MIN, self.NZ_MAX)

            # Apply Eq. 1 forward with CLAMPED n_z (μ unchanged)
            actual_eta_dot = (self.G / self.v) * (nz * np.cos(mu) - cos_eta)
            if abs(cos_eta) > 1e-8:
                actual_psi_dot = self.G * nz * np.sin(mu) / (self.v * cos_eta)
            else:
                actual_psi_dot = 0.0

            # Integrate position
            self.x += self.v * cos_eta * np.sin(self.psi) * h
            self.y += self.v * cos_eta * np.cos(self.psi) * h
            self.z += self.v * np.sin(self.eta) * h

            # Integrate state
            self.v += acc * h
            self.eta += actual_eta_dot * h
            self.psi += actual_psi_dot * h
            last_mu = mu

            # Intra-substep crash detection (ground)
            if self.z < self.CRASH_ALT:
                self.crashed = True
                self.z = max(self.z, 0.0)
                self.pitch_rate = actual_eta_dot
                self.yaw_rate = actual_psi_dot
                self.mu = last_mu
                self.v = np.clip(self.v, self.V_MIN, self.V_MAX)
                self.eta = np.clip(self.eta, -np.pi / 2, np.pi / 2)
                self.psi = (self.psi + np.pi) % (2 * np.pi) - np.pi
                return

            # Hard altitude ceiling DISABLED — replaced with soft penalty in
            # environment._evaluate() above 8km. Hard ceiling was causing
            # 75-85% of episodes to terminate as zero-reward draws, destroying
            # learning signal for crash induction strategy.

        # Store actual rates for state observation (Table 1)
        self.pitch_rate = actual_eta_dot
        self.yaw_rate = actual_psi_dot
        self.mu = last_mu

        # State limits
        self.v = np.clip(self.v, self.V_MIN, self.V_MAX)
        self.eta = np.clip(self.eta, -np.pi / 2, np.pi / 2)
        self.psi = (self.psi + np.pi) % (2 * np.pi) - np.pi

        if self.z < self.CRASH_ALT:
            self.crashed = True
            self.z = max(self.z, 0.0)
        # Hard ceiling disabled — soft penalty in environment._evaluate handles altitude

    def pos(self):
        return np.array([self.x, self.y, self.z])

    def heading(self):
        """World direction vector w_i (Table 1)."""
        return np.array([
            np.cos(self.eta) * np.sin(self.psi),
            np.cos(self.eta) * np.cos(self.psi),
            np.sin(self.eta)
        ])
