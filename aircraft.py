"""
Aircraft dynamics -- Paper Eq. 1: 3D gravity-coupled coordinated turn model.

    x_dot = v cos eta sin psi
    y_dot = v cos eta cos psi
    z_dot = v sin eta
    eta_dot = (g/v)(n_z cos mu - cos eta)
    psi_dot = g n_z sin mu / (v cos eta)

with eta = flight path angle (pitch), psi = heading angle (yaw),
mu = bank angle (roll), n_z = normal load factor (G).

Action space (Section 3.6) is commanded (yaw_rate, pitch_rate, accel).
Inner-loop controller inverts Eq. 1 to get the (n_z, mu) that would
produce those rates, clamps n_z to Table B1's [-3, +9] G envelope, then
re-applies Eq. 1 with the clamped n_z. When n_z saturates, both rates
shrink proportionally -- this is the gravity coupling that makes high-G
turns lose energy/altitude (paper's crash-induction physics).

Integration: Table B1's 100 substeps x 1 ms = 100 ms decision step.
"""
import numpy as np


class Aircraft:
    G = 9.81
    NZ_MAX = 9.0   # Table B1: maximum overload
    NZ_MIN = -3.0  # Table B1: minimum overload
    V_MAX = 800.0
    V_MIN = 100.0
    MAX_YAW_RATE = np.radians(30)    # Table B1: max sideslip angular velocity
    MAX_PITCH_RATE = np.radians(60)  # Table B1: max pitch angular velocity
    CRASH_ALT = 100.0
    CEILING_ALT = 10000.0  # Constrained-RL: hard service ceiling (paper's max init altitude)
    N_SUBSTEPS = 100  # Table B1: 1 ms substep x 100 = 100 ms decision step

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
        self.acc = 0.0
        self.yaw_rate = 0.0
        self.pitch_rate = 0.0

    def reset(self, x, y, z, v, eta, psi):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.v, self.eta, self.psi = float(v), float(eta), float(psi)
        self.mu = 0.0
        self.health = 100.0
        self.crashed = False
        self.acc = self.yaw_rate = self.pitch_rate = 0.0

    def step(self, acc, desired_yaw_rate, desired_pitch_rate, dt=0.1):
        """Advance state by dt seconds using paper Eq. 1.

        Inner-loop controller inverts Eq. 1 to find (n_z, mu) for the
        commanded rates, clamps n_z to [-3, +9] G, and re-applies Eq. 1
        with the clamped value. Saturation produces less rate than commanded,
        leaking energy/altitude on unsustainable maneuvers.
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

            # Invert Eq. 1 for required (n_z, mu):
            #   eta_dot = (g/v)(n_z cos mu - cos eta)
            #     => n_z cos mu = (v eta_dot/g) + cos eta  =: A
            #   psi_dot = g n_z sin mu / (v cos eta)
            #     => n_z sin mu = v cos eta psi_dot / g    =: B
            #   n_z = sign(A) * sqrt(A^2 + B^2),  mu = atan2(B, |A|)
            A = self.v * desired_pitch / self.G + cos_eta
            B = self.v * cos_eta * desired_yaw / self.G if abs(cos_eta) > 1e-8 else 0.0

            nz_magnitude = np.sqrt(A * A + B * B)
            nz_required = -nz_magnitude if A < 0 else nz_magnitude
            mu = np.arctan2(B, abs(A))

            # Clamp n_z to physical envelope (Table B1)
            nz = np.clip(nz_required, self.NZ_MIN, self.NZ_MAX)

            # Re-apply Eq. 1 forward with clamped n_z
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

            # Constrained-RL: hard service ceiling.
            # When z >= 10km, clamp z and force eta <= 0 so the aircraft
            # cannot continue climbing. The pitch_rate state observation
            # still reflects the commanded rate, so the agent can see that
            # its climb command had no effect (useful learning signal).
            if self.z >= self.CEILING_ALT:
                self.z = self.CEILING_ALT
                if self.eta > 0:
                    self.eta = 0.0

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

        self.pitch_rate = actual_eta_dot
        self.yaw_rate = actual_psi_dot
        self.mu = last_mu

        self.v = np.clip(self.v, self.V_MIN, self.V_MAX)
        self.eta = np.clip(self.eta, -np.pi / 2, np.pi / 2)
        self.psi = (self.psi + np.pi) % (2 * np.pi) - np.pi

        # Constrained-RL: also enforce ceiling at end-of-step (in case the
        # final substep landed exactly at z = CEILING_ALT with eta > 0)
        if self.z >= self.CEILING_ALT:
            self.z = self.CEILING_ALT
            if self.eta > 0:
                self.eta = 0.0

        if self.z < self.CRASH_ALT:
            self.crashed = True
            self.z = max(self.z, 0.0)

    def pos(self):
        return np.array([self.x, self.y, self.z])

    def heading(self):
        """World direction vector w_i (Table 1)."""
        return np.array([
            np.cos(self.eta) * np.sin(self.psi),
            np.cos(self.eta) * np.cos(self.psi),
            np.sin(self.eta),
        ])
