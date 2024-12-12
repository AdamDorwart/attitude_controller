"""Spacecraft Attitude Controller

This module implements a quaternion-based attitude controller using Linear Quadratic
Regulator (LQR) control. The implementation uses a reduced-state formulation that
converts quaternion error to an Euler angle representation.

Mathematical Foundation:
---------------------
1. State Representation:
   The controller uses a reduced 6-state vector: x = [attitude_error (3); ω_error (3)]
   where attitude_error is derived from quaternion error through:
   a) Small angle approximation: 2*q_vec_error ≈ [ϕ, θ, ψ]
   OR
   b) Full conversion:
      ϕ = atan2(2(q₀q₁ + q₂q₃), 1 - 2(q₁² + q₂²))
      θ = asin(2(q₀q₂ - q₃q₁))
      ψ = atan2(2(q₀q₃ + q₁q₂), 1 - 2(q₂² + q₃²))

2. Linear System:
   The linearized dynamics around equilibrium [q = (1,0,0,0), ω = 0]:
   ẋ = Ax + Bu
   where:
   A = [0₃ₓ₃    -I₃ₓ₃]  # Euler angle error kinematics
       [0₃ₓ₃     0₃ₓ₃]
   B = [  0₃ₓₙ  ]
       [J⁻¹B_act]

3. LQR Control:
   The optimal control law minimizes: J = ∫(x^T Q x + u^T R u)dt
   resulting in u = -Kx where K is the optimal gain matrix

Key Properties:
-------------
1. Performance:
   - Optimal for small errors near the linearization point
   - Simple 2*q_vec_error approximation typically provides faster convergence
     than full Euler conversion due to better matching with LQR assumptions

2. Limitations:
   - Subject to gimbal lock (inherent to Euler angle representation)
   - Performance may degrade with large attitude errors
   - Assumes small angular velocities for linearization

Conventions:
-----------
- Quaternions: [scalar; vector] format
- Angular velocities: expressed in body frame
- Error quaternion: q_error = q_current^(-1) ⊗ q_desired

"""

from dataclasses import dataclass

import control
import numpy as np
import numpy.typing as npt


@dataclass
class SpacecraftConfig:
    """Physical configuration of the spacecraft

    This class defines the physical properties and actuator configuration
    of the spacecraft needed for control.

    Attributes:
        J: 3x3 inertia matrix [kg*m^2]
        actuator_matrix: Mapping from actuator commands to body torques
            Shape is (3, n_actuators)
    """

    J: npt.NDArray[np.float64]  # 3x3 inertia matrix
    actuator_matrix: npt.NDArray[np.float64]  # 3xN actuator influence matrix

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.J.shape != (3, 3):
            raise ValueError(f"Inertia matrix must be 3x3, got {self.J.shape}")
        if not np.allclose(self.J, self.J.T):
            raise ValueError("Inertia matrix must be symmetric")
        if self.actuator_matrix.shape[0] != 3:
            raise ValueError(
                f"Actuator matrix must have 3 rows, got {self.actuator_matrix.shape}"
            )

    @classmethod
    def three_reaction_wheels(cls, J: npt.NDArray[np.float64]) -> "SpacecraftConfig":
        """Creates config for 3 orthogonal reaction wheels

        Args:
            J: 3x3 inertia matrix

        Returns:
            SpacecraftConfig with 3 reaction wheels aligned with body axes
        """
        actuator_matrix = np.eye(3)  # Identity mapping for orthogonal wheels
        return cls(J=J, actuator_matrix=actuator_matrix)

    @classmethod
    def four_reaction_wheels(cls, J: npt.NDArray[np.float64]) -> "SpacecraftConfig":
        """Creates config for 4 reaction wheels in tetrahedral configuration

        Args:
            J: 3x3 inertia matrix

        Returns:
            SpacecraftConfig with 4 reaction wheels in tetrahedral pattern
        """
        # Normalized tetrahedral configuration
        actuator_matrix = (
            1
            / np.sqrt(3)
            * np.array(
                [
                    [1, -1, 0, 0],  # x-axis torques
                    [0, 0, 1, -1],  # y-axis torques
                    [-1, -1, 1, 1],  # z-axis torques
                ]
            )
        )
        return cls(J=J, actuator_matrix=actuator_matrix)


@dataclass
class ControllerGains:
    """LQR Controller gains"""

    K: npt.NDArray[np.float64]


class AttitudeController:
    """LQR-based spacecraft attitude controller

    This controller:
    1. Uses quaternion error and angular velocity error as states
    2. Computes optimal LQR gains for the linearized system
    3. Generates actuator commands to track desired attitude

    The controller assumes all states are measured (no estimation).
    """

    def __init__(
        self,
        spacecraft: SpacecraftConfig,
    ):
        """Initialize controller

        Args:
            spacecraft: Physical spacecraft configuration
            conventions: Attitude conventions to use (optional)
        """
        self.spacecraft = spacecraft

    def compute_gains(
        self, Q: npt.NDArray[np.float64], R: npt.NDArray[np.float64]
    ) -> ControllerGains:
        """Compute optimal LQR gains for the system

        Args:
            Q: State cost matrix (7x7) for [quat_error (4); ang_vel_error (3)]
            R: Control cost matrix (NxN) where N is number of actuators

        Returns:
            ControllerGains object with optimal feedback gains
        """
        # Build state space matrices
        A, B = self._build_state_matrices()

        # Verify system properties
        if not np.allclose(Q, Q.T) or not np.allclose(R, R.T):
            raise ValueError("Q and R matrices must be symmetric")

        # Check controllability
        C = control.ctrb(A, B)
        rank = np.linalg.matrix_rank(C)
        if rank < A.shape[0]:
            raise ValueError(f"System not controllable! Rank {rank} < {A.shape[0]}")

        # Solve LQR problem
        K, _, _ = control.lqr(A, B, Q, R)
        print(f"Computed LQR gains: {K}")
        return ControllerGains(K=K)

    def _build_state_matrices(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Construct linearized state space matrices

        The system uses a reduced state vector where quaternion error is approximated
        as Euler angles through: 2*q_vec_error ≈ [ϕ, θ, ψ]

        State vector: x = [2*q_vec_error (3); angular_velocity_error (3)]

        The linearized system matrices represent:
        1. Kinematics: d/dt(2*q_vec_error) = -ω_error
        2. Dynamics: ω̇_error = J⁻¹τ

        Returns:
            A: System matrix (6x6)
            B: Input matrix (6xN) where N is number of actuators
        """
        n_states = 6
        n_inputs = self.spacecraft.actuator_matrix.shape[1]

        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_inputs))

        # Kinematics linearization: d/dt(2*q_vec_error) = -w_error
        A[0:3, 3:6] = -np.eye(3)

        # Dynamics
        J_inv = np.linalg.inv(self.spacecraft.J)
        B[3:6, :] = J_inv @ self.spacecraft.actuator_matrix

        return A, B

    def quaternion_error(
        self, q_current: np.ndarray, q_desired: np.ndarray
    ) -> np.ndarray:
        """Compute quaternion error in a way consistent with the linearization"""
        q_cur_inv = np.array(
            [q_current[0], -q_current[1], -q_current[2], -q_current[3]]
        )

        # Note the order change here
        q_error = np.array(
            [
                q_cur_inv[0] * q_desired[0]
                - q_cur_inv[1] * q_desired[1]
                - q_cur_inv[2] * q_desired[2]
                - q_cur_inv[3] * q_desired[3],
                q_cur_inv[0] * q_desired[1]
                + q_cur_inv[1] * q_desired[0]
                + q_cur_inv[2] * q_desired[3]
                - q_cur_inv[3] * q_desired[2],
                q_cur_inv[0] * q_desired[2]
                - q_cur_inv[1] * q_desired[3]
                + q_cur_inv[2] * q_desired[0]
                + q_cur_inv[3] * q_desired[1],
                q_cur_inv[0] * q_desired[3]
                + q_cur_inv[1] * q_desired[2]
                - q_cur_inv[2] * q_desired[1]
                + q_cur_inv[3] * q_desired[0],
            ]
        )

        # Ensure we take the shortest path
        if q_error[0] < 0:
            q_error = -q_error

        return q_error

    def compute_control(
        self,
        q_current: npt.NDArray[np.float64],
        q_desired: npt.NDArray[np.float64],
        w_current: npt.NDArray[np.float64],
        w_desired: npt.NDArray[np.float64],
        gains: ControllerGains,
    ) -> npt.NDArray[np.float64]:
        """Compute control inputs for actuators using reduced error state LQR

        Uses the approximation 2*q_vec_error ≈ [ϕ, θ, ψ] to represent attitude error
        in a form suitable for linear control. This approximation is most accurate
        near the linearization point (small errors).

        Args:
            q_current: Current quaternion [scalar; vector]
            q_desired: Desired quaternion [scalar; vector]
            w_current: Current angular velocity [rad/s]
            w_desired: Desired angular velocity [rad/s]
            gains: Controller gains to use

        Returns:
            u: Control inputs for actuators
        """
        # Compute errors
        q_error = self.quaternion_error(q_current, q_desired)
        w_error = w_current - w_desired

        # small error approximation for quaternion -> euler angles
        q_vec_error = 2 * q_error[1:4]

        # Form error state
        x = np.concatenate([q_vec_error, w_error])

        # Compute control
        u = -gains.K @ x

        return u
