"""Unit tests for spacecraft attitude controller with error state formulation"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation

from attitude_controller import (AttitudeController, ControllerGains,
                                 SpacecraftConfig)
from utils import quaternion_to_euler


class TestConfigAndSetup:
    """Tests for configuration and initialization"""

    @pytest.fixture
    def simple_spacecraft(self):
        """Returns spacecraft with unit inertia"""
        J = np.eye(3)
        return SpacecraftConfig.three_reaction_wheels(J)

    def test_state_matrices_shape(self, simple_spacecraft):
        """Test the dimensions of state space matrices"""
        controller = AttitudeController(simple_spacecraft)
        A, B = controller._build_state_matrices()

        assert A.shape == (6, 6), "A matrix should be 6x6"
        assert B.shape == (6, 3), "B matrix for 3 wheels should be 6x3"

        # Check structure of A matrix
        assert np.allclose(A[0:3, 0:3], 0), "Upper left should be zero"
        assert np.allclose(A[0:3, 3:6], -np.eye(3)), "Upper right should be I"
        assert np.allclose(A[3:6, 0:3], 0), "Lower left should be zero"

    def test_controllability(self, simple_spacecraft):
        """Test that the system is controllable"""
        controller = AttitudeController(simple_spacecraft)
        A, B = controller._build_state_matrices()
        C = np.concatenate(
            [
                B,
                A @ B,
                A @ A @ B,
                A @ A @ A @ B,
                A @ A @ A @ A @ B,
                A @ A @ A @ A @ A @ B,
            ],
            axis=1,
        )
        rank = np.linalg.matrix_rank(C)
        assert rank == 6, f"System should be controllable, rank={rank}"

    def test_cost_matrix_validation(self, simple_spacecraft):
        """Test validation of Q and R matrices"""
        controller = AttitudeController(simple_spacecraft)

        # Non-symmetric Q
        Q = np.eye(6)
        Q[0, 1] = 1.0  # Make non-symmetric
        R = np.eye(3)

        with pytest.raises(ValueError):
            controller.compute_gains(Q, R)

        # Wrong size Q
        Q = np.eye(5)
        with pytest.raises(ValueError):
            controller.compute_gains(Q, R)


class TestQuaternionError:
    """Tests for quaternion error computation"""

    @pytest.fixture
    def controller(self):
        J = np.eye(3)
        spacecraft = SpacecraftConfig.three_reaction_wheels(J)
        return AttitudeController(spacecraft)

    def test_quaternion_error_normalization(self, controller):
        """Test that quaternion error remains normalized"""
        angles = [30, 90, 180, 270]
        for angle in angles:
            r = Rotation.from_euler("x", angle, degrees=True)
            q_desired = np.roll(r.as_quat(), 1)  # Convert to scalar-first
            q_current = np.array([1.0, 0.0, 0.0, 0.0])

            q_error = controller.quaternion_error(q_current, q_desired)
            norm = np.linalg.norm(q_error)
            assert (
                np.abs(norm - 1.0) < 1e-10
            ), f"Error quaternion not normalized for {angle}° rotation"

    def test_quaternion_error_direction(self, controller):
        """Verify quaternion error produces correct rotation direction"""
        # 45° rotation about x-axis
        r = Rotation.from_euler("x", 45, degrees=True)
        q_desired = np.roll(r.as_quat(), 1)
        q_current = np.array([1.0, 0.0, 0.0, 0.0])

        q_error = controller.quaternion_error(q_current, q_desired)
        euler_error = quaternion_to_euler(q_error)

        # Error should indicate positive x rotation needed
        assert euler_error[0] > 0, "Error indicates wrong rotation direction"

    def test_quaternion_error_edge_cases(self, controller):
        """Test quaternion error computation in challenging cases"""
        q_current = np.array([1.0, 0.0, 0.0, 0.0])

        # 1. Almost 180° rotation
        r = Rotation.from_euler("x", 179.9, degrees=True)
        q_almost_180 = np.roll(r.as_quat(), 1)
        e1 = controller.quaternion_error(q_current, q_almost_180)
        assert not np.any(np.isnan(e1)), "Error shouldn't be NaN near 180°"
        assert np.abs(np.linalg.norm(e1) - 1.0) < 1e-10, "Error should be normalized"

        # 2. Exactly 180° rotation
        q_180 = np.array([0.0, 1.0, 0.0, 0.0])
        e2 = controller.quaternion_error(q_current, q_180)
        assert not np.any(np.isnan(e2)), "Error shouldn't be NaN at 180°"
        assert np.abs(np.linalg.norm(e2) - 1.0) < 1e-10, "Error should be normalized"

        # 3. Very small rotation
        r = Rotation.from_euler("x", 0.001, degrees=True)
        q_tiny = np.roll(r.as_quat(), 1)
        e3 = controller.quaternion_error(q_current, q_tiny)
        assert not np.any(np.isnan(e3)), "Error shouldn't be NaN for tiny rotation"
        assert np.abs(e3[0]) > 0.999, "Scalar part should be near 1 for small error"

    def test_error_state_mapping(self, controller):
        """Test mapping between quaternion error and error state"""
        angles = [1, 10, 45]  # degrees - focusing on small angle cases
        for angle in angles:
            r = Rotation.from_euler("x", angle, degrees=True)
            q_desired = np.roll(r.as_quat(), 1)
            q_current = np.array([1.0, 0.0, 0.0, 0.0])

            q_error = controller.quaternion_error(q_current, q_desired)
            error_vec = 2 * q_error[1:4]
            actual_rot = angle * np.array([1, 0, 0])  # degrees

            # For small angles, error_vec should approximate rotation vector
            np.testing.assert_allclose(error_vec, np.radians(actual_rot), rtol=0.1)


class TestControlComputation:
    """Tests for control law computation"""

    @pytest.fixture
    def simple_controller(self):
        J = np.eye(3)
        spacecraft = SpacecraftConfig.three_reaction_wheels(J)
        return AttitudeController(spacecraft)

    def test_compute_basic_gains(self, simple_controller):
        """Test basic LQR gain computation"""
        Q = np.eye(6)  # Equal state costs
        R = np.eye(3)  # Equal control costs
        gains = simple_controller.compute_gains(Q, R)

        assert gains.K.shape == (3, 6), "Wrong gain matrix shape"
        assert np.allclose(
            gains.K[0, 0], gains.K[1, 1]
        ), "Unequal diagonal position gains"
        assert np.allclose(gains.K[0, 3], gains.K[1, 4]), "Unequal diagonal rate gains"

    def test_zero_control_at_target(self, simple_controller):
        """Test that control is zero when at desired attitude"""
        K = np.ones((3, 6))
        gains = ControllerGains(K=K)

        q = np.array([1.0, 0.0, 0.0, 0.0])
        w = np.zeros(3)

        u = simple_controller.compute_control(q, q, w, w, gains)
        assert_array_almost_equal(u, np.zeros(3), decimal=10)

    def test_control_near_target(self, simple_controller):
        """Test controller behavior when very close to target attitude"""
        Q = np.eye(6)
        R = np.eye(3)
        gains = simple_controller.compute_gains(Q, R)

        # Very small angle error (0.1 degrees)
        r = Rotation.from_euler("x", 0.1, degrees=True)
        q_desired = np.roll(r.as_quat(), 1)
        q_current = np.array([1.0, 0.0, 0.0, 0.0])

        # Small but non-zero angular velocity
        w_current = np.array([0.01, 0.0, 0.0])
        w_desired = np.zeros(3)

        u = simple_controller.compute_control(
            q_current, q_desired, w_current, w_desired, gains
        )

        assert np.all(np.abs(u) < 1.0), "Control too large for small error"
        assert not np.allclose(u, 0), "Control should not be zero for non-zero error"


class TestDynamicsAndPhysics:
    """Tests for physical behavior"""

    def test_dynamics_matrix(self):
        """Test dynamics with non-diagonal inertia"""
        J = np.array([[1, 0.1, 0], [0.1, 2, -0.1], [0, -0.1, 3]])
        spacecraft = SpacecraftConfig.three_reaction_wheels(J)
        controller = AttitudeController(spacecraft)

        A, B = controller._build_state_matrices()
        J_inv = np.linalg.inv(J)

        assert_array_almost_equal(A[3:6, 3:6], 0)  # No velocity feedback
        assert_array_almost_equal(B[3:6, :], J_inv @ spacecraft.actuator_matrix)

    def test_energy_analysis(self):
        """Test that control reduces system energy"""
        J = np.eye(3)
        spacecraft = SpacecraftConfig.three_reaction_wheels(J)
        controller = AttitudeController(spacecraft)

        Q = np.diag([10, 10, 10, 1, 1, 1])
        R = np.eye(3)
        gains = controller.compute_gains(Q, R)

        test_cases = [
            (np.radians(5), 0.1),  # Small angle and velocity
            (np.radians(10), 0.2),  # Larger angle and velocity
        ]

        for angle, vel in test_cases:
            q_current = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
            q_desired = np.array([1.0, 0.0, 0.0, 0.0])
            w_current = np.array([vel, 0.0, 0.0])
            w_desired = np.zeros(3)

            u = controller.compute_control(
                q_current, q_desired, w_current, w_desired, gains
            )
            acc = np.linalg.inv(J) @ (spacecraft.actuator_matrix @ u)

            # For non-zero velocity, kinetic energy should decrease
            if abs(vel) > 0:
                dKE_dt = w_current.T @ J @ acc
                assert dKE_dt < 0, f"Kinetic energy should decrease when velocity={vel}"

            # For non-zero angle, torque should oppose position error
            if abs(angle) > 0:
                assert acc[0] * angle < 0, f"Torque should oppose position error"


class TestMultiAxisControl:
    """Tests for multi-axis behavior"""

    def test_multi_axis_coupling(self):
        """Test controller behavior with simultaneous multi-axis rotation"""
        J = np.diag([1.0, 2.0, 3.0])
        spacecraft = SpacecraftConfig.three_reaction_wheels(J)
        controller = AttitudeController(spacecraft)

        Q = np.eye(6)
        R = np.eye(3)
        gains = controller.compute_gains(Q, R)

        r = Rotation.from_euler("xy", [45, 45], degrees=True)
        q_desired = np.roll(r.as_quat(), 1)
        q_current = np.array([1.0, 0.0, 0.0, 0.0])

        w_current = np.zeros(3)
        w_desired = np.zeros(3)

        u = controller.compute_control(
            q_current, q_desired, w_current, w_desired, gains
        )

        assert u[0] != 0 and u[1] != 0, "Should command both x and y control"
        assert abs(u[2]) < abs(u[0]) + abs(u[1]), "Z-axis control should be smaller"

    def test_four_wheel_control_distribution(self):
        """Test control distribution with redundant actuators"""
        J = np.eye(3)
        spacecraft = SpacecraftConfig.four_reaction_wheels(J)
        controller = AttitudeController(spacecraft)

        K = np.zeros((4, 6))
        K[0:4, 0:4] = -np.eye(4)
        gains = ControllerGains(K=K)

        q_current = np.array([1.0, 0.0, 0.0, 0.0])
        q_desired = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])
        w = np.zeros(3)

        u = controller.compute_control(q_current, q_desired, w, w, gains)
        tau = spacecraft.actuator_matrix @ u

        torque_magnitude = np.linalg.norm(tau)
        assert (
            torque_magnitude < 0.5
        ), f"Total torque magnitude {torque_magnitude} too large"
        assert not np.any(np.abs(u) > 2 * np.abs(tau[0])), "Poor control distribution"


class TestStabilityAndConvergence:
    """Tests for system stability and convergence properties"""

    @pytest.fixture
    def simple_system(self):
        J = np.eye(3)
        spacecraft = SpacecraftConfig.three_reaction_wheels(J)
        controller = AttitudeController(spacecraft)
        Q = np.diag([10, 10, 10, 1, 1, 1])
        R = np.eye(3)
        gains = controller.compute_gains(Q, R)
        return spacecraft, controller, gains

    def test_momentum_conservation(self, simple_system):
        """Test that total angular momentum is conserved"""
        spacecraft, controller, gains = simple_system

        # Initial conditions
        q_current = np.array([1.0, 0.0, 0.0, 0.0])
        w_current = np.array([0.1, 0.2, -0.3])  # Initial angular velocity
        q_desired = np.array([1.0, 0.0, 0.0, 0.0])
        w_desired = np.zeros(3)

        # Compute control and resulting acceleration
        u = controller.compute_control(
            q_current, q_desired, w_current, w_desired, gains
        )
        tau = spacecraft.actuator_matrix @ u

        # Check that internal torques sum to zero
        assert_array_almost_equal(
            np.sum(tau), 0, decimal=10, err_msg="Internal torques must sum to zero"
        )

    def test_lyapunov_stability(self, simple_system):
        """Test Lyapunov stability condition"""
        spacecraft, controller, gains = simple_system

        test_cases = [
            (np.radians(5), 0.1),  # Small angle, small velocity
            (np.radians(10), 0.0),  # Larger angle, zero velocity
            (0.0, 0.2),  # Zero angle, nonzero velocity
        ]

        for angle, vel in test_cases:
            q_current = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
            w_current = np.array([vel, 0.0, 0.0])
            q_desired = np.array([1.0, 0.0, 0.0, 0.0])
            w_desired = np.zeros(3)

            u = controller.compute_control(
                q_current, q_desired, w_current, w_desired, gains
            )
            tau = spacecraft.actuator_matrix @ u
            acc = np.linalg.inv(spacecraft.J) @ tau

            # Get quaternion error vector part
            eq = 2 * controller.quaternion_error(q_current, q_desired)[1:4]

            if np.linalg.norm(w_current) > 1e-10:
                # If there's velocity, check V_dot
                V_dot = -2 * np.dot(eq, w_current) + np.dot(w_current, tau)
                assert (
                    V_dot < 0
                ), f"Lyapunov derivative not negative for angle={angle}, vel={vel}"
            else:
                # If zero velocity, check that acceleration is in correct direction
                # Check only non-zero components
                non_zero = np.abs(eq) > 1e-10
                if np.any(non_zero):
                    # They should be equal signs for stability (positive error -> positive acc)
                    assert np.all(
                        np.sign(acc[non_zero]) == np.sign(eq[non_zero])
                    ), f"Acceleration not correcting error for angle={angle}"

        def test_closed_loop_convergence(self, simple_system):
            """Test convergence over multiple timesteps"""
            spacecraft, controller, gains = simple_system

            # Initial conditions
            angle = np.radians(10)
            q_current = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
            w_current = np.zeros(3)
            q_desired = np.array([1.0, 0.0, 0.0, 0.0])
            w_desired = np.zeros(3)

            dt = 0.01
            num_steps = 100

            # Track error evolution
            errors = []
            velocities = []

            for _ in range(num_steps):
                # Store current state
                eq = 2 * controller.quaternion_error(q_current, q_desired)[1:4]
                errors.append(np.linalg.norm(eq))
                velocities.append(np.linalg.norm(w_current))

                # Compute control
                u = controller.compute_control(
                    q_current, q_desired, w_current, w_desired, gains
                )
                tau = spacecraft.actuator_matrix @ u
                acc = np.linalg.inv(spacecraft.J) @ tau

                # Simple euler integration (just for testing)
                w_next = w_current + acc * dt

                # Update quaternion using quaternion kinematics
                w_norm = np.linalg.norm(w_next)
                if w_norm > 0:
                    axis = w_next / w_norm
                    angle = w_norm * dt
                    dq = np.array(
                        [
                            np.cos(angle / 2),
                            axis[0] * np.sin(angle / 2),
                            axis[1] * np.sin(angle / 2),
                            axis[2] * np.sin(angle / 2),
                        ]
                    )

                    # Quaternion multiplication
                    q_next = np.array(
                        [
                            q_current[0] * dq[0] - np.dot(q_current[1:4], dq[1:4]),
                            q_current[0] * dq[1]
                            + dq[0] * q_current[1]
                            + q_current[2] * dq[3]
                            - q_current[3] * dq[2],
                            q_current[0] * dq[2]
                            + dq[0] * q_current[2]
                            + q_current[3] * dq[1]
                            - q_current[1] * dq[3],
                            q_current[0] * dq[3]
                            + dq[0] * q_current[3]
                            + q_current[1] * dq[2]
                            - q_current[2] * dq[1],
                        ]
                    )
                else:
                    q_next = q_current

                # Normalize quaternion
                q_next = q_next / np.linalg.norm(q_next)

                # Update state
                q_current = q_next
                w_current = w_next

            # Check convergence properties
            assert errors[-1] < errors[0], "Error should decrease"
            assert velocities[-1] < 1.0, "Velocity should remain bounded"
            assert np.all(
                np.diff(errors[:10]) < 0
            ), "Error should decrease monotonically initially"

        def test_gain_scaling_stability(self, simple_system):
            """Test stability with different gain scales"""
            spacecraft, controller, _ = simple_system

            scales = [1.0, 2.0, 4.0]
            for scale in scales:
                Q = scale * np.diag([10, 10, 10, 1, 1, 1])
                R = np.eye(3)
                gains = controller.compute_gains(Q, R)

                angle = np.radians(5)
                q_current = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
                w_current = np.array([0.1, 0.0, 0.0])
                q_desired = np.array([1.0, 0.0, 0.0, 0.0])
                w_desired = np.zeros(3)

                u = controller.compute_control(
                    q_current, q_desired, w_current, w_desired, gains
                )
                tau = spacecraft.actuator_matrix @ u
                acc = np.linalg.inv(spacecraft.J) @ tau

                # Check that acceleration scales roughly with sqrt(scale)
                assert np.all(
                    np.abs(acc) < 2 * np.sqrt(scale)
                ), f"Acceleration too large for scale {scale}"

                # Check acceleration direction
                eq = 2 * controller.quaternion_error(q_current, q_desired)[1:4]
                non_zero = np.abs(eq) > 1e-10
                if np.any(non_zero):
                    # They should be equal signs for stability
                    assert np.all(
                        np.sign(acc[non_zero]) == np.sign(eq[non_zero])
                    ), f"Acceleration not correcting error for scale {scale}"


class TestRobustness:
    """Tests for controller robustness"""

    def test_numerical_conditioning(self):
        """Test behavior with poorly conditioned inertia matrix"""
        # Create poorly conditioned inertia matrix
        J = np.diag([1.0, 1000.0, 0.001])
        spacecraft = SpacecraftConfig.three_reaction_wheels(J)
        controller = AttitudeController(spacecraft)

        Q = np.eye(6)
        R = np.eye(3)

        # Should still be able to compute gains
        gains = controller.compute_gains(Q, R)
        assert not np.any(np.isnan(gains.K)), "Gains contain NaN values"
        assert not np.any(np.isinf(gains.K)), "Gains contain infinite values"

    def test_disturbance_rejection(self):
        """Test response to external disturbances"""
        J = np.eye(3)
        spacecraft = SpacecraftConfig.three_reaction_wheels(J)
        controller = AttitudeController(spacecraft)

        Q = np.diag([10, 10, 10, 1, 1, 1])
        R = np.eye(3)
        gains = controller.compute_gains(Q, R)

        # Add disturbance velocity
        q_current = np.array([1.0, 0.0, 0.0, 0.0])
        w_current = np.array([0.1, 0.0, 0.0])  # Disturbance velocity
        q_desired = np.array([1.0, 0.0, 0.0, 0.0])
        w_desired = np.zeros(3)

        u = controller.compute_control(
            q_current, q_desired, w_current, w_desired, gains
        )
        tau = spacecraft.actuator_matrix @ u

        # Control should oppose disturbance
        assert np.dot(tau, w_current) < 0, "Control should oppose disturbance velocity"

    def test_multiple_quaternion_representations(self):
        """Test that controller handles equivalent quaternion representations"""
        J = np.eye(3)
        spacecraft = SpacecraftConfig.three_reaction_wheels(J)
        controller = AttitudeController(spacecraft)

        Q = np.eye(6)
        R = np.eye(3)
        gains = controller.compute_gains(Q, R)

        # Test equivalent quaternions (q and -q represent same rotation)
        q1 = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0])
        q2 = -q1
        w = np.zeros(3)
        q_desired = np.array([1.0, 0.0, 0.0, 0.0])

        u1 = controller.compute_control(q1, q_desired, w, w, gains)
        u2 = controller.compute_control(q2, q_desired, w, w, gains)

        # Should get same control for equivalent representations
        assert_array_almost_equal(u1, u2, decimal=10)
