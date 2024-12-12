"""Spacecraft attitude control simulation and analysis framework."""

import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from attitude_controller import AttitudeController, SpacecraftConfig
from utils import quaternion_multiply, quaternion_to_euler


@dataclass
class SimulationConfig:
    """Configuration parameters for simulation."""

    t_start: float = 0.0
    t_end: float = 40.0
    dt: float = 0.05
    rtol: float = 1e-8
    atol: float = 1e-11
    max_step: float = 0.1


@dataclass
class ControllerConfig:
    """LQR controller configuration."""

    Q: np.ndarray
    R: np.ndarray


@dataclass
class TestCase:
    """Complete test scenario definition."""

    name: str
    spacecraft_config: SpacecraftConfig
    controller_config: ControllerConfig
    q0: np.ndarray
    w0: np.ndarray
    euler_desired: np.ndarray
    disturbance: Optional[Callable[[float], np.ndarray]] = None
    requirements: dict = None


@dataclass
class SimulationResults:
    """Container for simulation results and metrics."""

    t: np.ndarray
    q_history: np.ndarray
    w_history: np.ndarray
    u_history: np.ndarray
    metrics: dict


def plot_simulation_results(results: SimulationResults, case: TestCase) -> None:
    """Create interactive plots of simulation results with detailed analysis.

    Args:
        results: Simulation results containing time history and metrics
        case: Test case configuration
    """
    # Convert quaternion history to euler angles
    euler_history = np.array([quaternion_to_euler(q) for q in results.q_history])

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Euler Angles",
            "Quaternions",
            "Angular Velocity",
            "Control Torques",
            "Pointing Error",
            "Phase Portrait",
            "Control Effort",
            "Energy",
        ),
        vertical_spacing=0.1,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
    )

    # 1. Euler Angles
    axes = ["Roll", "Pitch", "Yaw"]
    for i in range(3):
        # Actual angles
        fig.add_trace(
            go.Scatter(
                x=results.t,
                y=euler_history[:, i],
                name=f"Current {axes[i]}",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )
        # Target angles
        fig.add_trace(
            go.Scatter(
                x=[results.t[0], results.t[-1]],
                y=[case.euler_desired[i], case.euler_desired[i]],
                name=f"Target {axes[i]}",
                line=dict(dash="dash"),
            ),
            row=1,
            col=1,
        )

    # 2. Quaternions
    for i in range(4):
        fig.add_trace(
            go.Scatter(x=results.t, y=results.q_history[:, i], name=f"q{i}"),
            row=1,
            col=2,
        )

    # 3. Angular Velocity
    for i in range(3):
        fig.add_trace(
            go.Scatter(x=results.t, y=results.w_history[:, i], name=f"ω{axes[i]}"),
            row=2,
            col=1,
        )

    # 4. Control Torques
    for i in range(3):
        fig.add_trace(
            go.Scatter(x=results.t, y=results.u_history[:, i], name=f"τ{i+1}"),
            row=2,
            col=2,
        )

    # 5. Pointing Error
    error_history = euler_history - case.euler_desired
    for i in range(3):
        fig.add_trace(
            go.Scatter(x=results.t, y=error_history[:, i], name=f"{axes[i]} Error"),
            row=3,
            col=1,
        )

    # 6. Phase Portrait (Angular Velocity vs Error)
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=error_history[:, i],
                y=results.w_history[:, i],
                name=f"{axes[i]} Phase",
                mode="lines",
            ),
            row=3,
            col=2,
        )

    # 7. Control Effort
    cumulative_effort = np.cumsum(np.linalg.norm(results.u_history, axis=1)) * (
        results.t[1] - results.t[0]
    )
    fig.add_trace(
        go.Scatter(x=results.t, y=cumulative_effort, name="Cumulative Control Effort"),
        row=4,
        col=1,
    )

    # 8. System Energy
    # Kinetic energy
    J = case.spacecraft_config.J
    kinetic_energy = [0.5 * w.T @ J @ w for w in results.w_history]
    fig.add_trace(
        go.Scatter(x=results.t, y=kinetic_energy, name="Kinetic Energy"), row=4, col=2
    )

    # Update layout
    fig.update_layout(
        height=1200,
        width=1200,
        title=f"Simulation Results: {case.name}",
        showlegend=True,
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_xaxes(title_text="Time (s)", row=4, col=2)

    fig.update_yaxes(title_text="Angle (deg)", row=1, col=1)
    fig.update_yaxes(title_text="Quaternion Value", row=1, col=2)
    fig.update_yaxes(title_text="Angular Velocity (rad/s)", row=2, col=1)
    fig.update_yaxes(title_text="Torque (N⋅m)", row=2, col=2)
    fig.update_yaxes(title_text="Error (deg)", row=3, col=1)
    fig.update_yaxes(title_text="Angular Velocity (rad/s)", row=3, col=2)
    fig.update_yaxes(title_text="Cumulative Effort", row=4, col=1)
    fig.update_yaxes(title_text="Energy (J)", row=4, col=2)

    fig.show()


class SpacecraftSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def simulate(self, case: TestCase) -> SimulationResults:
        """Run simulation for given test case."""
        # Initialize controller with spacecraft config
        controller = AttitudeController(case.spacecraft_config)

        # Compute gains using specified Q and R matrices
        gains = controller.compute_gains(
            case.controller_config.Q, case.controller_config.R
        )

        # Convert euler angles to quaternion
        r = Rotation.from_euler("xyz", case.euler_desired, degrees=True)
        q_desired = np.roll(r.as_quat(), 1)

        # Initialize last logging time
        last_log_time = -float("inf")
        log_interval = 1.0  # Log every 1 second, adjust as needed

        def log_state(t: float, q: np.ndarray, w: np.ndarray, u: np.ndarray) -> None:
            nonlocal last_log_time
            if t - last_log_time >= log_interval:
                # Convert quaternion to euler angles for readable output
                euler = Rotation.from_quat(np.roll(q, -1)).as_euler("xyz", degrees=True)
                print(f"\nTime: {t:.2f} s")
                print(f"Attitude (euler deg): {euler}")
                print(f"Angular velocity (rad/s): {w}")
                print(f"Control input: {u}")
                last_log_time = t

        def dynamics(t: float, state: np.ndarray) -> np.ndarray:
            q = state[0:4]
            w = state[4:7]

            # Compute control
            u = controller.compute_control(q, q_desired, w, np.zeros(3), gains)
            dynamics.u_history.append(u)

            # Add disturbance if specified
            disturbance = case.disturbance(t) if case.disturbance else np.zeros(3)

            # Quaternion kinematics
            q_dot = 0.5 * quaternion_multiply(q, np.concatenate(([0], w)))

            # Angular dynamics
            J = controller.spacecraft.J
            w_cross = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            w_dot = np.linalg.inv(J) @ (
                controller.spacecraft.actuator_matrix @ u
                + disturbance
                - w_cross @ J @ w
            )

            log_state(t, q, w, u)
            return np.concatenate([q_dot, w_dot])

        dynamics.u_history = []

        # Simulate
        t_eval = np.arange(self.config.t_start, self.config.t_end, self.config.dt)
        sol = solve_ivp(
            dynamics,
            (self.config.t_start, self.config.t_end),
            np.concatenate([case.q0, case.w0]),
            t_eval=t_eval,
            method="RK45",
            rtol=self.config.rtol,
            atol=self.config.atol,
            max_step=self.config.max_step,
        )

        results = SimulationResults(
            t=sol.t,
            q_history=sol.y[0:4, :].T,
            w_history=sol.y[4:7, :].T,
            u_history=np.array(dynamics.u_history),
            metrics=self._analyze_performance(
                sol.t, sol.y[0:4, :].T, sol.y[4:7, :].T, q_desired, case.euler_desired
            ),
        )

        return results

    def _analyze_performance(self, t, q_history, w_history, q_desired, euler_desired):
        """Enhanced performance analysis with comprehensive metrics."""
        metrics = {}

        euler_history = np.array([quaternion_to_euler(q) for q in q_history])
        error_history = euler_history - euler_desired

        # Per-axis metrics
        for i, axis in enumerate(["roll", "pitch", "yaw"]):
            metrics[f"{axis}_max_error"] = np.max(np.abs(error_history[:, i]))
            metrics[f"{axis}_rms_error"] = np.sqrt(np.mean(error_history[:, i] ** 2))

        # Total error metrics
        metrics["total_max_error"] = np.max(np.sqrt(np.sum(error_history**2, axis=1)))
        metrics["total_rms_error"] = np.sqrt(np.mean(np.sum(error_history**2, axis=1)))

        # Energy metrics
        metrics["total_energy"] = np.sum(np.abs(w_history)) * (t[1] - t[0])

        # Stability metrics
        metrics["final_rate_stability"] = np.all(np.abs(w_history[-10:]) < 0.01)
        metrics["final_pointing_stability"] = np.all(np.abs(error_history[-10:]) < 0.5)

        # Settling time (when total error stays below 2% of maximum error)
        threshold = 0.02 * metrics["total_max_error"]
        total_error = np.sqrt(np.sum(error_history**2, axis=1))
        settled_indices = np.where(total_error < threshold)[0]
        metrics["settling_time"] = (
            t[settled_indices[0]] if len(settled_indices) > 0 else np.inf
        )

        return metrics


class TestSuite:
    """Test suite for spacecraft controller verification."""

    @staticmethod
    def get_all_test_cases() -> dict[str, TestCase]:
        """Return all test cases in a dictionary for easy access."""

        # Common spacecraft configurations
        J_symmetric = 10.0 * np.eye(3)
        J_asymmetric = np.array(
            [[100.0, -5.0, 2.0], [-5.0, 85.0, -3.0], [2.0, -3.0, 75.0]]
        )

        # Common controller configurations
        nominal_control = ControllerConfig(
            Q=np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0]), R=0.1 * np.eye(3)
        )

        aggressive_control = ControllerConfig(
            Q=np.diag([1000.0, 1000.0, 1000.0, 10.0, 10.0, 10.0]), R=0.01 * np.eye(4)
        )

        return {
            "small_slew": TestCase(
                name="Small Angle Slew - Symmetric Spacecraft",
                spacecraft_config=SpacecraftConfig.three_reaction_wheels(J_symmetric),
                controller_config=nominal_control,
                q0=np.array([1.0, 0.0, 0.0, 0.0]),
                w0=np.zeros(3),
                euler_desired=np.array([15.0, 5.0, 20.0]),
                requirements={"total_max_error": 30.0, "settling_time": 10.0},
            ),
            "large_slew_asymmetric": TestCase(
                name="Large Angle Slew - Asymmetric Spacecraft",
                spacecraft_config=SpacecraftConfig.four_reaction_wheels(J_asymmetric),
                controller_config=aggressive_control,
                q0=np.array([1.0, 0.0, 0.0, 0.0]),
                w0=np.zeros(3),
                # -90 degrees is gimbal lock
                euler_desired=np.array([170.0, -89.0, 45.0]),
                requirements={"total_max_error": 5.0, "settling_time": 30.0},
            ),
            "disturbance_rejection": TestCase(
                name="Disturbance Rejection - Robust Control",
                spacecraft_config=SpacecraftConfig.three_reaction_wheels(J_asymmetric),
                controller_config=ControllerConfig(
                    Q=np.diag([500.0, 500.0, 500.0, 50.0, 50.0, 50.0]),
                    R=0.05 * np.eye(3),
                ),
                q0=np.array([1.0, 0.0, 0.0, 0.0]),
                w0=np.array([0.1, -0.05, 0.02]),
                euler_desired=np.array([0.0, 0.0, 0.0]),
                disturbance=lambda t: 0.1
                * np.sin(2 * np.pi * 0.1 * t)
                * np.array([1.0, 0.5, -0.3]),
                requirements={"total_max_error": 3.0, "settling_time": 20.0},
            ),
        }


# For pytest-style automated testing
def test_controller_verification():
    """Run all test cases in automated mode."""
    config = SimulationConfig()
    simulator = SpacecraftSimulator(config)

    for name, case in TestSuite.get_all_test_cases().items():
        results = simulator.simulate(case)

        print(f"\nTesting {name}:")
        print(f"Spacecraft inertia matrix:\n{case.spacecraft_config.J}")
        print(f"Controller Q matrix:\n{case.controller_config.Q}")
        print(f"Controller R matrix:\n{case.controller_config.R}")

        for metric, requirement in case.requirements.items():
            assert (
                results.metrics[metric] < requirement
            ), f"Failed requirement {metric} in test case {name}"


def run_test_case(simulator, case):
    """Run a single test case and display results."""
    print(f"\nRunning test case: {case.name}")
    print(case)

    results = simulator.simulate(case)

    # Print metrics
    print("\nPerformance Metrics:")
    for metric, value in results.metrics.items():
        print(f"{metric}: {value}")

    # Show plots
    plot_simulation_results(results, case)


def print_test_cases(test_cases):
    """Print available test cases with details."""
    print("\nAvailable test cases:")
    for i, (name, case) in enumerate(test_cases.items()):
        print(f"\n{i+1}. {name}")
        print(f"   Description: {case.name}")
        print(
            f"   Spacecraft: {'Symmetric' if np.allclose(case.spacecraft_config.J, case.spacecraft_config.J.T) else 'Asymmetric'}"
        )
        print(
            f"   Control: {'Aggressive' if np.trace(case.controller_config.Q) > 1000 else 'Nominal'}"
        )


def run_interactive_simulation(test_number=None):
    """
    Interactive simulation runner with case selection and plotting.

    Args:
        test_number (int, optional): Specific test number to run. If provided, runs only that test.
    """
    config = SimulationConfig()
    simulator = SpacecraftSimulator(config)
    test_cases = TestSuite.get_all_test_cases()

    if test_number is not None:
        idx = int(test_number) - 1
        if 0 <= idx < len(test_cases):
            case = list(test_cases.values())[idx]
            run_test_case(simulator, case)
            return
        else:
            print(
                f"Error: Test number {test_number} is out of range. Available tests: 1-{len(test_cases)}"
            )
            sys.exit(1)

    # Interactive mode
    while True:
        try:
            print_test_cases(test_cases)
            selection = input("\nSelect test case number (or 'q' to quit): ")
            if selection.lower() == "q":
                break

            idx = int(selection) - 1
            if 0 <= idx < len(test_cases):
                case = list(test_cases.values())[idx]
                run_test_case(simulator, case)

                if input("\nRun another case? (y/n): ").lower() != "y":
                    break
            else:
                print("Invalid selection. Please try again.")
        except ValueError as e:
            print(f"Invalid input. {e}")


if __name__ == "__main__":
    # Check if test number was provided as command line argument
    if len(sys.argv) > 1:
        test_num = int(sys.argv[1])
        run_interactive_simulation(test_num)
    else:
        run_interactive_simulation()
