# Spacecraft Attitude Control System

A reference implementation of a quaternion-based spacecraft attitude controller using Linear Quadratic Regulator (LQR) control theory. The controller is designed for precision pointing using reaction wheels, with support for both 3-wheel orthogonal and 4-wheel tetrahedral configurations. Additional accuator configurations are possible with minimal modification (ex: RCS thrusters).

## Features

- Optimal LQR control using reduced-state quaternion error formulation
- Configurable for different spacecraft inertia properties and actuator layouts  
- Rigorous stability analysis and performance metrics
- Interactive simulation with detailed visualization of system response
- Comprehensive test suite validating performance across various scenarios

## Usage
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Run the interactive simulation:
```bash
python simulation.py
```
To run a specific test scenario directly:
```bash
python simulation.py <test_number>
```

The visualization tool provides real-time plots of:
- Attitude quaternions and Euler angles
- Angular rates
- Control torques
- Pointing error
- Phase portraits
- System energy

## Mathematical Foundation

The controller uses a reduced-state formulation that converts quaternion error to an Euler angle representation for linear control. The system is modeled as:

dx/dt = Ax + Bu

Where:
- x = [attitude_error (3); w_error (3)]
- A captures the linearized rigid body dynamics
- B represents the actuator influence matrix

You can find more complete documentation in `attitude_controller.py`
