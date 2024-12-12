import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to euler angles (XYZ sequence)"""  # Changed from ZYX
    return Rotation.from_quat(np.roll(q, -1)).as_euler("xyz", degrees=True)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in scalar-first format [q0, q1, q2, q3]

    Args:
        q1: First quaternion [q0, q1, q2, q3]
        q2: Second quaternion [q0, q1, q2, q3]

    Returns:
        Product quaternion [q0, q1, q2, q3]
    """
    # Extract scalar and vector parts
    s1, v1 = q1[0], q1[1:4]
    s2, v2 = q2[0], q2[1:4]

    # Compute product
    s = s1 * s2 - np.dot(v1, v2)
    v = s1 * v2 + s2 * v1 + np.cross(v1, v2)

    return np.array([s, v[0], v[1], v[2]])
