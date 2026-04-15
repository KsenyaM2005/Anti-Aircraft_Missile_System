import numpy as np
import math
from typing import Union, Tuple


def dist(lhs: Union[list, tuple, np.ndarray], rhs: Union[list, tuple, np.ndarray]) -> float:
    """Calculate Euclidean distance between two points."""
    lhs = np.array(lhs, dtype=np.float64) if not isinstance(lhs, np.ndarray) else lhs
    rhs = np.array(rhs, dtype=np.float64) if not isinstance(rhs, np.ndarray) else rhs
    return float(np.sqrt(np.sum((lhs - rhs) ** 2)))


def normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


def spherical_to_cartesian(r: float, phi: float, theta: float) -> Tuple[float, float, float]:
    """Convert spherical coordinates to Cartesian."""
    x = r * math.cos(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.cos(theta)
    z = r * math.sin(theta)
    return x, y, z


def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert Cartesian coordinates to spherical."""
    r = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y, x)
    theta = math.asin(z / r) if r > 0 else 0.0
    return r, phi, theta


def add_gaussian_noise(value: Union[float, np.ndarray], sigma: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Add Gaussian noise to a value or array."""
    return value + np.random.normal(0, sigma, size=np.shape(value))


class Vector3:
    """Simple 3D vector class for cleaner code."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
    
    @classmethod
    def from_array(cls, arr: Union[list, tuple, np.ndarray]) -> 'Vector3':
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3':
        mag = self.magnitude()
        if mag > 0:
            return Vector3(self.x / mag, self.y / mag, self.z / mag)
        return Vector3(0, 0, 0)
    
    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3') -> 'Vector3':
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
