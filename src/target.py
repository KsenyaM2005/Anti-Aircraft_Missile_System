import numpy as np
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum, auto
from trajectory import Trajectory


class TargetStatus(Enum):
    """Status of a target."""
    INACTIVE = auto()      # Not yet spawned
    ACTIVE = auto()        # Flying normally
    DESTROYED = auto()     # Hit by missile
    EXPIRED = auto()       # Left simulation area
    LOST = auto()          # Lost by radar


class TargetType(Enum):
    """Type of aerial target."""
    UNKNOWN = auto()
    FIGHTER = auto()
    BOMBER = auto()
    CRUISE_MISSILE = auto()
    BALLISTIC_MISSILE = auto()
    DRONE = auto()
    HELICOPTER = auto()


@dataclass
class TargetSignature:
    """Radar signature characteristics of a target."""
    rcs: float = 1.0              # Radar Cross Section (m²)
    rcs_fluctuation: float = 0.1  # RCS fluctuation factor
    thermal_signature: float = 1.0
    visual_signature: float = 1.0
    
    def get_current_rcs(self) -> float:
        """Get current RCS with fluctuation."""
        fluctuation = 1.0 + np.random.normal(0, self.rcs_fluctuation)
        return max(0.01, self.rcs * fluctuation)


class Target:
    """
    Aerial target class.
    Represents an enemy aircraft or missile in the simulation.
    """
    
    _id_counter = 0
    
    def __init__(self, 
                 id: Optional[int] = None,
                 trajectory: Optional[Trajectory] = None,
                 target_type: TargetType = TargetType.UNKNOWN,
                 signature: Optional[TargetSignature] = None,
                 max_speed: float = 1000.0,
                 max_acceleration: float = 10.0):
        
        self.id = id if id is not None else Target._id_counter
        Target._id_counter = max(Target._id_counter, self.id + 1)
        
        self.trajectory = trajectory
        self.position = trajectory.get_position() if trajectory else np.zeros(3)
        self.velocity = trajectory.get_velocity() if trajectory else np.zeros(3)
        
        self.target_type = target_type
        self.signature = signature or TargetSignature()
        self.status = TargetStatus.ACTIVE
        self.destroyed = False  # Legacy compatibility
        
        # Performance characteristics
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        
        # Tracking information
        self.is_tracked = False
        self.tracking_radar_id: Optional[str] = None
        self.track_quality: float = 0.0  # 0.0 to 1.0
        self.time_since_last_detection: float = 0.0
        
        # Statistics
        self.spawn_time: float = 0.0
        self.destruction_time: Optional[float] = None
        self.distance_traveled: float = 0.0
        
        # History for visualization
        self.position_history: list = []
        self.max_history_length = 1000
    
    def update(self, time_step: float, current_time: Optional[float] = None):
        """Update target state."""
        if self.status != TargetStatus.ACTIVE:
            return
        
        old_position = self.position.copy()
        
        if self.trajectory:
            self.trajectory.update(time_step)
            self.position = self.trajectory.get_position()
            self.velocity = self.trajectory.get_velocity()
        
        # Update statistics
        self.distance_traveled += np.linalg.norm(self.position - old_position)
        self.time_since_last_detection += time_step
        
        # Update history
        self.position_history.append(self.position.copy())
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        # Check if target is lost
        if self.time_since_last_detection > 10.0:  # 10 seconds without detection
            self.status = TargetStatus.LOST
            self.is_tracked = False
    
    def mark_detected(self, radar_id: str, quality: float = 1.0):
        """Mark target as detected by radar."""
        self.is_tracked = True
        self.tracking_radar_id = radar_id
        self.track_quality = quality
        self.time_since_last_detection = 0.0
        
        if self.status == TargetStatus.LOST:
            self.status = TargetStatus.ACTIVE
    
    def destroy(self, current_time: Optional[float] = None):
        """Mark target as destroyed."""
        self.status = TargetStatus.DESTROYED
        self.destroyed = True
        self.destruction_time = current_time
        self.is_tracked = False
    
    def get_noisy_position(self, noise_sigma: Union[float, np.ndarray] = 10.0) -> np.ndarray:
        """
        Get target position with simulated radar noise.
        This is called by the Air Environment to simulate radar measurements.
        """
        if isinstance(noise_sigma, (int, float)):
            noise = np.random.normal(0, noise_sigma, 3)
        else:
            noise = np.random.normal(0, noise_sigma, 3)
        
        return self.position + noise
    
    def get_noisy_rcs(self) -> float:
        """Get current RCS with fluctuation."""
        return self.signature.get_current_rcs()
    
    def get_state(self) -> Dict[str, Any]:
        """Get target state for serialization."""
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "type": self.target_type.name,
            "status": self.status.name,
            "destroyed": self.destroyed,
            "is_tracked": self.is_tracked,
            "track_quality": self.track_quality,
            "rcs": self.signature.get_current_rcs()
        }
    
    def get_predicted_position(self, time_ahead: float) -> np.ndarray:
        """Predict target position at future time."""
        if self.trajectory is None:
            return self.position + self.velocity * time_ahead
        
        # Simple linear prediction
        return self.position + self.velocity * time_ahead
    
    def is_alive(self) -> bool:
        """Check if target is still active."""
        return self.status == TargetStatus.ACTIVE
    
    def get_speed(self) -> float:
        """Get current speed."""
        return float(np.linalg.norm(self.velocity))
    
    def get_altitude(self) -> float:
        """Get current altitude (Z coordinate)."""
        return float(self.position[2])
    
    def get_heading(self) -> float:
        """Get heading angle in degrees (0 = North/Y-axis)."""
        speed = self.get_speed()
        if speed < 0.01:
            return 0.0
        return float(np.degrees(np.arctan2(self.velocity[0], self.velocity[1])))
    
    def __repr__(self) -> str:
        return f"Target(id={self.id}, pos={self.position.round(1)}, status={self.status.name})"

