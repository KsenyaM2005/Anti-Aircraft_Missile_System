import numpy as np
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
from enum import Enum, auto
from misc import dist, normalize


class MissileStatus(Enum):
    """Status of a missile."""
    STORED = auto()          # In launcher magazine
    PREPARING = auto()       # Preparing for launch
    BOOSTING = auto()        # Initial boost phase
    CRUISING = auto()        # Mid-course flight
    TERMINAL = auto()        # Terminal guidance phase
    DETONATED = auto()       # Warhead detonated
    MISSED = auto()          # Missed target (self-destruct)
    MALFUNCTION = auto()     # System failure


class GuidanceMode(Enum):
    """Missile guidance modes."""
    COMMAND = auto()         # Command guidance from ground
    SEMI_ACTIVE = auto()     # Semi-active radar homing
    ACTIVE = auto()          # Active radar homing
    INFRARED = auto()        # Infrared homing
    INERTIAL = auto()        # Inertial navigation only
    PROPORTIONAL = auto()    # Proportional navigation


@dataclass
class MissileParameters:
    """Missile performance parameters."""
    max_speed: float = 1000.0           # m/s
    acceleration: float = 180.0         # m/s² (≈18g — adequate for chasing aerial targets)
    turn_rate: float = 35.0             # degrees/s
    fuel_duration: float = 60.0         # seconds
    warhead_weight: float = 50.0        # kg
    blast_radius: float = 100.0         # meters
    trigger_distance: float = 10.0      # Proximity fuse trigger distance


class Missile:
    """
    Guided missile (ZUR) class.
    Represents a surface-to-air missile in the simulation.
    """
    
    _id_counter = 0
    
    def __init__(self,
                 id: Optional[int] = None,
                 position: Union[list, tuple, np.ndarray] = None,
                 target_position: Union[list, tuple, np.ndarray] = None,
                 launcher_id: Optional[int] = None,
                 parameters: Optional[MissileParameters] = None,
                 guidance_mode: GuidanceMode = GuidanceMode.COMMAND):
        
        self.id = id if id is not None else Missile._id_counter
        Missile._id_counter = max(Missile._id_counter, self.id + 1)
        
        self.position = np.array(position, dtype=np.float64) if position is not None else np.zeros(3)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)
        
        self.target_position = np.array(target_position, dtype=np.float64) if target_position is not None else None
        self.launcher_id = launcher_id
        
        self.params = parameters or MissileParameters()
        self.guidance_mode = guidance_mode
        
        self.status = MissileStatus.STORED
        self.exploded = False  # Legacy compatibility
        
        # Flight data
        self.launch_time: Optional[float] = None
        self.flight_time: float = 0.0
        self.distance_flown: float = 0.0
        
        # Guidance data
        self.guidance_commands: List[Dict[str, Any]] = []
        self.last_guidance_update: float = 0.0
        self.intercept_point: Optional[np.ndarray] = None
        self.desired_direction: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        
        # Navigation constant for proportional navigation
        self.N = 3.0
        
        # History
        self.position_history: list = []
        self.max_history_length = 500
        
        # Target tracking
        self.assigned_target_id: Optional[int] = None
        
        # Telemetry
        self.telemetry: Dict[str, Any] = {}
    
    def launch(self, current_time: float, target_position: np.ndarray, assigned_target_id: Optional[int] = None):
        """Launch the missile."""
        self.status = MissileStatus.BOOSTING
        self.launch_time = current_time
        self.target_position = target_position.copy()
        self.assigned_target_id = assigned_target_id
        
        # Initial velocity towards target
        direction = normalize(self.target_position - self.position)
        self.velocity = direction * (self.params.max_speed * 0.65)  # Initial boost speed
        
        self.exploded = False
    
    def update(self, time_step: float, new_target: Optional[np.ndarray] = None, 
               guidance_command: Optional[Dict[str, Any]] = None):
        """
        Update missile state.
        
        Args:
            time_step: Simulation time step
            new_target: Updated target position (from radar/PBU)
            guidance_command: Guidance command from PBU
        """
        if self.status in [MissileStatus.STORED, MissileStatus.DETONATED, MissileStatus.MISSED]:
            return
        
        old_position = self.position.copy()
        
        # Update target position if provided
        if new_target is not None:
            self.target_position = np.array(new_target, dtype=np.float64)
        
        # Process guidance command
        if guidance_command is not None:
            self.guidance_commands.append(guidance_command)
            if len(self.guidance_commands) > 10:
                self.guidance_commands.pop(0)
            
            if "intercept_point" in guidance_command:
                self.intercept_point = np.array(guidance_command["intercept_point"])
            if "target_position" in guidance_command:
                self.target_position = np.array(guidance_command["target_position"], dtype=np.float64)
            self.last_guidance_update = guidance_command.get("timestamp", self.last_guidance_update)
        
        # Update flight phase based on time
        self.flight_time += time_step
        self._update_flight_phase()
        
        # Apply guidance
        if self.target_position is not None:
            self._apply_guidance(time_step)
        
        # Update physics
        self._update_physics(time_step)
        
        # Check for impact/detonation
        self._check_detonation()
        
        # Check fuel expiration
        if self.flight_time > self.params.fuel_duration:
            self.status = MissileStatus.MISSED
            self.exploded = True
        
        # Update statistics
        self.distance_flown += np.linalg.norm(self.position - old_position)
        
        # Update history
        self.position_history.append(self.position.copy())
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        # Update telemetry
        self._update_telemetry()
    
    def _update_flight_phase(self):
        """Update missile flight phase based on time."""
        if self.flight_time < 3.0:
            self.status = MissileStatus.BOOSTING
        elif self.flight_time < self.params.fuel_duration * 0.8:
            self.status = MissileStatus.CRUISING
        else:
            self.status = MissileStatus.TERMINAL
    
    def _apply_guidance(self, time_step: float):
        """Apply guidance laws to steer missile towards target."""
        if self.target_position is None:
            return
        
        # Line of sight vector
        los = self.target_position - self.position
        los_distance = np.linalg.norm(los)
        
        if los_distance < 1.0:
            return
        
        los_rate = los / los_distance
        
        if self.guidance_mode == GuidanceMode.COMMAND:
            # Command guidance: simply steer towards target or intercept point
            target = self.intercept_point if self.intercept_point is not None else self.target_position
            desired_direction = normalize(target - self.position)
            
        elif self.guidance_mode == GuidanceMode.PROPORTIONAL:
            # Proportional navigation
            if len(self.position_history) > 1:
                los_prev = self.target_position - self.position_history[-2]
                los_prev_dist = np.linalg.norm(los_prev)
                if los_prev_dist > 0:
                    los_prev_rate = los_prev / los_prev_dist
                    los_rate_derivative = (los_rate - los_prev_rate) / time_step
                    
                    # Acceleration command
                    closing_velocity = -np.dot(self.velocity, los_rate)
                    desired_accel = self.N * closing_velocity * np.cross(los_rate, los_rate_derivative)
                    
                    self.acceleration = np.clip(desired_accel, -self.params.acceleration, self.params.acceleration)
                    return
        
        # Simple pursuit guidance (fallback)
        if "desired_direction" not in locals():
            desired_direction = normalize(self.target_position - self.position)
        self.desired_direction = desired_direction
        
        # Calculate required acceleration
        current_direction = normalize(self.velocity) if np.linalg.norm(self.velocity) > 0 else desired_direction
        direction_error = desired_direction - current_direction
        
        # Apply acceleration with limits
        max_accel = self.params.acceleration
        if self.status == MissileStatus.BOOSTING:
            max_accel *= 1.5

        turn_accel = direction_error * max_accel * 6.0
        thrust_accel = desired_direction * max_accel * (1.1 if self.status == MissileStatus.BOOSTING else 0.65)
        commanded_accel = turn_accel + thrust_accel
        accel_norm = np.linalg.norm(commanded_accel)
        if accel_norm > max_accel * 1.5:
            commanded_accel = commanded_accel / accel_norm * max_accel * 1.5
        self.acceleration = commanded_accel
    
    def _update_physics(self, time_step: float):
        """Update missile position and velocity."""
        # Apply acceleration
        self.velocity += self.acceleration * time_step
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        max_speed = self.params.max_speed
        
        if self.status == MissileStatus.BOOSTING:
            max_speed = self.params.max_speed * (0.7 + 0.3 * min(1.0, self.flight_time / 1.5))
        
        if speed > max_speed:
            self.velocity = self.velocity / speed * max_speed
            speed = max_speed

        if self.target_position is not None:
            minimum_speed = self.params.max_speed * (0.65 if self.status == MissileStatus.BOOSTING else 0.7)
            if speed < minimum_speed:
                direction = self.desired_direction if np.linalg.norm(self.desired_direction) > 0 else normalize(self.velocity)
                self.velocity = direction * minimum_speed
                speed = minimum_speed
        
        # Update position
        self.position += self.velocity * time_step
        
        # Apply gravity (simplified)
        self.velocity[2] -= 9.81 * time_step * 0.1  # Reduced gravity effect
    
    def _check_detonation(self):
        """Check if missile should detonate."""
        if self.target_position is None:
            return
        
        distance_to_target = dist(self.position, self.target_position)
        
        # Proximity fuse
        if distance_to_target < self.params.trigger_distance:
            self.status = MissileStatus.DETONATED
            self.exploded = True
        
        # Direct hit
        if distance_to_target < 1.0:
            self.status = MissileStatus.DETONATED
            self.exploded = True
    
    def _update_telemetry(self):
        """Update telemetry data."""
        self.telemetry = {
            "id": self.id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "speed": float(np.linalg.norm(self.velocity)),
            "altitude": float(self.position[2]),
            "status": self.status.name,
            "flight_time": self.flight_time,
            "distance_to_target": float(dist(self.position, self.target_position)) if self.target_position is not None else None,
            "assigned_target": self.assigned_target_id
        }
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get current telemetry data."""
        return self.telemetry.copy()
    
    def get_state(self) -> Dict[str, Any]:
        """Get missile state for serialization."""
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "status": self.status.name,
            "exploded": self.exploded,
            "flight_time": self.flight_time,
            "target_position": self.target_position.tolist() if self.target_position is not None else None,
            "launcher_id": self.launcher_id
        }
    
    def is_active(self) -> bool:
        """Check if missile is active in flight."""
        return self.status in [MissileStatus.BOOSTING, MissileStatus.CRUISING, MissileStatus.TERMINAL]
    
    def __repr__(self) -> str:
        return f"Missile(id={self.id}, status={self.status.name}, pos={self.position.round(1)})"


# Legacy compatibility classes
class Projectile(Missile):
    """Legacy compatibility class."""
    def __init__(self, position=None, target=None, id=None, trigger_distance=10.0, 
                 explosion_range=100.0, max_velocity=1000.0, **kwargs):
        params = MissileParameters(
            max_speed=max_velocity,
            trigger_distance=trigger_distance,
            blast_radius=explosion_range
        )
        super().__init__(id=id, position=position, target_position=target, parameters=params)
        self.explosion_range = explosion_range


class GuidedMissile(Missile):
    """Legacy compatibility class for guided missiles."""
    def __init__(self, **kwargs):
        params = MissileParameters(
            max_speed=kwargs.get('max_velocity', 1000.0),
            trigger_distance=kwargs.get('trigger_distance', 10.0),
            blast_radius=kwargs.get('explosion_range', 100.0)
        )
        super().__init__(
            id=kwargs.get('id'),
            position=kwargs.get('position'),
            target_position=kwargs.get('target'),
            parameters=params,
            guidance_mode=GuidanceMode.COMMAND
        )
        self.explosion_range = kwargs.get('explosion_range', 100.0)
    
    def update_target(self, target):
        """Update target position."""
        self.target_position = np.array(target, dtype=np.float64) if target is not None else None


class PreemptiveMissile(GuidedMissile):
    """Legacy compatibility class for preemptive missiles."""
    def __init__(self, preemption=0.0, **kwargs):
        super().__init__(**kwargs)
        self.preemption = preemption
        self.prev_target = None
    
    def update_target(self, target):
        if self.prev_target is not None and target is not None:
            # Predict future position
            target_velocity = target - self.prev_target
            predicted = target + target_velocity * self.preemption
            super().update_target(predicted)
        else:
            super().update_target(target)
        
        self.prev_target = target.copy() if target is not None else None


# Factory mapping
projectile_typename_to_class = {
    'simple projectile': Projectile,
    'guided missile': GuidedMissile,
    'preemptive missile': PreemptiveMissile
}
