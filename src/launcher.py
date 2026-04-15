import numpy as np
import math
from typing import Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from projectile import Missile, MissileParameters, GuidanceMode
from logs import launcher_logger as logger
from event_types import EventBus, SimulationEvent, EventType


class LauncherStatus(Enum):
    """Status of a launcher."""
    IDLE = auto()              # Ready, waiting for command
    PREPARING = auto()         # Preparing for launch
    READY_TO_FIRE = auto()     # Target locked, ready
    FIRING = auto()            # Missile launching
    RELOADING = auto()         # Reloading magazine
    MAINTENANCE = auto()       # Under maintenance
    EMPTY = auto()             # No missiles left
    MALFUNCTION = auto()       # System error


@dataclass
class LauncherParameters:
    """Launcher configuration parameters."""
    max_azimuth: float = 360.0      # degrees
    min_azimuth: float = 0.0
    max_elevation: float = 85.0     # degrees
    min_elevation: float = 5.0
    traverse_speed: float = 30.0    # deg/s
    elevation_speed: float = 20.0   # deg/s
    reload_time: float = 10.0       # seconds
    salvo_size: int = 1             # Missiles per salvo
    salvo_interval: float = 0.5     # seconds between missiles


@dataclass
class MissileMagazine:
    """Missile magazine storage."""
    capacity: int = 8
    missiles: Dict[int, Missile] = field(default_factory=dict)
    missile_types: Dict[int, str] = field(default_factory=dict)
    
    def add_missile(self, missile: Missile, missile_type: str = "guided") -> bool:
        """Add a missile to magazine."""
        if len(self.missiles) >= self.capacity:
            return False
        
        self.missiles[missile.id] = missile
        self.missile_types[missile.id] = missile_type
        return True
    
    def remove_missile(self, missile_id: int) -> Optional[Missile]:
        """Remove and return a missile from magazine."""
        missile = self.missiles.pop(missile_id, None)
        if missile is not None:
            self.missile_types.pop(missile_id, None)
        return missile
    
    def get_next_missile(self, preferred_type: Optional[str] = None) -> Optional[Tuple[int, Missile, str]]:
        """Get next missile to launch."""
        if not self.missiles:
            return None
        
        if preferred_type:
            for mid, mtype in self.missile_types.items():
                if mtype == preferred_type:
                    return mid, self.missiles[mid], mtype
        
        # Return first available
        mid = next(iter(self.missiles.keys()))
        return mid, self.missiles[mid], self.missile_types[mid]
    
    def get_count(self) -> int:
        """Get current missile count."""
        return len(self.missiles)
    
    def is_empty(self) -> bool:
        """Check if magazine is empty."""
        return len(self.missiles) == 0


class Launcher:
    """
    Launcher / Пусковая Установка (ПУ) class.
    Manages missile inventory, launch preparation, and firing.
    """
    
    _id_counter = 0
    
    def __init__(self,
                 launcher_id: Optional[int] = None,
                 position: Union[list, tuple, np.ndarray] = (0, 0, 0),
                 parameters: Optional[LauncherParameters] = None,
                 magazine_capacity: int = 8,
                 event_bus: Optional[EventBus] = None):
        
        self.id = launcher_id if launcher_id is not None else Launcher._id_counter
        Launcher._id_counter = max(Launcher._id_counter, self.id + 1)
        
        self.position = np.array(position, dtype=np.float64)
        self.params = parameters or LauncherParameters()
        self.status = LauncherStatus.IDLE
        
        # Magazine
        self.magazine = MissileMagazine(capacity=magazine_capacity)
        self.initial_missile_count = magazine_capacity  # Legacy compatibility
        self.missile_amount = magazine_capacity  # Legacy compatibility
        
        # Current orientation
        self.current_azimuth: float = 0.0
        self.current_elevation: float = 45.0
        self.target_azimuth: float = 0.0
        self.target_elevation: float = 45.0
        
        # Targeting
        self.assigned_target_id: Optional[int] = None
        self.assigned_target_position: Optional[np.ndarray] = None
        self.assigned_missile_type: str = "guided missile"
        
        # Launch state
        self.launch_countdown: float = 0.0
        self.reload_timer: float = 0.0
        self.salvo_remaining: int = 0
        self.salvo_timer: float = 0.0
        self.auto_launch_requested: bool = False
        self.current_time: float = 0.0
        self.bound_environment = None
        self._last_reported_status: Optional[LauncherStatus] = None

        # Default missile parameters
        self.default_missile_speed: float = 1000.0  # Legacy compatibility
        self.speed: float = 1000.0  # Legacy compatibility
        self.trigger_distance: float = 10.0
        self.explosion_range: float = 100.0
        
        # Statistics
        self.total_launches: int = 0
        self.successful_launches: int = 0
        self.failed_launches: int = 0
        
        # Event bus
        self.event_bus = event_bus or EventBus()

        self._setup_event_handlers()

        logger.info(f"Launcher {self.id} initialized at {self.position}")
        self._report_status(force=True)

    def _setup_event_handlers(self) -> None:
        """Subscribe to architecture-level launcher commands."""
        self.event_bus.subscribe(EventType.LAUNCHER_COMMAND, self._handle_command_event)

    def bind_environment(self, environment) -> None:
        """Bind launcher to the shared air environment."""
        self.bound_environment = environment
        logger.info(f"Launcher {self.id}: environment binding updated")

    def _handle_command_event(self, event: SimulationEvent) -> None:
        """Handle a command sent from the PBU or dispatcher."""
        data = event.data or {}
        launcher_id = data.get("launcher_id")

        if launcher_id is not None and launcher_id != self.id:
            return

        action = data.get("action", "assign")

        if action == "reload":
            self.reload()
            return

        if action == "cancel":
            self.cancel_launch()
            return

        target_position = (
            data.get("intercept_point")
            or data.get("target_position")
            or data.get("target")
        )

        if target_position is not None:
            self.assign_target(
                np.array(target_position, dtype=np.float64),
                target_id=data.get("target_id"),
                missile_type=data.get("missile_type")
            )

        if action in {"launch", "assign_and_launch"}:
            countdown = float(data.get("countdown", 0.0))
            self.start_launch_sequence(countdown=countdown)

        self._report_status(force=True)
    
    def initialize_magazine(self, missile_count: int, missile_type: str = "guided missile",
                           missile_speed: float = 1000.0) -> None:
        """Initialize magazine with missiles."""
        self.missile_amount = missile_count
        self.speed = missile_speed
        self.default_missile_speed = missile_speed
        self.assigned_missile_type = missile_type
        
        for i in range(missile_count):
            missile_id = self.id * 1000 + i
            params = MissileParameters(
                max_speed=missile_speed,
                trigger_distance=self.trigger_distance,
                blast_radius=self.explosion_range
            )
            missile = Missile(
                id=missile_id,
                position=self.position.copy(),
                launcher_id=self.id,
                parameters=params,
                guidance_mode=GuidanceMode.COMMAND
            )
            self.magazine.add_missile(missile, missile_type)
        
        self.initial_missile_count = missile_count
        logger.info(f"Launcher {self.id}: Magazine initialized with {missile_count} missiles")
        self._report_status(force=True)
    
    def update(self, time_step: float) -> None:
        """Update launcher state."""
        self.current_time += time_step

        # Update orientation
        self._update_orientation(time_step)

        # Update timers
        if self.launch_countdown > 0:
            self.launch_countdown -= time_step

        if self.salvo_timer > 0:
            self.salvo_timer -= time_step
            if self.salvo_timer <= 0 and self.salvo_remaining > 0:
                self._execute_launch()
        
        if self.reload_timer > 0:
            self.reload_timer -= time_step
            if self.reload_timer <= 0:
                self._complete_reload()

        if self.auto_launch_requested and self.launch_countdown <= 0 and self._is_aimed():
            self._execute_launch()

        # Update status
        self._update_status()
        self._report_status()
    
    def _update_orientation(self, time_step: float) -> None:
        """Update launcher orientation towards target."""
        if self.assigned_target_position is None:
            return
        
        # Calculate required angles
        relative = self.assigned_target_position - self.position
        r = np.linalg.norm(relative)
        
        if r > 0:
            target_az = math.degrees(math.atan2(relative[1], relative[0])) % 360
            target_el = math.degrees(math.asin(relative[2] / r))
            
            self.target_azimuth = np.clip(target_az, self.params.min_azimuth, self.params.max_azimuth)
            self.target_elevation = np.clip(target_el, self.params.min_elevation, self.params.max_elevation)
        
        # Smooth movement
        az_diff = self._angle_difference(self.target_azimuth, self.current_azimuth)
        el_diff = self.target_elevation - self.current_elevation
        
        max_az_move = self.params.traverse_speed * time_step
        max_el_move = self.params.elevation_speed * time_step
        
        self.current_azimuth += np.clip(az_diff, -max_az_move, max_az_move)
        self.current_elevation += np.clip(el_diff, -max_el_move, max_el_move)
        
        self.current_azimuth %= 360
    
    def _angle_difference(self, target: float, current: float) -> float:
        """Calculate shortest angle difference."""
        diff = (target - current + 180) % 360 - 180
        return diff
    
    def _update_status(self) -> None:
        """Update launcher status based on current state."""
        if self.magazine.is_empty():
            self.status = LauncherStatus.EMPTY
        elif self.reload_timer > 0:
            self.status = LauncherStatus.RELOADING
        elif self.auto_launch_requested or self.salvo_remaining > 0:
            if self._is_aimed() and self.launch_countdown <= 0:
                self.status = LauncherStatus.FIRING
            else:
                self.status = LauncherStatus.PREPARING
        elif self._is_aimed():
            self.status = LauncherStatus.READY_TO_FIRE
        elif self.assigned_target_position is not None:
            self.status = LauncherStatus.PREPARING
        else:
            self.status = LauncherStatus.IDLE
    
    def _is_aimed(self) -> bool:
        """Check if launcher is aimed at target."""
        if self.assigned_target_position is None:
            return True
        
        az_diff = abs(self._angle_difference(self.target_azimuth, self.current_azimuth))
        el_diff = abs(self.target_elevation - self.current_elevation)
        
        return az_diff < 1.0 and el_diff < 1.0
    
    def assign_target(self, target_position: np.ndarray, target_id: Optional[int] = None,
                      missile_type: Optional[str] = None) -> None:
        """Assign a target for engagement."""
        self.assigned_target_position = target_position.copy()
        self.assigned_target_id = target_id
        
        if missile_type:
            self.assigned_missile_type = missile_type
        
        logger.info(f"Launcher {self.id}: Target assigned at {target_position}")
        self._report_status(force=True)
    
    def prepare_launch(self) -> bool:
        """Prepare for launch."""
        if self.magazine.is_empty():
            logger.warning(f"Launcher {self.id}: Cannot prepare - magazine empty")
            return False

        if self.assigned_target_position is None:
            logger.warning(f"Launcher {self.id}: Cannot prepare - target not assigned")
            return False

        if not self._is_aimed():
            logger.info(f"Launcher {self.id}: Still aiming...")
            return False

        self.status = LauncherStatus.READY_TO_FIRE
        return True
    
    def launch(self, target_position: np.ndarray, missile_id_counter: int,
               environment) -> int:
        """
        Legacy launch method.
        Returns new missile ID counter.
        """
        if self.missile_amount <= 0:
            logger.warning(f"Launcher {self.id}: No missiles remaining")
            return missile_id_counter
        
        self.missile_amount -= 1
        self.total_launches += 1
        
        missile_id = missile_id_counter
        missile_id_counter += 1
        
        # Add missile to environment
        environment.add_projectile(
            self.assigned_missile_type,
            id=missile_id,
            position=self.position.copy(),
            target=target_position,
            trigger_distance=self.trigger_distance,
            explosion_range=self.explosion_range,
            max_velocity=self.speed
        )
        
        logger.info(f"Launcher {self.id}: Launched missile {missile_id}")
        
        self.event_bus.publish(SimulationEvent(
            event_type=EventType.MISSILE_LAUNCHED,
            source_id=f"launcher_{self.id}",
            target_id=f"missile_{missile_id}",
            data={
                "position": self.position.tolist(),
                "target": target_position.tolist(),
                "missile_id": missile_id
            }
        ))
        
        return missile_id_counter
    
    def launch_missile(self, environment, current_time: float) -> Optional[Missile]:
        """
        Launch a missile from magazine.
        Returns the launched missile or None.
        """
        if self.magazine.is_empty():
            logger.warning(f"Launcher {self.id}: Magazine empty")
            self.status = LauncherStatus.EMPTY
            return None
        
        if not self._is_aimed():
            logger.info(f"Launcher {self.id}: Not aimed, cannot launch")
            return None
        
        # Get missile from magazine
        result = self.magazine.get_next_missile(self.assigned_missile_type)
        if result is None:
            return None
        
        missile_id, missile, missile_type = result
        self.magazine.remove_missile(missile_id)
        
        # Launch the missile
        missile.launch(current_time, self.assigned_target_position, self.assigned_target_id)
        
        # Add to environment
        environment.missiles[missile_id] = missile
        environment.projectiles[missile_id] = missile  # Legacy
        
        self.total_launches += 1
        self.successful_launches += 1
        self.missile_amount = self.magazine.get_count()  # Legacy sync
        
        logger.info(f"Launcher {self.id}: Launched missile {missile_id} at target {self.assigned_target_id}")
        
        self.event_bus.publish(SimulationEvent(
            event_type=EventType.MISSILE_LAUNCHED,
            source_id=f"launcher_{self.id}",
            target_id=f"missile_{missile_id}",
            data={
                "missile_id": missile_id,
                "missile_type": missile_type,
                "target_id": self.assigned_target_id,
                "position": self.position.tolist(),
                "target_position": self.assigned_target_position.tolist()
            }
        ))
        
        return missile
    
    def start_launch_sequence(self, countdown: float = 3.0) -> bool:
        """Start launch countdown."""
        if self.magazine.is_empty():
            logger.warning(f"Launcher {self.id}: Cannot start launch - magazine empty")
            return False

        if self.assigned_target_position is None:
            logger.warning(f"Launcher {self.id}: Cannot start launch - target not assigned")
            return False

        self.auto_launch_requested = True
        self.launch_countdown = max(0.0, countdown)
        self.status = LauncherStatus.PREPARING if not self._is_aimed() else LauncherStatus.FIRING
        logger.info(f"Launcher {self.id}: Launch sequence started, T-{countdown}s")
        self._report_status(force=True)
        return True

    def start_salvo(self, count: int, interval: float = 0.5) -> bool:
        """Start a salvo launch."""
        if count > self.magazine.get_count():
            logger.warning(f"Launcher {self.id}: Not enough missiles for salvo")
            return False
        
        if self.assigned_target_position is None:
            logger.warning(f"Launcher {self.id}: Cannot start salvo - target not assigned")
            return False

        self.salvo_remaining = count
        self.salvo_timer = 0.0
        self.params.salvo_interval = interval
        self.auto_launch_requested = True
        self.launch_countdown = 0.0
        self.status = LauncherStatus.PREPARING if not self._is_aimed() else LauncherStatus.FIRING

        logger.info(f"Launcher {self.id}: Starting salvo of {count} missiles")
        self._report_status(force=True)
        return True

    def _execute_launch(self) -> None:
        """Execute the actual launch."""
        if self.bound_environment is None:
            logger.warning(f"Launcher {self.id}: No environment bound for launch")
            self.auto_launch_requested = False
            return

        if self.assigned_target_position is None:
            logger.warning(f"Launcher {self.id}: No target assigned for launch")
            self.auto_launch_requested = False
            return

        missile = self.launch_missile(self.bound_environment, self.current_time)
        if missile is None:
            self.failed_launches += 1
            self.auto_launch_requested = False
            return

        if self.salvo_remaining > 0:
            self.salvo_remaining -= 1

        if self.salvo_remaining > 0 and not self.magazine.is_empty():
            self.salvo_timer = self.params.salvo_interval
        else:
            self.salvo_timer = 0.0
            self.auto_launch_requested = False
            self.assigned_target_position = None
            self.assigned_target_id = None

        self.launch_countdown = 0.0
        self._update_status()
        self._report_status(force=True)
    
    def reload(self) -> None:
        """Start reloading process."""
        if self.status != LauncherStatus.EMPTY:
            logger.info(f"Launcher {self.id}: Starting reload")

        self.status = LauncherStatus.RELOADING
        self.reload_timer = self.params.reload_time
        self._report_status(force=True)
    
    def _complete_reload(self) -> None:
        """Complete reloading process."""
        # In a real simulation, this would add missiles from supply
        self.status = LauncherStatus.IDLE
        logger.info(f"Launcher {self.id}: Reload complete")
        self._report_status(force=True)
    
    def cancel_launch(self) -> None:
        """Cancel current launch sequence."""
        self.launch_countdown = 0.0
        self.salvo_remaining = 0
        self.salvo_timer = 0.0
        self.auto_launch_requested = False
        self.assigned_target_position = None
        self.assigned_target_id = None
        self.status = LauncherStatus.IDLE
        logger.info(f"Launcher {self.id}: Launch cancelled")
        self._report_status(force=True)

    def _report_status(self, force: bool = False) -> None:
        """Publish launcher status updates for PBU and GUI consumers."""
        if not force and self._last_reported_status == self.status:
            return

        status_payload = self.get_status()
        self.event_bus.publish(SimulationEvent(
            event_type=EventType.LAUNCHER_STATUS_UPDATED,
            source_id=f"launcher_{self.id}",
            target_id=f"launcher_{self.id}",
            data=status_payload
        ))

        if self.status == LauncherStatus.READY_TO_FIRE:
            self.event_bus.publish(SimulationEvent(
                event_type=EventType.LAUNCHER_READY,
                source_id=f"launcher_{self.id}",
                target_id=f"launcher_{self.id}",
                data=status_payload
            ))
        elif self.status == LauncherStatus.RELOADING:
            self.event_bus.publish(SimulationEvent(
                event_type=EventType.LAUNCHER_RELOADING,
                source_id=f"launcher_{self.id}",
                target_id=f"launcher_{self.id}",
                data=status_payload
            ))
        elif self.status == LauncherStatus.EMPTY:
            self.event_bus.publish(SimulationEvent(
                event_type=EventType.LAUNCHER_EMPTY,
                source_id=f"launcher_{self.id}",
                target_id=f"launcher_{self.id}",
                data=status_payload
            ))

        self._last_reported_status = self.status
    
    def get_status(self) -> Dict[str, Any]:
        """Get launcher status report."""
        return {
            "id": self.id,
            "status": self.status.name,
            "position": self.position.tolist(),
            "azimuth": self.current_azimuth,
            "elevation": self.current_elevation,
            "magazine_count": self.magazine.get_count(),
            "magazine_capacity": self.magazine.capacity,
            "assigned_target": self.assigned_target_id,
            "is_aimed": self._is_aimed(),
            "total_launches": self.total_launches,
            "missile_amount": self.missile_amount  # Legacy
        }
    
    def get_missile_count(self) -> int:
        """Get remaining missile count."""
        return self.magazine.get_count()
    
    def is_ready(self) -> bool:
        """Check if launcher is ready to fire."""
        return (self.status == LauncherStatus.READY_TO_FIRE and 
                not self.magazine.is_empty() and 
                self._is_aimed())
    
    def __repr__(self) -> str:
        return f"Launcher(id={self.id}, status={self.status.name}, missiles={self.magazine.get_count()})"
