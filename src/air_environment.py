import numpy as np
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from target import Target, TargetStatus, TargetType, TargetSignature
from projectile import Missile, MissileStatus
from trajectory import create_trajectory
from misc import dist
from logs import environment_logger as logger
from event_types import EventBus, SimulationEvent, EventType


class WeatherCondition(Enum):
    """Weather conditions affecting radar propagation."""
    CLEAR = auto()
    LIGHT_RAIN = auto()
    HEAVY_RAIN = auto()
    FOG = auto()
    STORM = auto()


@dataclass
class AtmosphericConditions:
    """Atmospheric conditions affecting target motion and radar."""
    temperature: float = 15.0  # Celsius
    pressure: float = 1013.25  # hPa
    humidity: float = 50.0  # %
    wind_speed: float = 0.0  # m/s
    wind_direction: float = 0.0  # degrees
    weather: WeatherCondition = WeatherCondition.CLEAR
    
    def get_radar_attenuation(self) -> float:
        """Get radar signal attenuation factor due to weather."""
        if self.weather == WeatherCondition.CLEAR:
            return 1.0
        elif self.weather == WeatherCondition.LIGHT_RAIN:
            return 0.9
        elif self.weather == WeatherCondition.HEAVY_RAIN:
            return 0.7
        elif self.weather == WeatherCondition.FOG:
            return 0.85
        elif self.weather == WeatherCondition.STORM:
            return 0.5
        return 1.0
    
    def get_noise_factor(self) -> float:
        """Get additional noise factor due to weather."""
        if self.weather == WeatherCondition.STORM:
            return 3.0
        elif self.weather == WeatherCondition.HEAVY_RAIN:
            return 2.0
        elif self.weather == WeatherCondition.LIGHT_RAIN:
            return 1.3
        return 1.0


class AirEnvironment:
    """
    Air Environment / Воздушная Среда.
    Manages all aerial objects (targets and missiles) and simulates:
    - Target trajectory generation
    - Weather effects
    - Target destruction handling
    """
    
    def __init__(self, 
                 initialization_type: str = 'empty',
                 config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None):
        
        self.targets: Dict[int, Target] = {}
        self.projectiles: Dict[int, Missile] = {}  # Legacy compatibility
        self.missiles: Dict[int, Missile] = {}
        
        self.time_step: float = 0.005
        
        # Environment conditions
        self.atmosphere = AtmosphericConditions()
        
        # Simulation boundaries
        self.boundary_x: Tuple[float, float] = (0, 10000)
        self.boundary_y: Tuple[float, float] = (0, 10000)
        self.boundary_z: Tuple[float, float] = (0, 5000)
        
        # Event system
        self.event_bus = event_bus or EventBus()
        self.pending_guidance_commands: Dict[int, Dict[str, Any]] = {}
        
        # Target spawn management
        self.target_spawn_queue: List[Dict[str, Any]] = []
        self.scenario_targets: List[Dict[str, Any]] = []
        self.scenario_time: float = 0.0
        
        # Exploded objects pending cleanup
        self.exploded_not_cleared_projectiles: List[int] = []
        self.exploded_not_cleared_targets: List[int] = []
        
        # Statistics
        self.total_targets_spawned: int = 0
        self.total_targets_destroyed: int = 0
        self.total_missiles_launched: int = 0

        self._setup_event_handlers()
        
        if initialization_type == 'config_file' and config is not None:
            self.initialize_with_file_data(config)
            logger.info("Environment: initialization performed using config file")
        elif initialization_type == 'empty':
            logger.info("Environment: initialized empty")
        else:
            logger.warning("Environment: initializing with empty field")

    def _setup_event_handlers(self) -> None:
        """Subscribe to guidance updates produced by the PBU."""
        self.event_bus.subscribe(
            EventType.MISSILE_GUIDANCE_COMMAND,
            self._handle_guidance_command
        )

    def _handle_guidance_command(self, event: SimulationEvent) -> None:
        """Store the latest guidance command for a missile."""
        missile_id = (event.data or {}).get("missile_id")
        if missile_id is None:
            return

        self.pending_guidance_commands[missile_id] = dict(event.data)
    
    def initialize_with_file_data(self, config: Dict[str, Any]) -> bool:
        """Initialize environment from configuration file."""
        if config is None:
            logger.error("Environment: initialization error: config not provided")
            return False
        
        self.time_step = config.get("time_step", 0.005)
        
        # Load atmosphere settings
        if "atmosphere" in config:
            atm = config["atmosphere"]
            self.atmosphere = AtmosphericConditions(
                temperature=atm.get("temperature", 15.0),
                pressure=atm.get("pressure", 1013.25),
                humidity=atm.get("humidity", 50.0),
                wind_speed=atm.get("wind_speed", 0.0),
                wind_direction=atm.get("wind_direction", 0.0),
                weather=WeatherCondition[atm.get("weather", "CLEAR").upper()]
            )
        
        # Load boundaries
        if "boundaries" in config:
            b = config["boundaries"]
            self.boundary_x = tuple(b.get("x", [0, 10000]))
            self.boundary_y = tuple(b.get("y", [0, 10000]))
            self.boundary_z = tuple(b.get("z", [0, 5000]))
        
        # Load targets
        for target_id_str, params in config.get("targets", {}).items():
            target_id = int(target_id_str) if isinstance(target_id_str, str) else target_id_str
            self.add_target(
                trajectory_type=params.get("trajectory_type", "uniform"),
                id=target_id,
                trajectory_arguments=params.get("trajectory_arguments", {}),
                type=params.get("type", "UNKNOWN"),
                rcs=params.get("rcs", 1.0),
                rcs_fluctuation=params.get("rcs_fluctuation", 0.1),
            )
        
        return True
    
    def update(self, time_step: float):
        """
        Main update method called by dispatcher.
        Updates all targets and missiles.
        """
        self.scenario_time += time_step
        
        # Process spawn queue
        self._process_spawn_queue()
        
        # Update all targets
        self.update_targets(time_step)
        
        # Update all missiles
        self.update_missiles(time_step)
        
        # Apply wind effects
        self._apply_wind_effects(time_step)
        
        # Check boundaries
        self._check_boundaries()
        
        # Check for explosions
        self.check_exploded()
        
        # Clean up destroyed objects
        self.clear_exploded()
        self._publish_environment_state()
    
    def update_targets(self, time_step: float):
        """Update all active targets."""
        for target in self.targets.values():
            if target.status == TargetStatus.ACTIVE:
                target.update(time_step, self.scenario_time)
    
    def update_missiles(self, time_step: float):
        """Update all active missiles."""
        for missile_id, missile in list(self.missiles.items()):
            if missile.is_active():
                old_status = missile.status
                guidance_command = self.pending_guidance_commands.pop(missile_id, None)
                missile.update(time_step, guidance_command=guidance_command)
                self._publish_missile_telemetry(missile, guidance_command)

                if missile.status != old_status:
                    self._publish_missile_status_change(missile, old_status)
    
    def update_projectiles(self, time_step: float, new_targets: Dict[int, np.ndarray] = None):
        """
        Legacy compatibility method for updating projectiles.
        """
        new_targets = new_targets or {}
        
        for missile_id, missile in self.missiles.items():
            if missile_id in new_targets:
                missile.update(time_step, new_target=new_targets[missile_id])
            else:
                missile.update(time_step)
            
            if missile.exploded:
                self.exploded_not_cleared_projectiles.append(missile_id)
                self._check_missile_damage(missile)
    
    def _check_missile_damage(self, missile: Missile):
        """Check if missile explosion destroys any targets."""
        for target in self.targets.values():
            if target.status == TargetStatus.ACTIVE:
                if dist(target.position, missile.position) < missile.params.blast_radius:
                    target.destroy(self.scenario_time)
                    self.exploded_not_cleared_targets.append(target.id)
                    
                    # Publish event
                    self.event_bus.publish(SimulationEvent(
                        event_type=EventType.TARGET_DESTROYED,
                        source_id=f"missile_{missile.id}",
                        target_id=f"target_{target.id}",
                        data={
                            "position": target.position.tolist(),
                            "missile_position": missile.position.tolist()
                        }
                    ))

    def _publish_missile_telemetry(
        self,
        missile: Missile,
        guidance_command: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publish live missile telemetry for PBU and GUI consumers."""
        telemetry = missile.get_telemetry()
        telemetry["guidance_command"] = guidance_command
        telemetry["launcher_id"] = missile.launcher_id

        self.event_bus.publish(SimulationEvent(
            event_type=EventType.MISSILE_TELEMETRY,
            source_id=f"missile_{missile.id}",
            target_id=f"target_{missile.assigned_target_id}" if missile.assigned_target_id is not None else None,
            data=telemetry
        ))

    def _publish_missile_status_change(
        self,
        missile: Missile,
        previous_status: MissileStatus
    ) -> None:
        """Publish status transitions for missiles."""
        telemetry = missile.get_telemetry()
        telemetry["previous_status"] = previous_status.name
        telemetry["launcher_id"] = missile.launcher_id

        if missile.status == MissileStatus.DETONATED:
            event_type = EventType.MISSILE_DETONATED
        elif missile.status == MissileStatus.MISSED:
            event_type = EventType.MISSILE_MISSED
        else:
            event_type = EventType.MISSILE_GUIDANCE_UPDATED

        self.event_bus.publish(SimulationEvent(
            event_type=event_type,
            source_id=f"missile_{missile.id}",
            target_id=f"target_{missile.assigned_target_id}" if missile.assigned_target_id is not None else None,
            data=telemetry
        ))
    
    def _apply_wind_effects(self, time_step: float):
        """Apply wind effects to targets."""
        if self.atmosphere.wind_speed <= 0:
            return
        
        wind_dir_rad = math.radians(self.atmosphere.wind_direction)
        wind_vector = np.array([
            self.atmosphere.wind_speed * math.sin(wind_dir_rad),
            self.atmosphere.wind_speed * math.cos(wind_dir_rad),
            0.0
        ])
        
        # Apply to targets (simplified)
        for target in self.targets.values():
            if target.status == TargetStatus.ACTIVE:
                # Small position drift due to wind
                target.position += wind_vector * time_step * 0.1
    
    def _check_boundaries(self):
        """Check if targets have left the simulation area."""
        for target in self.targets.values():
            if target.status == TargetStatus.ACTIVE:
                pos = target.position
                if (pos[0] < self.boundary_x[0] or pos[0] > self.boundary_x[1] or
                    pos[1] < self.boundary_y[0] or pos[1] > self.boundary_y[1] or
                    pos[2] < self.boundary_z[0] or pos[2] > self.boundary_z[1]):
                    target.status = TargetStatus.EXPIRED
                    logger.info(f"Target {target.id} left simulation area")
    
    def _process_spawn_queue(self):
        """Process pending target spawns."""
        to_remove = []
        
        for i, spawn_data in enumerate(self.target_spawn_queue):
            spawn_time = spawn_data.get("spawn_time", 0.0)
            
            if self.scenario_time >= spawn_time:
                self._spawn_target(spawn_data)
                to_remove.append(i)
        
        for i in reversed(to_remove):
            self.target_spawn_queue.pop(i)
    
    def _spawn_target(self, spawn_data: Dict[str, Any]):
        """Spawn a target from spawn data."""
        traj_type = spawn_data.get("trajectory_type", "uniform")
        traj_args = spawn_data.get("trajectory_arguments", {})
        
        target_id = spawn_data.get("id")
        if target_id is None:
            target_id = max(self.targets.keys()) + 1 if self.targets else 0
        
        self.add_target(traj_type, id=target_id, trajectory_arguments=traj_args)
        
        self.event_bus.publish(SimulationEvent(
            event_type=EventType.TARGET_SPAWNED,
            target_id=f"target_{target_id}",
            data=spawn_data
        ))
    
    def add_target(self, trajectory_type: str, **kwargs) -> bool:
        """Add a new target to the environment."""
        target_id = kwargs.get('id')
        
        if target_id in self.targets:
            logger.error(f"Environment: target {target_id} already exists")
            return False
        
        try:
            traj_args = kwargs.get('trajectory_arguments', {})
            traj_args['position'] = traj_args.get('position', [0, 0, 0])
            traj_args['velocity'] = traj_args.get('velocity', [0, 0, 0])
            
            trajectory = create_trajectory(trajectory_type, **traj_args)
            
            target_type = TargetType.UNKNOWN
            if 'type' in kwargs:
                target_type = TargetType[kwargs['type'].upper()]
            
            signature = TargetSignature(
                rcs=kwargs.get('rcs', 1.0),
                rcs_fluctuation=kwargs.get('rcs_fluctuation', 0.1)
            )
            
            target = Target(
                id=target_id,
                trajectory=trajectory,
                target_type=target_type,
                signature=signature
            )
            
            self.targets[target_id] = target
            self.total_targets_spawned += 1
            
            logger.info(f"Environment: target {target_id} successfully set")
            return True
            
        except Exception as e:
            logger.error(f"Environment: adding target failed: {e}")
            return False
    
    def add_projectile(self, projectile_type: str, **kwargs) -> bool:
        """Legacy method to add a projectile/missile."""
        missile_id = kwargs.get('id')
        
        if missile_id in self.missiles:
            logger.error(f"Environment: missile {missile_id} already exists")
            return False
        
        try:
            missile_class = projectiles_typename_to_class.get(projectile_type)
            if missile_class is None:
                logger.error(f"Environment: unknown projectile type {projectile_type}")
                return False
            
            missile = missile_class(**kwargs)
            self.missiles[missile_id] = missile
            self.projectiles[missile_id] = missile  # Legacy compatibility
            self.total_missiles_launched += 1
            
            # Publish event
            self.event_bus.publish(SimulationEvent(
                event_type=EventType.MISSILE_LAUNCHED,
                source_id=f"missile_{missile_id}",
                data={"position": missile.position.tolist()}
            ))
            
            logger.info(f"Environment: projectile {missile_id} successfully set")
            return True
            
        except Exception as e:
            logger.error(f"Environment: adding projectile failed: {e}")
            return False
    
    def get_target_truth(self, target_id: int) -> Optional[Dict[str, Any]]:
        """
        Return the true target state for radar-owned measurement models.
        Simulates the "Зашумленные координаты" from the diagram.
        """
        target = self.targets.get(target_id)
        if target is None or target.status != TargetStatus.ACTIVE:
            return None

        return {
            "target_id": target_id,
            "position": target.position.copy(),
            "true_position": target.position.copy(),
            "rcs": target.signature.get_current_rcs(),
            "velocity": target.velocity.copy(),
            "target_type": target.target_type.name,
            "timestamp": self.scenario_time
        }

    def get_noisy_measurement(self, target_id: int, radar_position: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Legacy compatibility wrapper.
        Measurement noise now belongs to the radar receiver, so the environment
        only exposes true target state.
        """
        del radar_position
        return self.get_target_truth(target_id)
    
    def check_exploded(self):
        """Check for exploded missiles and affected targets."""
        for missile_id, missile in self.missiles.items():
            if missile.exploded and missile_id not in self.exploded_not_cleared_projectiles:
                self.exploded_not_cleared_projectiles.append(missile_id)
                self._register_missile_impact(missile)

    def _register_missile_impact(self, missile: Missile) -> None:
        """Apply the blast effect of a detonated missile."""
        for target_id, target in self.targets.items():
            if target.status != TargetStatus.ACTIVE:
                continue

            if dist(target.position, missile.position) >= missile.params.blast_radius:
                continue

            target.destroy(self.scenario_time)
            if target_id not in self.exploded_not_cleared_targets:
                self.exploded_not_cleared_targets.append(target_id)

            self.event_bus.publish(SimulationEvent(
                event_type=EventType.TARGET_DESTROYED,
                source_id=f"missile_{missile.id}",
                target_id=f"target_{target_id}",
                data={
                    "target_id": target_id,
                    "missile_id": missile.id,
                    "position": target.position.tolist(),
                    "missile_position": missile.position.tolist()
                }
            ))
    
    def clear_exploded(self):
        """Clean up exploded missiles and destroyed targets."""
        for missile_id in self.exploded_not_cleared_projectiles:
            if missile_id in self.missiles:
                del self.missiles[missile_id]
                if missile_id in self.projectiles:
                    del self.projectiles[missile_id]
        
        for target_id in self.exploded_not_cleared_targets:
            if target_id in self.targets:
                self.total_targets_destroyed += 1
                del self.targets[target_id]
        
        self.exploded_not_cleared_projectiles.clear()
        self.exploded_not_cleared_targets.clear()

    def _publish_environment_state(self) -> None:
        """Publish a compact snapshot of the current air environment."""
        active_targets = [
            {
                "id": target.id,
                "position": target.position.tolist(),
                "velocity": target.velocity.tolist(),
                "type": target.target_type.name,
                "status": target.status.name
            }
            for target in self.targets.values()
            if target.status == TargetStatus.ACTIVE
        ]

        self.event_bus.publish(SimulationEvent(
            event_type=EventType.ENVIRONMENT_STATE_UPDATED,
            source_id="environment",
            data={
                "time": self.scenario_time,
                "active_targets": active_targets,
                "active_missiles": len([m for m in self.missiles.values() if m.is_active()])
            }
        ))
    
    def get_targets(self) -> Dict[int, Target]:
        """Get all targets."""
        return self.targets
    
    def get_projectiles(self) -> Dict[int, Missile]:
        """Legacy: Get all projectiles."""
        return self.projectiles
    
    def get_missiles(self) -> Dict[int, Missile]:
        """Get all missiles."""
        return self.missiles
    
    def get_active_targets(self) -> List[Target]:
        """Get all active targets."""
        return [t for t in self.targets.values() if t.status == TargetStatus.ACTIVE]
    
    def get_state(self) -> Dict[str, Any]:
        """Get environment state for serialization."""
        return {
            "time": self.scenario_time,
            "atmosphere": {
                "weather": self.atmosphere.weather.name,
                "wind_speed": self.atmosphere.wind_speed,
                "wind_direction": self.atmosphere.wind_direction
            },
            "targets": [t.get_state() for t in self.targets.values()],
            "missiles": [m.get_state() for m in self.missiles.values()],
            "statistics": {
                "targets_spawned": self.total_targets_spawned,
                "targets_destroyed": self.total_targets_destroyed,
                "missiles_launched": self.total_missiles_launched
            }
        }
    
    def load_scenario(self, scenario: Dict[str, Any]):
        """Load a scenario with timed target spawns."""
        self.scenario_targets = scenario.get("targets", [])
        self.target_spawn_queue = self.scenario_targets.copy()
        self.scenario_time = 0.0
        logger.info(f"Environment: loaded scenario with {len(self.scenario_targets)} targets")


# Legacy compatibility imports
from projectile import projectile_typename_to_class as projectiles_typename_to_class
