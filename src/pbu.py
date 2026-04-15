import numpy as np
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from target import TargetType
from launcher import Launcher, LauncherStatus
from misc import dist
from logs import pbu_logger as logger
from event_types import EventBus, SimulationEvent, EventType


class ThreatLevel(Enum):
    """Target threat assessment levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class EngagementStatus(Enum):
    """Status of target engagement."""
    UNENGAGED = auto()
    ASSIGNED = auto()
    ENGAGED = auto()
    INTERCEPTED = auto()
    MISSED = auto()


@dataclass
class TargetInfo:
    """Information about a tracked target."""
    target_id: int
    track_id: Optional[int] = None
    radar_id: Optional[int] = None
    
    # Position and motion
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_update: float = 0.0
    
    # Classification
    target_type: TargetType = TargetType.UNKNOWN
    estimated_rcs: float = 1.0
    
    # Threat assessment
    threat_level: ThreatLevel = ThreatLevel.NONE
    priority: float = 0.0
    time_to_impact: Optional[float] = None
    closest_approach: Optional[float] = None
    
    # Engagement
    engagement_status: EngagementStatus = EngagementStatus.UNENGAGED
    assigned_launcher_id: Optional[int] = None
    assigned_missile_id: Optional[int] = None
    engagement_time: Optional[float] = None
    predicted_intercept_point: Optional[np.ndarray] = None
    
    # History
    position_history: List[np.ndarray] = field(default_factory=list)
    max_history: int = 20


@dataclass
class EngagementPlan:
    """Plan for engaging a target."""
    target_id: int
    launcher_id: int
    missile_type: str = "guided missile"
    intercept_point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    estimated_intercept_time: float = 0.0
    probability_of_kill: float = 0.8
    command_time: float = 0.0


class ThreatAssessor:
    """Assesses threat levels of targets."""
    
    def __init__(self, defended_position: np.ndarray = None):
        if defended_position is None:
            self.defended_position = np.array([5000, 5000, 0])
        else:
            self.defended_position = defended_position
        self.speed_threshold_high = 300.0  # m/s
        self.speed_threshold_critical = 500.0
        self.range_threshold_low = 5000.0   # m
        self.range_threshold_high = 2000.0
        self.altitude_threshold_low = 100.0
    
    def assess(self, target_info: TargetInfo) -> ThreatLevel:
        """Assess threat level of a target."""
        score = 0.0
        
        # Distance to defended asset
        distance = dist(target_info.position, self.defended_position)
        if distance < self.range_threshold_high:
            score += 3.0
        elif distance < self.range_threshold_low:
            score += 2.0
        else:
            score += 1.0
        
        # Speed (faster = more threatening)
        speed = np.linalg.norm(target_info.velocity)
        if speed > self.speed_threshold_critical:
            score += 3.0
        elif speed > self.speed_threshold_high:
            score += 2.0
        else:
            score += 1.0
        
        # Heading towards defended asset
        if self._is_heading_towards_defense(target_info):
            score += 2.0
        
        # Target type
        if target_info.target_type == TargetType.BALLISTIC_MISSILE:
            score += 3.0
        elif target_info.target_type == TargetType.CRUISE_MISSILE:
            score += 2.0
        elif target_info.target_type == TargetType.BOMBER:
            score += 1.5
        elif target_info.target_type == TargetType.FIGHTER:
            score += 1.0
        
        # RCS (smaller = stealthier = more threatening if close)
        if target_info.estimated_rcs < 0.5 and distance < self.range_threshold_low:
            score += 1.0
        
        # Determine threat level
        if score >= 8.0:
            return ThreatLevel.CRITICAL
        elif score >= 6.0:
            return ThreatLevel.HIGH
        elif score >= 4.0:
            return ThreatLevel.MEDIUM
        elif score > 0:
            return ThreatLevel.LOW
        
        return ThreatLevel.NONE
    
    def _is_heading_towards_defense(self, target_info: TargetInfo) -> bool:
        """Check if target is heading towards defended position."""
        if np.linalg.norm(target_info.velocity) < 1.0:
            return False
        
        to_defense = self.defended_position - target_info.position
        velocity_dir = target_info.velocity / np.linalg.norm(target_info.velocity)
        to_defense_dir = to_defense / np.linalg.norm(to_defense)
        
        dot_product = np.dot(velocity_dir, to_defense_dir)
        return dot_product > 0.5  # Within ~60 degrees


class EngagementPlanner:
    """Plans target engagements."""
    
    def __init__(self):
        self.min_intercept_altitude = 50.0
        self.max_engagement_range = 5000.0
        self.preferred_engagement_range = 3000.0
    
    def plan_engagement(self, target_info: TargetInfo, 
                        launchers: Dict[int, Launcher]) -> Optional[EngagementPlan]:
        """Create an engagement plan for a target."""
        # Find suitable launcher
        best_launcher_id = None
        best_intercept_time = float('inf')
        best_intercept_point = None
        
        for launcher_id, launcher in launchers.items():
            if launcher.status == LauncherStatus.EMPTY:
                continue
            if launcher.get_missile_count() <= 0:
                continue
            
            # Check if launcher can reach target
            distance = dist(launcher.position, target_info.position)
            if distance > self.max_engagement_range:
                continue
            
            # Calculate intercept
            intercept_result = self._calculate_intercept(
                launcher.position, target_info
            )
            
            if intercept_result is None:
                continue
            
            intercept_point, intercept_time = intercept_result
            
            # Check if this is better than current best
            if intercept_time < best_intercept_time:
                best_intercept_time = intercept_time
                best_intercept_point = intercept_point
                best_launcher_id = launcher_id
        
        if best_launcher_id is None:
            return None
        
        return EngagementPlan(
            target_id=target_info.target_id,
            launcher_id=best_launcher_id,
            intercept_point=best_intercept_point,
            estimated_intercept_time=best_intercept_time
        )
    
    def _calculate_intercept(self, launcher_pos: np.ndarray, 
                             target_info: TargetInfo) -> Optional[Tuple[np.ndarray, float]]:
        """Calculate intercept point and time."""
        # Simplified intercept calculation
        missile_speed = 1000.0  # m/s
        
        target_pos = target_info.position
        target_vel = target_info.velocity
        
        # Relative position
        rel_pos = target_pos - launcher_pos
        
        # Solve for intercept time using quadratic equation
        a = np.dot(target_vel, target_vel) - missile_speed**2
        b = 2 * np.dot(rel_pos, target_vel)
        c = np.dot(rel_pos, rel_pos)
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None
        
        t1 = (-b + math.sqrt(discriminant)) / (2*a)
        t2 = (-b - math.sqrt(discriminant)) / (2*a)
        
        intercept_time = min(t for t in [t1, t2] if t > 0)
        
        if intercept_time <= 0:
            return None
        
        intercept_point = target_pos + target_vel * intercept_time
        
        # Check altitude
        if intercept_point[2] < self.min_intercept_altitude:
            return None
        
        return intercept_point, intercept_time


class Pbu:
    """
    Command Post / Пункт Боевого Управления (ПБУ).
    Central command and control for air defense.
    
    Responsibilities:
    - Receiving tracks from radars
    - Threat assessment and prioritization
    - Target assignment to launchers
    - Generating guidance commands for missiles
    - Managing overall battle situation
    """
    
    def __init__(self, 
                 initialization_type: str = 'empty',
                 config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None):
        
        # Tracked targets
        self.targets: Dict[int, TargetInfo] = {}
        self._next_target_id = 0
        
        # Launchers under command
        self.launchers: Dict[int, Launcher] = {}
        
        # Active engagements
        self.engagements: Dict[int, EngagementPlan] = {}
        
        # Missile guidance
        self.missile_assignments: Dict[int, int] = {}  # missile_id -> target_id
        self.missile_telemetry: Dict[int, Dict[str, Any]] = {}
        self.launcher_statuses: Dict[int, Dict[str, Any]] = {}
        self.order_log: List[Dict[str, Any]] = []

        # Configuration
        self.time_step: float = 0.005
        self.add_distance: float = 300.0  # Minimum distance between targets
        self.min_engagement_interval: float = 2.0  # Seconds between engagements
        self.last_engagement_time: float = 0.0
        
        # Components
        self.threat_assessor = ThreatAssessor()
        self.engagement_planner = EngagementPlanner()
        
        # Situation awareness
        self.defended_assets: List[np.ndarray] = [np.array([5000, 5000, 0])]
        self.situation_picture: Dict[str, Any] = {}
        
        # Statistics
        self.targets_detected: int = 0
        self.targets_engaged: int = 0
        self.targets_destroyed: int = 0
        self.missiles_expended: int = 0
        
        # Timing
        self.current_time: float = 0.0
        
        # Event bus
        self.event_bus = event_bus or EventBus()
        self._setup_event_handlers()
        
        # Legacy compatibility
        self.exploded_not_cleared_targets: List[int] = []
        
        if initialization_type == 'config_file' and config is not None:
            self.initialize_with_file_data(config)
            logger.info("PBU: initialization performed using config file")
        elif initialization_type == 'empty':
            logger.info("PBU: initialized empty")
        else:
            logger.warning("PBU: initializing with empty field")

    def _setup_event_handlers(self) -> None:
        """Subscribe to architecture-level events from other components."""
        self.event_bus.subscribe(EventType.TARGET_TRACK_UPDATED, self._handle_track_update_event)
        self.event_bus.subscribe(EventType.LAUNCHER_STATUS_UPDATED, self._handle_launcher_status_event)
        self.event_bus.subscribe(EventType.MISSILE_LAUNCHED, self._handle_missile_launched_event)
        self.event_bus.subscribe(EventType.MISSILE_TELEMETRY, self._handle_missile_telemetry_event)
        self.event_bus.subscribe(EventType.MISSILE_DETONATED, self._handle_missile_result_event)
        self.event_bus.subscribe(EventType.MISSILE_MISSED, self._handle_missile_result_event)
        self.event_bus.subscribe(EventType.TARGET_DESTROYED, self._handle_target_destroyed_event)
        self.event_bus.subscribe(EventType.OPERATOR_COMMAND, self._handle_operator_command_event)

    def _parse_target_id(self, raw_target_id: Any) -> Optional[int]:
        """Normalize target identifiers from different event producers."""
        if raw_target_id is None:
            return None
        if isinstance(raw_target_id, int):
            return raw_target_id
        if isinstance(raw_target_id, str) and raw_target_id.startswith("target_"):
            try:
                return int(raw_target_id.split("_", 1)[1])
            except ValueError:
                return None
        try:
            return int(raw_target_id)
        except (TypeError, ValueError):
            return None

    def _handle_track_update_event(self, event: SimulationEvent) -> None:
        """Handle track data produced by radars."""
        track_data = dict(event.data or {})
        radar_id = track_data.pop("radar_id", None)
        if radar_id is None:
            source_id = event.source_id or ""
            if source_id.startswith("radar_"):
                try:
                    radar_id = int(source_id.split("_", 1)[1])
                except ValueError:
                    radar_id = -1
        if radar_id is None:
            return

        self.process_radar_track(radar_id=radar_id, track_data=track_data)

    def _handle_launcher_status_event(self, event: SimulationEvent) -> None:
        """Cache launcher status reports for decision making and GUI export."""
        status = dict(event.data or {})
        launcher_id = status.get("id")
        if launcher_id is None:
            return
        self.launcher_statuses[launcher_id] = status

    def _handle_missile_launched_event(self, event: SimulationEvent) -> None:
        """Bind a launched missile to its assigned target."""
        data = event.data or {}
        missile_id = data.get("missile_id")
        target_id = self._parse_target_id(data.get("target_id") or event.target_id)
        if missile_id is None or target_id is None:
            return

        self.missile_assignments[missile_id] = target_id
        self.missiles_expended += 1

        target_info = self.targets.get(target_id)
        if target_info is not None:
            target_info.assigned_missile_id = missile_id
            target_info.engagement_status = EngagementStatus.ENGAGED

    def _handle_missile_telemetry_event(self, event: SimulationEvent) -> None:
        """Store live telemetry for active missiles."""
        telemetry = dict(event.data or {})
        missile_id = telemetry.get("id")
        if missile_id is None:
            return
        self.missile_telemetry[missile_id] = telemetry

    def _handle_missile_result_event(self, event: SimulationEvent) -> None:
        """Update engagement state when a missile detonates or misses."""
        telemetry = dict(event.data or {})
        missile_id = telemetry.get("id")
        target_id = self._parse_target_id(telemetry.get("assigned_target") or event.target_id)

        if missile_id is not None:
            self.missile_telemetry[missile_id] = telemetry

        if target_id is None:
            return

        target_info = self.targets.get(target_id)
        if target_info is None:
            return

        if event.event_type == EventType.MISSILE_MISSED:
            target_info.engagement_status = EngagementStatus.MISSED
            target_info.assigned_missile_id = None
            if missile_id in self.missile_assignments:
                del self.missile_assignments[missile_id]
            self.engagements.pop(target_id, None)

    def _handle_target_destroyed_event(self, event: SimulationEvent) -> None:
        """Close the engagement loop after a confirmed target kill."""
        target_id = self._parse_target_id((event.data or {}).get("target_id") or event.target_id)
        if target_id is None:
            return

        target_info = self.targets.get(target_id)
        if target_info is not None:
            if target_info.engagement_status != EngagementStatus.INTERCEPTED:
                self.targets_destroyed += 1
            target_info.engagement_status = EngagementStatus.INTERCEPTED
            target_info.assigned_missile_id = None
        else:
            self.targets_destroyed += 1
        self.engagements.pop(target_id, None)
        for missile_id, assigned_target_id in list(self.missile_assignments.items()):
            if assigned_target_id == target_id:
                del self.missile_assignments[missile_id]

    def _handle_operator_command_event(self, event: SimulationEvent) -> None:
        """Apply operator actions that directly affect PBU decisions."""
        data = event.data or {}
        command = data.get("command")

        if command == "manual_launch":
            launcher_id = data.get("launcher_id")
            target_id = data.get("target_id")
            target_info = self.targets.get(target_id)
            if launcher_id is None or target_info is None:
                return

            intercept_point = target_info.predicted_intercept_point
            if intercept_point is None:
                intercept_point = target_info.position.copy()

            self.event_bus.publish(SimulationEvent(
                event_type=EventType.LAUNCHER_COMMAND,
                source_id="pbu",
                target_id=f"launcher_{launcher_id}",
                data={
                    "action": "assign_and_launch",
                    "launcher_id": launcher_id,
                    "target_id": target_id,
                    "target_position": target_info.position.tolist(),
                    "intercept_point": intercept_point.tolist(),
                    "missile_type": data.get("missile_type", "guided missile"),
                    "countdown": float(data.get("countdown", 0.0))
                }
            ))
    
    def initialize_with_file_data(self, config: Dict[str, Any]) -> bool:
        """Initialize PBU from configuration."""
        if config is None:
            logger.error("PBU: initialization error: config not provided")
            return False
        
        self.time_step = config.get("time_step", 0.005)
        self.add_distance = config.get("add_distance", 300.0)
        self.min_engagement_interval = config.get("min_engagement_interval", self.min_engagement_interval)

        threat_cfg = config.get("threat_assessment", {})
        self.threat_assessor.speed_threshold_high = threat_cfg.get(
            "speed_threshold_high",
            self.threat_assessor.speed_threshold_high
        )
        self.threat_assessor.speed_threshold_critical = threat_cfg.get(
            "speed_threshold_critical",
            self.threat_assessor.speed_threshold_critical
        )
        self.threat_assessor.range_threshold_low = threat_cfg.get(
            "range_threshold_low",
            self.threat_assessor.range_threshold_low
        )
        self.threat_assessor.range_threshold_high = threat_cfg.get(
            "range_threshold_high",
            self.threat_assessor.range_threshold_high
        )

        # Initialize launchers
        for launcher_id_str, params in config.get("launchers", {}).items():
            launcher_id = int(launcher_id_str) if isinstance(launcher_id_str, str) else launcher_id_str
            self.add_launcher(
                id=launcher_id,
                launcher_pos=params.get("launcher_pos", [0, 0, 0]),
                missile_amount=params.get("missile_amount", 5),
                speed=params.get("speed", 1000),
                missile_type=params.get("missile_type", "guided missile"),
                trigger_distance=params.get("trigger_distance", 10.0),
                explosion_range=params.get("explosion_range", 100.0)
            )
        
        # Initialize defended assets
        if "defended_assets" in config:
            self.defended_assets = [np.array(asset) for asset in config["defended_assets"]]
            if self.defended_assets:
                self.threat_assessor.defended_position = self.defended_assets[0]

        return True
    
    def update(self, time_step: float) -> None:
        """Main update method called by dispatcher."""
        self.time_step = time_step
        self.current_time += time_step
        
        # Update threat assessments
        self._update_threat_assessments()
        
        # Make engagement decisions
        self._make_engagement_decisions()
        
        # Update guidance for missiles
        self._update_missile_guidance()
        
        # Update situation picture
        self._update_situation_picture()
        
        # Clean up old engagements
        self._cleanup_engagements()

    def process_radar_track(self, radar_id: int, track_data: Dict[str, Any]) -> None:
        """Process track data from a radar."""
        target_id = track_data.get("target_id")

        if target_id is None:
            target_id = self._register_new_target(track_data)

        if target_id not in self.targets:
            self.targets[target_id] = TargetInfo(target_id=target_id)

        target_info = self.targets[target_id]
        target_info.track_id = track_data.get("track_id")
        target_info.radar_id = radar_id
        target_info.position = np.array(track_data["position"])
        target_info.velocity = np.array(track_data.get("velocity", [0, 0, 0]))
        target_info.estimated_rcs = track_data.get("estimated_rcs", 1.0)
        target_info.last_update = self.current_time

        target_info.position_history.append(target_info.position.copy())
        if len(target_info.position_history) > target_info.max_history:
            target_info.position_history.pop(0)

        self.event_bus.publish(SimulationEvent(
            event_type=EventType.TARGET_DETECTED,
            source_id=f"radar_{radar_id}",
            target_id=f"target_{target_id}",
            data={
                "radar_id": radar_id,
                "track_id": target_info.track_id,
                "position": target_info.position.tolist(),
                "velocity": target_info.velocity.tolist(),
                "estimated_rcs": target_info.estimated_rcs
            }
        ))
    
    def _register_new_target(self, track_data: Dict[str, Any]) -> int:
        """Register a new target and assign an ID."""
        new_pos = np.array(track_data["position"])
        
        # Check if this matches an existing target
        for existing_id, existing_info in self.targets.items():
            if dist(existing_info.position, new_pos) < self.add_distance:
                # Might be the same target
                logger.info(f"PBU: Target at {new_pos} matches existing target {existing_id}")
                return existing_id
        
        # Create new target
        target_id = self._next_target_id
        self._next_target_id += 1
        
        # Create TargetInfo for the new target
        target_info = TargetInfo(target_id=target_id)
        target_info.position = new_pos
        target_info.velocity = np.array(track_data.get("velocity", [0, 0, 0]))
        target_info.last_update = self.current_time
        
        # Add to targets dictionary
        self.targets[target_id] = target_info
        
        logger.info(f"PBU: Registered new target {target_id} at {new_pos}")
        
        self.event_bus.publish(SimulationEvent(
            event_type=EventType.TARGET_DETECTED,
            source_id="pbu",
            target_id=f"target_{target_id}",
            data={"position": new_pos.tolist()}
        ))
        
        self.targets_detected += 1
        
        return target_id

    
    def _update_threat_assessments(self) -> None:
        """Update threat assessments for all targets."""
        for target_info in self.targets.values():
            # Skip if data is stale
            if self.current_time - target_info.last_update > 5.0:
                continue
            
            target_info.threat_level = self.threat_assessor.assess(target_info)
            
            # Calculate priority
            threat_scores = {
                ThreatLevel.CRITICAL: 100.0,
                ThreatLevel.HIGH: 70.0,
                ThreatLevel.MEDIUM: 40.0,
                ThreatLevel.LOW: 10.0,
                ThreatLevel.NONE: 0.0
            }
            
            base_priority = threat_scores[target_info.threat_level]
            
            # Adjust for distance (closer = higher priority)
            distance = dist(target_info.position, self.defended_assets[0])
            distance_factor = max(0, 1.0 - distance / 10000.0)
            
            target_info.priority = base_priority * (1.0 + distance_factor)
            self.event_bus.publish(SimulationEvent(
                event_type=EventType.THREAT_ASSESSED,
                source_id="pbu",
                target_id=f"target_{target_info.target_id}",
                data={
                    "target_id": target_info.target_id,
                    "threat_level": target_info.threat_level.name,
                    "priority": target_info.priority,
                    "distance_to_asset": distance
                }
            ))
    
    def _make_engagement_decisions(self) -> None:
        """Make decisions about which targets to engage."""
        if self.current_time - self.last_engagement_time < self.min_engagement_interval:
            return
        
        # Get unengaged targets sorted by priority
        unengaged_targets = [
            (tid, info) for tid, info in self.targets.items()
            if info.engagement_status == EngagementStatus.UNENGAGED
            and info.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            and self.current_time - info.last_update < 5.0
        ]
        
        unengaged_targets.sort(key=lambda x: x[1].priority, reverse=True)
        
        for target_id, target_info in unengaged_targets:
            # Check if we have available launchers
            available_launchers = {
                lid: l for lid, l in self.launchers.items()
                if l.get_missile_count() > 0
                and l.status not in [LauncherStatus.EMPTY, LauncherStatus.RELOADING]
            }
            
            if not available_launchers:
                break
            
            # Plan engagement
            plan = self.engagement_planner.plan_engagement(target_info, available_launchers)
            
            if plan is not None:
                self._execute_engagement(plan, target_info)
                self.last_engagement_time = self.current_time
                self.targets_engaged += 1
                break
    
    def _execute_engagement(self, plan: EngagementPlan, target_info: TargetInfo) -> None:
        """Execute an engagement plan."""
        launcher = self.launchers.get(plan.launcher_id)
        if launcher is None:
            return
        
        # Update target info
        target_info.engagement_status = EngagementStatus.ASSIGNED
        target_info.assigned_launcher_id = plan.launcher_id
        target_info.predicted_intercept_point = plan.intercept_point
        target_info.engagement_time = self.current_time
        plan.command_time = self.current_time
        
        # Store engagement plan
        self.engagements[plan.target_id] = plan

        if target_info.radar_id is not None:
            self.event_bus.publish(SimulationEvent(
                event_type=EventType.RADAR_CONTROL_COMMAND,
                source_id="pbu",
                target_id=f"radar_{target_info.radar_id}",
                data={
                    "radar_id": target_info.radar_id,
                    "mode": "TRACK",
                    "track_id": target_info.track_id,
                    "target_id": plan.target_id
                }
            ))

        self.event_bus.publish(SimulationEvent(
            event_type=EventType.LAUNCHER_COMMAND,
            source_id="pbu",
            target_id=f"launcher_{plan.launcher_id}",
            data={
                "action": "assign_and_launch",
                "launcher_id": plan.launcher_id,
                "target_id": plan.target_id,
                "target_position": target_info.position.tolist(),
                "intercept_point": plan.intercept_point.tolist(),
                "missile_type": plan.missile_type,
                "countdown": 0.0
            }
        ))

        self.order_log.append({
            "time": self.current_time,
            "target_id": plan.target_id,
            "launcher_id": plan.launcher_id,
            "radar_id": target_info.radar_id,
            "order": "assign_and_launch"
        })

        logger.info(f"PBU: Target {plan.target_id} assigned to launcher {plan.launcher_id}")

        self.event_bus.publish(SimulationEvent(
            event_type=EventType.TARGET_ASSIGNED,
            source_id="pbu",
            target_id=f"target_{plan.target_id}",
            data={
                "launcher_id": plan.launcher_id,
                "intercept_point": plan.intercept_point.tolist(),
                "estimated_time": plan.estimated_intercept_time
            }
        ))
        self.event_bus.publish(SimulationEvent(
            event_type=EventType.ENGAGEMENT_DECISION,
            source_id="pbu",
            target_id=f"target_{plan.target_id}",
            data={
                "target_id": plan.target_id,
                "launcher_id": plan.launcher_id,
                "radar_id": target_info.radar_id,
                "missile_type": plan.missile_type
            }
        ))
    
    def _update_missile_guidance(self) -> None:
        """Update guidance commands for in-flight missiles."""
        for missile_id, target_id in list(self.missile_assignments.items()):
            target_info = self.targets.get(target_id)
            if target_info is None:
                continue

            telemetry = self.missile_telemetry.get(missile_id)
            if telemetry is None:
                continue

            launcher = self.launchers.get(target_info.assigned_launcher_id)
            if launcher is None:
                continue

            guidance = self.get_guidance_command(
                missile_id=missile_id,
                missile_position=np.array(telemetry["position"]),
                target_id=target_id
            )
            if guidance is None:
                continue

            self.event_bus.publish(SimulationEvent(
                event_type=EventType.MISSILE_GUIDANCE_COMMAND,
                source_id="pbu",
                target_id=f"missile_{missile_id}",
                data=guidance
            ))
    
    def get_guidance_command(self, missile_id: int, missile_position: np.ndarray,
                            target_id: int) -> Optional[Dict[str, Any]]:
        """Generate guidance command for a missile."""
        target_info = self.targets.get(target_id)
        if target_info is None:
            return None
        
        # Recalculate intercept point
        launcher = self.launchers.get(target_info.assigned_launcher_id)
        if launcher is None:
            return None
        
        intercept_result = self.engagement_planner._calculate_intercept(
            launcher.position, target_info
        )
        
        if intercept_result is None:
            # Use predicted position
            intercept_point = target_info.position + target_info.velocity * 1.0
        else:
            intercept_point, _ = intercept_result
        
        return {
            "missile_id": missile_id,
            "target_id": target_id,
            "intercept_point": intercept_point,
            "target_position": target_info.position,
            "target_velocity": target_info.velocity,
            "timestamp": self.current_time
        }
    
    def _update_situation_picture(self) -> None:
        """Update the overall situation picture."""
        active_targets = []
        for target_info in self.targets.values():
            if self.current_time - target_info.last_update < 10.0:
                active_targets.append({
                    "id": target_info.target_id,
                    "position": target_info.position.tolist(),
                    "threat_level": target_info.threat_level.name,
                    "status": target_info.engagement_status.name
                })
        
        self.situation_picture = {
            "time": self.current_time,
            "active_targets": len(active_targets),
            "targets": active_targets,
            "launchers": [l.get_status() for l in self.launchers.values()],
            "launcher_statuses": self.launcher_statuses,
            "orders": self.order_log[-20:],
            "missiles": list(self.missile_telemetry.values())[-20:],
            "missiles_expended": self.missiles_expended,
            "targets_destroyed": self.targets_destroyed
        }
        self.event_bus.publish(SimulationEvent(
            event_type=EventType.SITUATION_UPDATED,
            source_id="pbu",
            data=self.situation_picture.copy()
        ))
    
    def _cleanup_engagements(self) -> None:
        """Clean up completed or failed engagements."""
        to_remove = []
        
        for target_id, plan in self.engagements.items():
            target_info = self.targets.get(target_id)
            
            # Remove if target is gone or engagement is old
            if target_info is None:
                to_remove.append(target_id)
            elif self.current_time - plan.command_time > max(30.0, plan.estimated_intercept_time + 5.0):
                # Engagement probably missed
                to_remove.append(target_id)
                target_info.engagement_status = EngagementStatus.MISSED
                logger.info(f"PBU: Engagement for target {target_id} timed out")
        
        for target_id in to_remove:
            del self.engagements[target_id]
    
    def add_launcher(self, **kwargs) -> bool:
        """Add a launcher to PBU control."""
        launcher_id = kwargs.get('id')
        
        if launcher_id in self.launchers:
            logger.warning(f"PBU: Launcher {launcher_id} already exists")
            return False
        
        try:
            launcher = Launcher(
                launcher_id=launcher_id,
                position=kwargs.get('launcher_pos', [0, 0, 0]),
                magazine_capacity=kwargs.get('missile_amount', 5),
                event_bus=self.event_bus
            )
            
            launcher.speed = kwargs.get('speed', 1000)
            launcher.trigger_distance = kwargs.get('trigger_distance', 10.0)
            launcher.explosion_range = kwargs.get('explosion_range', 100.0)
            launcher.params.traverse_speed = kwargs.get('traverse_speed', launcher.params.traverse_speed)
            launcher.params.elevation_speed = kwargs.get('elevation_speed', launcher.params.elevation_speed)
            launcher.params.reload_time = kwargs.get('reload_time', launcher.params.reload_time)
            
            missile_type = kwargs.get('missile_type', 'guided missile')
            launcher.initialize_magazine(
                missile_count=kwargs.get('missile_amount', 5),
                missile_type=missile_type,
                missile_speed=kwargs.get('speed', 1000)
            )
            
            self.launchers[launcher_id] = launcher
            
            logger.info(f"PBU: Launcher {launcher_id} successfully added")
            return True
            
        except Exception as e:
            logger.error(f"PBU: Adding launcher failed: {e}")
            return False
    
    def add_target(self, pos: np.ndarray, env) -> Tuple[int, int]:
        """
        Legacy method for adding a target.
        Returns (pbu_target_id, missile_id).
        """
        # Check if target already exists
        for target_id, target_info in self.targets.items():
            if dist(target_info.position, pos) < self.add_distance:
                logger.info(f"PBU: Target already exists at {pos}")
                return -1, -1
        
        # Create new target
        target_id = self._next_target_id
        self._next_target_id += 1
        
        self.targets[target_id] = TargetInfo(
            target_id=target_id,
            position=pos.copy(),
            last_update=self.current_time
        )
        
        # Find nearest launcher with missiles
        min_dist = float('inf')
        best_launcher_id = None
        
        for launcher_id, launcher in self.launchers.items():
            if launcher.missile_amount > 0:
                d = dist(launcher.position, pos)
                if d < min_dist:
                    min_dist = d
                    best_launcher_id = launcher_id
        
        if best_launcher_id is not None:
            # Launch missile
            missile_id = self.targets_detected + self.targets_engaged
            self.missiles_expended += 1
            return target_id, missile_id
        
        logger.warning(f"PBU: No available launcher for target {target_id}")
        return -1, -1
    
    def update_targets(self, target_id: int, pos: np.ndarray) -> None:
        """Legacy method to update target position."""
        if target_id in self.targets:
            self.targets[target_id].position = pos.copy()
            self.targets[target_id].last_update = self.current_time
    
    def clear_exploded(self, target_id: int) -> None:
        """Legacy method to clear destroyed target."""
        if target_id in self.targets:
            del self.targets[target_id]
            self.targets_destroyed += 1
            logger.info(f"PBU: Target {target_id} destroyed")
    
    def get_launchers(self) -> Dict[int, Launcher]:
        """Get all launchers."""
        return self.launchers
    
    def get_targets(self) -> Dict[int, TargetInfo]:
        """Get all tracked targets."""
        return self.targets
    
    def get_situation_picture(self) -> Dict[str, Any]:
        """Get current situation picture for display."""
        return self.situation_picture.copy()
    
    def get_engagement_status(self) -> Dict[str, Any]:
        """Get engagement status report."""
        return {
            "active_engagements": len(self.engagements),
            "targets_detected": self.targets_detected,
            "targets_engaged": self.targets_engaged,
            "targets_destroyed": self.targets_destroyed,
            "missiles_expended": self.missiles_expended,
            "launchers_available": sum(1 for l in self.launchers.values() if l.get_missile_count() > 0)
        }
    
    def __repr__(self) -> str:
        return f"PBU(targets={len(self.targets)}, launchers={len(self.launchers)}, engagements={len(self.engagements)})"


# Alias for compatibility
Pbu = Pbu
