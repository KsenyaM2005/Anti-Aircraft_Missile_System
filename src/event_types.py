from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from time import time


class EventType(Enum):
    """Types of events in the simulation."""
    # System events
    SIMULATION_START = auto()
    SIMULATION_PAUSE = auto()
    SIMULATION_RESET = auto()
    SIMULATION_TICK = auto()
    
    # Target events
    TARGET_SPAWNED = auto()
    TARGET_DESTROYED = auto()
    TARGET_DETECTED = auto()
    TARGET_LOST = auto()
    TARGET_TRACK_UPDATED = auto()
    
    # Radar events
    RADAR_SCAN_STARTED = auto()
    RADAR_SCAN_COMPLETED = auto()
    RADAR_TRACK_INITIATED = auto()
    RADAR_TRACK_DROPPED = auto()
    RADAR_CONTROL_COMMAND = auto()

    # Launcher events
    LAUNCHER_READY = auto()
    LAUNCHER_RELOADING = auto()
    LAUNCHER_EMPTY = auto()
    LAUNCHER_COMMAND = auto()
    LAUNCHER_STATUS_UPDATED = auto()

    # Missile events
    MISSILE_LAUNCHED = auto()
    MISSILE_GUIDANCE_UPDATED = auto()
    MISSILE_GUIDANCE_COMMAND = auto()
    MISSILE_TELEMETRY = auto()
    MISSILE_DETONATED = auto()
    MISSILE_MISSED = auto()
    MISSILE_SELF_DESTRUCTED = auto()

    # PBU events
    THREAT_ASSESSED = auto()
    TARGET_ASSIGNED = auto()
    ENGAGEMENT_DECISION = auto()
    SITUATION_UPDATED = auto()

    # Shared data snapshots
    ENVIRONMENT_STATE_UPDATED = auto()

    # Operator events
    OPERATOR_COMMAND = auto()
    MANUAL_LAUNCH = auto()
    SCENARIO_SELECTED = auto()


@dataclass
class SimulationEvent:
    """Base event class for simulation events."""
    event_type: EventType
    timestamp: float = field(default_factory=time)
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Event({self.event_type.name}, src={self.source_id}, tgt={self.target_id})"


class EventBus:
    """Central event bus for simulation communication."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, list] = {}
        self._event_history: list[SimulationEvent] = []
        self._max_history = 10000
    
    def subscribe(self, event_type: EventType, callback):
        """Subscribe to a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback):
        """Unsubscribe from a specific event type."""
        if event_type in self._subscribers:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
    
    def publish(self, event: SimulationEvent):
        """Publish an event to all subscribers."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                callback(event)
    
    def get_history(self, event_type: Optional[EventType] = None) -> list:
        """Get event history, optionally filtered by type."""
        if event_type is None:
            return self._event_history.copy()
        return [e for e in self._event_history if e.event_type == event_type]
    
    def clear_history(self):
        """Clear event history."""
        self._event_history.clear()
