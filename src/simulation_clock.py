import time
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass
from enum import Enum, auto
from logs import dispatcher_logger as logger
from event_types import EventBus, SimulationEvent, EventType


class SimulationState(Enum):
    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STEP = auto()


@dataclass
class TickableComponent:
    name: str
    update_func: Callable[[float], None]
    priority: int = 0
    enabled: bool = True


class SimulationClock:
    def __init__(self, time_step: float = 0.005, event_bus: Optional[EventBus] = None):
        self.time_step = time_step
        self.current_time: float = 0.0
        self.tick_count: int = 0
        self.state: SimulationState = SimulationState.STOPPED
        self._components: List[TickableComponent] = []
        self._component_map: Dict[str, TickableComponent] = {}
        self._real_time_factor: float = 1.0
        self._last_real_time: Optional[float] = None
        self.event_bus = event_bus or EventBus()
        self._tick_times: List[float] = []
        logger.info(f"SimulationClock initialized with time_step={time_step}")
    
    def register_component(self, name: str, update_func: Callable[[float], None], priority: int = 0):
        component = TickableComponent(name=name, update_func=update_func, priority=priority)
        self._components.append(component)
        self._component_map[name] = component
        self._components.sort(key=lambda c: c.priority)
        logger.info(f"Registered component: {name} (priority={priority})")
    
    def start(self):
        self.state = SimulationState.RUNNING
        self._last_real_time = time.time()
        self.event_bus.publish(SimulationEvent(event_type=EventType.SIMULATION_START))
        logger.info("Simulation started")
    
    def pause(self):
        self.state = SimulationState.PAUSED
        self._last_real_time = None
        logger.info("Simulation paused")
    
    def stop(self):
        self.state = SimulationState.STOPPED
        logger.info("Simulation stopped")
    
    def tick(self) -> bool:
        if self.state == SimulationState.STOPPED or self.state == SimulationState.PAUSED:
            return False
        tick_start = time.time()
        for component in self._components:
            if component.enabled:
                try:
                    component.update_func(self.time_step)
                except Exception as e:
                    logger.error(f"Error updating {component.name}: {e}")
        self.current_time += self.time_step
        self.tick_count += 1
        tick_duration = time.time() - tick_start
        self._tick_times.append(tick_duration)
        self.event_bus.publish(SimulationEvent(event_type=EventType.SIMULATION_TICK))
        return True
    
    def get_time(self) -> float:
        return self.current_time
    
    def get_tick_count(self) -> int:
        return self.tick_count
    
    def is_running(self) -> bool:
        return self.state == SimulationState.RUNNING
    
    def is_paused(self) -> bool:
        return self.state == SimulationState.PAUSED
    
    def get_average_tick_time(self) -> float:
        if not self._tick_times:
            return 0.0
        return sum(self._tick_times) / len(self._tick_times)
