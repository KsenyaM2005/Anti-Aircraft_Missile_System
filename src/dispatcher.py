from __future__ import annotations

from typing import List, Optional

import numpy as np

from air_environment import AirEnvironment
from event_types import EventBus, EventType, SimulationEvent
from gui import GUI
from logs import dispatcher_logger as logger
from pbu import Pbu
from radar import RadarSystem
from simulation_clock import SimulationClock
from target import TargetStatus


class SimulationDispatcher:
    """
    Dispatcher coordinating components according to the provided architecture.

    Responsibilities:
    - Distribute ticks to environment, radar, PBU, and launchers.
    - Bind launchers to the shared environment.
    - Handle operator commands that affect the clock or scenario.
    """

    def __init__(
        self,
        clock: SimulationClock,
        event_bus: EventBus,
        environment: AirEnvironment,
        pbu: Pbu,
        radars: List[RadarSystem],
        gui: Optional[GUI] = None
    ):
        self.clock = clock
        self.event_bus = event_bus
        self.environment = environment
        self.pbu = pbu
        self.radars = radars
        self.gui = gui
        self._registered = False

        self._bind_components()
        self._setup_event_handlers()

    def _bind_components(self) -> None:
        """Bind cross-component runtime dependencies."""
        for launcher in self.pbu.launchers.values():
            launcher.bind_environment(self.environment)

    def _setup_event_handlers(self) -> None:
        """Subscribe to operator commands emitted by the GUI."""
        self.event_bus.subscribe(EventType.OPERATOR_COMMAND, self._handle_operator_command)

    def _handle_operator_command(self, event: SimulationEvent) -> None:
        """Handle operator commands at dispatcher level."""
        data = event.data or {}
        command = data.get("command")

        if command == "toggle_pause":
            if self.clock.is_paused():
                self.clock.start()
            elif self.clock.is_running():
                self.clock.pause()
            return

        if command == "pause":
            self.clock.pause()
            return

        if command == "start":
            self.clock.start()
            return

        if command == "stop":
            self.clock.stop()
            return

        if command == "add_target":
            trajectory_type = data.get("trajectory_type", "uniform")
            target_id = data.get("id")
            trajectory_arguments = data.get("trajectory_arguments", {})
            self.environment.add_target(
                trajectory_type=trajectory_type,
                id=target_id,
                trajectory_arguments=trajectory_arguments,
                type=data.get("type", "UNKNOWN"),
                rcs=data.get("rcs", 1.0),
                rcs_fluctuation=data.get("rcs_fluctuation", 0.1)
            )
            return

        if command == "select_scenario":
            scenario = data.get("scenario")
            if isinstance(scenario, dict):
                self.environment.load_scenario(scenario)

    def register_default_components(self) -> None:
        """Register all architecture blocks with the simulation clock."""
        if self._registered:
            return

        self.clock.register_component("Environment", self.environment.update, 10)

        for index, radar in enumerate(self.radars):
            # Создаём замыкание для каждого радара
            def make_update(radar_instance):
                def update(dt):
                    radar_instance.dt = dt
                    radar_instance.update_scan()
                    # Здесь нужно реализовать логику обнаружения целей
                    # через environment.get_noisy_measurement
                    self._update_radar(radar_instance, dt)

                return update

            self.clock.register_component(
                name=f"Radar_{radar.id}",
                update_func=make_update(radar),
                priority=20 + index
            )

        self.clock.register_component("PBU", self.pbu.update, 30)

        for launcher_id, launcher in self.pbu.launchers.items():
            self.clock.register_component(
                name=f"Launcher_{launcher_id}",
                update_func=launcher.update,
                priority=40
            )

        self._registered = True
        logger.info("Dispatcher: default architecture components registered")

    def _update_radar(self, radar: RadarSystem, dt: float) -> None:
        """Обновление одного радара."""
        radar.dt = dt
        radar.update_scan()

        az, el = radar.get_beam_direction()

        # Сканирование по дальности
        for r in np.arange(0, radar.r_max, radar.dr):
            beam_pos = radar.get_beam_position(r)

            for target_id, target in self.environment.targets.items():
                if target.status != TargetStatus.ACTIVE:
                    continue

                if radar.is_target_in_beam(target.position, r):
                    measurement = self.environment.get_noisy_measurement(target_id, radar.position)
                    if measurement is not None:
                        track_id = radar.process_measurement(measurement)
                        if track_id is not None:
                            # Отправка данных в PBU
                            for track_data in radar.get_track_data():
                                if track_data["track_id"] == track_id:
                                    self.pbu.process_radar_track(radar.id, track_data)
                                    break
                        break

        radar.update_tracks()

    def update_gui_cache(self) -> None:
        """Push the latest state snapshot into the GUI cache."""
        if self.gui is None:
            return

        self.gui.update_cache(
            targets=self.environment.targets,
            missiles=self.environment.missiles,
            radars=self.radars,
            launchers=self.pbu.launchers,
            defended_assets=self.pbu.defended_assets,
            simulation_time=self.clock.current_time,
        )
