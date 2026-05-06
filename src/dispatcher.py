from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from air_environment import AirEnvironment
from event_types import EventBus, EventType, SimulationEvent
from gui import GUI
from logs import dispatcher_logger as logger
from pbu import Pbu
from radar import Radar
from simulation_clock import SimulationClock

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
        radars: List[Radar],
        gui: Optional[GUI] = None
    ):
        self.clock = clock
        self.event_bus = event_bus
        self.environment = environment
        self.pbu = pbu
        self.radars = radars
        self.gui = gui
        self._registered = False
        self.variants_dir = Path(__file__).resolve().parents[1] / "variants"

        self._bind_components()
        self._setup_event_handlers()

    def _bind_components(self) -> None:
        """Bind cross-component runtime dependencies."""
        for launcher in self.pbu.launchers.values():
            launcher.bind_environment(self.environment)

    def _setup_event_handlers(self) -> None:
        """Subscribe to operator commands emitted by the GUI."""
        self.event_bus.subscribe(EventType.OPERATOR_COMMAND, self._handle_operator_command)

    def _set_gui_status(self, message: str) -> None:
        """Push a short operator feedback message into the GUI HUD."""
        if self.gui is not None:
            self.gui.set_status_message(message)

    def _sanitize_variant_name(self, raw_name: Any) -> str:
        """Convert a user-provided variant label into a safe filename stem."""
        text = str(raw_name or "baseline").strip()
        safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)
        return safe or "baseline"

    def _find_radar(self, radar_id: int) -> Optional[Radar]:
        """Locate a radar by its runtime identifier."""
        for radar in self.radars:
            if radar.id == radar_id:
                return radar
        return None

    def _update_component_position(self, component_type: str, component_id: int, position: List[float]) -> None:
        """Apply a GUI-driven component relocation."""
        position_array = np.array(position, dtype=np.float64)

        if component_type == "radar":
            radar = self._find_radar(component_id)
            if radar is None:
                self._set_gui_status(f"Radar {component_id} not found")
                return
            radar.set_position(position_array)
            self._set_gui_status(f"Radar R{component_id} moved")
            return

        if component_type == "launcher":
            launcher = self.pbu.launchers.get(component_id)
            if launcher is None:
                self._set_gui_status(f"Launcher {component_id} not found")
                return
            launcher.set_position(position_array)
            self._set_gui_status(f"Launcher L{component_id} moved")
            return

        if component_type == "asset":
            if component_id < 0 or component_id >= len(self.pbu.defended_assets):
                self._set_gui_status(f"Asset {component_id} not found")
                return
            self.pbu.defended_assets[component_id] = position_array
            if component_id == 0:
                self.pbu.threat_assessor.defended_position = position_array.copy()
            self._set_gui_status(f"Asset A{component_id} moved")
            return

        self._set_gui_status(f"Unsupported component type: {component_type}")

    def _build_variant_snapshot(self) -> Dict[str, Any]:
        """Capture GUI-editable runtime settings into a serializable snapshot."""
        return {
            "radars": {
                str(radar.id): radar.get_configuration_snapshot()
                for radar in self.radars
            },
            "launchers": {
                str(launcher_id): {
                    "position": launcher.position.tolist(),
                }
                for launcher_id, launcher in self.pbu.launchers.items()
            },
            "defended_assets": [asset.tolist() for asset in self.pbu.defended_assets],
        }

    def _save_variant(self, raw_name: Any) -> None:
        """Persist the current GUI-editable settings to disk."""
        variant_name = self._sanitize_variant_name(raw_name)
        self.variants_dir.mkdir(parents=True, exist_ok=True)
        variant_path = self.variants_dir / f"{variant_name}.yaml"
        with open(variant_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(self._build_variant_snapshot(), file, sort_keys=False, allow_unicode=True)
        self._set_gui_status(f"Variant saved: {variant_name}")

    def _load_variant(self, raw_name: Any) -> None:
        """Load a previously saved GUI-editable settings snapshot."""
        variant_name = self._sanitize_variant_name(raw_name)
        variant_path = self.variants_dir / f"{variant_name}.yaml"
        if not variant_path.exists():
            self._set_gui_status(f"Variant not found: {variant_name}")
            return

        with open(variant_path, "r", encoding="utf-8") as file:
            snapshot = yaml.safe_load(file) or {}

        for radar_id_str, radar_snapshot in (snapshot.get("radars") or {}).items():
            radar = self._find_radar(int(radar_id_str))
            if radar is not None:
                radar.apply_configuration_snapshot(radar_snapshot)

        for launcher_id_str, launcher_snapshot in (snapshot.get("launchers") or {}).items():
            launcher = self.pbu.launchers.get(int(launcher_id_str))
            if launcher is not None and "position" in launcher_snapshot:
                launcher.set_position(launcher_snapshot["position"])

        defended_assets = snapshot.get("defended_assets")
        if defended_assets:
            self.pbu.defended_assets = [np.array(asset, dtype=np.float64) for asset in defended_assets]
            self.pbu.threat_assessor.defended_position = self.pbu.defended_assets[0].copy()

        self._set_gui_status(f"Variant loaded: {variant_name}")

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
            return

        if command == "update_component_position":
            component_type = str(data.get("component_type", ""))
            component_id = data.get("component_id")
            position = data.get("position")
            if component_id is not None and isinstance(position, list) and len(position) == 3:
                self._update_component_position(component_type, int(component_id), position)
            return

        if command == "save_variant":
            self._save_variant(data.get("name"))
            return

        if command == "load_variant":
            self._load_variant(data.get("name"))
            return

    def register_default_components(self) -> None:
        """Register all architecture blocks with the simulation clock."""
        if self._registered:
            return

        self.clock.register_component("Environment", self.environment.update, 10)

        for index, radar in enumerate(self.radars):
            self.clock.register_component(
                name=f"Radar_{radar.id}",
                update_func=lambda dt, r=radar: r.update(dt, self.environment),
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

    def update_gui_cache(self) -> None:
        """Push the latest state snapshot into the GUI cache."""
        if self.gui is None:
            return

        self.gui.update_cache(
            targets=self.environment.targets,
            missiles=self.environment.missiles,
            radars=self.radars,
            launchers=self.pbu.launchers,
            track_estimates=self.pbu.get_display_targets(),
            defended_assets=self.pbu.defended_assets,
            simulation_time=self.clock.current_time,
        )
