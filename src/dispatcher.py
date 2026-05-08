from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from air_environment import AirEnvironment
from event_types import EventBus, EventType, SimulationEvent
from gui import GUI
from logs import dispatcher_logger as logger
from pbu import Pbu
from radar import Radar
from simulation_clock import SimulationClock


@dataclass
class ScenarioState:
    """UI-staged configuration that will be applied on the next RESET."""
    radars: List[Dict[str, Any]] = field(default_factory=list)
    launchers: List[Dict[str, Any]] = field(default_factory=list)
    targets: List[Dict[str, Any]] = field(default_factory=list)
    assets: List[Dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "ScenarioState":
        return ScenarioState(
            radars=copy.deepcopy(self.radars),
            launchers=copy.deepcopy(self.launchers),
            targets=copy.deepcopy(self.targets),
            assets=copy.deepcopy(self.assets),
        )


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
        gui: Optional[GUI] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.clock = clock
        self.event_bus = event_bus
        self.environment = environment
        self.pbu = pbu
        self.radars = radars
        self.gui = gui
        self._registered = False
        self.variants_dir = Path(__file__).resolve().parents[1] / "variants"

        self.config: Dict[str, Any] = copy.deepcopy(config or {})
        self.scenario_state = ScenarioState()
        self._populate_scenario_state_from_runtime()

        self._bind_components()
        self._setup_event_handlers()

    def _populate_scenario_state_from_runtime(self) -> None:
        """Capture current radars/launchers/targets/assets into scenario_state."""
        radar_cfg_map = {}
        if self.config:
            for radar_id_str, radar_cfg in (self.config.get("Locator") or {}).items():
                try:
                    radar_cfg_map[int(radar_id_str)] = copy.deepcopy(radar_cfg)
                except (TypeError, ValueError):
                    continue

        launcher_cfg_map = {}
        if self.config:
            for launcher_id_str, launcher_cfg in ((self.config.get("Pbu") or {}).get("launchers") or {}).items():
                try:
                    launcher_cfg_map[int(launcher_id_str)] = copy.deepcopy(launcher_cfg)
                except (TypeError, ValueError):
                    continue

        target_cfg_map = {}
        if self.config:
            for target_id_str, target_cfg in ((self.config.get("Environment") or {}).get("targets") or {}).items():
                try:
                    target_cfg_map[int(target_id_str)] = copy.deepcopy(target_cfg)
                except (TypeError, ValueError):
                    continue

        radar_entries = []
        for radar in self.radars:
            position = radar.position.tolist()
            cfg = radar_cfg_map.get(radar.id, {})
            radar_entries.append({
                "id": radar.id,
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "config": cfg,
            })
        self.scenario_state.radars = radar_entries

        launcher_entries = []
        for launcher_id, launcher in self.pbu.launchers.items():
            position = launcher.position.tolist()
            cfg = launcher_cfg_map.get(launcher_id, {})
            launcher_entries.append({
                "id": launcher_id,
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "missile_amount": int(launcher.magazine.capacity),
                "speed": float(getattr(launcher, "speed", 1000.0)),
                "missile_type": cfg.get("missile_type", "guided missile"),
                "trigger_distance": float(cfg.get("trigger_distance", launcher.trigger_distance)),
                "explosion_range": float(cfg.get("explosion_range", launcher.explosion_range)),
                "config": cfg,
            })
        self.scenario_state.launchers = launcher_entries

        target_entries = []
        for target_id, target in self.environment.targets.items():
            cfg = target_cfg_map.get(target_id, {})
            traj_args = cfg.get("trajectory_arguments", {}) if cfg else {}
            position = traj_args.get("position", target.position.tolist())
            velocity = traj_args.get("velocity", target.velocity.tolist())
            target_entries.append({
                "id": target_id,
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "vx": float(velocity[0]),
                "vy": float(velocity[1]),
                "vz": float(velocity[2]),
                "type": cfg.get("type", target.target_type.name) if cfg else target.target_type.name,
                "rcs": float(cfg.get("rcs", target.signature.rcs)) if cfg else float(target.signature.rcs),
                "rcs_fluctuation": float(cfg.get("rcs_fluctuation", 0.1)) if cfg else 0.1,
                "trajectory_type": cfg.get("trajectory_type", "uniform") if cfg else "uniform",
            })
        self.scenario_state.targets = target_entries

        self.scenario_state.assets = [
            {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])}
            for a in self.pbu.defended_assets
        ]

        if self.gui is not None:
            self.gui.set_scenario_state(self.scenario_state)

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
        """Serialize the staged scenario_state for variant save."""
        return {
            "radars": copy.deepcopy(self.scenario_state.radars),
            "launchers": copy.deepcopy(self.scenario_state.launchers),
            "targets": copy.deepcopy(self.scenario_state.targets),
            "assets": copy.deepcopy(self.scenario_state.assets),
        }

    def _save_variant(self, raw_name: Any) -> None:
        """Persist the staged scenario_state to disk."""
        variant_name = self._sanitize_variant_name(raw_name)
        if variant_name == "baseline":
            self._set_gui_status("Cannot overwrite the baseline variant. Pick a different name.")
            return
        self.variants_dir.mkdir(parents=True, exist_ok=True)
        variant_path = self.variants_dir / f"{variant_name}.yaml"
        with open(variant_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(self._build_variant_snapshot(), file, sort_keys=False, allow_unicode=True)
        self._set_gui_status(f"Variant saved: {variant_name}")
        if self.gui is not None:
            self.gui.refresh_variant_list()

    def _load_variant(self, raw_name: Any) -> None:
        """Load a saved scenario_state snapshot. Apply on next RESET."""
        variant_name = self._sanitize_variant_name(raw_name)

        if variant_name == "baseline":
            self.scenario_state = ScenarioState()
            self._populate_scenario_state_from_runtime_config()
            if self.gui is not None:
                self.gui.set_scenario_state(self.scenario_state)
            self._set_gui_status("Loaded baseline. Press RESET to apply.")
            return

        variant_path = self.variants_dir / f"{variant_name}.yaml"
        if not variant_path.exists():
            self._set_gui_status(f"Variant not found: {variant_name}")
            return

        with open(variant_path, "r", encoding="utf-8") as file:
            snapshot = yaml.safe_load(file) or {}

        self.scenario_state = ScenarioState(
            radars=list(snapshot.get("radars") or []),
            launchers=list(snapshot.get("launchers") or []),
            targets=list(snapshot.get("targets") or []),
            assets=list(snapshot.get("assets") or []),
        )
        if self.gui is not None:
            self.gui.set_scenario_state(self.scenario_state)
        self._set_gui_status(f"Variant '{variant_name}' loaded. Press RESET to apply.")

    def _coerce_position(self, raw: Any, default: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
        """Accept positions written either as [x, y, z] or as {x: ..., y: ..., z: ...}."""
        if raw is None:
            return default
        if isinstance(raw, dict):
            return (
                float(raw.get("x", default[0])),
                float(raw.get("y", default[1])),
                float(raw.get("z", default[2])),
            )
        try:
            return (float(raw[0]), float(raw[1]), float(raw[2]))
        except (KeyError, IndexError, TypeError, ValueError):
            return default

    def _populate_scenario_state_from_runtime_config(self) -> None:
        """Reconstruct scenario_state from the original config (baseline)."""
        radar_entries = []
        for radar_id_str, radar_cfg in (self.config.get("Locator") or {}).items():
            try:
                radar_id = int(radar_id_str)
            except (TypeError, ValueError):
                continue
            x, y, z = self._coerce_position(radar_cfg.get("position"))
            radar_entries.append({
                "id": radar_id,
                "x": x,
                "y": y,
                "z": z,
                "config": copy.deepcopy(radar_cfg),
            })
        self.scenario_state.radars = radar_entries

        launcher_entries = []
        for launcher_id_str, launcher_cfg in ((self.config.get("Pbu") or {}).get("launchers") or {}).items():
            try:
                launcher_id = int(launcher_id_str)
            except (TypeError, ValueError):
                continue
            x, y, z = self._coerce_position(launcher_cfg.get("launcher_pos"))
            launcher_entries.append({
                "id": launcher_id,
                "x": x,
                "y": y,
                "z": z,
                "missile_amount": int(launcher_cfg.get("missile_amount", 8)),
                "speed": float(launcher_cfg.get("speed", 1000.0)),
                "missile_type": launcher_cfg.get("missile_type", "guided missile"),
                "trigger_distance": float(launcher_cfg.get("trigger_distance", 30.0)),
                "explosion_range": float(launcher_cfg.get("explosion_range", 120.0)),
                "config": copy.deepcopy(launcher_cfg),
            })
        self.scenario_state.launchers = launcher_entries

        target_entries = []
        for target_id_str, target_cfg in ((self.config.get("Environment") or {}).get("targets") or {}).items():
            try:
                target_id = int(target_id_str)
            except (TypeError, ValueError):
                continue
            traj_args = target_cfg.get("trajectory_arguments", {})
            x, y, z = self._coerce_position(traj_args.get("position"))
            vx, vy, vz = self._coerce_position(traj_args.get("velocity"))
            target_entries.append({
                "id": target_id,
                "x": x, "y": y, "z": z,
                "vx": vx, "vy": vy, "vz": vz,
                "type": target_cfg.get("type", "UNKNOWN"),
                "rcs": float(target_cfg.get("rcs", 1.0)),
                "rcs_fluctuation": float(target_cfg.get("rcs_fluctuation", 0.1)),
                "trajectory_type": target_cfg.get("trajectory_type", "uniform"),
            })
        self.scenario_state.targets = target_entries

        self.scenario_state.assets = []
        for asset in ((self.config.get("Pbu") or {}).get("defended_assets") or []):
            ax, ay, az = self._coerce_position(asset)
            self.scenario_state.assets.append({"x": ax, "y": ay, "z": az})

    def _next_id(self, entries: List[Dict[str, Any]]) -> int:
        used = {entry.get("id") for entry in entries if "id" in entry}
        candidate = 0
        while candidate in used:
            candidate += 1
        return candidate

    def _stage_radar(self, payload: Dict[str, Any]) -> None:
        template = self.scenario_state.radars[0] if self.scenario_state.radars else {}
        new_radar = {
            "id": payload.get("id", self._next_id(self.scenario_state.radars)),
            "x": float(payload["x"]),
            "y": float(payload["y"]),
            "z": float(payload.get("z", 0.0)),
            "config": copy.deepcopy(template.get("config", {})),
        }
        self.scenario_state.radars.append(new_radar)
        self._set_gui_status(f"Radar staged at ({new_radar['x']:.0f}, {new_radar['y']:.0f}). Press RESET to apply.")

    def _stage_launcher(self, payload: Dict[str, Any]) -> None:
        template = self.scenario_state.launchers[0] if self.scenario_state.launchers else {}
        new_launcher = {
            "id": payload.get("id", self._next_id(self.scenario_state.launchers)),
            "x": float(payload["x"]),
            "y": float(payload["y"]),
            "z": float(payload.get("z", 0.0)),
            "missile_amount": int(payload.get("missile_amount", template.get("missile_amount", 8))),
            "speed": float(payload.get("speed", template.get("speed", 1000.0))),
            "missile_type": payload.get("missile_type", template.get("missile_type", "guided missile")),
            "trigger_distance": float(payload.get("trigger_distance", template.get("trigger_distance", 30.0))),
            "explosion_range": float(payload.get("explosion_range", template.get("explosion_range", 120.0))),
            "config": copy.deepcopy(template.get("config", {})),
        }
        self.scenario_state.launchers.append(new_launcher)
        self._set_gui_status(f"Launcher staged at ({new_launcher['x']:.0f}, {new_launcher['y']:.0f}). Press RESET to apply.")

    def _stage_target(self, payload: Dict[str, Any]) -> None:
        new_target = {
            "id": payload.get("id", self._next_id(self.scenario_state.targets)),
            "x": float(payload["x"]),
            "y": float(payload["y"]),
            "z": float(payload.get("z", 1000.0)),
            "vx": float(payload.get("vx", 200.0)),
            "vy": float(payload.get("vy", 0.0)),
            "vz": float(payload.get("vz", 0.0)),
            "type": str(payload.get("type", "FIGHTER")).upper(),
            "rcs": float(payload.get("rcs", 2.0)),
            "rcs_fluctuation": float(payload.get("rcs_fluctuation", 0.1)),
            "trajectory_type": payload.get("trajectory_type", "uniform"),
        }
        self.scenario_state.targets.append(new_target)
        self._set_gui_status(f"Target staged at ({new_target['x']:.0f}, {new_target['y']:.0f}, {new_target['z']:.0f}). Press RESET to apply.")

    def _stage_asset(self, payload: Dict[str, Any]) -> None:
        new_asset = {
            "x": float(payload["x"]),
            "y": float(payload["y"]),
            "z": float(payload.get("z", 0.0)),
        }
        self.scenario_state.assets.append(new_asset)
        self._set_gui_status(f"Asset staged at ({new_asset['x']:.0f}, {new_asset['y']:.0f}). Press RESET to apply.")

    def _remove_staged(self, kind: str, identifier: Any) -> None:
        if kind == "radar":
            self.scenario_state.radars = [r for r in self.scenario_state.radars if r.get("id") != identifier]
        elif kind == "launcher":
            self.scenario_state.launchers = [l for l in self.scenario_state.launchers if l.get("id") != identifier]
        elif kind == "target":
            self.scenario_state.targets = [t for t in self.scenario_state.targets if t.get("id") != identifier]
        elif kind == "asset":
            try:
                index = int(identifier)
                if 0 <= index < len(self.scenario_state.assets):
                    self.scenario_state.assets.pop(index)
            except (TypeError, ValueError):
                return
        self._set_gui_status(f"{kind.capitalize()} removed from staging. Press RESET to apply.")

    def _reset_simulation(self) -> None:
        """Tear down runtime state and rebuild from scenario_state."""
        was_running = self.clock.is_running() or self.clock.is_paused()
        self.clock.stop()

        self.environment.reset_state(self._build_target_specs())
        self.pbu.reset_state(
            launchers_spec=self._build_launcher_specs(),
            assets_spec=[(a["x"], a["y"], a["z"]) for a in self.scenario_state.assets],
        )

        # Rebuild radars in place (mutate the shared list).
        self.radars.clear()
        for entry in self.scenario_state.radars:
            radar_cfg = copy.deepcopy(entry.get("config") or {})
            radar_cfg["position"] = {"x": float(entry["x"]), "y": float(entry["y"]), "z": float(entry["z"])}
            radar = Radar(radar_id=int(entry["id"]), event_bus=self.event_bus)
            radar.initialize_with_file_data(radar_cfg)
            self.radars.append(radar)

        for launcher in self.pbu.launchers.values():
            launcher.bind_environment(self.environment)

        self.clock.clear_components()
        self.clock.reset()
        self._registered = False
        self.register_default_components()

        if self.gui is not None:
            self.gui.cache.events.clear()
            self.gui.cache.trails.clear()
            self.gui.selected_object_id = None
            self.gui.selected_object_type = None

        if was_running:
            self.clock.start()

        self._set_gui_status("Simulation reset.")

    def _build_target_specs(self) -> List[Dict[str, Any]]:
        specs = []
        for entry in self.scenario_state.targets:
            specs.append({
                "id": int(entry["id"]),
                "trajectory_type": entry.get("trajectory_type", "uniform"),
                "trajectory_arguments": {
                    "position": [float(entry["x"]), float(entry["y"]), float(entry["z"])],
                    "velocity": [float(entry["vx"]), float(entry["vy"]), float(entry["vz"])],
                },
                "type": str(entry.get("type", "UNKNOWN")).upper(),
                "rcs": float(entry.get("rcs", 1.0)),
                "rcs_fluctuation": float(entry.get("rcs_fluctuation", 0.1)),
            })
        return specs

    def _build_launcher_specs(self) -> List[Dict[str, Any]]:
        specs = []
        for entry in self.scenario_state.launchers:
            cfg = entry.get("config") or {}
            specs.append({
                "id": int(entry["id"]),
                "launcher_pos": [float(entry["x"]), float(entry["y"]), float(entry["z"])],
                "missile_amount": int(entry.get("missile_amount", cfg.get("missile_amount", 8))),
                "speed": float(entry.get("speed", cfg.get("speed", 1000.0))),
                "missile_type": entry.get("missile_type", cfg.get("missile_type", "guided missile")),
                "trigger_distance": float(entry.get("trigger_distance", cfg.get("trigger_distance", 30.0))),
                "explosion_range": float(entry.get("explosion_range", cfg.get("explosion_range", 120.0))),
                "traverse_speed": float(cfg.get("traverse_speed", 150.0)),
                "elevation_speed": float(cfg.get("elevation_speed", 70.0)),
                "reload_time": float(cfg.get("reload_time", 12.0)),
            })
        return specs

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

        if command == "add_radar":
            self._stage_radar(data)
            return

        if command == "remove_radar":
            self._remove_staged("radar", data.get("id"))
            return

        if command == "add_launcher":
            self._stage_launcher(data)
            return

        if command == "remove_launcher":
            self._remove_staged("launcher", data.get("id"))
            return

        if command == "add_target_staged":
            self._stage_target(data)
            return

        if command == "remove_target":
            self._remove_staged("target", data.get("id"))
            return

        if command == "add_asset":
            self._stage_asset(data)
            return

        if command == "remove_asset":
            self._remove_staged("asset", data.get("index"))
            return

        if command == "reset_simulation":
            self._reset_simulation()
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

    def get_variant_names(self) -> List[str]:
        """List available variants for the GUI dropdown."""
        names = ["baseline"]
        if self.variants_dir.exists():
            for path in sorted(self.variants_dir.glob("*.yaml")):
                stem = path.stem
                if stem and stem not in names:
                    names.append(stem)
        return names

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
