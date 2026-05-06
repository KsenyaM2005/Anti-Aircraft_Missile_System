"""Compact GUI for the ADS simulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider, TextBox

from event_types import EventBus, EventType, SimulationEvent
from logs import gui_logger as logger


class ViewMode(Enum):
    TOP_DOWN = auto()
    SIDE = auto()
    PERSPECTIVE_3D = auto()
    SPLIT = auto()


class DisplayLayer(Enum):
    TARGETS = auto()
    MISSILES = auto()
    RADARS = auto()
    LAUNCHERS = auto()
    TRAILS = auto()
    DETECTION_RANGES = auto()
    THREAT_INDICATORS = auto()
    GRID = auto()
    LABELS = auto()


@dataclass
class RenderCache:
    targets: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    tracks: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    missiles: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    radars: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    launchers: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    defended_assets: List[np.ndarray] = field(default_factory=list)
    trails: Dict[Any, deque] = field(default_factory=dict)
    events: List[SimulationEvent] = field(default_factory=list)
    simulation_time: float = 0.0
    status_message: str = ""
    dirty: bool = True


class Camera:
    def __init__(self):
        self.center_x = 5000.0
        self.center_y = 5000.0
        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.xlim = (0.0, 10000.0)
        self.ylim = (0.0, 10000.0)
        self.zlim = (0.0, 5000.0)
        self.elevation = 30.0
        self.azimuth = -60.0
        self.follow_target_id: Optional[int] = None

    def pan(self, dx: float, dy: float) -> None:
        self.center_x -= dx / self.zoom
        self.center_y -= dy / self.zoom
        self._update_bounds()

    def zoom_in(self, factor: float = 1.2) -> None:
        self.zoom = min(self.zoom * factor, self.max_zoom)
        self._update_bounds()

    def zoom_out(self, factor: float = 1.2) -> None:
        self.zoom = max(self.zoom / factor, self.min_zoom)
        self._update_bounds()

    def _update_bounds(self) -> None:
        width = 10000 / self.zoom
        height = 10000 / self.zoom
        self.xlim = (self.center_x - width / 2, self.center_x + width / 2)
        self.ylim = (self.center_y - height / 2, self.center_y + height / 2)

    def follow_target(self, position: np.ndarray) -> None:
        self.center_x = float(position[0])
        self.center_y = float(position[1])
        self._update_bounds()

    def reset(self) -> None:
        self.center_x = 5000.0
        self.center_y = 5000.0
        self.zoom = 1.0
        self._update_bounds()
        self.follow_target_id = None


class GUI:
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or EventBus()
        self.cache = RenderCache()
        self.camera = Camera()
        self.view_mode = ViewMode.SPLIT
        self.layer_visibility = {layer: True for layer in DisplayLayer}
        self.fig: Optional[plt.Figure] = None
        self.ax_xy: Optional[plt.Axes] = None
        self.ax_xz: Optional[plt.Axes] = None
        self.ax_3d: Optional[plt.Axes] = None
        self.ax_hud: Optional[plt.Axes] = None
        self.widgets: Dict[str, Any] = {}
        self.connection_ids: List[int] = []
        self.is_panning = False
        self.pan_start: Optional[Tuple[float, float]] = None
        self.selected_object_id: Optional[int] = None
        self.selected_object_type: Optional[str] = None
        self.pause_callback: Optional[Callable[[], None]] = None
        self._suspend_position_events = False
        self.variant_name = "baseline"
        self.show_hud = True
        self.event_display_count = 4
        self.colors = {
            "background": "#0a0a2e",
            "grid": "#2a2a5e",
            "radar": "#4169E1",
            "launcher": "#FF8C00",
            "target_low": "#FFD700",
            "target_medium": "#FFA500",
            "target_high": "#FF4500",
            "target_critical": "#8B0000",
            "missile": "#00FFFF",
            "missile_trail": "#00CED1",
            "defended_asset": "#FFD700",
            "text": "#FFFFFF",
            "hud_background": "#1a1a4e",
        }
        self.display_event_types = {
            EventType.TARGET_DETECTED,
            EventType.TARGET_ASSIGNED,
            EventType.TARGET_DESTROYED,
            EventType.MISSILE_LAUNCHED,
            EventType.MISSILE_DETONATED,
            EventType.MISSILE_MISSED,
        }
        self.event_labels = {
            EventType.TARGET_DETECTED: "Target detected",
            EventType.TARGET_ASSIGNED: "Target assigned",
            EventType.TARGET_DESTROYED: "Target destroyed",
            EventType.MISSILE_LAUNCHED: "Missile launched",
            EventType.MISSILE_DETONATED: "Missile detonated",
            EventType.MISSILE_MISSED: "Missile missed",
        }
        for event_type in self.display_event_types:
            self.event_bus.subscribe(event_type, self._on_event)
        logger.info("GUI initialized")

    def _on_event(self, event: SimulationEvent) -> None:
        if self.cache.events:
            last = self.cache.events[-1]
            if last.event_type == event.event_type and last.source_id == event.source_id and last.target_id == event.target_id:
                return
        self.cache.events.append(event)
        if len(self.cache.events) > 30:
            self.cache.events.pop(0)
        self.cache.dirty = True

    def initialize(self, figsize: Tuple[int, int] = (16, 10)) -> None:
        if self.fig is None:
            self.fig = plt.figure(figsize=figsize, facecolor=self.colors["background"])
        else:
            self.fig.clf()
            self.fig.set_size_inches(*figsize, forward=True)
            self.fig.set_facecolor(self.colors["background"])
        self.widgets = {}
        self.ax_xy = self.ax_xz = self.ax_3d = self.ax_hud = None
        if self.view_mode == ViewMode.SPLIT:
            self.ax_xy = self.fig.add_axes([0.05, 0.18, 0.28, 0.62])
            self.ax_xz = self.fig.add_axes([0.39, 0.18, 0.28, 0.62])
        elif self.view_mode == ViewMode.TOP_DOWN:
            self.ax_xy = self.fig.add_axes([0.05, 0.18, 0.62, 0.62])
        elif self.view_mode == ViewMode.SIDE:
            self.ax_xz = self.fig.add_axes([0.05, 0.18, 0.62, 0.62])
        else:
            self.ax_3d = self.fig.add_axes([0.05, 0.18, 0.62, 0.62], projection="3d")
        self.ax_hud = self.fig.add_axes([0.71, 0.50, 0.26, 0.30])
        self.ax_hud.set_facecolor(self.colors["hud_background"])
        self.ax_hud.patch.set_alpha(0.85)
        self.ax_hud.axis("off")
        self.widgets["pause"] = Button(self.fig.add_axes([0.71, 0.88, 0.12, 0.04]), "Pause", color="#2a2a5e", hovercolor="#3a3a7e")
        self.widgets["reset"] = Button(self.fig.add_axes([0.85, 0.88, 0.12, 0.04]), "Reset", color="#2a2a5e", hovercolor="#3a3a7e")
        self.widgets["zoom"] = Slider(self.fig.add_axes([0.73, 0.83, 0.22, 0.03]), "Zoom", 0.1, 5.0, valinit=self.camera.zoom, color="#4169E1")
        self.widgets["variant_name"] = TextBox(self.fig.add_axes([0.73, 0.43, 0.22, 0.04]), "Variant", initial=self.variant_name, color="#0f1037", hovercolor="#181b4e")
        self.widgets["save_variant"] = Button(self.fig.add_axes([0.73, 0.37, 0.10, 0.04]), "Save", color="#2a2a5e", hovercolor="#3a3a7e")
        self.widgets["load_variant"] = Button(self.fig.add_axes([0.85, 0.37, 0.10, 0.04]), "Load", color="#2a2a5e", hovercolor="#3a3a7e")
        self.widgets["pos_x"] = Slider(self.fig.add_axes([0.73, 0.29, 0.22, 0.03]), "X", 0.0, 12000.0, valinit=0.0, color="#FF8C00")
        self.widgets["pos_y"] = Slider(self.fig.add_axes([0.73, 0.24, 0.22, 0.03]), "Y", 0.0, 12000.0, valinit=0.0, color="#FF8C00")
        self.widgets["pos_z"] = Slider(self.fig.add_axes([0.73, 0.19, 0.22, 0.03]), "Z", 0.0, 6000.0, valinit=0.0, color="#FF8C00")
        self._connect_handlers()

    def _connect_handlers(self) -> None:
        if self.fig is None:
            return
        for cid in self.connection_ids:
            self.fig.canvas.mpl_disconnect(cid)
        self.connection_ids = [
            self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press),
            self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release),
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move),
            self.fig.canvas.mpl_connect("scroll_event", self._on_scroll),
            self.fig.canvas.mpl_connect("key_press_event", self._on_key_press),
        ]
        if self.pause_callback is not None:
            self.widgets["pause"].on_clicked(lambda _e: self.pause_callback())
        self.widgets["reset"].on_clicked(lambda _e: self.camera.reset())
        self.widgets["zoom"].on_changed(self._on_zoom_changed)
        self.widgets["variant_name"].on_submit(self._on_variant_name_changed)
        self.widgets["save_variant"].on_clicked(lambda _e: self._save_variant())
        self.widgets["load_variant"].on_clicked(lambda _e: self._load_variant())
        self.widgets["pos_x"].on_changed(lambda value: self._on_position_slider_changed("x", value))
        self.widgets["pos_y"].on_changed(lambda value: self._on_position_slider_changed("y", value))
        self.widgets["pos_z"].on_changed(lambda value: self._on_position_slider_changed("z", value))

    def _on_mouse_press(self, event) -> None:
        if event.inaxes not in [self.ax_xy, self.ax_xz]:
            return
        if event.button == 2:
            self.is_panning = True
            self.pan_start = (event.x, event.y)
        elif event.button == 1 and event.xdata is not None and event.ydata is not None:
            self._select_object_at(event.xdata, event.ydata, event.inaxes)

    def _on_mouse_release(self, event) -> None:
        if event.button == 2:
            self.is_panning = False
            self.pan_start = None

    def _on_mouse_move(self, event) -> None:
        if not self.is_panning or self.pan_start is None or event.x is None or event.y is None:
            return
        self.camera.pan(event.x - self.pan_start[0], event.y - self.pan_start[1])
        self.pan_start = (event.x, event.y)

    def _on_scroll(self, event) -> None:
        if event.button == "up":
            self.camera.zoom_in()
        else:
            self.camera.zoom_out()
        self.widgets["zoom"].set_val(self.camera.zoom)

    def _on_key_press(self, event) -> None:
        if event.key == " ":
            self.event_bus.publish(SimulationEvent(event_type=EventType.OPERATOR_COMMAND, data={"command": "toggle_pause"}))
        elif event.key == "r":
            self.camera.reset()
        elif event.key == "t":
            modes = list(ViewMode)
            self.view_mode = modes[(modes.index(self.view_mode) + 1) % len(modes)]
            self.initialize(tuple(self.fig.get_size_inches()))
        elif event.key == "h":
            self.show_hud = not self.show_hud
        elif event.key == "1":
            self.view_mode = ViewMode.TOP_DOWN
            self.initialize(tuple(self.fig.get_size_inches()))
        elif event.key == "2":
            self.view_mode = ViewMode.SIDE
            self.initialize(tuple(self.fig.get_size_inches()))
        elif event.key == "3":
            self.view_mode = ViewMode.SPLIT
            self.initialize(tuple(self.fig.get_size_inches()))
        elif event.key == "f" and self.selected_object_type == "target":
            self.camera.follow_target_id = self.selected_object_id
        elif event.key == "escape":
            self.selected_object_id = None
            self.selected_object_type = None
            self.camera.follow_target_id = None

    def _on_zoom_changed(self, value: float) -> None:
        self.camera.zoom = float(value)
        self.camera._update_bounds()

    def _on_variant_name_changed(self, value: str) -> None:
        self.variant_name = value.strip() or "baseline"

    def _emit_operator_command(self, command: str, **payload: Any) -> None:
        self.event_bus.publish(
            SimulationEvent(
                event_type=EventType.OPERATOR_COMMAND,
                source_id="gui",
                data={"command": command, **payload},
            )
        )

    def _save_variant(self) -> None:
        self._emit_operator_command("save_variant", name=self.variant_name)

    def _load_variant(self) -> None:
        self._emit_operator_command("load_variant", name=self.variant_name)

    def _selected_position(self) -> Optional[np.ndarray]:
        if self.selected_object_type == "radar":
            radar = self.cache.radars.get(self.selected_object_id)
            return None if radar is None else np.array(radar["position"], dtype=np.float64)
        if self.selected_object_type == "launcher":
            launcher = self.cache.launchers.get(self.selected_object_id)
            return None if launcher is None else np.array(launcher["position"], dtype=np.float64)
        if self.selected_object_type == "asset":
            if self.selected_object_id is None or self.selected_object_id >= len(self.cache.defended_assets):
                return None
            return np.array(self.cache.defended_assets[self.selected_object_id], dtype=np.float64)
        return None

    def _refresh_selection_controls(self) -> None:
        if not {"pos_x", "pos_y", "pos_z"}.issubset(self.widgets):
            return
        position = self._selected_position()
        if position is None:
            return
        self._suspend_position_events = True
        self.widgets["pos_x"].set_val(float(position[0]))
        self.widgets["pos_y"].set_val(float(position[1]))
        self.widgets["pos_z"].set_val(float(position[2]))
        self._suspend_position_events = False

    def _on_position_slider_changed(self, axis: str, value: float) -> None:
        if self._suspend_position_events:
            return
        position = self._selected_position()
        if position is None or self.selected_object_id is None or self.selected_object_type is None:
            return
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        position[axis_index] = float(value)
        self._emit_operator_command(
            "update_component_position",
            component_type=self.selected_object_type,
            component_id=self.selected_object_id,
            position=position.tolist(),
        )

    def _select_object_at(self, x: float, y: float, ax: plt.Axes) -> None:
        min_dist = 50 / self.camera.zoom
        selected = None
        for radar_id, data in self.cache.radars.items():
            pos = data["position"]
            dist = np.sqrt((pos[0] - x) ** 2 + ((pos[1] if ax == self.ax_xy else pos[2]) - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                selected = ("radar", radar_id)
        for asset_id, pos in enumerate(self.cache.defended_assets):
            dist = np.sqrt((pos[0] - x) ** 2 + ((pos[1] if ax == self.ax_xy else pos[2]) - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                selected = ("asset", asset_id)
        for target_id, data in self.cache.targets.items():
            pos = data["position"]
            dist = np.sqrt((pos[0] - x) ** 2 + ((pos[1] if ax == self.ax_xy else pos[2]) - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                selected = ("target", target_id)
        for launcher_id, data in self.cache.launchers.items():
            pos = data["position"]
            dist = np.sqrt((pos[0] - x) ** 2 + ((pos[1] if ax == self.ax_xy else pos[2]) - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                selected = ("launcher", launcher_id)
        if selected is not None:
            self.selected_object_type, self.selected_object_id = selected
            self._refresh_selection_controls()

    def update_cache(
        self,
        targets: Dict[int, Any],
        missiles: Dict[int, Any],
        radars: List[Any],
        launchers: Dict[int, Any],
        defended_assets: List[np.ndarray],
        track_estimates: Optional[Dict[int, Any]] = None,
        simulation_time: Optional[float] = None
    ) -> None:
        self.cache.targets = {tid: {"position": target.position.copy(), "velocity": target.velocity.copy(), "status": target.status.name, "threat_level": getattr(target, "threat_level", "NONE"), "type": getattr(target, "target_type", "UNKNOWN").name} for tid, target in targets.items()}
        self.cache.missiles = {mid: {"position": missile.position.copy(), "velocity": missile.velocity.copy(), "status": missile.status.name, "target_id": getattr(missile, "assigned_target_id", None)} for mid, missile in missiles.items()}
        self.cache.radars = {
            radar.id: {
                "position": radar.position.copy(),
                "mode": radar.mode.name,
                "r_max": radar.r_max,
                "beam_points": {
                    "x": radar.curr_ray_x.copy() if radar.curr_ray_x else [],
                    "y": radar.curr_ray_y.copy() if radar.curr_ray_y else [],
                    "z": radar.curr_ray_z.copy() if radar.curr_ray_z else []
                },
                "beam_polygon_xy": [tuple(point) for point in getattr(radar, "curr_beam_xy", [])],
                "beam_polygon_xz": [tuple(point) for point in getattr(radar, "curr_beam_xz", [])],
            }
            for radar in radars
        }
        self.cache.launchers = {lid: {"position": launcher.position.copy(), "status": launcher.status.name, "missile_count": launcher.get_missile_count()} for lid, launcher in launchers.items()}
        self.cache.tracks = {}
        for target_id, track in (track_estimates or {}).items():
            history = [np.array(position, dtype=np.float64) for position in getattr(track, "position_history", [])]
            current_position = np.array(getattr(track, "position", np.zeros(3)), dtype=np.float64)
            if not history:
                history = [current_position.copy()]
            self.cache.tracks[target_id] = {
                "position": current_position,
                "history": history,
                "threat_level": getattr(getattr(track, "threat_level", None), "name", "NONE"),
                "engagement_status": getattr(getattr(track, "engagement_status", None), "name", "UNENGAGED"),
                "assigned_launcher_id": getattr(track, "assigned_launcher_id", None),
                "radar_id": getattr(track, "radar_id", None),
                "source_count": len(getattr(track, "source_tracks", {})),
                "source_radars": sorted(getattr(track, "source_tracks", {}).keys()),
            }
        self.cache.defended_assets = [np.array(asset, dtype=np.float64) for asset in defended_assets]
        if simulation_time is not None:
            self.cache.simulation_time = float(simulation_time)
        for target_id, target in targets.items():
            self.cache.trails.setdefault(target_id, deque(maxlen=50)).append(target.position.copy())
        for trail_id in [key for key in self.cache.trails if isinstance(key, int) and key not in targets]:
            self.cache.trails.pop(trail_id, None)
        for missile_id, missile in missiles.items():
            self.cache.trails.setdefault(f"missile_{missile_id}", deque(maxlen=30)).append(missile.position.copy())
        for trail_id in [key for key in self.cache.trails if isinstance(key, str) and key.startswith("missile_") and int(key.split("_", 1)[1]) not in missiles]:
            self.cache.trails.pop(trail_id, None)
        self._refresh_selection_controls()
        self.cache.dirty = True

    def render(self) -> None:
        if self.fig is None:
            return
        if self.camera.follow_target_id in self.cache.targets:
            self.camera.follow_target(self.cache.targets[self.camera.follow_target_id]["position"])
        if self.ax_xy is not None:
            self.ax_xy.clear()
            self._style_2d(self.ax_xy, "X (m)", "Y (m)", "XY Plan View")
            self.ax_xy.set_xlim(*self.camera.xlim)
            self.ax_xy.set_ylim(*self.camera.ylim)
            self.ax_xy.set_aspect("equal")
            self._render_top_down(self.ax_xy)
        if self.ax_xz is not None:
            self.ax_xz.clear()
            self._style_2d(self.ax_xz, "X (m)", "Z (m)", "XZ Altitude View")
            self.ax_xz.set_xlim(*self.camera.xlim)
            self.ax_xz.set_ylim(*self.camera.zlim)
            self.ax_xz.set_aspect("equal")
            self._render_side(self.ax_xz)
        if self.ax_3d is not None:
            self.ax_3d.clear()
            self.ax_3d.set_facecolor(self.colors["background"])
            self.ax_3d.set_xlim(*self.camera.xlim)
            self.ax_3d.set_ylim(*self.camera.ylim)
            self.ax_3d.set_zlim(*self.camera.zlim)
            self.ax_3d.set_xlabel("X (m)", color=self.colors["text"])
            self.ax_3d.set_ylabel("Y (m)", color=self.colors["text"])
            self.ax_3d.set_zlabel("Z (m)", color=self.colors["text"])
            self.ax_3d.set_title("3D Perspective", color=self.colors["text"])
            for radar in self.cache.radars.values():
                pos = radar["position"]
                self.ax_3d.scatter(pos[0], pos[1], pos[2], marker="s", s=100, color=self.colors["radar"])
            for launcher in self.cache.launchers.values():
                pos = launcher["position"]
                self.ax_3d.scatter(pos[0], pos[1], pos[2], marker="^", s=120, color=self.colors["launcher"])
            for asset in self.cache.defended_assets:
                self.ax_3d.scatter(asset[0], asset[1], asset[2], marker="*", s=220, color=self.colors["defended_asset"])
            for target_id, target in self.cache.targets.items():
                pos = target["position"]
                self.ax_3d.scatter(pos[0], pos[1], pos[2], marker="o", s=50, color=self._get_target_color(target_id))
            for missile in self.cache.missiles.values():
                pos = missile["position"]
                self.ax_3d.scatter(pos[0], pos[1], pos[2], marker="x", s=35, color=self.colors["missile"])
        if self.show_hud and self.ax_hud is not None:
            self._render_hud()
        self.cache.dirty = False

    def _style_2d(self, ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
        ax.set_facecolor(self.colors["background"])
        ax.grid(self.layer_visibility[DisplayLayer.GRID], alpha=0.3, color=self.colors["grid"])
        ax.tick_params(colors=self.colors["text"])
        ax.set_xlabel(xlabel, color=self.colors["text"])
        ax.set_ylabel(ylabel, color=self.colors["text"])
        ax.set_title(title, color=self.colors["text"])
        for spine in ax.spines.values():
            spine.set_color(self.colors["text"])

    def _render_top_down(self, ax: plt.Axes) -> None:
        for radar_id, radar in sorted(self.cache.radars.items()):
            pos = radar["position"]
            selected = radar_id == self.selected_object_id and self.selected_object_type == "radar"
            ax.scatter(pos[0], pos[1], marker="s", s=135 if selected else 100, color=self.colors["radar"], edgecolors="white" if selected else "black", linewidth=2 if selected else 1)
            if self.layer_visibility[DisplayLayer.DETECTION_RANGES]:
                ax.add_patch(Circle((pos[0], pos[1]), radar["r_max"], fill=False, edgecolor=self.colors["radar"], alpha=0.25, linestyle="--"))
            if radar["beam_polygon_xy"]:
                beam_xy = np.array(radar["beam_polygon_xy"], dtype=np.float64)
                ax.fill(beam_xy[:, 0], beam_xy[:, 1], color=self.colors["radar"], alpha=0.10)
                ax.plot(beam_xy[:, 0], beam_xy[:, 1], color=self.colors["radar"], alpha=0.28, linewidth=0.7)
            if radar["beam_points"]["x"]:
                ax.plot(radar["beam_points"]["x"], radar["beam_points"]["y"], color=self.colors["radar"], alpha=0.35, linewidth=0.6)
            if self.layer_visibility[DisplayLayer.LABELS]:
                ax.annotate(f"R{radar_id}", xy=(pos[0], pos[1]), xytext=(6, 6), textcoords="offset points", color=self.colors["text"], fontsize=8)
        for launcher_id, launcher in sorted(self.cache.launchers.items()):
            pos = launcher["position"]
            selected = launcher_id == self.selected_object_id and self.selected_object_type == "launcher"
            ax.scatter(pos[0], pos[1], marker="^", s=145 if selected else 120, color=self.colors["launcher"], edgecolors="white" if selected else "black", linewidth=2 if selected else 1)
            if self.layer_visibility[DisplayLayer.LABELS]:
                ax.annotate(f"L{launcher_id}", xy=(pos[0], pos[1]), xytext=(6, 6), textcoords="offset points", color=self.colors["text"], fontsize=8)
        for asset_id, asset in enumerate(self.cache.defended_assets):
            selected = asset_id == self.selected_object_id and self.selected_object_type == "asset"
            ax.scatter(asset[0], asset[1], marker="*", s=260 if selected else 220, color=self.colors["defended_asset"], edgecolors="white" if selected else "black", linewidth=2 if selected else 1)
            if self.layer_visibility[DisplayLayer.LABELS]:
                ax.annotate(f"A{asset_id}", xy=(asset[0], asset[1]), xytext=(6, 6), textcoords="offset points", color=self.colors["defended_asset"], fontsize=8)
        for trail_id, trail in self.cache.trails.items():
            if len(trail) <= 1:
                continue
            trail_arr = np.array(trail)
            if isinstance(trail_id, int):
                ax.plot(trail_arr[:, 0], trail_arr[:, 1], color=self._get_target_color(trail_id), alpha=0.3, linewidth=1)
            elif isinstance(trail_id, str) and trail_id.startswith("missile_"):
                ax.plot(trail_arr[:, 0], trail_arr[:, 1], color=self.colors["missile_trail"], alpha=0.35, linewidth=1)
        for target_id, track in sorted(self.cache.tracks.items()):
            history = np.array(track["history"], dtype=np.float64)
            if len(history) > 1:
                ax.plot(history[:, 0], history[:, 1], color=self._get_target_color(target_id), alpha=0.85, linewidth=1.1, linestyle="--")
            pos = track["position"]
            ax.scatter(pos[0], pos[1], marker="D", s=34, facecolors="none", edgecolors=self._get_target_color(target_id), linewidth=1.2)
            launcher_id = track.get("assigned_launcher_id")
            launcher = self.cache.launchers.get(launcher_id)
            if launcher is not None:
                launcher_pos = launcher["position"]
                ax.plot([launcher_pos[0], pos[0]], [launcher_pos[1], pos[1]], color=self.colors["launcher"], alpha=0.18, linewidth=0.8, linestyle=":")
        for target_id, target in sorted(self.cache.targets.items()):
            pos = target["position"]
            selected = target_id == self.selected_object_id and self.selected_object_type == "target"
            ax.scatter(pos[0], pos[1], marker="o", s=95 if selected else 50, color=self._get_target_color(target_id), edgecolors="white" if selected else "black", linewidth=2 if selected else 1)
            if self.layer_visibility[DisplayLayer.LABELS]:
                ax.annotate(f"T{target_id}", xy=(pos[0], pos[1]), xytext=(6, 6), textcoords="offset points", color=self._get_target_color(target_id), fontsize=8)
        for missile in self.cache.missiles.values():
            pos = missile["position"]
            ax.scatter(pos[0], pos[1], marker="x", s=35, color=self.colors["missile"])

    def _render_side(self, ax: plt.Axes) -> None:
        for radar_id, radar in sorted(self.cache.radars.items()):
            pos = radar["position"]
            selected = radar_id == self.selected_object_id and self.selected_object_type == "radar"
            ax.scatter(pos[0], pos[2], marker="s", s=135 if selected else 100, color=self.colors["radar"], edgecolors="white" if selected else "black", linewidth=2 if selected else 1)
            if radar["beam_polygon_xz"]:
                beam_xz = np.array(radar["beam_polygon_xz"], dtype=np.float64)
                ax.fill(beam_xz[:, 0], beam_xz[:, 1], color=self.colors["radar"], alpha=0.10)
                ax.plot(beam_xz[:, 0], beam_xz[:, 1], color=self.colors["radar"], alpha=0.28, linewidth=0.7)
            if radar["beam_points"]["x"]:
                ax.plot(radar["beam_points"]["x"], radar["beam_points"]["z"], color=self.colors["radar"], alpha=0.35, linewidth=0.6)
            if self.layer_visibility[DisplayLayer.LABELS]:
                ax.annotate(f"R{radar_id}", xy=(pos[0], pos[2]), xytext=(6, 6), textcoords="offset points", color=self.colors["text"], fontsize=8)
        for launcher_id, launcher in sorted(self.cache.launchers.items()):
            pos = launcher["position"]
            selected = launcher_id == self.selected_object_id and self.selected_object_type == "launcher"
            ax.scatter(pos[0], pos[2], marker="^", s=145 if selected else 120, color=self.colors["launcher"], edgecolors="white" if selected else "black", linewidth=2 if selected else 1)
            if self.layer_visibility[DisplayLayer.LABELS]:
                ax.annotate(f"L{launcher_id}", xy=(pos[0], pos[2]), xytext=(6, 6), textcoords="offset points", color=self.colors["text"], fontsize=8)
        for asset_id, asset in enumerate(self.cache.defended_assets):
            selected = asset_id == self.selected_object_id and self.selected_object_type == "asset"
            ax.scatter(asset[0], asset[2], marker="*", s=260 if selected else 220, color=self.colors["defended_asset"], edgecolors="white" if selected else "black", linewidth=2 if selected else 1)
        for trail_id, trail in self.cache.trails.items():
            if len(trail) <= 1:
                continue
            trail_arr = np.array(trail)
            if isinstance(trail_id, int):
                ax.plot(trail_arr[:, 0], trail_arr[:, 2], color=self._get_target_color(trail_id), alpha=0.25, linewidth=1)
            elif isinstance(trail_id, str) and trail_id.startswith("missile_"):
                ax.plot(trail_arr[:, 0], trail_arr[:, 2], color=self.colors["missile_trail"], alpha=0.35, linewidth=1)
        for target_id, track in sorted(self.cache.tracks.items()):
            history = np.array(track["history"], dtype=np.float64)
            if len(history) > 1:
                ax.plot(history[:, 0], history[:, 2], color=self._get_target_color(target_id), alpha=0.85, linewidth=1.1, linestyle="--")
            pos = track["position"]
            ax.scatter(pos[0], pos[2], marker="D", s=34, facecolors="none", edgecolors=self._get_target_color(target_id), linewidth=1.2)
        for target_id, target in sorted(self.cache.targets.items()):
            pos = target["position"]
            ax.scatter(pos[0], pos[2], marker="o", s=50, color=self._get_target_color(target_id))
        for missile in self.cache.missiles.values():
            pos = missile["position"]
            ax.scatter(pos[0], pos[2], marker="x", s=35, color=self.colors["missile"])

    def _get_target_color(self, target_id: int) -> str:
        threat = self.cache.tracks.get(target_id, {}).get(
            "threat_level",
            self.cache.targets.get(target_id, {}).get("threat_level", "NONE")
        )
        if threat == "CRITICAL":
            return self.colors["target_critical"]
        if threat == "HIGH":
            return self.colors["target_high"]
        if threat == "MEDIUM":
            return self.colors["target_medium"]
        return self.colors["target_low"]

    def _render_hud(self) -> None:
        self.ax_hud.clear()
        self.ax_hud.set_facecolor(self.colors["hud_background"])
        self.ax_hud.patch.set_alpha(0.85)
        self.ax_hud.axis("off")
        active_targets = len([t for t in self.cache.targets.values() if t.get("status") == "ACTIVE"])
        active_tracks = len([track for track in self.cache.tracks.values() if track.get("engagement_status") != "INTERCEPTED"])
        active_missiles = len([m for m in self.cache.missiles.values() if m.get("status") in ["BOOSTING", "CRUISING", "TERMINAL"]])
        lines = [f"SIM TIME: {self.cache.simulation_time:.1f}s", "", f"TARGETS: {active_targets}", f"TRACKS: {active_tracks}", f"MISSILES: {active_missiles}", "", "LAUNCHERS:"]
        for launcher_id in sorted(self.cache.launchers):
            launcher = self.cache.launchers[launcher_id]
            lines.append(f"  L{launcher_id}: {launcher.get('missile_count', 0)} | {launcher.get('status', 'IDLE')}")
        if self.cache.events:
            lines.extend(["", "RECENT EVENTS:"])
            for event in self._recent_events():
                lines.append(f"  {self._format_event(event)}")
        if self.selected_object_type is not None and self.selected_object_id is not None:
            lines.extend([
                "",
                "SELECTION:",
                f"  {self.selected_object_type.upper()} #{self.selected_object_id}",
            ])
            if self.selected_object_type == "target":
                track = self.cache.tracks.get(self.selected_object_id)
                if track is not None:
                    lines.append(f"  SOURCES: {track.get('source_count', 0)}")
                    source_radars = track.get("source_radars") or []
                    if source_radars:
                        lines.append(f"  RADARS: {','.join(f'R{rid}' for rid in source_radars)}")
        if self.cache.status_message:
            lines.extend(["", "STATUS:", f"  {self.cache.status_message}"])
        y = 0.98
        for line in lines:
            self.ax_hud.text(0.03, y, line, transform=self.ax_hud.transAxes, color=self.colors["text"], fontsize=8, verticalalignment="top", fontfamily="monospace")
            y -= 0.045

    def _recent_events(self) -> List[SimulationEvent]:
        items: List[SimulationEvent] = []
        seen = set()
        for event in reversed(self.cache.events):
            key = (event.event_type, event.target_id, event.source_id)
            if key in seen:
                continue
            seen.add(key)
            items.append(event)
            if len(items) >= self.event_display_count:
                break
        items.reverse()
        return items

    def _format_event(self, event: SimulationEvent) -> str:
        label = self.event_labels.get(event.event_type, event.event_type.name.replace("_", " ").title())
        object_id = event.target_id or event.source_id or ""
        if isinstance(object_id, str) and "_" in object_id:
            prefix, suffix = object_id.split("_", 1)
            if prefix == "target":
                object_id = f"T{suffix}"
            elif prefix == "missile":
                object_id = f"M{suffix}"
            elif prefix == "launcher":
                object_id = f"L{suffix}"
        return f"{label}: {object_id}" if object_id else label

    def set_pause_callback(self, callback: Callable) -> None:
        self.pause_callback = callback
        if "pause" in self.widgets:
            self.widgets["pause"].on_clicked(lambda _e: callback())

    def set_status_message(self, message: str) -> None:
        self.cache.status_message = message
        self.cache.dirty = True

    def show(self) -> None:
        if self.fig is None:
            self.initialize()
        plt.show()

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
