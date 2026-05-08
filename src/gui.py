"""Compact GUI for the ADS simulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import Button

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
        self.ax_objects: Optional[plt.Axes] = None
        self.widgets: Dict[str, Any] = {}
        self.row_buttons: List[Button] = []
        self.connection_ids: List[int] = []
        self.is_panning = False
        self.pan_start: Optional[Tuple[float, float]] = None
        self.selected_object_id: Optional[int] = None
        self.selected_object_type: Optional[str] = None
        self.pause_callback: Optional[Callable[[], None]] = None
        self.variant_names: List[str] = ["baseline"]
        self.variant_index: int = 0
        self.scenario_state: Any = None
        self._config_window: Optional[plt.Figure] = None
        self._config_widgets: Dict[str, Any] = {}
        self._add_modal: Optional[plt.Figure] = None
        self._add_modal_widgets: Dict[str, Any] = {}
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

    # --- variant / scenario helpers -------------------------------------------------

    def set_scenario_state(self, scenario_state: Any) -> None:
        """Bind the dispatcher's ScenarioState to the GUI."""
        self.scenario_state = scenario_state
        self._render_objects_panel()

    def set_variant_provider(self, provider: Callable[[], List[str]]) -> None:
        """Function used to repopulate the variant list (baseline + variants/*.yaml)."""
        self._variant_provider = provider
        self.refresh_variant_list()

    def refresh_variant_list(self) -> None:
        if hasattr(self, "_variant_provider"):
            try:
                self.variant_names = list(self._variant_provider())
            except Exception:
                self.variant_names = ["baseline"]
        if not self.variant_names:
            self.variant_names = ["baseline"]
        self.variant_index = min(self.variant_index, len(self.variant_names) - 1)
        self._update_variant_label()

    def _current_variant(self) -> str:
        if 0 <= self.variant_index < len(self.variant_names):
            return self.variant_names[self.variant_index]
        return "baseline"

    def _update_variant_label(self) -> None:
        button = self.widgets.get("variant_dropdown")
        if button is None:
            return
        label = f"▼ {self._current_variant()}"
        button.label.set_text(label)
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    # --- event handling -------------------------------------------------

    def _on_event(self, event: SimulationEvent) -> None:
        if self.cache.events:
            last = self.cache.events[-1]
            if last.event_type == event.event_type and last.source_id == event.source_id and last.target_id == event.target_id:
                return
        self.cache.events.append(event)
        if len(self.cache.events) > 30:
            self.cache.events.pop(0)
        self.cache.dirty = True

    # --- layout ---------------------------------------------------------

    def initialize(self, figsize: Tuple[int, int] = (16, 10)) -> None:
        if self.fig is None:
            self.fig = plt.figure(figsize=figsize, facecolor=self.colors["background"])
        else:
            self.fig.clf()
            self.fig.set_size_inches(*figsize, forward=True)
            self.fig.set_facecolor(self.colors["background"])
        self.widgets = {}
        self.row_buttons = []
        self.ax_xy = self.ax_xz = self.ax_3d = self.ax_hud = self.ax_objects = None

        if self.view_mode == ViewMode.SPLIT:
            self.ax_xy = self.fig.add_axes([0.05, 0.18, 0.30, 0.70])
            self.ax_xz = self.fig.add_axes([0.38, 0.18, 0.30, 0.70])
        elif self.view_mode == ViewMode.TOP_DOWN:
            self.ax_xy = self.fig.add_axes([0.05, 0.18, 0.62, 0.70])
        elif self.view_mode == ViewMode.SIDE:
            self.ax_xz = self.fig.add_axes([0.05, 0.18, 0.62, 0.70])
        else:
            self.ax_3d = self.fig.add_axes([0.05, 0.18, 0.62, 0.70], projection="3d")

        self.ax_hud = self.fig.add_axes([0.71, 0.55, 0.26, 0.33])
        self.ax_hud.set_facecolor(self.colors["hud_background"])
        self.ax_hud.patch.set_alpha(0.85)
        self.ax_hud.axis("off")

        self.ax_objects = self.fig.add_axes([0.71, 0.04, 0.26, 0.40])
        self.ax_objects.set_facecolor(self.colors["hud_background"])
        self.ax_objects.patch.set_alpha(0.85)
        self.ax_objects.axis("off")

        self.widgets["pause"] = Button(
            self.fig.add_axes([0.71, 0.93, 0.12, 0.045]), "Pause",
            color="#2a2a5e", hovercolor="#3a3a7e",
        )
        self.widgets["reset"] = Button(
            self.fig.add_axes([0.85, 0.93, 0.12, 0.045]), "Reset",
            color="#5a2a2a", hovercolor="#7a3a3a",
        )

        self.widgets["variant_dropdown"] = Button(
            self.fig.add_axes([0.71, 0.495, 0.16, 0.04]), f"▼ {self._current_variant()}",
            color="#0f1037", hovercolor="#181b4e",
        )
        self.widgets["save_variant"] = Button(
            self.fig.add_axes([0.71, 0.45, 0.08, 0.04]), "Save",
            color="#2a2a5e", hovercolor="#3a3a7e",
        )
        self.widgets["load_variant"] = Button(
            self.fig.add_axes([0.80, 0.45, 0.08, 0.04]), "Load",
            color="#2a2a5e", hovercolor="#3a3a7e",
        )
        self.widgets["edit_objects"] = Button(
            self.fig.add_axes([0.89, 0.45, 0.08, 0.04]), "Edit",
            color="#2a5a2a", hovercolor="#3a7a3a",
        )

        self._connect_handlers()
        self._render_objects_panel()
        self._update_variant_label()

        # Persistent artist initialisation for the main 2D axes. This is what
        # keeps the render path cheap — instead of clearing and rebuilding all
        # collections every tick, we keep them around and call set_offsets /
        # set_segments / set_xy.
        self._axis_artists: Dict[str, Dict[str, Any]] = {}
        if self.ax_xy is not None:
            self._axis_artists["xy"] = self._init_axis_artists(self.ax_xy, "xy")
        if self.ax_xz is not None:
            self._axis_artists["xz"] = self._init_axis_artists(self.ax_xz, "xz")

    def _init_axis_artists(self, ax: plt.Axes, kind: str) -> Dict[str, Any]:
        """Create the long-lived artists for one 2D axis.

        ``kind`` is ``"xy"`` (top-down — uses Y as second dimension) or
        ``"xz"`` (side — uses Z).
        """
        is_xy = kind == "xy"

        # Static styling — set once, never re-applied per frame.
        ax.set_facecolor(self.colors["background"])
        ax.grid(self.layer_visibility[DisplayLayer.GRID], alpha=0.3, color=self.colors["grid"])
        ax.tick_params(colors=self.colors["text"])
        ax.set_xlabel("X (m)", color=self.colors["text"])
        ax.set_ylabel("Y (m)" if is_xy else "Z (m)", color=self.colors["text"])
        ax.set_title("XY Plan View" if is_xy else "XZ Altitude View", color=self.colors["text"])
        for spine in ax.spines.values():
            spine.set_color(self.colors["text"])
        ax.set_aspect("equal")

        empty = np.empty((0, 2))
        artists: Dict[str, Any] = {}

        artists["radars"] = ax.scatter(
            [], [], marker="s", s=100, color=self.colors["radar"],
            edgecolors="black", linewidths=1.0, zorder=4
        )
        artists["launchers"] = ax.scatter(
            [], [], marker="^", s=120, color=self.colors["launcher"],
            edgecolors="black", linewidths=1.0, zorder=4
        )
        artists["assets"] = ax.scatter(
            [], [], marker="*", s=220, color=self.colors["defended_asset"],
            edgecolors="black", linewidths=1.0, zorder=4
        )
        artists["targets"] = ax.scatter(
            [], [], marker="o", s=50, edgecolors="black", linewidths=1.0, zorder=5
        )
        artists["missiles"] = ax.scatter(
            [], [], marker="x", s=35, color=self.colors["missile"], zorder=5
        )
        artists["track_diamonds"] = ax.scatter(
            [], [], marker="D", s=34, facecolors="none", linewidths=1.2, zorder=4
        )

        artists["target_trails"] = LineCollection(
            [], linewidths=1, alpha=0.3, zorder=2
        )
        ax.add_collection(artists["target_trails"])

        artists["missile_trails"] = LineCollection(
            [], linewidths=1, alpha=0.35,
            colors=[self.colors["missile_trail"]], zorder=2
        )
        ax.add_collection(artists["missile_trails"])

        artists["track_history"] = LineCollection(
            [], linewidths=1.1, alpha=0.85, linestyles="--", zorder=3
        )
        ax.add_collection(artists["track_history"])

        artists["launcher_lines"] = LineCollection(
            [], linewidths=0.8, alpha=0.18, linestyles=":",
            colors=[self.colors["launcher"]], zorder=1
        )
        ax.add_collection(artists["launcher_lines"])

        # Variable-count patches/lines kept in lists; resized lazily.
        artists["radar_range_patches"] = []
        artists["radar_beam_polys"] = []
        artists["radar_beam_lines"] = []

        # Labels are Text artists keyed by id; created lazily, hidden when
        # the underlying object disappears.
        artists["labels_radars"] = {}
        artists["labels_launchers"] = {}
        artists["labels_assets"] = {}
        artists["labels_targets"] = {}

        return artists

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
        self.widgets["reset"].on_clicked(lambda _e: self._on_reset_clicked())
        self.widgets["variant_dropdown"].on_clicked(lambda _e: self._cycle_variant())
        self.widgets["save_variant"].on_clicked(lambda _e: self._save_variant())
        self.widgets["load_variant"].on_clicked(lambda _e: self._load_variant())
        self.widgets["edit_objects"].on_clicked(lambda _e: self._open_config_window())

    def _on_reset_clicked(self) -> None:
        self.camera.reset()
        self._emit_operator_command("reset_simulation")

    def _cycle_variant(self) -> None:
        """Open a Tk popup listing all available variants for selection."""
        self.refresh_variant_list()
        if not self.variant_names:
            self.set_status_message("No variants available.")
            return

        tk, _ = self._ensure_tk()
        if tk is None:
            # Fallback: cycle through if Tk is unavailable.
            self.variant_index = (self.variant_index + 1) % len(self.variant_names)
            self._update_variant_label()
            return

        try:
            root = tk._default_root or tk.Tk()
        except Exception:
            root = tk.Tk()
        if tk._default_root is None:
            root.withdraw()

        popup = tk.Toplevel(root)
        popup.title("Select variant")
        popup.geometry("280x260")
        popup.transient(root)

        tk.Label(
            popup,
            text="Variant = starting scenario.\n"
                 "'baseline' is the layout from config.yaml.\n"
                 "Other items are saved variants.",
            justify="left",
            padx=10, pady=8,
        ).pack(anchor="w")

        listbox = tk.Listbox(popup, height=8, font=("TkDefaultFont", 10))
        listbox.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        for name in self.variant_names:
            listbox.insert("end", name)
        if 0 <= self.variant_index < len(self.variant_names):
            listbox.selection_set(self.variant_index)
            listbox.see(self.variant_index)

        def apply_and_close() -> None:
            sel = listbox.curselection()
            if sel:
                self.variant_index = int(sel[0])
                self._update_variant_label()
                self.set_status_message(
                    f"Variant '{self._current_variant()}' selected. "
                    "Press Load to apply (then RESET)."
                )
            popup.destroy()

        listbox.bind("<Double-Button-1>", lambda _e: apply_and_close())

        btns = tk.Frame(popup)
        btns.pack(fill="x", padx=10, pady=(0, 10))
        tk.Button(btns, text="Select", command=apply_and_close).pack(side="left")
        tk.Button(btns, text="Cancel", command=popup.destroy).pack(side="left", padx=6)

    # --- mouse / keyboard --------------------------------------------------

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

    # --- operator commands ------------------------------------------------

    def _emit_operator_command(self, command: str, **payload: Any) -> None:
        self.event_bus.publish(
            SimulationEvent(
                event_type=EventType.OPERATOR_COMMAND,
                source_id="gui",
                data={"command": command, **payload},
            )
        )

    def _save_variant(self) -> None:
        """Open a Tk prompt for a variant name and save the staged scenario."""
        tk, _ = self._ensure_tk()
        if tk is None:
            self._emit_operator_command("save_variant", name=self._current_variant())
            return

        try:
            root = tk._default_root or tk.Tk()
        except Exception:
            root = tk.Tk()
        if tk._default_root is None:
            root.withdraw()

        popup = tk.Toplevel(root)
        popup.title("Save variant")
        popup.geometry("340x180")
        popup.transient(root)

        tk.Label(
            popup,
            text="Save current scenario_state as a new variant.\n"
                 "Pick any name except 'baseline'.",
            justify="left",
            padx=10, pady=8,
        ).pack(anchor="w")

        default_name = self._current_variant()
        if default_name == "baseline":
            default_name = "my_layout"
        var = tk.StringVar(value=default_name)
        entry = tk.Entry(popup, textvariable=var, width=30)
        entry.pack(padx=10, pady=4)
        entry.select_range(0, "end")
        entry.focus_set()

        def submit() -> None:
            name = var.get().strip()
            if not name or name == "baseline":
                self.set_status_message("Use a name other than 'baseline'.")
                return
            self._emit_operator_command("save_variant", name=name)
            self.refresh_variant_list()
            if name in self.variant_names:
                self.variant_index = self.variant_names.index(name)
                self._update_variant_label()
                self.set_status_message(f"Variant '{name}' saved.")
            popup.destroy()

        btns = tk.Frame(popup)
        btns.pack(pady=10)
        tk.Button(btns, text="Save", command=submit).pack(side="left", padx=4)
        tk.Button(btns, text="Cancel", command=popup.destroy).pack(side="left", padx=4)
        entry.bind("<Return>", lambda _e: submit())

    def _load_variant(self) -> None:
        self._emit_operator_command("load_variant", name=self._current_variant())

    # --- object selection -------------------------------------------------

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

    # --- object panel rendering -------------------------------------------

    def _render_objects_panel(self) -> None:
        if self.ax_objects is None:
            return
        ax = self.ax_objects
        ax.clear()
        ax.set_facecolor(self.colors["hud_background"])
        ax.patch.set_alpha(0.85)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        sections = self._scenario_sections()
        lines = [
            "OBJECTS  (click 'Edit' to change)",
            "Edits are auto-staged.",
            "Press RESET to apply them.",
            "",
        ]
        for label, entries in sections:
            lines.append(f"{label}: {len(entries)}")
            for entry in entries[:3]:
                lines.append(f"  {self._summarize_entry(label, entry)}")
            if len(entries) > 3:
                lines.append(f"  … {len(entries) - 3} more")
            lines.append("")

        y = 0.97
        for line in lines:
            ax.text(0.04, y, line, transform=ax.transAxes,
                    color=self.colors["text"], fontsize=8,
                    verticalalignment="top", fontfamily="monospace")
            y -= 0.045

    def _scenario_sections(self) -> List[Tuple[str, List[Dict[str, Any]]]]:
        if self.scenario_state is None:
            return [("Radars", []), ("Launchers", []), ("Targets", []), ("Assets", [])]
        return [
            ("Radars", list(getattr(self.scenario_state, "radars", []))),
            ("Launchers", list(getattr(self.scenario_state, "launchers", []))),
            ("Targets", list(getattr(self.scenario_state, "targets", []))),
            ("Assets", list(getattr(self.scenario_state, "assets", []))),
        ]

    def _summarize_entry(self, kind: str, entry: Dict[str, Any]) -> str:
        if kind == "Targets":
            return (
                f"T{entry.get('id')} {entry.get('type', '?')} "
                f"({entry.get('x', 0):.0f},{entry.get('y', 0):.0f},{entry.get('z', 0):.0f})"
            )
        if kind == "Assets":
            return f"({entry.get('x', 0):.0f},{entry.get('y', 0):.0f},{entry.get('z', 0):.0f})"
        prefix = {"Radars": "R", "Launchers": "L"}.get(kind, "?")
        return (
            f"{prefix}{entry.get('id')} "
            f"({entry.get('x', 0):.0f},{entry.get('y', 0):.0f},{entry.get('z', 0):.0f})"
        )

    # --- Tkinter-based scenario editor ----------------------------------------
    # Native Tk widgets sidestep matplotlib's reentrant-callback crashes and
    # render text/buttons reliably on every backend.

    def _ensure_tk(self):
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception as exc:
            logger.error(f"Tkinter unavailable: {exc}")
            return None, None
        return tk, ttk

    def pump_tk_events(self) -> None:
        """Drive Tk's event loop from the matplotlib animation tick.

        Pumps the default Tk root so every Toplevel (editor, add modal, variant
        popup) gets its events processed without anyone having to call
        ``mainloop()``. Also clears stale references for windows that were
        closed since last tick.
        """
        try:
            import tkinter as tk
        except Exception:
            return
        root = getattr(tk, "_default_root", None)
        if root is not None:
            try:
                root.update()
            except Exception:
                pass

        for attr in ("_config_window", "_add_modal"):
            window = getattr(self, attr, None)
            if window is None:
                continue
            try:
                alive = bool(window.winfo_exists())
            except Exception:
                alive = False
            if not alive:
                setattr(self, attr, None)
                if attr == "_config_window" and hasattr(self, "_tk_section_lists"):
                    self._tk_section_lists = {}

    def _open_config_window(self) -> None:
        existing = self._config_window
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.deiconify()
                    existing.lift()
                    return
            except Exception:
                pass
            self._config_window = None

        tk, ttk = self._ensure_tk()
        if tk is None:
            return

        try:
            root = tk._default_root or tk.Tk()
        except Exception:
            root = tk.Tk()
        if tk._default_root is None:
            root.withdraw()

        win = tk.Toplevel(root)
        win.title("Scenario editor — changes apply on RESET")
        win.geometry("760x560")
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_config_window())
        self._config_window = win

        header = tk.Label(
            win,
            text="Edits are auto-saved. Press RESET in the main window to apply.",
            font=("TkDefaultFont", 10, "bold"),
            anchor="w",
            padx=8, pady=6,
        )
        header.pack(fill="x")

        sections_frame = tk.Frame(win)
        sections_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self._tk_section_lists = {}
        layouts = [
            ("radars", "Radars", 0, 0),
            ("launchers", "Launchers", 0, 1),
            ("assets", "Defended assets", 1, 0),
            ("targets", "Targets", 1, 1),
        ]
        for kind, label, row, col in layouts:
            frame = tk.LabelFrame(sections_frame, text=label, padx=4, pady=4)
            frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            sections_frame.grid_rowconfigure(row, weight=1)
            sections_frame.grid_columnconfigure(col, weight=1)

            listbox = tk.Listbox(frame, height=8, font=("TkFixedFont", 10))
            listbox.pack(fill="both", expand=True, side="top")

            btn_frame = tk.Frame(frame)
            btn_frame.pack(fill="x", side="bottom", pady=(4, 0))
            add_btn = tk.Button(btn_frame, text="+ Add",
                                command=lambda k=kind: self._open_add_modal(k))
            add_btn.pack(side="left")
            del_btn = tk.Button(btn_frame, text="× Delete selected",
                                command=lambda k=kind: self._on_remove_selected(k))
            del_btn.pack(side="left", padx=(6, 0))

            self._tk_section_lists[kind] = listbox

        footer = tk.Frame(win)
        footer.pack(fill="x", padx=8, pady=6)
        close_btn = tk.Button(footer, text="Close",
                              command=lambda: self._close_config_window())
        close_btn.pack(side="right")

        self._refresh_config_window()

    def _on_remove_selected(self, kind: str) -> None:
        listbox = self._tk_section_lists.get(kind) if hasattr(self, "_tk_section_lists") else None
        if listbox is None:
            return
        selection = listbox.curselection()
        if not selection:
            self.set_status_message("Select a row first.")
            return
        index = int(selection[0])

        sections = self._scenario_sections()
        kind_to_section = {"radars": 0, "launchers": 1, "targets": 2, "assets": 3}
        entries = sections[kind_to_section[kind]][1]
        if index >= len(entries):
            return

        if kind == "assets":
            self._emit_operator_command("remove_asset", index=index)
        else:
            entry_id = entries[index].get("id")
            cmd = {"radars": "remove_radar", "launchers": "remove_launcher",
                   "targets": "remove_target"}[kind]
            self._emit_operator_command(cmd, id=entry_id)

        self._render_objects_panel()
        self._refresh_config_window()

    def _refresh_config_window(self) -> None:
        if self._config_window is None or not hasattr(self, "_tk_section_lists"):
            return
        try:
            if not self._config_window.winfo_exists():
                self._config_window = None
                return
        except Exception:
            self._config_window = None
            return

        sections = self._scenario_sections()
        data = {
            "radars": sections[0][1],
            "launchers": sections[1][1],
            "targets": sections[2][1],
            "assets": sections[3][1],
        }
        for kind, listbox in self._tk_section_lists.items():
            try:
                listbox.delete(0, "end")
            except Exception:
                continue
            for index, entry in enumerate(data[kind]):
                listbox.insert("end", self._format_row_tk(kind, entry, index))

    def _format_row_tk(self, kind: str, entry: Dict[str, Any], index: int) -> str:
        if kind == "radars":
            return f"R{entry.get('id'):>2}   ({entry.get('x'):>6.0f}, {entry.get('y'):>6.0f}, {entry.get('z'):>5.0f})"
        if kind == "launchers":
            return (
                f"L{entry.get('id'):>2}   "
                f"({entry.get('x'):>6.0f}, {entry.get('y'):>6.0f}, {entry.get('z'):>5.0f})   "
                f"ammo={entry.get('missile_amount', '-')}"
            )
        if kind == "targets":
            return (
                f"T{entry.get('id'):>2}  {str(entry.get('type', '?'))[:10]:<10}  "
                f"({entry.get('x'):>6.0f}, {entry.get('y'):>6.0f}, {entry.get('z'):>5.0f})  "
                f"v=({entry.get('vx'):>5.0f}, {entry.get('vy'):>5.0f}, {entry.get('vz'):>4.0f})"
            )
        return f"A{index:>2}   ({entry.get('x'):>6.0f}, {entry.get('y'):>6.0f}, {entry.get('z'):>5.0f})"

    def _close_config_window(self) -> None:
        win = self._config_window
        self._config_window = None
        if hasattr(self, "_tk_section_lists"):
            self._tk_section_lists = {}
        if win is None:
            return
        try:
            win.destroy()
        except Exception:
            pass
        self._render_objects_panel()

    def _open_add_modal(self, kind: str) -> None:
        existing = self._add_modal
        if existing is not None:
            try:
                existing.destroy()
            except Exception:
                pass
            self._add_modal = None

        tk, _ = self._ensure_tk()
        if tk is None:
            return

        parent = self._config_window if self._config_window is not None else (tk._default_root or tk.Tk())
        win = tk.Toplevel(parent)
        win.title(f"Add {kind[:-1] if kind != 'assets' else 'asset'}")
        win.transient(parent)
        win.geometry("360x320")
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_add_modal())
        self._add_modal = win

        if kind == "radars":
            fields = [("x", "1500"), ("y", "5000"), ("z", "5")]
        elif kind == "launchers":
            fields = [("x", "5000"), ("y", "5000"), ("z", "0"),
                      ("missile_amount", "8"), ("speed", "1000")]
        elif kind == "targets":
            fields = [("x", "1000"), ("y", "5000"), ("z", "1500"),
                      ("vx", "250"), ("vy", "0"), ("vz", "0"),
                      ("type", "FIGHTER")]
        else:
            fields = [("x", "5000"), ("y", "5000"), ("z", "0")]

        tk.Label(win, text=f"Add new {kind[:-1] if kind != 'assets' else 'asset'}",
                 font=("TkDefaultFont", 11, "bold")).pack(pady=(8, 8))

        form = tk.Frame(win)
        form.pack(padx=12, pady=4, fill="x")

        entries: Dict[str, Any] = {}
        for index, (name, default) in enumerate(fields):
            tk.Label(form, text=f"{name}:").grid(row=index, column=0, sticky="e", padx=(0, 6), pady=2)
            var = tk.StringVar(value=default)
            entry = tk.Entry(form, textvariable=var, width=20)
            entry.grid(row=index, column=1, sticky="we", pady=2)
            entries[name] = var
        form.grid_columnconfigure(1, weight=1)

        btns = tk.Frame(win)
        btns.pack(pady=10)
        tk.Button(btns, text="Add",
                  command=lambda: self._submit_add_modal(kind, entries)).pack(side="left", padx=4)
        tk.Button(btns, text="Cancel",
                  command=lambda: self._close_add_modal()).pack(side="left", padx=4)

    def _submit_add_modal(self, kind: str, entries: Dict[str, Any]) -> None:
        payload: Dict[str, Any] = {}
        for name, var in entries.items():
            value = var.get().strip()
            if name == "type":
                payload[name] = value.upper() or "FIGHTER"
            else:
                try:
                    payload[name] = float(value) if value else 0.0
                except ValueError:
                    self.set_status_message(f"Invalid {name}: {value}")
                    return

        if kind == "radars":
            self._emit_operator_command("add_radar", **payload)
        elif kind == "launchers":
            self._emit_operator_command("add_launcher", **payload)
        elif kind == "targets":
            self._emit_operator_command("add_target_staged", **payload)
        elif kind == "assets":
            self._emit_operator_command("add_asset", **payload)

        self._render_objects_panel()
        self._refresh_config_window()
        self._close_add_modal()

    def _close_add_modal(self) -> None:
        modal = self._add_modal
        self._add_modal = None
        if modal is None:
            return
        try:
            modal.destroy()
        except Exception:
            pass

    # --- cache + render --------------------------------------------------

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
        self.cache.dirty = True

    def render(self) -> None:
        if self.fig is None:
            return
        if self.camera.follow_target_id in self.cache.targets:
            self.camera.follow_target(self.cache.targets[self.camera.follow_target_id]["position"])
        if self.ax_xy is not None:
            self.ax_xy.set_xlim(*self.camera.xlim)
            self.ax_xy.set_ylim(*self.camera.ylim)
            self._update_axis_artists(self.ax_xy, self._axis_artists.get("xy"), dim_y=1)
        if self.ax_xz is not None:
            self.ax_xz.set_xlim(*self.camera.xlim)
            self.ax_xz.set_ylim(*self.camera.zlim)
            self._update_axis_artists(self.ax_xz, self._axis_artists.get("xz"), dim_y=2)
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

    def _update_axis_artists(self, ax: plt.Axes, artists: Optional[Dict[str, Any]],
                              dim_y: int) -> None:
        """Update persistent collections from cache. Hot path; keep cheap."""
        if artists is None:
            return

        # Helper that flattens cached positions to (x, dim) pairs.
        def offsets_from(positions: List[np.ndarray]) -> np.ndarray:
            if not positions:
                return np.empty((0, 2))
            arr = np.asarray(positions, dtype=np.float64)
            return np.column_stack((arr[:, 0], arr[:, dim_y]))

        # ---- radars
        radar_ids = sorted(self.cache.radars.keys())
        radar_positions = [self.cache.radars[rid]["position"] for rid in radar_ids]
        artists["radars"].set_offsets(offsets_from(radar_positions))
        edge = []
        sizes = []
        for rid in radar_ids:
            sel = (rid == self.selected_object_id and self.selected_object_type == "radar")
            edge.append("white" if sel else "black")
            sizes.append(135 if sel else 100)
        if radar_ids:
            artists["radars"].set_edgecolors(edge)
            artists["radars"].set_sizes(sizes)
        self._sync_radar_circles(ax, artists, radar_ids, dim_y)
        self._sync_radar_beams(ax, artists, radar_ids, dim_y)

        # ---- launchers
        launcher_ids = sorted(self.cache.launchers.keys())
        artists["launchers"].set_offsets(
            offsets_from([self.cache.launchers[lid]["position"] for lid in launcher_ids])
        )
        if launcher_ids:
            l_edge, l_sizes = [], []
            for lid in launcher_ids:
                sel = (lid == self.selected_object_id and self.selected_object_type == "launcher")
                l_edge.append("white" if sel else "black")
                l_sizes.append(145 if sel else 120)
            artists["launchers"].set_edgecolors(l_edge)
            artists["launchers"].set_sizes(l_sizes)

        # ---- assets
        asset_positions = [np.asarray(a) for a in self.cache.defended_assets]
        artists["assets"].set_offsets(offsets_from(asset_positions))
        if asset_positions:
            a_edge, a_sizes = [], []
            for index in range(len(asset_positions)):
                sel = (index == self.selected_object_id and self.selected_object_type == "asset")
                a_edge.append("white" if sel else "black")
                a_sizes.append(260 if sel else 220)
            artists["assets"].set_edgecolors(a_edge)
            artists["assets"].set_sizes(a_sizes)

        # ---- targets
        target_ids = sorted(self.cache.targets.keys())
        artists["targets"].set_offsets(
            offsets_from([self.cache.targets[tid]["position"] for tid in target_ids])
        )
        if target_ids:
            t_colors, t_edge, t_sizes = [], [], []
            for tid in target_ids:
                t_colors.append(self._get_target_color(tid))
                sel = (tid == self.selected_object_id and self.selected_object_type == "target")
                t_edge.append("white" if sel else "black")
                t_sizes.append(95 if sel else 50)
            artists["targets"].set_facecolors(t_colors)
            artists["targets"].set_edgecolors(t_edge)
            artists["targets"].set_sizes(t_sizes)

        # ---- missiles
        artists["missiles"].set_offsets(
            offsets_from([m["position"] for m in self.cache.missiles.values()])
        )

        # ---- track diamonds + history lines + launcher→target lines
        track_ids = sorted(self.cache.tracks.keys())
        track_positions = [self.cache.tracks[tid]["position"] for tid in track_ids]
        artists["track_diamonds"].set_offsets(offsets_from(track_positions))
        if track_ids:
            artists["track_diamonds"].set_edgecolors(
                [self._get_target_color(tid) for tid in track_ids]
            )

        history_segs: List[np.ndarray] = []
        history_colors: List[str] = []
        launcher_segs: List[List[Tuple[float, float]]] = []
        for tid in track_ids:
            track = self.cache.tracks[tid]
            history = track["history"]
            if len(history) >= 2:
                arr = np.asarray(history, dtype=np.float64)
                history_segs.append(np.column_stack((arr[:, 0], arr[:, dim_y])))
                history_colors.append(self._get_target_color(tid))
            launcher_id = track.get("assigned_launcher_id")
            launcher = self.cache.launchers.get(launcher_id)
            if launcher is not None:
                lp = launcher["position"]
                tp = track["position"]
                launcher_segs.append([(lp[0], lp[dim_y]), (tp[0], tp[dim_y])])
        artists["track_history"].set_segments(history_segs)
        if history_colors:
            artists["track_history"].set_color(history_colors)
        artists["launcher_lines"].set_segments(launcher_segs)

        # ---- trails
        target_trail_segs: List[np.ndarray] = []
        target_trail_colors: List[str] = []
        missile_trail_segs: List[np.ndarray] = []
        for trail_id, trail in self.cache.trails.items():
            if len(trail) < 2:
                continue
            arr = np.asarray(trail, dtype=np.float64)
            seg = np.column_stack((arr[:, 0], arr[:, dim_y]))
            if isinstance(trail_id, int):
                target_trail_segs.append(seg)
                target_trail_colors.append(self._get_target_color(trail_id))
            elif isinstance(trail_id, str) and trail_id.startswith("missile_"):
                missile_trail_segs.append(seg)
        artists["target_trails"].set_segments(target_trail_segs)
        if target_trail_colors:
            artists["target_trails"].set_color(target_trail_colors)
        artists["missile_trails"].set_segments(missile_trail_segs)

        # ---- labels (lazy text artists keyed by id)
        if self.layer_visibility[DisplayLayer.LABELS]:
            self._sync_labels(ax, artists["labels_radars"], "R", radar_ids,
                              [self.cache.radars[rid]["position"] for rid in radar_ids],
                              dim_y, self.colors["text"])
            self._sync_labels(ax, artists["labels_launchers"], "L", launcher_ids,
                              [self.cache.launchers[lid]["position"] for lid in launcher_ids],
                              dim_y, self.colors["text"])
            self._sync_labels(ax, artists["labels_assets"], "A",
                              list(range(len(asset_positions))), asset_positions,
                              dim_y, self.colors["defended_asset"])
            self._sync_labels(ax, artists["labels_targets"], "T", target_ids,
                              [self.cache.targets[tid]["position"] for tid in target_ids],
                              dim_y, None, color_per_id=self._get_target_color)
        else:
            for cache in (artists["labels_radars"], artists["labels_launchers"],
                          artists["labels_assets"], artists["labels_targets"]):
                for txt in cache.values():
                    txt.set_visible(False)

    def _sync_radar_circles(self, ax: plt.Axes, artists: Dict[str, Any],
                            radar_ids: List[int], dim_y: int) -> None:
        """Resize the persistent Circle list to match the current radar count."""
        patches = artists["radar_range_patches"]
        while len(patches) < len(radar_ids):
            circ = Circle((0, 0), 1.0, fill=False, edgecolor=self.colors["radar"],
                          alpha=0.25, linestyle="--", zorder=1)
            ax.add_patch(circ)
            patches.append(circ)
        while len(patches) > len(radar_ids):
            patches.pop().remove()

        visible = self.layer_visibility[DisplayLayer.DETECTION_RANGES]
        for circ, rid in zip(patches, radar_ids):
            radar = self.cache.radars[rid]
            pos = radar["position"]
            circ.center = (pos[0], pos[dim_y])
            circ.set_radius(radar["r_max"])
            circ.set_visible(visible)

    def _sync_radar_beams(self, ax: plt.Axes, artists: Dict[str, Any],
                          radar_ids: List[int], dim_y: int) -> None:
        """Update the persistent radar-beam polygons + ray lines."""
        polys = artists["radar_beam_polys"]
        rays = artists["radar_beam_lines"]
        while len(polys) < len(radar_ids):
            poly = Polygon([(0, 0), (0, 0), (0, 0)], closed=True,
                           color=self.colors["radar"], alpha=0.10, zorder=1)
            ax.add_patch(poly)
            polys.append(poly)
            line, = ax.plot([], [], color=self.colors["radar"], alpha=0.35, linewidth=0.6, zorder=1)
            rays.append(line)
        while len(polys) > len(radar_ids):
            polys.pop().remove()
            rays.pop().remove()

        beam_key = "beam_polygon_xy" if dim_y == 1 else "beam_polygon_xz"
        beam_axis = "y" if dim_y == 1 else "z"
        for poly, line, rid in zip(polys, rays, radar_ids):
            radar = self.cache.radars[rid]
            beam = radar.get(beam_key) or []
            if len(beam) >= 3:
                poly.set_xy(beam)
                poly.set_visible(True)
            else:
                poly.set_visible(False)
            ray_pts = radar.get("beam_points") or {}
            xs = ray_pts.get("x") or []
            ys = ray_pts.get(beam_axis) or []
            if xs and ys:
                line.set_data(xs, ys)
                line.set_visible(True)
            else:
                line.set_visible(False)

    def _sync_labels(self, ax: plt.Axes, label_cache: Dict[Any, Any], prefix: str,
                     ids: List[Any], positions: List[np.ndarray], dim_y: int,
                     default_color: Optional[str],
                     color_per_id: Optional[Callable[[Any], str]] = None) -> None:
        present = set()
        for ident, pos in zip(ids, positions):
            text = label_cache.get(ident)
            if text is None:
                colour = color_per_id(ident) if color_per_id else default_color
                text = ax.text(0, 0, f"{prefix}{ident}", color=colour or self.colors["text"],
                                fontsize=8, zorder=6)
                label_cache[ident] = text
            text.set_position((float(pos[0]) + 60, float(pos[dim_y]) + 60))
            text.set_text(f"{prefix}{ident}")
            if color_per_id:
                text.set_color(color_per_id(ident))
            text.set_visible(True)
            present.add(ident)
        for ident in list(label_cache):
            if ident not in present:
                label_cache[ident].set_visible(False)

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
