#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def run_command(command: list[str], env: dict[str, str] | None = None) -> int:
    """Run a command from the project root and stream output to the console."""
    print(f"[RUN] {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT_DIR, env=env)
    return completed.returncode


class ADSSimulatorGUI:
    """GUI-enabled ADS simulator."""

    def __init__(self, config_path: str = "config.yaml"):
        import yaml
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        from air_environment import AirEnvironment
        from dispatcher import SimulationDispatcher
        from event_types import EventBus, EventType, SimulationEvent
        from gui import GUI
        from pbu import Pbu
        from radar import Radar
        from simulation_clock import SimulationClock, SimulationState

        self.SimulationState = SimulationState

        self.plt = plt
        self.FuncAnimation = FuncAnimation
        self.EventType = EventType
        self.SimulationEvent = SimulationEvent

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        self.event_bus = EventBus()
        self.env = AirEnvironment("config_file", self.config["Environment"], self.event_bus)
        self.pbu = Pbu("config_file", self.config["Pbu"], self.event_bus)

        self.radars = []
        for radar_id, radar_cfg in self.config["Locator"].items():
            radar = Radar(radar_id=int(radar_id), event_bus=self.event_bus)
            radar.initialize_with_file_data(radar_cfg)
            self.radars.append(radar)

        self.clock = SimulationClock(
            time_step=self.config["Environment"]["time_step"],
            event_bus=self.event_bus,
        )
        self.gui = GUI(self.event_bus)
        self.gui.initialize(figsize=(16, 10))

        self.frame_count = 0
        self.skip_frames = 5
        self.animation = None

        self.dispatcher = SimulationDispatcher(
            clock=self.clock,
            event_bus=self.event_bus,
            environment=self.env,
            pbu=self.pbu,
            radars=self.radars,
            gui=self.gui,
            config=self.config,
        )
        self.dispatcher.register_default_components()
        self.gui.set_variant_provider(self.dispatcher.get_variant_names)

    def update(self, _frame: int):
        """Update callback for animation and smoke tests."""
        for _ in range(self.skip_frames):
            self.clock.tick()

        self.dispatcher.update_gui_cache()
        self.gui.render()
        # With TkAgg, Tk's mainloop drives every Toplevel automatically — no
        # explicit pump needed. We still pump once a second as a safety net in
        # case the user is on a weird backend.
        if self.frame_count % 20 == 0:
            self.gui.pump_tk_events()

        if self.clock.state == self.SimulationState.STOPPED:
            self.plt.close()
            return []

        self.frame_count += 1
        return []

    def run(self) -> None:
        """Run the interactive GUI."""
        print("\n" + "=" * 60)
        print(" AIR DEFENSE SIMULATOR - GUI MODE")
        print("=" * 60)
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  R     - Reset View")
        print("  1     - Top-Down View")
        print("  2     - Side View")
        print("  3     - Split View")
        print("  F     - Follow Selected Target")
        print("  H     - Toggle HUD")
        print("  ESC   - Clear Selection")
        print("  Left click   - Select radar / launcher / asset / target")
        print("  Middle drag  - Pan view")
        print("  Right panel  - Move selected object, save/load variants")
        print("=" * 60 + "\n")

        self.clock.start()
        self.animation = self.FuncAnimation(
            self.gui.fig,
            self.update,
            interval=50,
            blit=False,
            cache_frame_data=False,
            repeat=False,
        )

        def toggle_pause():
            self.event_bus.publish(
                self.SimulationEvent(
                    event_type=self.EventType.OPERATOR_COMMAND,
                    source_id="gui",
                    data={"command": "toggle_pause"},
                )
            )

        self.gui.set_pause_callback(toggle_pause)
        self.plt.show()
        self.clock.stop()
        self.print_summary()

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print(" SIMULATION SUMMARY")
        print("=" * 60)
        print(f"  Simulation time: {self.clock.current_time:.1f}s")
        print(f" Active targets: {len(self.env.get_active_targets())}")
        print(f" Targets destroyed: {self.env.total_targets_destroyed}")
        print(f" Missiles launched: {self.env.total_missiles_launched}")
        print(f" Tracks: {sum(len(radar.tracks) for radar in self.radars)}")
        print("=" * 60)


def run_gui(config_path: str) -> int:
    """Run the GUI mode."""
    # Force the TkAgg backend so the matplotlib figure and the Tk-based scenario
    # editor share the same event loop. Without this, on macOS matplotlib falls
    # back to the Cocoa backend and the editor's Toplevel window has no Tk
    # mainloop driving it — clicks and repaints never happen.
    import matplotlib
    try:
        matplotlib.use("TkAgg", force=True)
    except (ImportError, ModuleNotFoundError, ValueError):
        pass
    simulator = ADSSimulatorGUI(config_path)
    simulator.run()
    return 0


def run_headless(config_path: str, duration: float) -> int:
    """Run headless mode directly through the simulator."""
    from main import AirDefenseSimulator

    simulator = AirDefenseSimulator(config_path=config_path)
    simulator.initialize()
    simulator.run_headless(duration=duration)
    return 0


def run_quick(config_path: str) -> int:
    """Run a short console smoke simulation."""
    import yaml
    import numpy as np

    from air_environment import AirEnvironment
    from event_types import EventBus
    from pbu import Pbu
    from radar import Radar
    from simulation_clock import SimulationClock

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("Loading config...")
    print("Initializing components...")

    event_bus = EventBus()
    env = AirEnvironment("config_file", config["Environment"], event_bus)
    pbu = Pbu("config_file", config["Pbu"], event_bus)

    radars = []
    for radar_id, radar_cfg in config["Locator"].items():
        radar = Radar(radar_id=int(radar_id), event_bus=event_bus)
        radar.initialize_with_file_data(radar_cfg)
        radars.append(radar)
    for launcher in pbu.launchers.values():
        launcher.bind_environment(env)

    print(" Loaded:")
    print(f"  - {len(env.targets)} targets")
    print(f"  - {len(pbu.launchers)} launchers")
    print(f"  - {len(radars)} radars")

    print("\n Initial Target Positions:")
    for target_id, target in env.targets.items():
        pos = target.position
        vel = target.velocity
        print(
            f"  Target {target_id}: pos=({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}), "
            f"speed={np.linalg.norm(vel):.0f} m/s"
        )

    print("\n Launcher Status:")
    for launcher_id, launcher in pbu.launchers.items():
        print(
            f"  Launcher {launcher_id}: "
            f"pos=({launcher.position[0]:.0f}, {launcher.position[1]:.0f}, {launcher.position[2]:.0f}), "
            f"missiles={launcher.get_missile_count()}"
        )

    print("\n Radar Positions:")
    for radar in radars:
        print(
            f"  Radar {radar.id}: "
            f"pos=({radar.position[0]:.0f}, {radar.position[1]:.0f}, {radar.position[2]:.0f}), "
            f"range={radar.r_max}m"
        )

    clock = SimulationClock(time_step=config["Environment"]["time_step"], event_bus=event_bus)
    clock.register_component("Environment", env.update, 10)
    for index, radar in enumerate(radars):
        clock.register_component(
            f"Radar_{radar.id}",
            lambda dt, active_radar=radar: active_radar.update(dt, env),
            20 + index,
        )
    clock.register_component("PBU", pbu.update, 30)
    for launcher_id, launcher in pbu.launchers.items():
        clock.register_component(f"Launcher_{launcher_id}", launcher.update, 40)

    print("\n System ready for simulation!")
    quick_duration = 10.0
    quick_ticks = int(quick_duration / config["Environment"]["time_step"])
    progress_interval = max(1, quick_ticks // 4)
    print(f"\nRunning {quick_duration:.1f} seconds of simulation ({quick_ticks} ticks)...")

    clock.start()
    for tick_index in range(quick_ticks):
        clock.tick()
        if tick_index % progress_interval == 0:
            active_targets = sum(1 for target in env.targets.values() if not target.destroyed)
            print(
                f"  Tick {tick_index}: time={clock.current_time:.2f}s, "
                f"active targets={active_targets}, tracked={len(pbu.targets)}, launched={env.total_missiles_launched}"
            )
    clock.stop()

    print("\n Test complete!")
    print(f"  Final time: {clock.current_time:.2f}s")
    print(f"  Ticks processed: {clock.tick_count}")
    print(f"  Average tick time: {clock.get_average_tick_time() * 1000:.2f} ms")
    print(f"  Tracks formed: {len(pbu.targets)}")
    print(f"  Missiles launched: {env.total_missiles_launched}")
    print(f"  Targets destroyed: {env.total_targets_destroyed}")

    print("\n Final Target Positions:")
    for target_id, target in env.targets.items():
        if not target.destroyed:
            pos = target.position
            print(f"  Target {target_id}: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")

    return 0


def run_tests() -> int:
    """Run the automated tests."""
    return run_command([sys.executable, "-m", "pytest", "-q"])


def run_gui_smoke(config_path: str, ticks: int) -> int:
    """Run GUI rendering without opening a visible window."""
    import matplotlib

    matplotlib.use("Agg")
    simulator = ADSSimulatorGUI(config_path)
    simulator.clock.start()
    for tick_index in range(ticks):
        simulator.update(tick_index)
    simulator.clock.stop()
    simulator.gui.render()
    print("GUI_SMOKE_OK")
    return 0


def run_check(config_path: str, duration: float, gui_ticks: int) -> int:
    """Run the full verification sequence."""
    steps = [
        ("tests", lambda: run_tests()),
        ("quick", lambda: run_quick(config_path)),
        ("headless", lambda: run_headless(config_path, duration)),
        ("gui-smoke", lambda: run_gui_smoke(config_path, gui_ticks)),
    ]

    for name, step in steps:
        print(f"[STEP] {name}")
        exit_code = step()
        if exit_code != 0:
            print(f"[FAIL] {name} failed with exit code {exit_code}")
            return exit_code

    print("[OK] Project check completed successfully")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Single-command launcher for the ADS project.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="gui",
        choices=["gui", "headless", "quick", "test", "check", "gui-smoke"],
        help="Launch mode. Default: gui",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the config file relative to the project root.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Headless simulation duration in seconds.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=3,
        help="Tick count for gui-smoke mode.",
    )
    return parser


def resolve_config_path(raw_path: str) -> str:
    """Resolve config path relative to the project root."""
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((ROOT_DIR / path).resolve())


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)

    if args.mode == "gui":
        return run_gui(config_path)
    if args.mode == "headless":
        return run_headless(config_path, args.duration)
    if args.mode == "quick":
        return run_quick(config_path)
    if args.mode == "test":
        return run_tests()
    if args.mode == "gui-smoke":
        return run_gui_smoke(config_path, args.ticks)
    return run_check(config_path, args.duration, args.ticks)


if __name__ == "__main__":
    raise SystemExit(main())
