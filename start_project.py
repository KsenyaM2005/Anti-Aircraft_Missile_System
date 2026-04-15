#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np

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

    def __init__(self, config_path: str = "config.yaml", auto_start: bool = False):
        import yaml
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        from src.air_environment import AirEnvironment
        from src.dispatcher import SimulationDispatcher
        from src.event_types import EventBus, EventType, SimulationEvent
        from src.gui import GUI
        from src.pbu import Pbu
        from src.radar import RadarSystem, Transmitter, Receiver
        from src.simulation_clock import SimulationClock

        self.plt = plt
        self.FuncAnimation = FuncAnimation
        self.EventType = EventType
        self.SimulationEvent = SimulationEvent
        self.auto_start = auto_start

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        self.event_bus = EventBus()
        self.env = AirEnvironment("config_file", self.config["Environment"], self.event_bus)
        self.pbu = Pbu("config_file", self.config["Pbu"], self.event_bus)

        # Создание радаров с использованием новой архитектуры RadarSystem
        self.radars = []
        time_step = self.config["Environment"]["time_step"]

        for radar_id_str, radar_cfg in self.config["Locator"].items():
            radar_id = int(radar_id_str)

            # Получение позиции из конфига
            position = radar_cfg.get("position", {"x": 0, "y": 0, "z": 0})
            pos_array = np.array([position["x"], position["y"], position["z"]])

            # Создание RadarSystem
            radar = RadarSystem(radar_id=radar_id, position=pos_array, dt=time_step)

            # Настройка передатчика из конфига
            if "transmitter" in radar_cfg:
                tx = radar_cfg["transmitter"]
                radar.transmitter = Transmitter(
                    power_w=tx.get("power_w", 10000.0),
                    gain_db=tx.get("gain_db", 40.0),
                    frequency_hz=tx.get("frequency_hz", 3e9),
                    bandwidth_hz=tx.get("bandwidth_hz", 1e6)
                )

            # Настройка приёмника из конфига
            if "receiver" in radar_cfg:
                rx = radar_cfg["receiver"]
                radar.receiver = Receiver(
                    noise_temp_k=rx.get("noise_temp_k", 290.0),
                    losses_db=rx.get("losses_db", 3.0),
                    azimuth_beamwidth_rad=rx.get("azimuth_beamwidth_rad", 0.0175),
                    elevation_beamwidth_rad=rx.get("elevation_beamwidth_rad", 0.0175),
                    km=rx.get("km", 1.0)
                )

            # Настройка параметров сканирования
            radar.r_max = radar_cfg.get("r_max", 2000.0)
            radar.dr = radar_cfg.get("dr", 10.0)
            radar.omega_az = np.radians(radar_cfg.get("omega_az", 15.0)) * time_step
            radar.omega_el = np.radians(radar_cfg.get("omega_el", 8.0)) * time_step
            radar.beam_width_az = np.radians(radar_cfg.get("beam_width_az", 2.0))
            radar.beam_width_el = np.radians(radar_cfg.get("beam_width_el", 2.0))

            self.radars.append(radar)

        self.clock = SimulationClock(
            time_step=time_step,
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
        )
        self.dispatcher.register_default_components()

    def update(self, _frame: int):
        """Update callback for animation and smoke tests."""
        # Обновляем GUI кэш для отображения статической картинки
        self.dispatcher.update_gui_cache()
        self.gui.render()

        # Только если auto_start включен, выполняем тики симуляции
        if self.auto_start:
            for _ in range(self.skip_frames):
                self.clock.tick()

        if not self.clock.is_running() and self.auto_start:
            self.plt.close()
            return []

        self.frame_count += 1
        return []

    def run(self) -> None:
        """Run the interactive GUI."""
        print("\n" + "=" * 60)
        print(" AIR DEFENSE SIMULATOR - GUI MODE")
        print("=" * 60)

        if not self.auto_start:
            print(" [СТАТИЧЕСКИЙ РЕЖИМ] - Симуляция остановлена")
            print(" Нажмите SPACE для запуска")
        else:
            print(" [АВТОМАТИЧЕСКИЙ РЕЖИМ] - Симуляция запущена")

        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  R     - Reset View")
        print("  1     - Top-Down View")
        print("  2     - Side View")
        print("  3     - Split View")
        print("  F     - Follow Selected Target")
        print("  H     - Toggle HUD")
        print("  ESC   - Clear Selection")
        print("=" * 60 + "\n")

        # Запускаем часы только если auto_start включен
        if self.auto_start:
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
            if self.auto_start:
                # Если auto_start включен, переключаем паузу
                self.event_bus.publish(
                    self.SimulationEvent(
                        event_type=self.EventType.OPERATOR_COMMAND,
                        source_id="gui",
                        data={"command": "toggle_pause"},
                    )
                )
            else:
                # Если auto_start выключен, запускаем симуляцию при первом нажатии SPACE
                self.auto_start = True
                self.clock.start()
                print("\n [ЗАПУСК] Симуляция начата\n")

        self.gui.set_pause_callback(toggle_pause)

        # Первый рендер для отображения статической картинки
        self.dispatcher.update_gui_cache()
        self.gui.render()

        self.plt.show()

        if self.auto_start:
            self.clock.stop()

        self.print_summary()

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print(" SIMULATION SUMMARY")
        print("=" * 60)
        print(f"  Simulation time: {self.clock.current_time:.1f}s")
        print(f"  Active targets: {len(self.env.get_active_targets())}")
        print(f"  Targets destroyed: {self.env.total_targets_destroyed}")
        print(f"  Missiles launched: {self.env.total_missiles_launched}")
        print(f"  Tracks: {sum(len(radar.tracks) for radar in self.radars)}")
        print("=" * 60)


def run_gui(config_path: str, auto_start: bool = False) -> int:
    """Run the GUI mode."""
    simulator = ADSSimulatorGUI(config_path, auto_start=auto_start)
    simulator.run()
    return 0


def run_headless(config_path: str, duration: float) -> int:
    """Run headless mode directly through the simulator."""
    from src.main import AirDefenseSimulator

    simulator = AirDefenseSimulator(config_path=config_path)
    simulator.initialize()
    simulator.run_headless(duration=duration)
    return 0


def run_quick(config_path: str) -> int:
    """Run a short console smoke simulation."""
    import yaml
    import numpy as np

    from src.air_environment import AirEnvironment
    from src.event_types import EventBus
    from src.pbu import Pbu
    from src.radar import RadarSystem, Transmitter, Receiver
    from src.simulation_clock import SimulationClock

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("Loading config...")
    print("Initializing components...")

    event_bus = EventBus()
    env = AirEnvironment("config_file", config["Environment"], event_bus)
    pbu = Pbu("config_file", config["Pbu"], event_bus)

    time_step = config["Environment"]["time_step"]
    radars = []

    for radar_id_str, radar_cfg in config["Locator"].items():
        radar_id = int(radar_id_str)

        # Получение позиции
        position = radar_cfg.get("position", {"x": 0, "y": 0, "z": 0})
        pos_array = np.array([position["x"], position["y"], position["z"]])

        # Создание RadarSystem
        radar = RadarSystem(radar_id=radar_id, position=pos_array, dt=time_step)

        # Настройка передатчика
        if "transmitter" in radar_cfg:
            tx = radar_cfg["transmitter"]
            radar.transmitter = Transmitter(
                power_w=tx.get("power_w", 10000.0),
                gain_db=tx.get("gain_db", 40.0),
                frequency_hz=tx.get("frequency_hz", 3e9),
                bandwidth_hz=tx.get("bandwidth_hz", 1e6)
            )

        # Настройка приёмника
        if "receiver" in radar_cfg:
            rx = radar_cfg["receiver"]
            radar.receiver = Receiver(
                noise_temp_k=rx.get("noise_temp_k", 290.0),
                losses_db=rx.get("losses_db", 3.0),
                azimuth_beamwidth_rad=rx.get("azimuth_beamwidth_rad", 0.0175),
                elevation_beamwidth_rad=rx.get("elevation_beamwidth_rad", 0.0175)
            )

        # Настройка параметров
        radar.r_max = radar_cfg.get("r_max", 2000.0)
        radar.dr = radar_cfg.get("dr", 10.0)
        radar.omega_az = np.radians(radar_cfg.get("omega_az", 15.0)) * time_step
        radar.omega_el = np.radians(radar_cfg.get("omega_el", 8.0)) * time_step

        radars.append(radar)

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

    clock = SimulationClock(time_step=time_step, event_bus=event_bus)
    clock.register_component("Environment", env.update, 10)

    for index, radar in enumerate(radars):
        clock.register_component(
            f"Radar_{radar.id}",
            lambda dt, active_radar=radar: _update_radar(active_radar, dt, env, pbu),
            20 + index,
        )

    clock.register_component("PBU", pbu.update, 30)

    print("\n System ready for simulation!")
    print("\nRunning 1 second of simulation (200 ticks)...")

    clock.start()
    for tick_index in range(200):
        clock.tick()
        if tick_index % 50 == 0:
            active_targets = sum(1 for target in env.targets.values() if not target.destroyed)
            print(f"  Tick {tick_index}: time={clock.current_time:.2f}s, active targets={active_targets}")
    clock.stop()

    print("\n Test complete!")
    print(f"  Final time: {clock.current_time:.2f}s")
    print(f"  Ticks processed: {clock.tick_count}")
    print(f"  Average tick time: {clock.get_average_tick_time() * 1000:.2f} ms")

    print("\n Final Target Positions:")
    for target_id, target in env.targets.items():
        if not target.destroyed:
            pos = target.position
            print(f"  Target {target_id}: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")

    return 0


def _update_radar(radar, dt: float, env, pbu) -> None:
    """
    Вспомогательная функция для обновления радара.
    """
    radar.dt = dt
    radar.update_scan()

    az, el = radar.get_beam_direction()

    # Сканирование по дальности
    for r in np.arange(0, radar.r_max, radar.dr):
        beam_pos = radar.get_beam_position(r)

        for target_id, target in env.targets.items():
            if target.status.value != 2:  # TargetStatus.ACTIVE = 2
                continue

            if radar.is_target_in_beam(target.position, r):
                measurement = env.get_noisy_measurement(target_id, radar.position)
                if measurement is not None:
                    track_id = radar.process_measurement(measurement)
                    if track_id is not None:
                        for track_data in radar.get_track_data():
                            if track_data["track_id"] == track_id:
                                pbu.process_radar_track(radar.id, track_data)
                                break
                    break

    radar.update_tracks()


def run_tests() -> int:
    """Run the automated tests."""
    return run_command([sys.executable, "-m", "pytest", "-q"])


def run_gui_smoke(config_path: str, ticks: int) -> int:
    """Run GUI rendering without opening a visible window."""
    import matplotlib

    matplotlib.use("Agg")
    simulator = ADSSimulatorGUI(config_path, auto_start=False)
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
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        default=True,
        help="Don't start simulation automatically (show static view).",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        default=False,
        help="Start simulation automatically.",
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
        # По умолчанию auto_start = False (статический режим)
        auto_start = args.auto_start if hasattr(args, 'auto_start') else False
        return run_gui(config_path, auto_start=auto_start)
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