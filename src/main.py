#!/usr/bin/env python3
"""
Air Defense System Simulator - Main Entry Point
Architecture based on the ADS simulation diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
import random
import argparse

# Core modules
from simulation_clock import SimulationClock
from event_types import EventBus, SimulationEvent, EventType
from dispatcher import SimulationDispatcher
from air_environment import AirEnvironment
from radar import Radar
from pbu import Pbu
from launcher import Launcher
from target import TargetStatus


class AirDefenseSimulator:
    """
    Main simulator class that orchestrates all components.
    Matches the architecture from the diagrams:
    - Dispatcher (Time Block)
    - Air Environment
    - Radars (RLS)
    - Command Post (PBU)
    - Launchers (PU)
    - Missiles (ZUR)
    """
    
    def __init__(self, config_path: str = "./config.yaml"):
        self.config_path = config_path
        self.config = None
        
        # Event bus for inter-component communication
        self.event_bus = EventBus()
        
        # Core components
        self.clock: SimulationClock = None
        self.environment: AirEnvironment = None
        self.pbu: Pbu = None
        self.dispatcher: SimulationDispatcher = None
        
        # Subsystems
        self.radars: list[Radar] = []
        self.launchers: dict[int, Launcher] = {}
        
        # Visualization
        self.fig = None
        self.axes = None
        self.animation = None
        
        # Statistics
        self.stats = {
            "targets_spawned": 0,
            "targets_destroyed": 0,
            "missiles_launched": 0,
            "missiles_hit": 0
        }
        
        # Random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    def load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        print(f"Loaded configuration from {self.config_path}")
        return self.config
    
    def initialize(self) -> None:
        """Initialize all simulation components."""
        print("\n" + "="*60)
        print("INITIALIZING AIR DEFENSE SIMULATOR")
        print("="*60)
        
        if self.config is None:
            self.load_config()
        
        # 1. Initialize Simulation Clock (Dispatcher)
        time_step = self.config.get("Environment", {}).get("time_step", 0.005)
        self.clock = SimulationClock(time_step=time_step, event_bus=self.event_bus)
        print(f"[OK] Dispatcher initialized (time_step={time_step}s)")
        
        # 2. Initialize Air Environment
        env_config = self.config.get("Environment", {})
        self.environment = AirEnvironment(
            initialization_type='config_file',
            config=env_config,
            event_bus=self.event_bus
        )
        print(f"[OK] Air Environment initialized with {len(self.environment.targets)} targets")
        
        # 3. Initialize PBU (Command Post)
        pbu_config = self.config.get("Pbu", {})
        self.pbu = Pbu(
            initialization_type='config_file',
            config=pbu_config,
            event_bus=self.event_bus
        )
        print(f"[OK] PBU initialized with {len(self.pbu.launchers)} launchers")
        
        # Store launcher references
        self.launchers = self.pbu.launchers
        
        # 4. Initialize Radars
        locator_config = self.config.get("Locator", {})
        for radar_id_str, radar_cfg in locator_config.items():
            radar = Radar(
                radar_id=int(radar_id_str),
                event_bus=self.event_bus
            )
            radar.initialize_with_file_data(radar_cfg)
            self.radars.append(radar)
        print(f"[OK] {len(self.radars)} Radars initialized")
        
        # 5. Register components with dispatcher
        self.dispatcher = SimulationDispatcher(
            clock=self.clock,
            event_bus=self.event_bus,
            environment=self.environment,
            pbu=self.pbu,
            radars=self.radars
        )
        self.dispatcher.register_default_components()

        # 6. Subscribe to events
        self._setup_event_handlers()
        
        print("="*60)
        print("INITIALIZATION COMPLETE")
        print("="*60 + "\n")
    
    def _setup_event_handlers(self) -> None:
        """Set up event handlers for monitoring."""
        
        def on_target_destroyed(event: SimulationEvent):
            self.stats["targets_destroyed"] += 1
        
        def on_missile_launched(event: SimulationEvent):
            self.stats["missiles_launched"] += 1

        def on_missile_detonated(event: SimulationEvent):
            self.stats["missiles_hit"] += 1

        def on_target_spawned(event: SimulationEvent):
            self.stats["targets_spawned"] += 1

        self.event_bus.subscribe(EventType.TARGET_DESTROYED, on_target_destroyed)
        self.event_bus.subscribe(EventType.MISSILE_LAUNCHED, on_missile_launched)
        self.event_bus.subscribe(EventType.MISSILE_DETONATED, on_missile_detonated)
        self.event_bus.subscribe(EventType.TARGET_SPAWNED, on_target_spawned)
    
    def run_headless(self, duration: float = 60.0) -> None:
        """Run simulation without visualization."""
        print(f"\nRunning headless simulation for {duration}s...")
        
        self.clock.start()
        
        while self.clock.current_time < duration:
            self.clock.tick()
            
            # Print progress every 1000 ticks
            if self.clock.tick_count % 1000 == 0:
                print(f"  Time: {self.clock.current_time:.2f}s, "
                      f"Ticks: {self.clock.tick_count}, "
                      f"Targets: {len(self.environment.get_active_targets())}, "
                      f"Missiles: {len(self.environment.missiles)}")
        
        self.clock.stop()
        self._print_summary()
    
    def run_interactive(self) -> None:
        """Run simulation with matplotlib visualization."""
        print("\nStarting interactive visualization...")
        
        self._setup_visualization()
        
        # Start simulation
        self.clock.start()
        
        # Create animation
        self.animation = animation.FuncAnimation(
            self.fig, 
            self._animate,
            frames=None,
            interval=50,  # ms between frames
            blit=False,
            cache_frame_data=False
        )
        
        plt.show()
        
        self.clock.stop()
    
    def _setup_visualization(self) -> None:
        """Set up matplotlib figure for visualization."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        
        self.ax_xy = self.axes[0, 0]  # Top-down view (XY)
        self.ax_xz = self.axes[0, 1]  # Side view (XZ)
        self.ax_info = self.axes[1, 0]  # Information panel
        self.ax_stats = self.axes[1, 1]  # Statistics panel
        
        for ax in [self.ax_xy, self.ax_xz]:
            ax.set_xlim(0, 10000)
            ax.set_ylim(0, 10000)
            ax.grid(True, alpha=0.3)
        
        self.ax_xy.set_xlabel("X (m)")
        self.ax_xy.set_ylabel("Y (m)")
        self.ax_xy.set_title("Top-Down View")
        
        self.ax_xz.set_xlabel("X (m)")
        self.ax_xz.set_ylabel("Z (m)")
        self.ax_xz.set_title("Side View (Altitude)")
        
        self.ax_info.axis('off')
        self.ax_stats.axis('off')
        
        plt.tight_layout()
    
    def _animate(self, frame: int) -> list:
        """Animation callback for visualization."""
        # Execute multiple simulation ticks per frame
        skip_frames = 5
        
        for _ in range(skip_frames):
            if not self.clock.is_running():
                break
            self.clock.tick()
        
        # Clear axes
        self.ax_xy.clear()
        self.ax_xz.clear()
        self.ax_info.clear()
        self.ax_stats.clear()
        
        # Reset axes properties
        self.ax_xy.set_xlim(0, 10000)
        self.ax_xy.set_ylim(0, 10000)
        self.ax_xy.grid(True, alpha=0.3)
        self.ax_xy.set_xlabel("X (m)")
        self.ax_xy.set_ylabel("Y (m)")
        self.ax_xy.set_title("Top-Down View")
        
        self.ax_xz.set_xlim(0, 10000)
        self.ax_xz.set_ylim(0, 5000)
        self.ax_xz.grid(True, alpha=0.3)
        self.ax_xz.set_xlabel("X (m)")
        self.ax_xz.set_ylabel("Z (m)")
        self.ax_xz.set_title("Side View (Altitude)")
        
        # Draw radars
        for radar in self.radars:
            self.ax_xy.scatter(radar.position[0], radar.position[1], 
                              marker='s', s=100, color='blue', 
                              label=f'Radar {radar.id}' if radar.id == self.radars[0].id else "")
            self.ax_xz.scatter(radar.position[0], radar.position[2], 
                              marker='s', s=100, color='blue')
            
            # Draw radar beam
            if radar.curr_ray_x:
                self.ax_xy.plot(radar.curr_ray_x, radar.curr_ray_y, 
                               'b-', alpha=0.3, linewidth=0.5)
                self.ax_xz.plot(radar.curr_ray_x, radar.curr_ray_z, 
                               'b-', alpha=0.3, linewidth=0.5)
        
        # Draw launchers
        for launcher in self.launchers.values():
            self.ax_xy.scatter(launcher.position[0], launcher.position[1], 
                              marker='^', s=100, color='orange', label='Launcher')
            self.ax_xz.scatter(launcher.position[0], launcher.position[2], 
                              marker='^', s=100, color='orange')
        
        # Draw defended asset
        if self.pbu.defended_assets:
            asset = self.pbu.defended_assets[0]
            self.ax_xy.scatter(asset[0], asset[1], 
                              marker='*', s=200, color='gold', 
                              label='Defended Asset', edgecolors='black')
            self.ax_xz.scatter(asset[0], asset[2], 
                              marker='*', s=200, color='gold', edgecolors='black')
        
        # Draw targets
        for target in self.environment.targets.values():
            if target.status == TargetStatus.ACTIVE:
                color = 'red'
                if target.id in self.pbu.targets:
                    threat = self.pbu.targets[target.id].threat_level
                    if threat.name == 'CRITICAL':
                        color = 'darkred'
                    elif threat.name == 'HIGH':
                        color = 'red'
                    elif threat.name == 'MEDIUM':
                        color = 'orange'
                    else:
                        color = 'yellow'
                
                self.ax_xy.scatter(target.position[0], target.position[1], 
                                  marker='o', s=50, color=color, label='Target')
                self.ax_xz.scatter(target.position[0], target.position[2], 
                                  marker='o', s=50, color=color)
                
                # Draw target trail
                if len(target.position_history) > 1:
                    trail = np.array(target.position_history[-20:])
                    self.ax_xy.plot(trail[:, 0], trail[:, 1], 
                                   color=color, alpha=0.3, linewidth=1)
                
                # Target label
                self.ax_xy.annotate(f"T{target.id}", 
                                   xy=(target.position[0], target.position[1]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, color=color)
        
        # Draw missiles
        for missile in self.environment.missiles.values():
            if missile.is_active():
                self.ax_xy.scatter(missile.position[0], missile.position[1], 
                                  marker='x', s=30, color='cyan')
                self.ax_xz.scatter(missile.position[0], missile.position[2], 
                                  marker='x', s=30, color='cyan')
                
                # Missile trail
                if len(missile.position_history) > 1:
                    trail = np.array(missile.position_history[-10:])
                    self.ax_xy.plot(trail[:, 0], trail[:, 1], 
                                   'c-', alpha=0.3, linewidth=1)
        
        # Information panel
        info_text = [
            f"Simulation Time: {self.clock.current_time:.2f} s",
            f"Tick: {self.clock.tick_count}",
            f"State: {self.clock.state.name}",
            "",
            f"Active Targets: {len(self.environment.get_active_targets())}",
            f"Active Missiles: {len([m for m in self.environment.missiles.values() if m.is_active()])}",
            "",
            f"PBU Engagements: {len(self.pbu.engagements)}",
            f"Targets Destroyed: {self.stats['targets_destroyed']}",
            f"Missiles Launched: {self.stats['missiles_launched']}",
            "",
            "Launcher Status:"
        ]
        
        for lid, launcher in self.launchers.items():
            info_text.append(f"  L{lid}: {launcher.status.name} ({launcher.get_missile_count()} missiles)")
        
        self.ax_info.text(0.05, 0.95, "\n".join(info_text), 
                         transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top',
                         fontfamily='monospace')
        self.ax_info.set_title("System Status")
        
        # Statistics panel
        stats_text = [
            "ENGAGEMENT DETAILS",
            "="*25,
            ""
        ]
        
        for target_id, plan in self.pbu.engagements.items():
            target_info = self.pbu.targets.get(target_id)
            if target_info:
                stats_text.append(f"Target {target_id}:")
                stats_text.append(f"  Threat: {target_info.threat_level.name}")
                stats_text.append(f"  Status: {target_info.engagement_status.name}")
                stats_text.append(f"  Launcher: {plan.launcher_id}")
                stats_text.append(f"  Est. Intercept: {plan.estimated_intercept_time:.1f}s")
                stats_text.append("")
        
        if not self.pbu.engagements:
            stats_text.append("No active engagements")
        
        self.ax_stats.text(0.05, 0.95, "\n".join(stats_text),
                          transform=self.ax_stats.transAxes,
                          fontsize=9, verticalalignment='top',
                          fontfamily='monospace')
        self.ax_stats.set_title("Engagement Details")
        
        # Legend
        handles, labels = self.ax_xy.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_xy.legend(by_label.values(), by_label.keys(), 
                         loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        return []
    
    def _print_summary(self) -> None:
        """Print simulation summary."""
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        print(f"Total simulation time: {self.clock.current_time:.2f} s")
        print(f"Total ticks: {self.clock.tick_count}")
        print(f"Average tick time: {self.clock.get_average_tick_time()*1000:.2f} ms")
        print()
        print(f"Targets spawned: {self.stats['targets_spawned']}")
        print(f"Targets destroyed: {self.stats['targets_destroyed']}")
        print(f"Missiles launched: {self.stats['missiles_launched']}")
        print()
        print("Final Launcher Status:")
        for lid, launcher in self.launchers.items():
            print(f"  Launcher {lid}: {launcher.get_missile_count()} missiles remaining")
        print("="*60)
    
    def save_animation(self, filename: str = "simulation.gif", fps: int = 30) -> None:
        """Save animation to file."""
        if self.animation is None:
            print("No animation to save. Run interactive mode first.")
            return
        
        print(f"Saving animation to {filename}...")
        self.animation.save(filename, writer='imagemagick', fps=fps)
        print("Done!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Air Defense System Simulator")
    parser.add_argument("--config", "-c", type=str, default="./config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", "-m", choices=["interactive", "headless", "save"],
                       default="interactive", help="Simulation mode")
    parser.add_argument("--duration", "-d", type=float, default=60.0,
                       help="Simulation duration in seconds (headless mode)")
    parser.add_argument("--output", "-o", type=str, default="simulation.gif",
                       help="Output file for animation (save mode)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create and initialize simulator
    sim = AirDefenseSimulator(config_path=args.config)
    sim.initialize()
    
    # Run in selected mode
    if args.mode == "headless":
        sim.run_headless(duration=args.duration)
    elif args.mode == "save":
        sim._setup_visualization()
        sim.clock.start()
        sim.animation = animation.FuncAnimation(
            sim.fig, sim._animate, frames=500, interval=50, blit=False
        )
        sim.save_animation(filename=args.output)
        sim.clock.stop()
        sim._print_summary()
    else:
        sim.run_interactive()


if __name__ == "__main__":
    main()
