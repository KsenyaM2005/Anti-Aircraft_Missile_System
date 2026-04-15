"""
Integration tests for the complete ADS simulation.
"""

import unittest
import numpy as np
import tempfile
import yaml
from pathlib import Path

from air_environment import AirEnvironment
from radar import Radar
from pbu import Pbu
from launcher import Launcher
from simulation_clock import SimulationClock
from event_types import EventBus


class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.event_bus = EventBus()
        
        # Create minimal config
        self.config = {
            'Environment': {
                'time_step': 0.01,
                'targets': {
                    '0': {
                        'trajectory_type': 'uniform',
                        'trajectory_arguments': {
                            'position': [2000, 5000, 500],
                            'velocity': [100, 0, 0]
                        }
                    }
                }
            },
            'Pbu': {
                'time_step': 0.01,
                'add_distance': 300.0,
                'launchers': {
                    '1': {
                        'launcher_pos': [3000, 3000, 0],
                        'missile_amount': 3,
                        'speed': 1000,
                        'missile_type': 'guided missile',
                        'trigger_distance': 10.0,
                        'explosion_range': 100.0
                    }
                }
            },
            'Locator': {
                '1': {
                    'time_step': 0.01,
                    'coordinates': {'x': 4000, 'y': 5000, 'z': 0},
                    'omega_az': 15,
                    'omega_el': 8,
                    'r_max': 3000,
                    'dr': 10
                }
            }
        }
    
    def test_full_simulation_pipeline(self):
        """Test the complete simulation pipeline."""
        # Initialize components
        env = AirEnvironment('config_file', self.config['Environment'], self.event_bus)
        pbu = Pbu('config_file', self.config['Pbu'], self.event_bus)
        
        radar_config = self.config['Locator']['1']
        radar = Radar(radar_id=1, event_bus=self.event_bus)
        radar.initialize_with_file_data(radar_config)
        
        clock = SimulationClock(time_step=0.01, event_bus=self.event_bus)
        
        # Register components
        clock.register_component("Environment", env.update, 10)
        clock.register_component("Radar", lambda dt: radar.update(dt, env), 20)
        clock.register_component("PBU", pbu.update, 30)
        
        # Run a few ticks
        clock.start()
        for _ in range(10):
            clock.tick()
        clock.stop()
        
        # Verify target exists
        self.assertEqual(len(env.targets), 1)
        
        # Verify radar has scanned
        self.assertGreater(radar.current_azimuth, 0)
        
        # Verify PBU has launcher
        self.assertEqual(len(pbu.launchers), 1)
    
    def test_target_detection_flow(self):
        """Test target detection and tracking flow."""
        env = AirEnvironment('config_file', self.config['Environment'], self.event_bus)
        
        radar = Radar(radar_id=1, event_bus=self.event_bus)
        radar.initialize_with_file_data(self.config['Locator']['1'])
        
        # Update a few times
        for _ in range(5):
            env.update(0.01)
            radar.update(0.01, env)
        
        # Should have scanned area
        self.assertGreater(len(radar.curr_ray_x), 0)
    
    def test_launcher_missile_flow(self):
        """Test launcher and missile interaction."""
        # Create launcher
        launcher = Launcher(
            launcher_id=1,
            position=[3000, 3000, 0],
            magazine_capacity=5
        )
        launcher.initialize_magazine(5, 'guided missile', 1000)
        
        # Create target position
        target_pos = np.array([4000, 4000, 1000])
        launcher.assign_target(target_pos, target_id=1)
        
        # Verify launcher status
        self.assertEqual(launcher.get_missile_count(), 5)
        self.assertIsNotNone(launcher.assigned_target_position)
    
    def test_event_bus_communication(self):
        """Test inter-component communication via event bus."""
        received_events = []
        
        def event_handler(event):
            received_events.append(event)
        
        from event_types import EventType  # Add this at the top with other imports
        self.event_bus.subscribe(EventType.TARGET_DETECTED, event_handler)
        
        # Publish test event
        from event_types import SimulationEvent, EventType
        event = SimulationEvent(
            event_type=EventType.TARGET_DETECTED,
            source_id="radar_1",
            data={"position": [1000, 2000, 500]}
        )
        self.event_bus.publish(event)
        
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].event_type, EventType.TARGET_DETECTED)


class TestScenarioLoading(unittest.TestCase):
    
    def test_config_loading(self):
        """Test loading configuration from YAML."""
        config_data = {
            'Environment': {
                'time_step': 0.01,
                'targets': {}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f)
            
            self.assertIn('Environment', loaded)
            self.assertEqual(loaded['Environment']['time_step'], 0.01)
        finally:
            Path(config_path).unlink()


class TestPerformance(unittest.TestCase):
    
    def test_simulation_speed(self):
        """Test simulation performance."""
        import time
        
        event_bus = EventBus()
        clock = SimulationClock(time_step=0.01, event_bus=event_bus)
        
        # Create dummy component
        counter = 0
        def dummy_update(dt):
            nonlocal counter
            counter += 1
        
        clock.register_component("Dummy", dummy_update, 10)
        
        # Measure ticks per second
        clock.start()
        start_time = time.perf_counter()
        
        for _ in range(100):
            clock.tick()
        
        elapsed = time.perf_counter() - start_time
        clock.stop()
        
        tps = 100 / elapsed
        
        print(f"\nPerformance: {tps:.1f} ticks/second")
        
        # Should be reasonably fast
        self.assertGreater(tps, 100)


if __name__ == '__main__':
    unittest.main()
