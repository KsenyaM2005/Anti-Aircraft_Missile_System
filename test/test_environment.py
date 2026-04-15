"""
Unit tests for Air Environment module.
"""

import unittest
import numpy as np

from air_environment import AirEnvironment, WeatherCondition, AtmosphericConditions
from target import TargetStatus
from event_types import EventBus


class TestAirEnvironment(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = EventBus()
        self.env = AirEnvironment(
            initialization_type='empty',
            event_bus=self.event_bus
        )
    
    def test_initialization_empty(self):
        """Test empty initialization."""
        self.assertEqual(len(self.env.targets), 0)
        self.assertEqual(len(self.env.missiles), 0)
    
    def test_add_target(self):
        """Test adding a target."""
        result = self.env.add_target(
            trajectory_type='uniform',
            id=1,
            trajectory_arguments={
                'position': [1000, 2000, 500],
                'velocity': [100, 0, 0]
            }
        )
        
        self.assertTrue(result)
        self.assertIn(1, self.env.targets)
        self.assertEqual(self.env.targets[1].status, TargetStatus.ACTIVE)
    
    def test_add_duplicate_target(self):
        """Test adding duplicate target fails."""
        self.env.add_target('uniform', id=1, 
                           trajectory_arguments={'position': [0,0,0], 'velocity': [0,0,0]})
        
        result = self.env.add_target('uniform', id=1,
                                    trajectory_arguments={'position': [0,0,0], 'velocity': [0,0,0]})
        
        self.assertFalse(result)
    
    def test_update_targets(self):
        """Test target position update."""
        self.env.add_target('uniform', id=1,
                           trajectory_arguments={
                               'position': [0, 0, 0],
                               'velocity': [100, 0, 0]
                           })
        
        initial_pos = self.env.targets[1].position.copy()
        self.env.update_targets(1.0)
        
        expected_pos = initial_pos + np.array([100, 0, 0])
        np.testing.assert_array_almost_equal(self.env.targets[1].position, expected_pos)
    
    def test_atmospheric_conditions(self):
        """Test atmospheric effects."""
        atm = AtmosphericConditions(weather=WeatherCondition.HEAVY_RAIN)
        
        attenuation = atm.get_radar_attenuation()
        self.assertLess(attenuation, 1.0)
        
        noise_factor = atm.get_noise_factor()
        self.assertGreater(noise_factor, 1.0)
    
    def test_get_active_targets(self):
        """Test getting active targets."""
        self.env.add_target('uniform', id=1,
                           trajectory_arguments={'position': [0,0,0], 'velocity': [0,0,0]})
        self.env.add_target('uniform', id=2,
                           trajectory_arguments={'position': [0,0,0], 'velocity': [0,0,0]})
        
        self.env.targets[2].destroy()
        
        active = self.env.get_active_targets()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].id, 1)
    
    def test_boundary_check(self):
        """Test targets leaving simulation area."""
        self.env.boundary_x = (0, 1000)
        self.env.boundary_y = (0, 1000)
        self.env.boundary_z = (0, 1000)
        
        self.env.add_target('uniform', id=1,
                           trajectory_arguments={
                               'position': [500, 500, 500],
                               'velocity': [600, 0, 0]
                           })
        
        self.env.update(1.0)
        self.env._check_boundaries()
        
        self.assertEqual(self.env.targets[1].status, TargetStatus.EXPIRED)


class TestRadarNoiseModel(unittest.TestCase):
    
    def setUp(self):
        from air_environment import RadarNoiseModel
        self.model = RadarNoiseModel(
            range_noise=10.0,
            angle_noise=0.5,
            detection_probability=1.0
        )
        self.atm = AtmosphericConditions()
    
    def test_noise_application(self):
        """Test noise is applied to position."""
        true_pos = np.array([1000, 0, 0])
        radar_pos = np.array([0, 0, 0])
        
        noisy = self.model.apply_noise_to_position(true_pos, radar_pos, self.atm)
        
        self.assertIsNotNone(noisy)
        self.assertEqual(len(noisy), 3)
        # Position should be different but close
        self.assertNotEqual(noisy[0], true_pos[0])
        self.assertLess(abs(noisy[0] - true_pos[0]), 100)


if __name__ == '__main__':
    unittest.main()
