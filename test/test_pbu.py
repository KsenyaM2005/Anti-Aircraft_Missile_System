"""
Unit tests for PBU (Command Post) module.
"""

import unittest
import numpy as np
from pbu import Pbu, ThreatLevel, ThreatAssessor, EngagementPlanner
from event_types import EventBus


class TestThreatAssessor(unittest.TestCase):
    
    def setUp(self):
        self.assessor = ThreatAssessor(defended_position=np.array([5000, 5000, 0]))
    
    def test_high_threat(self):
        """Test high threat assessment."""
        from pbu import TargetInfo
        
        target = TargetInfo(target_id=1)
        target.position = np.array([5000, 4000, 500])  # Close to defended asset
        target.velocity = np.array([0, 600, 0])  # High speed
        
        threat = self.assessor.assess(target)
        
        self.assertIn(threat, [ThreatLevel.HIGH, ThreatLevel.CRITICAL])
    
    def test_low_threat(self):
        """Test low threat assessment."""
        from pbu import TargetInfo
        
        target = TargetInfo(target_id=1)
        target.position = np.array([0, 0, 5000])  # Far away
        target.velocity = np.array([0, 100, 0])  # Low speed
        
        threat = self.assessor.assess(target)
        
        self.assertLess(threat.value, ThreatLevel.HIGH.value)
    
    def test_heading_towards_defense(self):
        """Test heading towards defense detection."""
        from pbu import TargetInfo
        
        target = TargetInfo(target_id=1)
        target.position = np.array([4000, 4000, 0])
        target.velocity = np.array([100, 100, 0])  # Towards (5000, 5000)
        
        is_heading = self.assessor._is_heading_towards_defense(target)
        
        self.assertTrue(is_heading)


class TestEngagementPlanner(unittest.TestCase):
    
    def setUp(self):
        self.planner = EngagementPlanner()
    
    def test_intercept_calculation(self):
        """Test intercept point calculation."""
        from pbu import TargetInfo
        
        target = TargetInfo(target_id=1)
        target.position = np.array([5000, 5000, 1000])
        target.velocity = np.array([-100, 0, 0])
        
        launcher_pos = np.array([3000, 5000, 0])
        
        result = self.planner._calculate_intercept(launcher_pos, target)
        
        self.assertIsNotNone(result)
        intercept_point, intercept_time = result
        
        self.assertGreater(intercept_time, 0)
        self.assertEqual(len(intercept_point), 3)


class TestPbu(unittest.TestCase):
    
    def setUp(self):
        self.event_bus = EventBus()
        self.pbu = Pbu(initialization_type='empty', event_bus=self.event_bus)
    
    def test_add_launcher(self):
        """Test adding a launcher."""
        result = self.pbu.add_launcher(
            id=1,
            launcher_pos=[3000, 3000, 0],
            missile_amount=5,
            speed=1000,
            missile_type='guided missile',
            trigger_distance=10.0,
            explosion_range=100.0
        )
        
        self.assertTrue(result)
        self.assertIn(1, self.pbu.launchers)
        self.assertEqual(self.pbu.launchers[1].get_missile_count(), 5)
    
    def test_add_duplicate_launcher(self):
        """Test adding duplicate launcher fails."""
        self.pbu.add_launcher(id=1, launcher_pos=[0,0,0], missile_amount=5,
                             speed=1000, missile_type='guided',
                             trigger_distance=10.0, explosion_range=100.0)
        
        result = self.pbu.add_launcher(id=1, launcher_pos=[0,0,0], missile_amount=5,
                                       speed=1000, missile_type='guided',
                                       trigger_distance=10.0, explosion_range=100.0)
        
        self.assertFalse(result)
    
    def test_process_radar_track(self):
        """Test processing radar track data."""
        track_data = {
            'track_id': 1,
            'position': [2000, 3000, 500],
            'velocity': [100, 0, 0],
            'estimated_rcs': 1.0
        }
        
        self.pbu.process_radar_track(radar_id=1, track_data=track_data)
        
        self.assertEqual(len(self.pbu.targets), 1)
    
    def test_register_new_target(self):
        """Test registering a new target."""
        track_data = {
            'position': [1000, 2000, 500],
            'velocity': [0, 0, 0]
        }
        
        target_id = self.pbu._register_new_target(track_data)
        
        self.assertEqual(target_id, 0)  # First target should be ID 0
        self.assertIn(target_id, self.pbu.targets)  # Use the returned ID
    
    def test_duplicate_target_filtering(self):
        """Test filtering duplicate targets."""
        self.pbu.add_distance = 500.0
        
        # First target
        self.pbu._register_new_target({'position': [1000, 1000, 500], 'velocity': [0,0,0]})
        
        # Second target near first
        target_id = self.pbu._register_new_target({'position': [1100, 1000, 500], 'velocity': [0,0,0]})
        
        # Should return existing target ID
        self.assertEqual(target_id, 0)
        self.assertEqual(len(self.pbu.targets), 1)
    
    def test_clear_exploded(self):
        """Test clearing destroyed target."""
        self.pbu.targets[0] = type('TargetInfo', (), {})()
        self.pbu.targets[0].target_id = 0
        
        self.pbu.clear_exploded(0)
        
        self.assertNotIn(0, self.pbu.targets)


if __name__ == '__main__':
    unittest.main()
