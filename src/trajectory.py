import numpy as np
import math

class Trajectory:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self._update_functions = [self._update_position]
    
    def _update_position(self, time_step: float):
        pass
    
    def update(self, time_step: float):
        for func in self._update_functions:
            func(time_step)
    
    def get_position(self):
        return self.position.copy()
    
    def get_velocity(self):
        return self.velocity.copy()

class TrajectoryUniform(Trajectory):
    def __init__(self, position, velocity):
        super().__init__(position, velocity)
    
    def _update_position(self, time_step: float):
        self.position += self.velocity * time_step

class TrajectoryAccelerating(Trajectory):
    def __init__(self, position, velocity, acceleration):
        super().__init__(position, velocity)
        self.acceleration = np.array(acceleration, dtype=np.float64)
        self._update_functions.append(self._update_velocity)
    
    def _update_position(self, time_step: float):
        self.position += self.velocity * time_step + 0.5 * self.acceleration * time_step**2
    
    def _update_velocity(self, time_step: float):
        self.velocity += self.acceleration * time_step

class TrajectoryCircled(Trajectory):
    def __init__(self, position, velocity, center):
        super().__init__(position, velocity)
        self.center = np.array(center, dtype=np.float64)
        self.radius = np.sqrt((self.position[0] - self.center[0])**2 + 
                              (self.position[1] - self.center[1])**2)
        self.angle = math.atan2(self.position[1] - self.center[1],
                                self.position[0] - self.center[0])
        speed = np.linalg.norm(self.velocity[:2])
        self.angular_velocity = speed / self.radius
        cross = (self.position[0] - self.center[0]) * self.velocity[1] - \
                (self.position[1] - self.center[1]) * self.velocity[0]
        if cross < 0:
            self.angular_velocity = -self.angular_velocity
    
    def _update_position(self, time_step: float):
        self.angle += self.angular_velocity * time_step
        self.position[0] = self.center[0] + self.radius * math.cos(self.angle)
        self.position[1] = self.center[1] + self.radius * math.sin(self.angle)
        speed = self.radius * abs(self.angular_velocity)
        self.velocity[0] = -speed * math.sin(self.angle)
        self.velocity[1] = speed * math.cos(self.angle)

class TrajectorySinusoidal(Trajectory):
    def __init__(self, position, velocity, amplitude=100.0, frequency=0.1):
        super().__init__(position, velocity)
        self.amplitude = amplitude
        self.frequency = frequency
        self.base_velocity = self.velocity.copy()
        self.phase = 0.0
        self._initial_position = self.position.copy()
    
    def _update_position(self, time_step: float):
        self.phase += self.frequency * time_step
        velocity_norm = np.linalg.norm(self.base_velocity)
        if velocity_norm > 0:
            direction = self.base_velocity / velocity_norm
            perpendicular = np.array([-direction[1], direction[0], 0])
            self.position = self._initial_position + \
                           self.base_velocity * (self.phase / self.frequency) + \
                           perpendicular * (self.amplitude * math.sin(self.phase))
            self.velocity = self.base_velocity + \
                           perpendicular * (self.amplitude * self.frequency * math.cos(self.phase))

class TrajectoryComplex(Trajectory):
    def __init__(self, position, segments):
        super().__init__(position, velocity=(0, 0, 0))
        self.segments = segments
        self.current_segment_idx = 0
        self.segment_time = 0.0
        self.current_trajectory = None
        self._init_next_segment()
    
    def _init_next_segment(self):
        if self.current_segment_idx >= len(self.segments):
            self.current_segment_idx = 0
            self.segment_time = 0.0
        segment = self.segments[self.current_segment_idx]
        traj_type = segment["type"]
        params = segment["parameters"].copy()
        params["position"] = self.position
        if traj_type == "uniform":
            self.current_trajectory = TrajectoryUniform(**params)
        elif traj_type == "accelerating":
            self.current_trajectory = TrajectoryAccelerating(**params)
        elif traj_type == "circled":
            self.current_trajectory = TrajectoryCircled(**params)
        elif traj_type == "sinusoidal":
            self.current_trajectory = TrajectorySinusoidal(**params)
    
    def _update_position(self, time_step: float):
        self.current_trajectory.update(time_step)
        self.position = self.current_trajectory.get_position()
        self.velocity = self.current_trajectory.get_velocity()
        self.segment_time += time_step
        if self.segment_time >= self.segments[self.current_segment_idx]["duration"]:
            self.current_segment_idx += 1
            self.segment_time = 0.0
            self._init_next_segment()

def create_trajectory(traj_type: str, **kwargs) -> Trajectory:
    type_map = {
        "uniform": TrajectoryUniform,
        "accelerating": TrajectoryAccelerating,
        "circled": TrajectoryCircled,
        "sinusoidal": TrajectorySinusoidal,
        "complex": TrajectoryComplex
    }
    traj_class = type_map.get(traj_type.lower())
    if traj_class is None:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
    return traj_class(**kwargs)

trajectory_typename_to_class = {
    'uniform': TrajectoryUniform,
    'accelerating': TrajectoryAccelerating,
    'circled': TrajectoryCircled,
    'sinusoidal': TrajectorySinusoidal,
    'complex': TrajectoryComplex
}
