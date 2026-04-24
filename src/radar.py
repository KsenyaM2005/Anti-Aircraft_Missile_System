import numpy as np
import math
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from target import TargetStatus
from misc import dist, cartesian_to_spherical
from logs import radar_logger as logger
from event_types import EventBus, SimulationEvent, EventType


class RadarMode(Enum):
    """Radar operating modes."""
    SEARCH = auto()          # Scanning for new targets
    TRACK = auto()           # Tracking assigned targets
    MULTI_TRACK = auto()     # Track while scan
    IDLE = auto()            # Standby
    CALIBRATION = auto()     # Self-test/calibration


class TrackStatus(Enum):
    """Status of a radar track."""
    INITIATING = auto()      # Tentative track
    CONFIRMED = auto()       # Stable track
    COASTING = auto()        # Lost detection, predicting
    DROPPED = auto()         # Track terminated


@dataclass
class RadarMeasurement:
    """Raw radar measurement."""
    timestamp: float
    position: np.ndarray
    range: float
    azimuth: float
    elevation: float
    rcs: float
    snr: float  # Signal-to-noise ratio
    target_id: Optional[int] = None
    target_type: str = "UNKNOWN"


@dataclass
class Track:
    """Radar track of a target."""
    track_id: int
    target_id: Optional[int] = None
    target_type: str = "UNKNOWN"
    
    # State estimates
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Covariance matrices
    position_cov: np.ndarray = field(default_factory=lambda: np.eye(3) * 100)
    velocity_cov: np.ndarray = field(default_factory=lambda: np.eye(3) * 10)
    
    # Track quality
    status: TrackStatus = TrackStatus.INITIATING
    confidence: float = 0.5
    hits: int = 0
    misses: int = 0
    consecutive_misses: int = 0
    
    # RCS estimate
    estimated_rcs: float = 1.0
    
    # Timing
    first_detection: float = 0.0
    last_update: float = 0.0
    time_since_last_update: float = 0.0
    
    # History
    position_history: List[np.ndarray] = field(default_factory=list)
    max_history: int = 50
    
    def update_history(self):
        """Update position history."""
        self.position_history.append(self.position.copy())
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
    
    def predict(self, time_ahead: float) -> np.ndarray:
        """Predict future position."""
        return self.position + self.velocity * time_ahead + 0.5 * self.acceleration * time_ahead**2
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get track state as dictionary."""
        return {
            "track_id": self.track_id,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "status": self.status.name,
            "confidence": self.confidence,
            "estimated_rcs": self.estimated_rcs
        }


class KalmanFilter:
    """
    Kalman filter for target tracking.
    Implements a constant acceleration motion model.
    """
    
    def __init__(self, dim: int = 3, dt: float = 0.005):
        self.dim = dim
        self.dt = dt
        self.state_dim = dim * 3  # position, velocity, acceleration
        
        # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 100
        
        # State transition matrix
        self.F = np.eye(self.state_dim)
        for i in range(dim):
            self.F[i, i + dim] = dt
            self.F[i, i + 2*dim] = 0.5 * dt**2
            self.F[i + dim, i + 2*dim] = dt
        
        # Measurement matrix (only position measured)
        self.H = np.zeros((dim, self.state_dim))
        for i in range(dim):
            self.H[i, i] = 1.0
        
        # Process noise
        q_pos = 1.0
        q_vel = 0.1
        q_acc = 0.01
        self.Q = np.eye(self.state_dim)
        for i in range(dim):
            self.Q[i, i] = q_pos
            self.Q[i + dim, i + dim] = q_vel
            self.Q[i + 2*dim, i + 2*dim] = q_acc
        
        # Measurement noise (will be set per measurement)
        self.R = np.eye(dim) * 100
    
    def predict(self):
        """Prediction step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None):
        """Update step with measurement z."""
        if R is not None:
            self.R = R
        
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get position, velocity, acceleration estimates."""
        pos = self.x[0:self.dim]
        vel = self.x[self.dim:2*self.dim]
        acc = self.x[2*self.dim:3*self.dim]
        return pos, vel, acc
    
    def get_covariance(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get covariance matrices."""
        pos_cov = self.P[0:self.dim, 0:self.dim]
        vel_cov = self.P[self.dim:2*self.dim, self.dim:2*self.dim]
        return pos_cov, vel_cov


class Radar:
    """
    Radar / РЛС class.
    Implements:
    - Search and track modes
    - Kalman filtering
    - Track association and management
    - Noise handling
    """
    
    _id_counter = 0
    
    def __init__(self,
                 radar_id: Optional[int] = None,
                 position: Tuple[float, float, float] = (0, 0, 0),
                 config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None,
                 role: str = "search"):

        self.role = role  # "search" или "tracking"
        if role == "search":
            self.mode = RadarMode.SEARCH
            self.never_change_mode = True  # Нельзя переключать
        else:
            self.mode = RadarMode.SEARCH  # Изначально поиск, но может смениться
        
        self.id = radar_id if radar_id is not None else Radar._id_counter
        Radar._id_counter = max(Radar._id_counter, self.id + 1)
        
        self.position = np.array(position, dtype=np.float64)
        
        # Radar parameters
        self.time_step: float = 0.005
        self.omega_az = 120.0  # град/сек
        self.omega_el = 30.0  # град/сек
        self.r_max: float = 2000.0   # Maximum range (m)
        self.dr: float = 10.0        # Range resolution (m)
        self.beam_width_az: float = 2.0  # degrees
        self.beam_width_el: float = 2.0  # degrees
        
        # Current scan state
        self.mode: RadarMode = RadarMode.SEARCH
        self.current_azimuth: float = 0.0
        self.current_elevation: float = 0.0
        self.scan_azimuth_sector: Tuple[float, float] = (0, 360)
        self.scan_elevation_sector: Tuple[float, float] = (0, 90)
        self.current_range_window: Tuple[float, float] = (0.0, self.r_max)
        self._az_scan_direction: float = 1.0
        self._following_track_id: Optional[int] = None
        
        # Tracks
        self.tracks: Dict[int, Track] = {}
        self._next_track_id: int = 0
        
        # Raw measurements
        self.raw_measurements: List[RadarMeasurement] = []
        self.curr_ray_x: List[float] = []
        self.curr_ray_y: List[float] = []
        self.curr_ray_z: List[float] = []
        self.curr_beam_xy: List[Tuple[float, float]] = []
        self.curr_beam_xz: List[Tuple[float, float]] = []
        
        # Track association
        self.association_gate: float = 200.0  # Association gate (m)
        self.confirmation_threshold: int = 3  # Hits needed for confirmation
        self.drop_threshold: int = 5  # Consecutive misses to drop
        
        # Kalman filter parameters
        self.use_kalman: bool = True
        self.kalman_filters: Dict[int, KalmanFilter] = {}
        
        # Performance metrics
        self.detection_count: int = 0
        self.false_alarm_count: int = 0
        self.track_count: int = 0
        
        # Event bus
        self.event_bus = event_bus or EventBus()
        self._setup_event_handlers()
        
        # Legacy compatibility
        self.state: int = 0
        self.rays: List[Any] = []
        
        if config is not None:
            self._load_config(config)
        
        logger.info(f"Radar {self.id} initialized at {self.position}")

    def set_mode(self, mode):
        if getattr(self, 'never_change_mode', False):
            return  # Центральная РЛС не меняет режим
        self.mode = mode

    def _setup_event_handlers(self) -> None:
        """Subscribe to PBU-issued radar control commands."""
        self.event_bus.subscribe(
            EventType.RADAR_CONTROL_COMMAND,
            self._handle_control_command
        )

    def _handle_control_command(self, event: SimulationEvent) -> None:
        """Apply a control command from the dispatcher or PBU."""
        data = event.data or {}
        radar_id = data.get("radar_id")
        if radar_id is not None and radar_id != self.id:
            return

        mode_name = data.get("mode")
        if mode_name:
            try:
                self.set_mode(RadarMode[mode_name.upper()])
            except KeyError:
                logger.warning(f"Radar {self.id}: Unknown mode {mode_name}")

        scan_az = data.get("scan_sector_az")
        scan_el = data.get("scan_sector_el")
        if scan_az is not None and scan_el is not None:
            self.set_scan_sector(tuple(scan_az), tuple(scan_el))

        track_id = data.get("track_id")
        if track_id is not None and track_id in self.tracks:
            self.assign_track_for_following(track_id)

    def _load_config(self, config: Dict[str, Any]):
        """Load configuration from dictionary."""
        self.time_step = config.get("time_step", self.time_step)
        self.omega_az = config.get("omega_az", self.omega_az)
        self.omega_el = config.get("omega_el", self.omega_el)
        self.r_max = config.get("r_max", self.r_max)
        self.dr = config.get("dr", self.dr)
        self.beam_width_az = config.get("beam_width_az", self.beam_width_az)
        self.beam_width_el = config.get("beam_width_el", self.beam_width_el)
        self.current_range_window = (0.0, self.r_max)
        
        coords = config.get("position") or config.get("coordinates")
        if coords is not None:
            self.position = np.array([coords["x"], coords["y"], coords["z"]])

        if "scan_sector_az" in config:
            self.scan_azimuth_sector = tuple(config["scan_sector_az"])
        if "scan_sector_el" in config:
            self.scan_elevation_sector = tuple(config["scan_sector_el"])
        self.current_azimuth = config.get("initial_azimuth", self.scan_azimuth_sector[0]) % 360
        self.current_elevation = float(config.get("initial_elevation", self.scan_elevation_sector[0]))
        self._update_beam_geometry()
    
    def initialize_with_file_data(self, config: Dict[str, Any]) -> bool:
        """Legacy initialization method."""
        self._load_config(config)
        
        # Initialize legacy ray
        self.rays = [LegacyRay(self.omega_az, self.omega_el, self.r_max)]
        self.state = 0
        
        logger.info(f"Radar {self.id}: initialization performed using config file")
        return True
    
    def update(self, time_step: float, environment) -> None:
        """
        Main update method called by dispatcher.
        Performs radar scan and track updates.
        """
        self.time_step = time_step
        self._predict_tracks()
        
        if self.mode == RadarMode.SEARCH:
            self._perform_search_scan(environment)
        elif self.mode == RadarMode.TRACK:
            self._perform_track_update(environment)
        elif self.mode == RadarMode.MULTI_TRACK:
            self._perform_multi_track_scan(environment)

        # Clean up old tracks
        self._cleanup_tracks()
        self._publish_track_updates()

    def _perform_search_scan(self, environment) -> None:
        """Perform search scan pattern."""
        self._advance_search_pattern()
        self.current_range_window = (0.0, self.r_max)
        self._update_beam_geometry()

        targets = environment.get_targets()

        for target_id, target in targets.items():
            if target.status != TargetStatus.ACTIVE:
                continue

            if not self._is_target_in_beam(target.position):
                continue

            measurement = environment.get_noisy_measurement(target_id, self.position)
            if measurement is None:
                continue

            target.mark_detected(f"radar_{self.id}")
            self._process_detection(target_id, measurement)
    
    def _perform_track_update(self, environment) -> None:
        """Update existing tracks with new measurements."""
        targets = environment.get_targets()

        track_ids = self._get_tracks_for_following()
        if not track_ids:
            self._perform_search_scan(environment)
            return

        first_track = self.tracks.get(track_ids[0])
        if first_track is not None:
            self._point_beam_at_position(first_track.position, range_window=self._tracking_range_window(first_track.position))

        for track_id in track_ids:
            track = self.tracks.get(track_id)
            if track is None or track.target_id is None:
                continue

            target = targets.get(track.target_id)
            if target is None or target.status != TargetStatus.ACTIVE:
                self._mark_track_missed(track)
                continue

            predicted_pos = track.position.copy()
            self._point_beam_at_position(predicted_pos, range_window=self._tracking_range_window(predicted_pos))

            if not self._is_target_in_beam(target.position):
                self._mark_track_missed(track)
                continue

            measurement = environment.get_noisy_measurement(track.target_id, self.position)
            if measurement is None:
                self._mark_track_missed(track)
                continue

            target.mark_detected(f"radar_{self.id}")
            self._update_track_with_measurement(track_id, measurement)
    
    def _perform_multi_track_scan(self, environment) -> None:
        """Track while scan mode."""
        self._perform_search_scan(environment)
        self._perform_track_update(environment)
    
    def _process_detection(self, target_id: int, measurement: Dict[str, Any]) -> None:
        """Process a target detection."""
        self.detection_count += 1
        
        # Create measurement object
        r, az, el = cartesian_to_spherical(
            measurement["position"][0] - self.position[0],
            measurement["position"][1] - self.position[1],
            measurement["position"][2] - self.position[2]
        )
        
        radar_meas = RadarMeasurement(
            timestamp=measurement["timestamp"],
            position=measurement["position"],
            range=r,
            azimuth=math.degrees(az),
            elevation=math.degrees(el),
            rcs=measurement["rcs"],
            snr=self._calculate_snr(r, measurement["rcs"]),
            target_id=target_id,
            target_type=measurement.get("target_type", "UNKNOWN")
        )
        
        self.raw_measurements.append(radar_meas)
        if len(self.raw_measurements) > 1000:
            self.raw_measurements.pop(0)
        
        # Associate with existing tracks
        associated_track_id = self._associate_measurement(radar_meas)
        
        if associated_track_id is not None:
            self._update_track_with_measurement(associated_track_id, measurement)
        else:
            # Create new tentative track
            self._create_new_track(radar_meas)
    
    def _associate_measurement(self, meas: RadarMeasurement) -> Optional[int]:
        """Associate measurement with existing track."""
        best_track_id = None
        best_distance = self.association_gate
        
        for track_id, track in self.tracks.items():
            predicted_pos = track.position
            d = dist(predicted_pos, meas.position)
            
            if d < best_distance:
                best_distance = d
                best_track_id = track_id
        
        return best_track_id
    
    def _create_new_track(self, meas: RadarMeasurement) -> int:
        """Create a new tentative track."""
        track_id = self._next_track_id
        self._next_track_id += 1
        
        track = Track(
            track_id=track_id,
            target_id=meas.target_id,
            target_type=meas.target_type,
            position=meas.position.copy(),
            estimated_rcs=meas.rcs,
            hits=1,
            confidence=0.4,
            first_detection=meas.timestamp,
            last_update=meas.timestamp
        )
        track.update_history()
        
        self.tracks[track_id] = track
        
        if self.use_kalman:
            kf = KalmanFilter(dt=self.time_step)
            kf.x[0:3] = meas.position
            self.kalman_filters[track_id] = kf
        
        logger.debug(f"Radar {self.id}: Created new track {track_id}")
        return track_id
    
    def _update_track_with_measurement(self, track_id: int, 
                                       measurement: Dict[str, Any]) -> None:
        """Update track with new measurement."""
        track = self.tracks.get(track_id)
        if track is None:
            return
        
        meas_pos = measurement["position"]
        
        if self.use_kalman and track_id in self.kalman_filters:
            kf = self.kalman_filters[track_id]
            
            # Set measurement noise based on SNR
            snr = self._calculate_snr(
                dist(self.position, meas_pos),
                measurement["rcs"]
            )
            noise_std = 10.0 / (1.0 + snr)
            R = np.eye(3) * noise_std**2
            
            kf.update(meas_pos, R)
            pos, vel, acc = kf.get_state()
            pos_cov, vel_cov = kf.get_covariance()
            
            track.position = pos
            track.velocity = vel
            track.acceleration = acc
            track.position_cov = pos_cov
            track.velocity_cov = vel_cov
        else:
            # Simple alpha-beta filter
            alpha = 0.7
            beta = 0.3
            
            predicted_pos = track.position.copy()
            track.velocity = track.velocity + beta * (meas_pos - predicted_pos) / max(self.time_step, 1e-6)
            track.position = track.position + alpha * (meas_pos - track.position)
        
        track.target_type = measurement.get("target_type", track.target_type)
        track.estimated_rcs = 0.8 * track.estimated_rcs + 0.2 * measurement["rcs"]
        track.hits += 1
        track.confidence = min(1.0, 0.3 + 0.15 * track.hits - 0.05 * track.misses)
        track.last_update = measurement["timestamp"]
        track.time_since_last_update = 0.0
        track.consecutive_misses = 0
        track.update_history()
        
        # Update track status
        if track.status == TrackStatus.INITIATING:
            if track.hits >= self.confirmation_threshold:
                track.status = TrackStatus.CONFIRMED
                self._publish_track_confirmed(track)
        
        elif track.status == TrackStatus.COASTING:
            track.status = TrackStatus.CONFIRMED
    
    def _predict_tracks(self) -> None:
        """Predict all tracks forward in time."""
        for track_id, track in self.tracks.items():
            if track.status in [TrackStatus.DROPPED]:
                continue
            
            track.time_since_last_update += self.time_step
            
            if self.use_kalman and track_id in self.kalman_filters:
                self.kalman_filters[track_id].predict()
                pos, vel, acc = self.kalman_filters[track_id].get_state()
                track.position = pos
                track.velocity = vel
                track.acceleration = acc
            else:
                track.position = track.predict(self.time_step)

            if track.status == TrackStatus.CONFIRMED and track.time_since_last_update > self.time_step * 1.5:
                track.status = TrackStatus.COASTING
    
    def _cleanup_tracks(self) -> None:
        """Remove dropped tracks."""
        to_remove = []
        
        for track_id, track in self.tracks.items():
            if track.consecutive_misses >= self.drop_threshold:
                track.status = TrackStatus.DROPPED
                to_remove.append(track_id)
                self._publish_track_dropped(track)
        
        for track_id in to_remove:
            del self.tracks[track_id]
            if track_id in self.kalman_filters:
                del self.kalman_filters[track_id]
    
    def _beam_position(self, r: float) -> Tuple[float, float, float]:
        """Get beam position at range r."""
        az_rad = math.radians(self.current_azimuth)
        el_rad = math.radians(self.current_elevation)
        
        x = self.position[0] + r * math.cos(el_rad) * math.cos(az_rad)
        y = self.position[1] + r * math.cos(el_rad) * math.sin(az_rad)
        z = self.position[2] + r * math.sin(el_rad)
        
        return x, y, z

    def _is_target_in_beam(self, target_pos: np.ndarray, r: Optional[float] = None) -> bool:
        """Check if target is within radar beam."""
        relative = target_pos - self.position
        rel_r = np.linalg.norm(relative)

        if rel_r < 1.0 or rel_r > self.r_max:
            return False

        target_az = math.degrees(math.atan2(relative[1], relative[0])) % 360
        target_el = math.degrees(math.asin(relative[2] / rel_r))

        az_diff = abs((target_az - self.current_azimuth + 180) % 360 - 180)
        el_diff = abs(target_el - self.current_elevation)

        range_min, range_max = self.current_range_window
        if r is not None:
            range_min = max(range_min, max(0.0, r - self.dr))
            range_max = min(range_max, r + self.dr)

        return (
            az_diff <= self.beam_width_az / 2 and
            el_diff <= self.beam_width_el / 2 and
            range_min <= rel_r <= range_max
        )
    
    def _can_see_position(self, pos: np.ndarray) -> bool:
        """Check if position is within radar coverage."""
        relative = pos - self.position
        r = np.linalg.norm(relative)
        
        if r > self.r_max:
            return False
        
        az = math.degrees(math.atan2(relative[1], relative[0])) % 360
        el = math.degrees(math.asin(relative[2] / r))
        
        return (
            self.scan_azimuth_sector[0] <= az <= self.scan_azimuth_sector[1] and
            self.scan_elevation_sector[0] <= el <= self.scan_elevation_sector[1]
        )
    
    def _calculate_snr(self, range_m: float, rcs: float) -> float:
        """Calculate signal-to-noise ratio."""
        # Simplified radar equation
        if range_m < 1.0:
            range_m = 1.0
        
        snr = (rcs / (range_m ** 4)) * 1e12
        return max(0.1, snr)
    
    def _publish_track_confirmed(self, track: Track) -> None:
        """Publish track confirmed event."""
        self.event_bus.publish(SimulationEvent(
            event_type=EventType.RADAR_TRACK_INITIATED,
            source_id=f"radar_{self.id}",
            target_id=f"track_{track.track_id}",
            data=track.get_state_dict()
        ))
        logger.info(f"Radar {self.id}: Track {track.track_id} confirmed")

    def _publish_track_updates(self) -> None:
        """Publish confirmed track data for PBU and GUI consumers."""
        for track in self.get_tracks().values():
            self.event_bus.publish(SimulationEvent(
                event_type=EventType.TARGET_TRACK_UPDATED,
                source_id=f"radar_{self.id}",
                target_id=f"target_{track.target_id}" if track.target_id is not None else f"track_{track.track_id}",
                data={
                    "radar_id": self.id,
                    **track.get_state_dict()
                }
            ))
    
    def _publish_track_dropped(self, track: Track) -> None:
        """Publish track dropped event."""
        self.event_bus.publish(SimulationEvent(
            event_type=EventType.RADAR_TRACK_DROPPED,
            source_id=f"radar_{self.id}",
            target_id=f"track_{track.track_id}"
        ))
        logger.info(f"Radar {self.id}: Track {track.track_id} dropped")
    
    def get_tracks(self) -> Dict[int, Track]:
        """Get all confirmed tracks."""
        return {tid: t for tid, t in self.tracks.items() 
                if t.status == TrackStatus.CONFIRMED}
    
    def get_track_data_for_pbu(self) -> List[Dict[str, Any]]:
        """Get track data formatted for PBU."""
        tracks_data = []
        
        for track_id, track in self.get_tracks().items():
            tracks_data.append({
                "track_id": track_id,
                "target_id": track.target_id,
                "position": track.position.tolist(),
                "velocity": track.velocity.tolist(),
                "confidence": track.confidence,
                "estimated_rcs": track.estimated_rcs,
                "status": track.status.name
            })
        
        return tracks_data
    
    # def set_mode(self, mode: RadarMode) -> None:
    #     """Set radar operating mode."""
    #     self.mode = mode
    #     logger.info(f"Radar {self.id}: Mode set to {mode.name}")
    
    def set_scan_sector(self, azimuth: Tuple[float, float], 
                        elevation: Tuple[float, float]) -> None:
        """Set scan sector."""
        self.scan_azimuth_sector = azimuth
        self.scan_elevation_sector = elevation
        logger.info(f"Radar {self.id}: Scan sector set to az={azimuth}, el={elevation}")
    
    def assign_track_for_following(self, track_id: int) -> None:
        """Assign a track for dedicated following."""
        if track_id in self.tracks:
            self._following_track_id = track_id
            if self.mode != RadarMode.MULTI_TRACK:
                self.mode = RadarMode.TRACK
            logger.info(f"Radar {self.id}: Following track {track_id}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get radar state for serialization."""
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "mode": self.mode.name,
            "current_azimuth": self.current_azimuth,
            "current_elevation": self.current_elevation,
            "tracks": [t.get_state_dict() for t in self.tracks.values()],
            "statistics": {
                "detections": self.detection_count,
                "false_alarms": self.false_alarm_count,
                "active_tracks": len(self.tracks)
            }
        }

    def _advance_search_pattern(self):
        az_start, az_end = self.scan_azimuth_sector
        el_start, el_end = self.scan_elevation_sector

        # Сохраняем старый азимут для проверки пересечения границы
        old_azimuth = self.current_azimuth

        # Движение по азимуту
        self.current_azimuth += self.omega_az * self.time_step * self._az_scan_direction

        # ПРОВЕРКА: пересекли ли границу азимута?
        crossed_boundary = False

        if self.current_azimuth > az_end:
            self.current_azimuth = az_end
            self._az_scan_direction = -1.0
            crossed_boundary = True

        elif self.current_azimuth < az_start:
            self.current_azimuth = az_start
            self._az_scan_direction = 1.0
            crossed_boundary = True

        # ЕСЛИ пересекли границу - увеличиваем угол места!
        if crossed_boundary:
            self.current_elevation += self.omega_el * self.time_step
            #self.current_elevation += 0.5*self.beam_width_el
            print(
                f"Radar {self.id}: boundary crossed! elevation += {self.omega_el * self.time_step:.2f} -> {self.current_elevation:.1f}°")

        # Сброс угла места при достижении максимума
        if self.current_elevation >= el_end:
            self.current_elevation = el_start
            print(f"Radar {self.id}: elevation reset to {self.current_elevation:.1f}°")

    def _point_beam_at_position(
        self,
        position: np.ndarray,
        range_window: Optional[Tuple[float, float]] = None
    ) -> None:
        """Point the beam center at a specific position."""
        relative = position - self.position
        r = np.linalg.norm(relative)
        if r < 1.0:
            return

        self.current_azimuth = math.degrees(math.atan2(relative[1], relative[0])) % 360
        self.current_elevation = math.degrees(math.asin(relative[2] / r))
        self.current_range_window = range_window or (0.0, min(self.r_max, r))
        self._update_beam_geometry()

    def _tracking_range_window(self, position: np.ndarray) -> Tuple[float, float]:
        """Use a narrow radial window for dedicated tracking."""
        distance = dist(self.position, position)
        spread = max(150.0, self.dr * 10.0)
        return (max(0.0, distance - spread), min(self.r_max, distance + spread))

    def _mark_track_missed(self, track: Track) -> None:
        """Update bookkeeping when a track was not observed on this tick."""
        track.consecutive_misses += 1
        track.misses += 1
        track.confidence = max(0.0, track.confidence - 0.08)
        if track.status == TrackStatus.CONFIRMED:
            track.status = TrackStatus.COASTING

    def _get_tracks_for_following(self) -> List[int]:
        """Resolve which tracks should be serviced in tracking mode."""
        if self._following_track_id is not None and self._following_track_id in self.tracks:
            return [self._following_track_id]

        confirmed_tracks = sorted(self.get_tracks().keys())
        return confirmed_tracks[:1]

    def _update_beam_geometry(self) -> None:
        """Build beam sector polygons for GUI rendering."""
        range_min, range_max = self.current_range_window
        range_max = min(range_max, self.r_max)
        if range_max <= 0:
            self.curr_beam_xy = []
            self.curr_beam_xz = []
            self.curr_ray_x = []
            self.curr_ray_y = []
            self.curr_ray_z = []
            return

        az_min = math.radians(self.current_azimuth - self.beam_width_az / 2)
        az_max = math.radians(self.current_azimuth + self.beam_width_az / 2)
        el_min = math.radians(self.current_elevation - self.beam_width_el / 2)
        el_max = math.radians(self.current_elevation + self.beam_width_el / 2)
        el_center = math.radians(self.current_elevation)
        az_center = math.radians(self.current_azimuth)

        samples = 12
        az_values = np.linspace(az_min, az_max, samples)
        outer_xy = [
            (
                self.position[0] + range_max * math.cos(el_center) * math.cos(az),
                self.position[1] + range_max * math.cos(el_center) * math.sin(az),
            )
            for az in az_values
        ]

        if range_min > 0:
            inner_xy = [
                (
                    self.position[0] + range_min * math.cos(el_center) * math.cos(az),
                    self.position[1] + range_min * math.cos(el_center) * math.sin(az),
                )
                for az in reversed(az_values)
            ]
            self.curr_beam_xy = outer_xy + inner_xy
        else:
            self.curr_beam_xy = [tuple(self.position[:2])] + outer_xy

        el_values = np.linspace(el_min, el_max, samples)
        outer_xz = [
            (
                self.position[0] + range_max * math.cos(el) * math.cos(az_center),
                self.position[2] + range_max * math.sin(el),
            )
            for el in el_values
        ]
        if range_min > 0:
            inner_xz = [
                (
                    self.position[0] + range_min * math.cos(el) * math.cos(az_center),
                    self.position[2] + range_min * math.sin(el),
                )
                for el in reversed(el_values)
            ]
            self.curr_beam_xz = outer_xz + inner_xz
        else:
            self.curr_beam_xz = [(self.position[0], self.position[2])] + outer_xz

        ray_distances = np.linspace(range_min, range_max, max(3, int(range_max / max(self.dr, 1.0))))
        self.curr_ray_x = []
        self.curr_ray_y = []
        self.curr_ray_z = []
        for distance in ray_distances:
            x, y, z = self.position + np.array([
                distance * math.cos(el_center) * math.cos(az_center),
                distance * math.cos(el_center) * math.sin(az_center),
                distance * math.sin(el_center)
            ])
            self.curr_ray_x.append(float(x))
            self.curr_ray_y.append(float(y))
            self.curr_ray_z.append(float(z))
    
    # Legacy compatibility methods
    def do_step(self, env, pbu) -> None:
        """Legacy step method."""
        self.update(self.time_step, env)
        
        # Send tracks to PBU
        for track_data in self.get_track_data_for_pbu():
            pbu.process_radar_track(self.id, track_data)
    
    def add_ray(self, phi, teta, m_id, id, pbu_target_id) -> None:
        """Legacy method for adding tracking ray."""
        pass
    
    def del_ray(self, index, PBU) -> None:
        """Legacy method for deleting tracking ray."""
        pass


class LegacyRay:
    """Legacy ray class for compatibility."""
    def __init__(self, omega_az, omega_el, r_max):
        self.omega_az = omega_az
        self.omega_el = omega_el
        self.phi = random.random() * 2 * math.pi
        self.teta = random.random() * math.pi / 2
        self.Rmax = r_max
    
    def upd_coord(self):
        self.phi = (self.phi + self.omega_az) % (2 * math.pi)
        self.teta = (self.teta + self.omega_el) % (math.pi / 2)


# Alias for compatibility
Locator = Radar
