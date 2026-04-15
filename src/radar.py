import numpy as np
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class TrackType(Enum):
    """Тип траектории сопровождения."""
    INITIATING = auto()  # Начальная траектория
    TENTATIVE = auto()  # Пробная траектория
    CONFIRMED = auto()  # Подтверждённая траектория
    TERMINATED = auto()  # Завершённая траектория


class RadarMode(Enum):
    """Режимы работы РЛС."""
    SEARCH = auto()  # Поиск
    TRACK = auto()  # Сопровождение
    MULTI_TRACK = auto()  # Сопровождение со сканированием
    IDLE = auto()  # Ожидание
    CALIBRATION = auto()  # Калибровка


@dataclass
class Measurement:
    """Измерение от РЛС."""
    timestamp: float
    position: np.ndarray  # Позиция в декартовых координатах
    range_m: float  # Дальность, м
    azimuth_rad: float  # Азимут, рад
    elevation_rad: float  # Угол места, рад
    rcs: float  # ЭПР
    snr_db: float  # Отношение сигнал/шум, дБ
    target_id: Optional[int] = None


@dataclass
class PriorMissileInfo:
    """Предварительная информация о ракете."""
    launch_time: float
    estimated_range: float
    estimated_velocity: float
    estimated_heading: float
    target_id: Optional[int] = None


class Transmitter:
    """
    Передатчик РЛС.
    P_t - мощность передатчика
    G - коэффициент усиления антенны
    λ - длина волны
    """

    def __init__(self, power_w: float = 10000.0, gain_db: float = 40.0,
                 frequency_hz: float = 3e9, bandwidth_hz: float = 1e6):
        self.P_t = power_w  # Мощность передатчика, Вт
        self.G = 10 ** (gain_db / 10)  # Коэффициент усиления
        self.frequency = frequency_hz  # Частота, Гц
        self.B = bandwidth_hz  # Полоса пропускания, Гц
        self.lam = 3e8 / frequency_hz  # Длина волны, м


class Receiver:
    """
    Приёмник РЛС.
    T_s - шумовая температура
    L - потери
    k - постоянная Больцмана
    θ_0.5,β - ширина луча по азимуту
    θ_0.5,ε - ширина луча по углу места
    k_m - коэффициент для ошибок углов
    """

    def __init__(self, noise_temp_k: float = 290.0, losses_db: float = 3.0,
                 azimuth_beamwidth_rad: float = 0.0175,
                 elevation_beamwidth_rad: float = 0.0175,
                 km: float = 1.0):
        self.T_s = noise_temp_k  # Шумовая температура, К
        self.L = 10 ** (losses_db / 10)  # Потери
        self.theta_0_5_beta = azimuth_beamwidth_rad  # Ширина луча по азимуту
        self.theta_0_5_epsilon = elevation_beamwidth_rad  # Ширина луча по углу места
        self.k_m = km  # Коэффициент для ошибок углов
        self.k = 1.38e-23  # Постоянная Больцмана

    def calculate_snr(self, R: float, transmitter: Transmitter, sigma: float = 1.0) -> float:
        """
        Расчёт SNR по уравнению радиолокации.

        SNR = (P_t * G^2 * λ^2 * σ) / ((4π)^3 * R^4 * k * T_s * B * L)
        """
        P_t = transmitter.P_t
        G = transmitter.G
        lam = transmitter.lam
        B = transmitter.B

        numerator = P_t * (G ** 2) * (lam ** 2) * sigma
        denominator = ((4 * np.pi) ** 3) * (R ** 4) * self.k * self.T_s * B * self.L

        if denominator == 0:
            return 0.0

        snr_linear = numerator / denominator
        snr_db = 10 * np.log10(max(snr_linear, 1e-10))

        return snr_db

    def calculate_errors(self, S: float, transmitter: Transmitter) -> Tuple[float, float, float]:
        """
        Расчёт ошибок измерений:
        σ_R = c / (2B√(2S))
        σ_β = θ_0.5,β / (k_m√(2S))
        σ_ε = θ_0.5,ε / (k_m√(2S))

        где S = SNR (линейное значение)
        """
        if S <= 0:
            return (float('inf'), float('inf'), float('inf'))

        c = 3e8  # Скорость света
        B = transmitter.B

        sqrt_2S = np.sqrt(2 * S)

        sigma_R = c / (2 * B * sqrt_2S)
        sigma_beta = self.theta_0_5_beta / (self.k_m * sqrt_2S)
        sigma_epsilon = self.theta_0_5_epsilon / (self.k_m * sqrt_2S)

        return (sigma_R, sigma_beta, sigma_epsilon)


class Association:
    """
    Класс ассоциации измерений с траекториями.
    """

    def __init__(self, gate_threshold: float = 200.0):
        self.gate_threshold = gate_threshold

    def associate(self, measurement: Measurement, tracks: Dict[int, 'Track']) -> Optional[int]:
        """Ассоциация измерения с существующей траекторией."""
        best_track_id = None
        best_distance = self.gate_threshold

        for track_id, track in tracks.items():
            predicted_pos = track.get_predicted_position()
            d = np.linalg.norm(measurement.position - predicted_pos)

            if d < best_distance:
                best_distance = d
                best_track_id = track_id

        return best_track_id


class KalmanFilter:
    """
    Фильтр Калмана для сопровождения цели.
    """

    def __init__(self, dt: float = 0.005):
        self.dt = dt
        self.dim = 3
        self.state_dim = 9  # [x, y, z, vx, vy, vz, ax, ay, az]

        # Матрица перехода состояния F
        self.F = self._build_F()

        # Матрица наблюдения H
        self.H = self._build_H()

        # Матрица шума процесса Q
        self.Q = self._build_Q()

        # Матрица шума измерений R
        self.R = self._build_R()

        # Состояние
        self.x = None  # Вектор состояния
        self.P = None  # Ковариационная матрица

    def _build_F(self) -> np.ndarray:
        """Построение матрицы перехода состояния."""
        dt, dt2_2 = self.dt, self.dt ** 2 / 2

        F = np.eye(self.state_dim)

        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        F[0, 6] = dt2_2
        F[1, 7] = dt2_2
        F[2, 8] = dt2_2

        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt

        return F

    def _build_H(self) -> np.ndarray:
        """Построение матрицы наблюдения."""
        H = np.zeros((self.dim, self.state_dim))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        return H

    def _build_Q(self) -> np.ndarray:
        """Построение матрицы шума процесса."""
        dt, dt2_2, dt3_6 = self.dt, self.dt ** 2 / 2, self.dt ** 3 / 6
        q = 1.0

        Q = np.zeros((self.state_dim, self.state_dim))

        for i in range(self.dim):
            Q[i, i] = q * dt3_6
            Q[i, i + 3] = q * dt2_2
            Q[i + 3, i] = q * dt2_2
            Q[i + 3, i + 3] = q * dt

        return Q

    def _build_R(self) -> np.ndarray:
        """Построение матрицы шума измерений."""
        sigma_range = 10.0
        sigma_angle = 0.01
        return np.diag([sigma_range ** 2, sigma_angle ** 2, sigma_angle ** 2])

    def initialize(self, measurement: Measurement) -> None:
        """Инициализация фильтра."""
        self.x = np.zeros(self.state_dim)
        self.x[0:3] = measurement.position

        self.P = np.eye(self.state_dim)
        self.P[0:3, 0:3] *= 100.0
        self.P[3:6, 3:6] *= 1000.0
        self.P[6:9, 6:9] *= 100.0

    def time_update(self) -> None:
        """
        1. Time update:
           Predicted state: x̂_{k|k-1} = F_{k-1} x̂_{k-1|k-1}
           Predicted covariance: P_{k|k-1} = F_{k-1} P_{k-1|k-1} (F_{k-1})^T + Q_{k-1}
        """
        if self.x is None:
            return

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def measurement_update(self, z: np.ndarray, R: Optional[np.ndarray] = None) -> np.ndarray:
        """
        2. Measurement update:
           Measurement residual: ẑ_k = z_k - H_k x̂_{k|k-1}
           Residual covariance: S_k = H_k P_{k|k-1} (H_k)^T + R_k
           Filter gain: K_k = P_{k|k-1} (H_k)^T S_k^{-1}
           Update state: x̂_{k|k} = x̂_{k|k-1} + K_k ẑ_k
           Update covariance: P_{k|k} = P_{k|k-1} - K_k S_k (K_k)^T
        """
        if self.x is None:
            return z

        if R is not None:
            self.R = R

        z_tilde = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ z_tilde
        self.P = self.P - K @ S @ K.T

        return self.x[:3]

    def get_position(self) -> np.ndarray:
        """Получение оценки позиции."""
        return self.x[0:3] if self.x is not None else np.zeros(3)

    def get_velocity(self) -> np.ndarray:
        """Получение оценки скорости."""
        return self.x[3:6] if self.x is not None else np.zeros(3)


class Track:
    """
    Траектория сопровождения цели.
    """

    def __init__(self, track_id: int, first_measurement: Measurement, dt: float = 0.005):
        self.id = track_id
        self.type = TrackType.INITIATING
        self.kalman_filter = KalmanFilter(dt=dt)
        self.kalman_filter.initialize(first_measurement)
        self.measurements: List[Measurement] = [first_measurement]
        self.hits_count = 1
        self.misses_count = 0
        self.last_update = first_measurement.timestamp
        self.target_id = first_measurement.target_id

    def update(self, measurement: Measurement) -> None:
        """Обновление траектории новым измерением."""
        self.kalman_filter.time_update()
        self.kalman_filter.measurement_update(measurement.position)
        self.measurements.append(measurement)
        self.hits_count += 1
        self.misses_count = 0
        self.last_update = measurement.timestamp

        if self.type == TrackType.INITIATING and self.hits_count >= 3:
            self.type = TrackType.TENTATIVE
        elif self.type == TrackType.TENTATIVE and self.hits_count >= 5:
            self.type = TrackType.CONFIRMED

    def miss(self) -> None:
        """Пропуск измерения."""
        self.misses_count += 1
        if self.misses_count >= 5:
            self.type = TrackType.TERMINATED

    def predict(self, dt: float) -> None:
        """Предсказание состояния."""
        self.kalman_filter.time_update()

    def get_predicted_position(self) -> np.ndarray:
        """Получение предсказанной позиции."""
        return self.kalman_filter.get_position()

    def get_position(self) -> np.ndarray:
        """Получение текущей позиции."""
        return self.kalman_filter.get_position()

    def get_velocity(self) -> np.ndarray:
        """Получение текущей скорости."""
        return self.kalman_filter.get_velocity()

    def is_confirmed(self) -> bool:
        """Проверка, подтверждена ли траектория."""
        return self.type == TrackType.CONFIRMED


class RadarSystem:
    """
    Основной класс радиолокационной системы.
    Агрегирует все компоненты архитектуры.
    """

    def __init__(self, radar_id: int = 0, position: np.ndarray = None, dt: float = 0.005):
        self.id = radar_id
        self.position = position if position is not None else np.zeros(3)
        self.dt = dt

        # Компоненты архитектуры
        self.transmitter = Transmitter()
        self.receiver = Receiver()
        self.association = Association()
        self.prior_missile_info: Optional[PriorMissileInfo] = None

        # Траектории
        self.tracks: Dict[int, Track] = {}
        self._next_track_id = 0

        # Параметры сканирования
        self.current_azimuth = 0.0
        self.current_elevation = 0.0
        self.omega_az = np.radians(15.0) * dt
        self.omega_el = np.radians(8.0) * dt
        self.r_max = 2000.0
        self.dr = 10.0
        self.beam_width_az = np.radians(2.0)
        self.beam_width_el = np.radians(2.0)

        # Режим работы (для совместимости с GUI)
        self.mode = RadarMode.SEARCH

        # Для визуализации лучей (для совместимости с GUI)
        self.curr_ray_x: List[float] = []
        self.curr_ray_y: List[float] = []
        self.curr_ray_z: List[float] = []

        # Пороги
        self.confirmation_threshold = 3
        self.drop_threshold = 5

        # Статистика
        self.detection_count = 0

    def set_prior_missile_info(self, info: PriorMissileInfo) -> None:
        """Установка предварительной информации о ракете."""
        self.prior_missile_info = info

    def set_mode(self, mode: RadarMode) -> None:
        """Установка режима работы."""
        self.mode = mode

    def update_scan(self) -> None:
        """Обновление позиции сканирования."""
        self.current_azimuth = (self.current_azimuth + self.omega_az) % (2 * np.pi)
        self.current_elevation = (self.current_elevation + self.omega_el) % (np.pi / 2)

        # Обновление лучей для визуализации (для совместимости с GUI)
        self._update_beam_points()

    def _update_beam_points(self) -> None:
        """Обновление точек луча для визуализации."""
        self.curr_ray_x = []
        self.curr_ray_y = []
        self.curr_ray_z = []

        az = self.current_azimuth
        el = self.current_elevation

        for r in np.arange(0, self.r_max, self.dr):
            x = self.position[0] + r * np.cos(el) * np.cos(az)
            y = self.position[1] + r * np.cos(el) * np.sin(az)
            z = self.position[2] + r * np.sin(el)

            self.curr_ray_x.append(x)
            self.curr_ray_y.append(y)
            self.curr_ray_z.append(z)

    def get_beam_direction(self) -> Tuple[float, float]:
        """Получение текущего направления луча."""
        return self.current_azimuth, self.current_elevation

    def get_beam_position(self, r: float) -> np.ndarray:
        """Получение позиции луча на дальности r."""
        az = self.current_azimuth
        el = self.current_elevation
        x = self.position[0] + r * np.cos(el) * np.cos(az)
        y = self.position[1] + r * np.cos(el) * np.sin(az)
        z = self.position[2] + r * np.sin(el)
        return np.array([x, y, z])

    def is_target_in_beam(self, target_pos: np.ndarray, r: float) -> bool:
        """Проверка, находится ли цель в луче."""
        rel = target_pos - self.position
        target_r = np.linalg.norm(rel)
        target_az = math.atan2(rel[1], rel[0])
        target_el = math.asin(rel[2] / target_r) if target_r > 0 else 0

        az_diff = abs((target_az - self.current_azimuth + np.pi) % (2 * np.pi) - np.pi)
        el_diff = abs(target_el - self.current_elevation)
        range_diff = abs(target_r - r)

        return (az_diff < self.beam_width_az / 2 and
                el_diff < self.beam_width_el / 2 and
                range_diff < self.dr)

    def process_measurement(self, measurement: Measurement) -> Optional[int]:
        """
        Обработка измерения.
        Возвращает ID траектории, с которой ассоциировано измерение.
        """
        # Расчёт SNR
        snr = self.receiver.calculate_snr(measurement.range_m, self.transmitter, measurement.rcs)
        measurement.snr_db = snr

        # Ассоциация с существующей траекторией
        associated_track_id = self.association.associate(measurement, self.tracks)

        if associated_track_id is not None:
            track = self.tracks[associated_track_id]
            track.update(measurement)
            return associated_track_id
        else:
            new_track = self._create_track(measurement)
            return new_track.id

    def _create_track(self, measurement: Measurement) -> Track:
        """Создание новой траектории."""
        track_id = self._next_track_id
        self._next_track_id += 1

        track = Track(track_id, measurement, self.dt)
        self.tracks[track_id] = track
        self.detection_count += 1

        return track

    def update_tracks(self) -> None:
        """Обновление всех траекторий (предсказание)."""
        to_remove = []

        for track_id, track in self.tracks.items():
            track.predict(self.dt)

            if track.type == TrackType.TERMINATED:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

    def get_confirmed_tracks(self) -> List[Track]:
        """Получение подтверждённых траекторий."""
        return [track for track in self.tracks.values() if track.is_confirmed()]

    def get_all_tracks(self) -> Dict[int, Track]:
        """Получение всех траекторий."""
        return self.tracks.copy()

    def get_track_data(self) -> List[Dict[str, Any]]:
        """Получение данных траекторий для внешних систем."""
        tracks_data = []

        for track_id, track in self.tracks.items():
            tracks_data.append({
                "track_id": track_id,
                "target_id": track.target_id,
                "position": track.get_position().tolist(),
                "velocity": track.get_velocity().tolist(),
                "status": track.type.name,
                "confidence": track.hits_count / max(5, track.hits_count + track.misses_count),
                "estimated_rcs": 1.0
            })

        return tracks_data

    def get_state(self) -> Dict[str, Any]:
        """Получение состояния радара для сериализации."""
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "mode": self.mode.name,
            "current_azimuth": np.degrees(self.current_azimuth),
            "current_elevation": np.degrees(self.current_elevation),
            "tracks": self.get_track_data(),
            "statistics": {
                "detections": self.detection_count,
                "active_tracks": len(self.tracks)
            }
        }