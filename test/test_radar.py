import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestKalmanFilter:

    def test_init_and_initialize(self):
        from src.radar import KalmanFilter, Measurement

        m = Measurement(0.0, np.array([100, 200, 300]), 374, 1.1, 0.93, 1, 20, 1)
        kf = KalmanFilter(dt=0.1)
        kf.initialize(m)

        assert kf.x is not None
        assert np.allclose(kf.x[0:3], [100, 200, 300])

    def test_time_update_changes_state_when_velocity_nonzero(self):
        from src.radar import KalmanFilter, Measurement

        m = Measurement(0.0, np.array([100, 200, 300]), 374, 1.1, 0.93, 1, 20, 1)
        kf = KalmanFilter(dt=0.1)
        kf.initialize(m)

        m2 = Measurement(0.1, np.array([110, 200, 300]), 374, 1.1, 0.93, 1, 20, 1)
        kf.time_update()
        kf.measurement_update(m2.position)

        x_before = kf.x.copy()
        kf.time_update()

        assert not np.allclose(kf.x, x_before)

    def test_measurement_update_returns_position(self):
        from src.radar import KalmanFilter, Measurement

        m = Measurement(0.0, np.array([100, 200, 300]), 374, 1.1, 0.93, 1, 20, 1)
        kf = KalmanFilter(dt=0.1)
        kf.initialize(m)
        kf.time_update()

        result = kf.measurement_update(np.array([102, 198, 301]))

        assert result.shape == (3,)


class TestAssociation:

    def test_associate_with_existing_track(self):
        from src.radar import Association, Track, Measurement

        m1 = Measurement(0.0, np.array([100, 200, 300]), 374, 1.1, 0.93, 1, 20, 1)
        track = Track(0, m1, dt=0.1)

        m2 = Measurement(0.1, np.array([102, 201, 299]), 374, 1.1, 0.93, 1, 20, 1)
        assoc = Association(gate_threshold=100.0)

        result = assoc.associate(m2, {0: track})
        assert result == 0

    def test_associate_no_match(self):
        from src.radar import Association, Track, Measurement

        m1 = Measurement(0.0, np.array([100, 200, 300]), 374, 1.1, 0.93, 1, 20, 1)
        track = Track(0, m1, dt=0.1)

        m2 = Measurement(0.1, np.array([500, 600, 700]), 1044, 0.88, 0.71, 1, 20, 2)
        assoc = Association(gate_threshold=50.0)

        result = assoc.associate(m2, {0: track})
        assert result is None

    def test_empty_tracks_returns_none(self):
        from src.radar import Association, Measurement

        m = Measurement(0.0, np.array([100, 200, 300]), 374, 1.1, 0.93, 1, 20, 1)
        assoc = Association()

        result = assoc.associate(m, {})
        assert result is None


class TestReceiverSNR:

    def test_snr_returns_positive(self):
        from src.radar import Transmitter, Receiver

        tx = Transmitter()
        rx = Receiver()
        snr = rx.calculate_snr(1000.0, tx, sigma=1.0)

        assert snr > 0

    def test_snr_decreases_with_range(self):
        from src.radar import Transmitter, Receiver

        tx = Transmitter()
        rx = Receiver()

        snr_1000 = rx.calculate_snr(1000.0, tx, sigma=1.0)
        snr_2000 = rx.calculate_snr(2000.0, tx, sigma=1.0)

        assert snr_2000 < snr_1000

    def test_snr_increases_with_rcs(self):
        from src.radar import Transmitter, Receiver

        tx = Transmitter()
        rx = Receiver()

        snr_small = rx.calculate_snr(1000.0, tx, sigma=0.1)
        snr_large = rx.calculate_snr(1000.0, tx, sigma=10.0)

        assert snr_large > snr_small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])