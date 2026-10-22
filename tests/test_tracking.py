import numpy as np
from kg_tsrdeblur.tracking import MultiObjectTracker, TrackerConfig, KalmanTrack


def detection(dx=0, confidence=0.9):
    return [[20+dx, 20, 60+dx, 60, confidence]]


def test_confirmation_and_five_frame_gap():
    tracker = MultiObjectTracker()
    for i in range(3):
        candidates = tracker.step(detection(i), (200, 100))
        assert candidates[0]['track_id'] == 1
        assert (candidates[0]['reliability'] > 0) == (i >= 2)
    values = [candidates[0]['reliability']]
    for i in range(5):
        c = tracker.step([], (200, 100))
        assert c[0]['missing'] == i+1
        values.append(c[0]['reliability'])
    assert all(x > y for x, y in zip(values, values[1:]))
    assert tracker.step([], (200, 100)) == []


def test_unconfirmed_missing_not_propagated():
    tracker = MultiObjectTracker()
    tracker.step(detection(), (200, 100))
    assert tracker.step([], (200, 100)) == []


def test_descriptor_equations_and_joseph_psd():
    cfg = TrackerConfig()
    t = KalmanTrack(1, np.array(detection()[0]), cfg)
    for i in range(1, 5):
        t.predict(1/30)
        t.update(np.array(detection(i)[0]))
    descriptor, r = t.prior(200, 100, 1/30)
    assert descriptor.shape == (16,) and np.isfinite(descriptor).all()
    assert 0 <= r <= .9
    assert np.linalg.eigvalsh(t.P).min() >= -1e-8
    assert np.allclose(descriptor[:4], t.x[:4]/[200, 100, 200, 100])


def test_innovation_rejects_inconsistent_box():
    tracker = MultiObjectTracker(TrackerConfig(innovation_threshold=0.01))
    tracker.step(detection(), (200, 100))
    c = tracker.step(detection(10), (200, 100))
    assert len(c) == 1 and c[0]['track_id'] == 2


def test_independent_instances_do_not_share_state():
    a, b = MultiObjectTracker(), MultiObjectTracker()
    a.step(detection(), (200, 100))
    assert not b.tracks
