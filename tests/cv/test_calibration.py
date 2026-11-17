import pytest

np = pytest.importorskip("numpy")
C = pytest.importorskip("floodsitrep.cv.eval.calibration")


def test_ece_hand_example():
    # 4 samples, all confidence 0.8 (one bin); 2 correct -> |0.5 - 0.8| = 0.3
    probs = np.array([[0.8, 0.2]] * 4)
    labels = np.array([0, 0, 1, 1])
    assert C.ece(probs, labels) == pytest.approx(0.3)


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 20000, 5
    true_logits = rng.normal(0, 1.5, (n, k))
    labels = np.array([rng.choice(k, p=p) for p in C.softmax(true_logits)])
    overconfident = true_logits * 3.0  # a model 3x too sharp
    rep = C.calibration_report(overconfident, labels)
    assert rep["temperature"] == pytest.approx(3.0, rel=0.1)
    assert rep["ece_after"] < rep["ece_before"]
    assert rep["nll_after"] < rep["nll_before"]


def test_temperature_does_not_change_argmax():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(100, 5))
    assert (C.softmax(z, 1.0).argmax(1) == C.softmax(z, 2.7).argmax(1)).all()
