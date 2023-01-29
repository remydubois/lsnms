import numpy as np
import pytest


@pytest.fixture
def instances():
    topleft = np.random.uniform(0.0, high=1_000, size=(10_000, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)

    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = np.random.uniform(0.01, 1.0, size=len(topleft))

    return boxes, scores


@pytest.fixture
def instances_subset():
    topleft = np.random.uniform(0.0, high=1_000, size=(100, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)

    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = np.random.uniform(0.01, 1.0, size=len(topleft))

    return boxes, scores


@pytest.fixture
def score_threshold():
    return 0.5


@pytest.fixture
def nms_signature():
    sig = "(array(float64, 2d, C), array(float64, 1d, C), float64, float64, int64, int64)"
    return sig
