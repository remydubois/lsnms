import numpy as np
import pytest


@pytest.fixture
def instances():
    rng = np.random.RandomState(seed=0)
    topleft = rng.uniform(0.0, high=1_000, size=(10_000, 2))
    wh = rng.uniform(15, 45, size=topleft.shape)

    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = rng.uniform(0.01, 1.0, size=len(topleft))

    return boxes, scores


@pytest.fixture
def instances_subset():
    topleft = np.random.uniform(0.0, high=1_000, size=(15, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)

    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = np.random.uniform(0.01, 1.0, size=len(topleft))

    return boxes, scores


@pytest.fixture
def score_threshold():
    return 0.5


@pytest.fixture
def nms_signature():
    sig = (
        "(Array(float64, 2, 'C', False, aligned=True),"
        " Array(float64, 1, 'C', False, aligned=True),"
        " float64, float64, int64, int64)"
    )
    return sig
