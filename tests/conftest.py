import pytest
import numpy as np


@pytest.fixture
def instances():
    topleft = np.random.uniform(0.0, high=1_000, size=(10_000, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)

    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = np.random.uniform(0.01, 1.0, size=len(topleft))

    return boxes, scores


@pytest.fixture
def score_threshold():
    return 0.5
