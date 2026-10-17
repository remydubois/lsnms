import numpy as np
import pytest


@pytest.fixture(
    params=[
        pytest.param("scattered", marks=pytest.mark.intra_framework),
        pytest.param("crowded", marks=pytest.mark.inter_framework),
    ]
)
def scenario(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(10, marks=pytest.mark.tiny),
        pytest.param(100, marks=pytest.mark.very_small),
        pytest.param(1000, marks=pytest.mark.small),
        pytest.param(10000, marks=pytest.mark.medium),
        pytest.param(100_000, marks=pytest.mark.large),
    ],
    ids=lambda val: f"n_instances={val}",
)
def n_instances(request):
    return request.param


@pytest.fixture
def instances(scenario, n_instances):
    rng = np.random.RandomState(seed=0)
    # scenario, n_instances = request.param
    if scenario == "scattered":
        topleft = rng.uniform(0.0, high=1_000, size=(n_instances, 2))
        wh = rng.uniform(15, 45, size=topleft.shape)

    elif scenario == "crowded":
        im_width = int((n_instances**0.5) * 10)
        topleft = (
            np.mgrid[5:im_width:10, 5:im_width:10].astype(float).reshape(2, -1).T
        )  # 99x99 instances
        n_detec_per_instance = np.repeat(
            5, topleft.shape[0]
        )  # Each instance is crowded with 5 detections
        topleft = np.repeat(
            topleft, n_detec_per_instance, axis=0
        )  # At this point we have repeated each instance n_detec_per_instance times
        topleft += rng.uniform(0.1, 0.5, size=topleft.shape)  # Add some random noise
        wh = rng.uniform(1, 2, size=topleft.shape)

    # Create the bboxes
    boxes = np.concatenate([topleft, topleft + wh], axis=1)

    # Generate scores
    scores = rng.uniform(0.9, 1.0, size=len(topleft))
    return boxes, scores


@pytest.fixture
def score_threshold():
    return 0.5


@pytest.fixture
def iou_threshold():
    return 0.1
