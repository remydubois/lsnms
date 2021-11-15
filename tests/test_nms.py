import numpy as np
import pytest
import torch
from lsnms import nms
from lsnms.nms import naive_nms
from torchvision.ops import boxes as box_ops


def datagen():
    topleft = np.random.uniform(0.0, high=1_000, size=(10_000, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)

    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = np.random.uniform(0.1, 1.0, size=len(topleft))

    return boxes, scores


def test_balltree_nms():

    boxes, scores = datagen()

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()

    # Compare naive NMS
    k2 = naive_nms(boxes, scores, 0.5, 0.0)
    # Compare naive NMS
    k3 = nms(boxes, scores, 0.5, 0.0)

    assert np.allclose(k1, k2) and np.allclose(k1, k3)


def test_boxes_dtype():
    boxes, scores = datagen()
    boxes = boxes.astype(np.float32)
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1)


def test_scores_dtype():
    boxes, scores = datagen()
    scores = scores.astype(np.float32)
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1)


def test_boxes_shape():
    boxes, scores = datagen()
    boxes = boxes[None]
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1)


def test_scores_shape():
    boxes, scores = datagen()
    scores = scores[..., None]
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1)


def test_box_encoding():
    boxes, scores = datagen()
    # Make the first box odd: x1 > x2
    boxes[0, 0] = boxes[0, 2] + 1
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1)
