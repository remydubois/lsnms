from socket import IP_DEFAULT_MULTICAST_LOOP
import numpy as np
import pytest
import torch
from lsnms import nms
from lsnms.nms import naive_nms
from torchvision.ops import boxes as box_ops


def datagen(n=10_000):
    rng = np.random.RandomState(0)
    topleft = rng.uniform(0.0, high=1_000, size=(n, 2))
    wh = rng.uniform(15, 45, size=topleft.shape)

    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = rng.uniform(0.1, 1.0, size=len(topleft))

    return boxes, scores


def test_rtree_nms():
    boxes, scores = datagen(n=1_000)

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()

    # Compare sparse NMS
    k2 = nms(boxes, scores, 0.5, 0.0)

    assert np.allclose(k1, k2)


def test_empty_nms():
    boxes, scores = datagen()
    # Put all the scores to zero artificially
    keep = nms(boxes, scores * 0.)

    assert keep.size == 0
    

def test_rtree_multiclass_nms():

    boxes, scores = datagen()
    class_ids = np.random.randint(0, 50, size=len(boxes))

    # Compare against torch
    k1 = box_ops.batched_nms(
        torch.tensor(boxes), torch.tensor(scores), torch.tensor(class_ids), 0.5
    ).numpy()

    # Compare sparse NMS
    k2 = nms(boxes, scores, 0.5, 0.0, class_ids=class_ids)

    assert np.allclose(k1, k2)


def test_naive_nms():

    boxes, scores = datagen()

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()

    # Compare naive NMS
    k2 = naive_nms(boxes, scores, 0.5, 0.0)

    assert np.allclose(k1, k2)


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


def test_warning_dist_arg():
    boxes, scores = datagen()
    with pytest.warns(None):
        nms(boxes, scores, 0.5, 0.1, cutoff_distance=64)


def test_warning_tree_arg():
    boxes, scores = datagen()
    with pytest.warns(None):
        nms(boxes, scores, 0.5, 0.1, tree="faketree")
