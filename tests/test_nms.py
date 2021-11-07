import numpy as np
import torch
from lsnms import nms, wbc
from lsnms.balltree import BallTree
from torchvision.ops import boxes as box_ops
from multiprocessing import Process
from timeit import Timer


def test_balltree_nms():

    topleft = np.random.uniform(0.0, high=1_000, size=(10_000, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)
    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = np.random.uniform(0.1, 1.0, size=len(topleft))

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()

    # Compare naive NMS
    k2 = nms(boxes, scores, 0.5, 0.1)
    # Compare naive NMS
    k3 = nms(boxes, scores, 0.5, 0.1, 64, tree="balltree")

    assert np.allclose(k1, k2) and np.allclose(k1, k3)


def test_kdtree_nms():

    topleft = np.random.uniform(0.0, high=1_000, size=(10_000, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)
    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = np.random.uniform(0.1, 1.0, size=len(topleft))

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()

    # Compare naive NMS
    k2 = nms(boxes, scores, 0.5, 0.1)
    # Compare naive NMS
    k3 = nms(boxes, scores, 0.5, 0.1, 64, tree="kdtree")

    assert np.allclose(k1, k2) and np.allclose(k1, k3)
