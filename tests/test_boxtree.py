import numpy as np
from lsnms.rtree import RTree


def intersection(boxA, boxB):
    xA = np.maximum(boxA[..., 0], boxB[..., 0])
    xB = np.minimum(boxA[..., 2], boxB[..., 2])
    dx = np.maximum(xB - xA, 0.0)

    yA = np.maximum(boxA[..., 1], boxB[..., 1])
    yB = np.minimum(boxA[..., 3], boxB[..., 3])
    dy = np.maximum(yB - yA, 0.0)

    # compute the area of intersection rectangle
    return dx * dy


def test_intersect_tree():

    topleft = np.random.uniform(0.0, high=1_000, size=(10_000, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)
    boxes = np.concatenate([topleft, topleft + wh], axis=1)

    tree = RTree(boxes, 16)

    queries = boxes[0]

    for min_area in np.arange(10, 300, 30):
        indices, intersections = tree.intersect(queries, min_area)

        np_intersect = intersection(boxes, queries)

        in_inter = np_intersect[indices]
        out_inter = np.array([inter for i, inter in enumerate(np_intersect) if i not in indices])
        np.testing.assert_allclose(in_inter, intersections)
        np.testing.assert_array_less(out_inter, min_area)
