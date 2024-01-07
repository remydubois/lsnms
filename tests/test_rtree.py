import numpy as np
import pytest

from lsnms.rtree import RNode, RTree


def intersection(boxA, boxB):
    xA = np.maximum(boxA[..., 0], boxB[..., 0])
    xB = np.minimum(boxA[..., 2], boxB[..., 2])
    dx = np.maximum(xB - xA, 0.0)

    yA = np.maximum(boxA[..., 1], boxB[..., 1])
    yB = np.minimum(boxA[..., 3], boxB[..., 3])
    dy = np.maximum(yB - yA, 0.0)

    # compute the area of intersection rectangle
    return dx * dy


def test_intersect_tree(instances, benchmark):

    boxes, _ = instances

    tree = RTree(boxes, 16)

    queries = boxes[0]

    for min_area in np.arange(10, 300, 30):
        indices, intersections = tree.intersect(queries, min_area)

        np_intersect = intersection(boxes, queries)

        in_inter = np_intersect[indices]
        out_inter = np.array([inter for i, inter in enumerate(np_intersect) if i not in indices])
        np.testing.assert_allclose(in_inter, intersections)
        np.testing.assert_array_less(out_inter, min_area)

    benchmark(tree.intersect, queries, 0)


def test_build_odd_tree(instances):
    boxes, _ = instances
    with pytest.raises(AssertionError):
        _ = RTree(boxes, leaf_size=0)


def test_build_tree(instances, benchmark):
    boxes, _ = instances
    _ = benchmark(RTree, boxes, leaf_size=1)


def test_build_rnode_default_args(instances):
    boxes, _ = instances

    _ = RNode(boxes)
