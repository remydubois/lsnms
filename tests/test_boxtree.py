import numpy as np
from lsnms.boxtree import BoxTree
# from lsnms.util import intersection
from timeit import Timer
import matplotlib.pyplot as plt
from lsnms.util import intersection


def test_intersect_tree():

    topleft = np.random.uniform(0.0, high=1_000, size=(10_000, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)
    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    # scores = np.random.uniform(0.1, 1.0, size=len(topleft))

    tree = BoxTree(boxes, 16)

    queries = boxes[0]

    for min_area in np.arange(1, 10) / 10:
        indices, intersections = tree.intersect(queries, min_area)

        np_intersect = intersection(boxes, queries)
        
        in_inter = np_intersect[indices]
        out_inter = np.array([inter for i, inter in enumerate(np_intersect) if i not in indices])
        
        np.testing.assert_allclose(in_inter, intersections)
        np.testing.assert_array_less(out_inter, min_area)