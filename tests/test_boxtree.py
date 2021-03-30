import numpy as np
from lsnms.boxtree import BoxTree
# from lsnms.util import intersection
from timeit import Timer
import matplotlib.pyplot as plt


def intersection(boxA, boxB):
    xA = np.maximum(boxA[..., 0], boxB[..., 0])
    xB = np.minimum(boxA[..., 2], boxB[..., 2])
    dx = np.maximum(xB - xA, 0.0)
    # if dx <= 0:
    #     return 0.0

    yA = np.maximum(boxA[..., 1], boxB[..., 1])
    yB = np.minimum(boxA[..., 3], boxB[..., 3])
    dy = np.maximum(yB - yA, 0.0)
    # if dy <= 0.0:
    #     return 0.0

    # compute the area of intersection rectangle
    return dx * dy


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


def test_timing_boxtree():

    ns = np.arange(1000, 50000, 2500)
    ts = []
    naive_ts = []
    repeats = 100
    for n in ns:
        topleft = np.random.uniform(0.0, high=1_000, size=(n, 2))
        wh = np.random.uniform(15, 45, size=topleft.shape)
        boxes = np.concatenate([topleft, topleft + wh], axis=1)
        
        tree = BoxTree(boxes, leaf_size=16)
        _ = tree.intersect(boxes[0])
        timer = Timer(lambda: tree.intersect(boxes[0]))
        ts.append(timer.timeit(number=repeats) / repeats * 1000)
        naive_timer = Timer(lambda: intersection(boxes[0], boxes))
        naive_ts.append(naive_timer.timeit(repeats) / repeats * 1000)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(ns, ts, label="box tree", marker="o")
        ax.plot(ns, naive_ts, label="naive", marker="o")
        ax.set_xlabel("Number of datapoints", c="k")
        ax.set_ylabel(f"Elapsed time (ms) (mean of {repeats} runs)")
        ax.set_title("sklearn versus LSNMS tree radius query timing")
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.9)
        ax.legend()
        ax.set_yscale('log')
        f.show()