from timeit import Timer

import matplotlib.pyplot as plt
import torch
import numpy as np
from lsnms import nms
from lsnms.nms import naive_nms

from lsnms.kdtree import KDTree
from lsnms.boxtree import BoxTree
from sklearn.neighbors import KDTree as skKDT
from torchvision.ops import boxes as box_ops
from tqdm import tqdm
import time
from pathlib import Path


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


def test_speed_nms():

    repeats = 1
    timings = []
    width = 10_000
    ns = [100, 1_000, 2_000, 5_000, 8_000, 10_000, 15_000]

    for n in tqdm(ns):
        topleft = np.random.uniform(0.0, high=width, size=(n, 2))
        wh = np.random.uniform(15, 45, size=topleft.shape)
        boxes = np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)
        scores = np.random.uniform(0.1, 1.0, size=len(topleft)).astype(np.float64)

        _ = nms(
            boxes[:50],
            scores[:50]
        )
        _ = naive_nms(boxes, scores, 0.5, 0.0)

        timer_box = Timer(
            lambda: nms(
                boxes,
                scores,
            )
        )

        timer_torch = Timer(lambda: box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5))
        timer_naive = Timer(lambda: naive_nms(boxes, scores, 0.5, 0.0))

        timings.append(
            (
                timer_box.timeit(number=repeats) / repeats * 1000,
                timer_torch.timeit(number=repeats) / repeats * 1000,
                timer_naive.timeit(number=repeats) / repeats * 1000,
            )
        )

    tboxs, ttorchs, tnaives = zip(*timings)

    with plt.xkcd():
        f, ax = plt.subplots(figsize=(12, 12))
        ax.plot(ns, tboxs, label="lsnms (boxtree)", marker="o")
        ax.plot(ns, ttorchs, label="torch nms", marker="o")
        ax.plot(ns, tnaives, label="naive greedy nms", marker="o")
        ax.set_xlabel("Number of instances", c="k")
        ax.set_ylabel(f"Elapsed time (ms) (mean of {repeats} runs)")
        ax.set_title(f"LSNMS versus Torch timing\n(image of size {width})")
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
        ax.legend()
        f.show()


def test_tree_query_timing():

    ns = np.arange(50, 5_000, 50)
    ts = []
    leaf_size = 128
    naive_ts = []
    linear_ts = []
    repeats = 50
    for n in ns:
        topleft = np.random.uniform(0.0, high=1_000, size=(n, 2))
        wh = np.random.uniform(15, 45, size=topleft.shape)
        data = np.concatenate([topleft, topleft + wh], axis=1)
        box_tree = BoxTree(data, leaf_size=leaf_size)
        _ = box_tree.intersect(data[0])
        # timer = Timer(lambda: tree.query_radius(data[0], 100.0))
        # ts.append(timer.timeit(number=repeats) / repeats * 1_000_000)
        # naive_timer = Timer(lambda: box_tree.intersect(data[np.random.randint(0, len(data)-1, 1)[0]]))
        # naive_ts.append(naive_timer.timeit(repeats) / repeats * 1_000_000)
        linear_timer = Timer(lambda: intersection(data[0], data))
        _ts = []
        for i in np.random.randint(0, len(data) - 1, repeats):
            st = time.time()
            _ = box_tree.intersect(data[i])
            d = time.time() - st
            _ts.append(d)
        naive_ts.append(np.mean(_ts) * 1_000_000)
        linear_ts.append(linear_timer.timeit(repeats) / repeats * 1_000_000)

    with plt.xkcd():
        f, ax = plt.subplots()
        # ax.plot(ns, ts, label="kd tree", marker="o")
        ax.plot(ns, naive_ts, label="box tree", marker="o")
        ax.plot(ns, linear_ts, label="linear", marker="o")
        ax.set_xlabel("Number of datapoints", c="k")
        ax.set_ylabel(f"Elapsed time (us) (mean of {repeats} runs)")
        ax.set_title("LSNMS box tree intersect vs naive intersect timing")
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.9)
        ax.legend()
        f.show()


def test_tree_building_timing():
    ns = np.arange(500, 10_000, 500)
    ts = []
    repeats = 50
    leaf_size = 256
    naive_ts = []
    for n in ns:
        topleft = np.random.uniform(0.0, high=1_000, size=(n, 2))
        wh = np.random.uniform(15, 45, size=topleft.shape)
        data = np.concatenate([topleft, topleft + wh], axis=1)
        _ = skKDT(data[:, :2], leaf_size)
        _ = BoxTree(data, leaf_size)
        timer = Timer(lambda: skKDT(data[:, :2], int(leaf_size * 0.67)))
        ts.append(timer.timeit(number=repeats) / repeats * 1000)
        # Sklearn allows leafs to be twice as big as the leaf_size given, see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
        naive_timer = Timer(lambda: BoxTree(data, leaf_size))
        naive_ts.append(naive_timer.timeit(repeats) / repeats * 1000)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(ns, ts, label="kd tree", marker="o")
        ax.plot(ns, naive_ts, label="box tree", marker="o")
        ax.set_xlabel("Number of datapoints", c="k")
        ax.set_ylabel(f"Elapsed time (ms) (mean of {repeats} runs)")
        ax.set_title("sklearn KDTree versus LSNMS boxtree building timing")
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
        ax.legend()
        f.show()
