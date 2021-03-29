from timeit import Timer

import matplotlib.pyplot as plt
import numpy as np
import torch
from lsnms import nms
from lsnms.kdtree import KDTree
from sklearn.neighbors import KDTree as skKDT
from torchvision.ops import boxes as box_ops
from tqdm import tqdm


def test_speed_nms():

    ns = np.arange(1000, 80_000, 5000)
    width = 10_000
    timings = []
    repeats = 7

    for n in tqdm(ns):
        topleft = np.random.uniform(0.0, high=width, size=(n, 2))
        wh = np.random.uniform(15, 45, size=topleft.shape)
        boxes = np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)
        scores = np.random.uniform(0.1, 1.0, size=len(topleft)).astype(np.float64)

        if n == ns[0]:
            _ = nms(
                boxes[:5], scores[:5], iou_threshold=0.5, score_threshold=0.1, cutoff_distance=-1
            )
            _ = nms(
                boxes[:50],
                scores[:50],
                iou_threshold=0.5,
                score_threshold=0.1,
                cutoff_distance=16,
                tree="kdtree",
            )
            _ = nms(
                boxes[:50],
                scores[:50],
                iou_threshold=0.5,
                score_threshold=0.1,
                cutoff_distance=16,
                tree="balltree",
            )

        timer1 = Timer(
            lambda: nms(boxes, scores, iou_threshold=0.5, score_threshold=0.1, cutoff_distance=-1)
        )
        timer2 = Timer(
            lambda: nms(
                boxes,
                scores,
                iou_threshold=0.5,
                score_threshold=0.1,
                cutoff_distance=64,
                tree="kdtree",
            )
        )
        timer2bis = Timer(
            lambda: nms(
                boxes,
                scores,
                iou_threshold=0.5,
                score_threshold=0.1,
                cutoff_distance=64,
                tree="balltree",
            )
        )
        timer3 = Timer(lambda: box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5))

        timings.append(
            (
                timer1.timeit(number=repeats) / repeats,
                # 0.,
                # 0.,
                timer2.timeit(number=repeats) / repeats,
                timer2bis.timeit(number=repeats) / repeats,
                timer3.timeit(number=repeats) / repeats,
                # timer4.timeit(number=repeats) / repeats,
                # 0.0,
                0.0,
            )
        )

    t1, t2, t2bis, t3, t4 = zip(*timings)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(ns, t1, marker="o", label="lsnms (cutoff_distance=-1)")
        ax.plot(ns, t2, marker="o", label="lsnms (kdtree; dist < 64)")
        ax.plot(ns, t2bis, marker="o", label="lsnms (balltree; dist < 64)")
        ax.plot(ns, t3, marker="o", label="torch nms")
        # ax.plot(ns, t4, marker='o', label="naive greedy nms")
        ax.set_xlabel("Number of instances", c="k")
        ax.set_ylabel(f"Elapsed time (s) (mean of {repeats} runs)")
        ax.set_title(f"LSNMS versus Torch timing\n(image of size {width})")
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
        ax.legend()
        f.show()


def test_tree_query_timing():

    ns = np.arange(1000, 200000, 10000)
    ts = []
    naive_ts = []
    repeats = 100
    for n in ns:
        data = np.random.uniform(0, 1000, (n, 2))
        sk_tree = skKDT(data, leaf_size=16)
        tree = KDTree(data, leaf_size=16)
        _ = tree.query_radius(data[0], 200.0)
        timer = Timer(lambda: tree.query_radius(data[0], 100.0))
        ts.append(timer.timeit(number=repeats) / repeats * 1000)
        naive_timer = Timer(lambda: sk_tree.query_radius(data[0:1], 100.0))
        naive_ts.append(naive_timer.timeit(repeats) / repeats * 1000)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(ns, ts, label="lsnms tree", marker="o")
        ax.plot(ns, naive_ts, label="sklearn", marker="o")
        ax.set_xlabel("Number of datapoints", c="k")
        ax.set_ylabel(f"Elapsed time (ms) (mean of {repeats} runs)")
        ax.set_title("sklearn versus LSNMS tree radius query timing")
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.9)
        ax.legend()
        f.show()


def test_tree_building_timing():
    ns = np.arange(1000, 300000, 25000)
    ts = []
    naive_ts = []
    for n in ns:
        data = np.random.uniform(0, n, (n, 2))
        _ = KDTree(data, 16)
        timer = Timer(lambda: KDTree(data, 16))
        ts.append(timer.timeit(number=5) / 5)
        naive_timer = Timer(lambda: skKDT(data, 16))
        naive_ts.append(naive_timer.timeit(5) / 5)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(ns, ts, label="lsnms tree", marker="o")
        ax.plot(ns, naive_ts, label="sklearn", marker="o")
        ax.set_xlabel("Number of datapoints", c="k")
        ax.set_ylabel("Elapsed time (s) (mean of 5 runs)")
        ax.set_title("sklearn KDTree versus LSNMS tree building timing")
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
        ax.legend()
        f.show()
