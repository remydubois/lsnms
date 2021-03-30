from timeit import Timer

import matplotlib.pyplot as plt
import numpy as np
import torch
from lsnms import nms
from lsnms.nms import sparse_nms

from lsnms.kdtree import KDTree
from lsnms.boxtree import BoxTree
# from sklearn.neighbors import KDTree as skKDT
from torchvision.ops import boxes as box_ops
from tqdm import tqdm
from pathlib import Path


def test_speed_nms():

    instances_folder = Path('/Users/rdubois/Desktop/instances')
    repeats = 5
    timings = []

    # for n in tqdm(ns):
    #     topleft = np.random.uniform(0.0, high=width, size=(n, 2))
    #     wh = np.random.uniform(15, 45, size=topleft.shape)
    #     boxes = np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)
    #     scores = np.random.uniform(0.1, 1.0, size=len(topleft)).astype(np.float64)
    for boxes_path in instances_folder.glob('*boxes*'):
        scores_path = str(boxes_path).replace('boxes', 'scores')
        
        boxes = np.load(boxes_path).astype(np.float64)
        scores = np.load(scores_path).astype(np.float64).ravel()
        # if n == ns[0]:
        _ = sparse_nms(
            boxes[:50],
            scores[:50],
            iou_threshold=0.5,
            score_threshold=0.,
        )

        _ = nms(
            boxes[:5], scores[:5], iou_threshold=0.5, score_threshold=0., cutoff_distance=-1
        )
        _ = nms(
            boxes[:50],
            scores[:50],
            iou_threshold=0.5,
            score_threshold=0.,
            cutoff_distance=16,
            tree="kdtree",
        )
        _ = nms(
            boxes[:50],
            scores[:50],
            iou_threshold=0.5,
            score_threshold=0.,
            cutoff_distance=16,
            tree="balltree",
        )

        timer1 = Timer(
            lambda: nms(boxes, scores, iou_threshold=0.5, score_threshold=0., cutoff_distance=-1)
        )
        timer2 = Timer(
            lambda: nms(
                boxes,
                scores,
                0.5,
                0.,
                64,
                "kdtree",
            )
        )
        timer2bis = Timer(
            lambda: nms(
                boxes,
                scores,
                0.5,
                0.,
                64,
                "balltree",
            )
        )
        timer3 = Timer(lambda: box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5))

        # timer4 = Timer(lambda: naive_nms(boxes, scores, 0.5, 0.))
        timer5 = Timer(lambda: sparse_nms(boxes, scores, 0.5, 0., 64))

        timings.append(
            ( 
                timer1.timeit(number=repeats) / repeats * 1000 ,
                0.,
                0.,
                # timer2.timeit(number=repeats) / repeats * 1000,
                # timer2bis.timeit(number=repeats) / repeats * 1000,
                timer3.timeit(number=repeats) / repeats * 1000,
                # timer4.timeit(number=repeats) / repeats * 1000,
                # 0.0,
                0.0,
                timer5.timeit(repeats) / repeats * 1000
            )
        )

    t1, t2, t2bis, t3, t4, t5 = zip(*timings)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.bar(1, np.mean(t1), label="lsnms (cutoff_distance=-1)")
        ax.bar(2, np.mean(t2), label="lsnms (kdtree; dist < 64)")
        ax.bar(3, np.mean(t2bis), label="lsnms (balltree; dist < 64)")
        ax.bar(4, np.mean(t3), label="torch nms")
        ax.bar(5, np.mean(t5), label="lsnms (boxtree)")
        # ax.plot(ns, t4, marker='o', label="naive greedy nms")
        ax.set_xlabel("Number of instances", c="k")
        ax.set_ylabel(f"Elapsed time (ms) (mean of {repeats} runs)")
        # ax.set_title(f"LSNMS versus Torch timing\n(image of size {width})")
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
        ax.legend()
        f.show()


def test_tree_query_timing():

    ns = np.arange(1000, 400000, 10000)
    ts = []
    leaf_size = 128
    naive_ts = []
    repeats = 50
    for n in ns:
        data = np.random.uniform(0, 1000, (n, 2))
        box_tree = BoxTree(data, leaf_size=leaf_size)
        tree = KDTree(data, leaf_size=leaf_size)
        _ = tree.query_radius(data[0], 200.0)
        _ = box_tree.intersect(data[0])
        timer = Timer(lambda: tree.query_radius(data[0], 100.0))
        ts.append(timer.timeit(number=repeats) / repeats * 1000)
        naive_timer = Timer(lambda: box_tree.intersect(data[0]))
        naive_ts.append(naive_timer.timeit(repeats) / repeats * 1000)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(ns, ts, label="kd tree", marker="o")
        ax.plot(ns, naive_ts, label="box tree", marker="o")
        ax.set_xlabel("Number of datapoints", c="k")
        ax.set_ylabel(f"Elapsed time (ms) (mean of {repeats} runs)")
        ax.set_title("sklearn versus LSNMS tree radius query timing")
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.9)
        ax.legend()
        f.show()


def test_tree_building_timing():
    ns = np.arange(1000, 100_000, 5000)
    ts = []
    repeats = 10
    leaf_size = 128
    naive_ts = []
    for n in ns:
        data = np.random.uniform(0, n, (n, 2))
        _ = KDTree(data, leaf_size)
        _ = BoxTree(data, leaf_size)
        timer = Timer(lambda: KDTree(data, leaf_size))
        ts.append(timer.timeit(number=repeats) / repeats)
        # Sklearn allows leafs to be twice as big as the leaf_size given, see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
        naive_timer = Timer(lambda: BoxTree(data, leaf_size))
        naive_ts.append(naive_timer.timeit(repeats) / repeats)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(ns, ts, label="kd tree", marker="o")
        ax.plot(ns, naive_ts, label="box tree", marker="o")
        ax.set_xlabel("Number of datapoints", c="k")
        ax.set_ylabel(f"Elapsed time (s) (mean of {repeats} runs)")
        # ax.set_title("sklearn KDTree versus LSNMS tree building timing")
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
        ax.legend()
        f.show()
