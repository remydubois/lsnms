from timeit import Timer

import matplotlib.pyplot as plt
import numpy as np
import torch
from lsnms import nms
from lsnms.nms import sparse_nms, naive_nms

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

    instances_folder = Path('/Users/rdubois/Desktop/instances')
    repeats = 1
    timings = []

    # for n in tqdm(ns):
    #     topleft = np.random.uniform(0.0, high=width, size=(n, 2))
    #     wh = np.random.uniform(15, 45, size=topleft.shape)
    #     boxes = np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)
    #     scores = np.random.uniform(0.1, 1.0, size=len(topleft)).astype(np.float64)
    for boxes_path in instances_folder.glob('*boxes*'):
        # if str(boxes_path) != '/Users/rdubois/Desktop/instances/080c127375d4d1d47752c6f8a542742b_boxes.npy':
        #     continue
        scores_path = str(boxes_path).replace('boxes', 'scores')
        
        boxes = np.load(boxes_path).astype(np.float64)
        scores = np.load(scores_path).astype(np.float64).ravel()
        # if n == ns[0]:
        _ = sparse_nms(
            boxes,
            scores,
            iou_threshold=0.5,
            score_threshold=0.,
            tree_leaf_size=128
        )
        _ = naive_nms(boxes, scores, 0.5, 0.)
        # return

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

        timer4 = Timer(lambda: naive_nms(boxes, scores, 0.5, 0.))
        timer5 = Timer(lambda: sparse_nms(boxes, scores, 0.5, 0., 128))

        timings.append(
            ( 
                timer1.timeit(number=repeats) / repeats * 1000 ,
                0.,
                0.,
                # timer2.timeit(number=repeats) / repeats * 1000,
                # timer2bis.timeit(number=repeats) / repeats * 1000,
                timer3.timeit(number=repeats) / repeats * 1000,
                timer4.timeit(number=repeats) / repeats * 1000,
                # 0.0,
                # 0.0,
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
        ax.bar(6, np.mean(t4), label="naive greedy nms")
        ax.set_xlabel("Number of instances", c="k")
        ax.set_ylabel(f"Elapsed time (ms) (mean of {repeats} runs)")
        # ax.set_title(f"LSNMS versus Torch timing\n(image of size {width})")
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
        # ax.legend()
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
        linear_timer = Timer(lambda : intersection(data[0], data))
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
