from collections import defaultdict
from timeit import Timer

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import KDTree as skKDT
from torchvision.ops import boxes as box_ops
from tqdm import tqdm

from lsnms import nms
from lsnms.nms import naive_nms
from lsnms.rtree import RTree


def intersection(boxA, boxB):
    xA = np.maximum(boxA[..., 0], boxB[..., 0])
    xB = np.minimum(boxA[..., 2], boxB[..., 2])
    dx = np.maximum(xB - xA, 0.0)

    yA = np.maximum(boxA[..., 1], boxB[..., 1])
    yB = np.minimum(boxA[..., 3], boxB[..., 3])
    dy = np.maximum(yB - yA, 0.0)

    return dx * dy


def generate_boxes(image_width, n):
    topleft = np.random.uniform(0.0, high=image_width, size=(n, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)
    boxes = np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)
    scores = np.random.uniform(0.1, 1.0, size=len(topleft)).astype(np.float64)

    return boxes, scores


def time_torch_nms(boxes, scores, n_repeats):
    timer = Timer(lambda: box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5))
    return timer.timeit(number=n_repeats) / n_repeats * 1000


def time_naive_nms(boxes, scores, n_repeats):
    # Ensure pre-compilation
    _ = naive_nms(boxes, scores, 0.5)
    timer = Timer(lambda: naive_nms(boxes, scores, 0.5))
    return timer.timeit(number=n_repeats) / n_repeats * 1000


def time_sparse_nms(boxes, scores, n_repeats):
    # Ensure pre-compilation
    _ = nms(boxes, scores, 0.5)
    timer = Timer(lambda: nms(boxes, scores, 0.5))
    return timer.timeit(number=n_repeats) / n_repeats * 1000


def test_speed_nms():

    repeats = 5
    timings = []
    width = 10_000
    ns = [100, 1_000, 2_000, 5_000, 8_000, 10_000, 15_000, 30_000]

    timings = defaultdict(list)

    for n in tqdm(ns):
        boxes, scores = generate_boxes(width, n)

        timings["naive"].append(time_naive_nms(boxes, scores, repeats))
        timings["torch"].append(time_torch_nms(boxes, scores, repeats))
        timings["lsnms"].append(time_sparse_nms(boxes, scores, repeats))

    with plt.xkcd():
        f, ax = plt.subplots(figsize=(8, 8))
        ax.plot(ns, timings["lsnms"], label="lsnms (rtree)", marker="o")
        ax.plot(ns, timings["torch"], label="torch nms", marker="o")
        # ax.plot(ns, timings['naive'], label="naive greedy nms", marker="o")

        ax.set_xlabel("Number of instances", c="k")
        ax.set_ylabel(f"Elapsed time (ms) (mean of {repeats} runs)")
        ax.set_title(f"LSNMS versus Torch timing\n(image of size {width})")

        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
        ax.legend()
        f.show()


def time_tree_query(tree, box, n_repeats):
    _ = tree.intersect(box)
    timer = Timer(lambda: tree.intersect(box))
    return timer.timeit(number=n_repeats) / n_repeats * 1000


def time_sktree_query(tree, box, n_repeats):
    _ = tree.query_radius(box[None, :2], 64)
    timer = Timer(lambda: tree.query_radius(box[None, :2], 64))
    return timer.timeit(number=n_repeats) / n_repeats * 1000


def time_naive_query(boxes, box, n_repeats):
    _ = intersection(boxes, box)
    timer = Timer(lambda: intersection(boxes, box))
    return timer.timeit(number=n_repeats) / n_repeats * 1000


def test_tree_query_timing():

    ns = np.arange(500, 50_000, 500)
    repeats = 50
    timings = defaultdict(list)
    for n in ns:
        boxes, scores = generate_boxes(1_000, n)
        tree = RTree(boxes, leaf_size=32)
        # sktree = skKDT(boxes[:, :2], 26)

        timings["linear"].append(time_naive_query(boxes, boxes[0], repeats))
        timings["tree"].append(time_tree_query(tree, boxes[0], repeats))
        # timings["sktree"].append(time_sktree_query(sktree, boxes[0], repeats))

    with plt.xkcd():
        f, ax = plt.subplots(figsize=(8, 8))
        ax.plot(ns, timings["tree"], label="box tree", marker="o")
        ax.plot(ns, timings["linear"], label="linear", marker="o")
        ax.set_xlabel("Number of boxes to intersect with", c="k")
        ax.set_ylabel(f"Elapsed time (us) (mean of {repeats} runs)")
        ax.set_title("LSNMS box tree intersect vs naive intersect timing")
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.9)
        ax.legend()
        f.show()


def time_sk_tree_build(boxes, n_repeats, leaf_size):
    timer = Timer(lambda: skKDT(boxes[:, :2], int(leaf_size * 0.67)))
    return timer.timeit(number=n_repeats) / n_repeats * 1000


def time_rtree_build(boxes, n_repeats, leaf_size):
    _ = RTree(boxes, leaf_size)
    timer = Timer(lambda: RTree(boxes, leaf_size))
    return timer.timeit(number=n_repeats) / n_repeats * 1000


def test_tree_building_timing():
    ns = np.arange(500, 15_000, 500)
    repeats = 50
    leaf_size = 256
    timings = defaultdict(list)
    for n in ns:
        boxes, _ = generate_boxes(10_000, n)
        timings["sklearn"].append(time_sk_tree_build(boxes, repeats, leaf_size))
        timings["rtree"].append(time_rtree_build(boxes, repeats, leaf_size))

    with plt.xkcd():
        f, ax = plt.subplots(figsize=(8, 8))
        ax.plot(ns, timings["sklearn"], label="sklearn's KDTree", marker="o")
        ax.plot(ns, timings["rtree"], label="lsnms' RTree", marker="o")
        ax.set_xlabel("Number of boxes to index", c="k")
        ax.set_ylabel(f"Elapsed time (ms) (mean of {repeats} runs)")
        ax.set_title("sklearn KDTree versus LSNMS RTree building timing")
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
        ax.legend()
        f.show()
