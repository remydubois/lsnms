"""
These tests for the ball tree only, no redundant nms timings.
See timings_kdtree.py for the nms and wbc timings.
"""
from timeit import Timer

import matplotlib.pyplot as plt
import numpy as np
from lsnms.balltree import BallTree
from sklearn.neighbors import BallTree as skBT


def test_tree_query_timing():

    ns = np.arange(1000, 200000, 10000)
    ts = []
    naive_ts = []
    repeats = 100
    for n in ns:
        data = np.random.uniform(0, 1000, (n, 2))
        sk_tree = skBT(data, leaf_size=16)
        tree = BallTree(data, leaf_size=16)
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
        _ = BallTree(data, 16)
        timer = Timer(lambda: BallTree(data, 16))
        ts.append(timer.timeit(number=5) / 5)
        naive_timer = Timer(lambda: skBT(data, 16))
        naive_ts.append(naive_timer.timeit(5) / 5)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(ns, ts, label="lsnms tree", marker="o")
        ax.plot(ns, naive_ts, label="sklearn", marker="o")
        ax.set_xlabel("Number of datapoints", c="k")
        ax.set_ylabel("Elapsed time (s) (mean of 5 runs)")
        ax.set_title("sklearn BallTree versus LSNMS tree building timing")
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
        ax.legend()
        f.show()
