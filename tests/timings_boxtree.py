from timeit import Timer

import matplotlib.pyplot as plt
import numpy as np
import torch
from lsnms import nms
from lsnms.nms import sparse_nms

# from lsnms.kdtree import KDTree
# from lsnms.boxtree import BoxTree
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
            score_threshold=0.1,
        )

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
                0.5,
                0.1,
                64,
                "kdtree",
            )
        )
        timer2bis = Timer(
            lambda: nms(
                boxes,
                scores,
                0.5,
                0.1,
                64,
                "balltree",
            )
        )
        timer3 = Timer(lambda: box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5))

        timer5 = Timer(lambda: sparse_nms(boxes, scores, 0.5, 0.1))

        timings.append(
            ( 
                timer1.timeit(number=repeats) / repeats * 1000 ,
                # 0.,
                # 0.,
                timer2.timeit(number=repeats) / repeats * 1000,
                timer2bis.timeit(number=repeats) / repeats * 1000,
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