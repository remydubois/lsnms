import numpy as np
import pytest
from numba import njit

from lsnms.util import box_englobing_boxes, intersection, offset_bboxes


def datagen(n=10_000):
    topleft = np.random.uniform(0.0, high=1_000, size=(n, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)

    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = np.random.uniform(0.1, 1.0, size=len(topleft))

    return boxes, scores


@njit
def intersect_many(tree, boxes):
    intersections = []
    for box in boxes:
        indices, _ = tree.intersect(box)
        intersections.extend([i for i in indices])

    return intersections


def assert_no_intersect(boxesA, boxesB):
    boxA = box_englobing_boxes(boxesA)
    boxB = box_englobing_boxes(boxesB)
    assert intersection(boxA, boxB) == 0.0


def test_offset_bboxes():
    boxes, _ = datagen()
    rng = np.random.default_rng(0)
    class_ids = rng.integers(0, 2, size=len(boxes))

    new_boxes = offset_bboxes(boxes, class_ids)

    # Assert that no boxes for class a intersect with class b
    unique_class_ids, class_index = np.unique(class_ids, return_inverse=True)
    for i in range(len(unique_class_ids) - 1):
        for j in range(i + 1, len(unique_class_ids)):
            class_i = unique_class_ids[i]
            class_j = unique_class_ids[j]
            boxes_i = new_boxes[class_ids == class_i]
            boxes_j = new_boxes[class_ids == class_j]
            assert_no_intersect(boxes_i, boxes_j)


@pytest.mark.skip(reason="Visual test")
def test_visual_offset_bboxes():
    import matplotlib.pyplot as plt

    boxes, _ = datagen()
    # boxes = boxes.reshape(-1, 2, 2).mean(1)
    rng = np.random.default_rng(0)
    class_ids = rng.integers(0, 3, size=len(boxes))

    f, (ax, ax1) = plt.subplots(figsize=(16, 7), ncols=2)
    for c in np.unique(class_ids):
        ax.plot(*boxes[class_ids == c, :2].T, linestyle="", marker="o", markersize=1)
    ax.set_title("Raw boxes centroids colored per class before offset")
    ax.set_xticks([0, 1_000])
    ax.set_yticks([0, 1_000])
    ax.set_xticklabels([0, 1_000])
    ax.set_yticklabels([0, 1_000])
    ax.set_xlim([0, 2_000])
    ax.set_ylim([0, 2_000])

    new_boxes = offset_bboxes(boxes, class_ids)
    for c in np.unique(class_ids):
        ax1.plot(*new_boxes[class_ids == c, :2].T, linestyle="", marker="o", markersize=1)
    ax1.set_title("Raw boxes centroids colored per class after offset")
    f.show()
