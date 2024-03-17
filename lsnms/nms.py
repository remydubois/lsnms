from typing import Optional

import numpy as np
from numba import njit

from lsnms.rtree import RNode
from lsnms.util import (
    area,
    check_correct_input,
    intersection,
    max_spread_axis,
    offset_bboxes,
)


@njit(cache=False)
def _nms(
    boxes: np.array,
    scores: np.array,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    rtree_leaf_size: int = 32,
) -> np.array:
    """
    See `lsnms.nms` docstring.
    """
    keep = []

    # Discard boxes and scores below score threshold right now to avoid building the tree on
    # useless boxes
    score_mask = scores > score_threshold
    boxes = boxes[score_mask]
    scores = scores[score_mask]

    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)

    # Build the RTree
    rtree = RNode(boxes, rtree_leaf_size, max_spread_axis(boxes), None)
    rtree.build()

    # Compute the areas once and for all: avoid recomputing it at each step
    areas = area(boxes)

    # Order by decreasing confidence
    order = np.argsort(scores)[::-1]
    # Create a mask to keep track of boxes which have already been visited
    to_consider = np.full(len(boxes), True)
    for current_idx in order:
        # If already visited or discarded
        if not to_consider[current_idx]:
            continue

        # If score is already below threshold then break
        if scores[current_idx] < score_threshold:
            break

        boxA = boxes[current_idx]

        # Query the overlapping boxes and return their intersection
        # return only boxes which have at least one pixel of overlap with the box of interest
        query, query_intersections = rtree.intersect(boxA, 0.0)

        for query_idx, overlap in zip(query, query_intersections):
            if not to_consider[query_idx]:
                continue
            sc = overlap / (areas[current_idx] + areas[query_idx] - overlap)
            to_consider[query_idx] = sc < iou_threshold

        # Add the current box
        keep.append(current_idx)
        to_consider[current_idx] = False

    keep = np.array(keep)
    return np.argwhere(score_mask)[:, 0][keep]


def nms(
    boxes: np.array,
    scores: np.array,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    class_ids: Optional[np.array] = None,
    rtree_leaf_size: int = 32,
) -> np.array:
    """
    Sparse NMS, will perform Non Maximum Suppression by only comparing overlapping boxes.
    This turns the usual O(n**2) complexity of the NMS into a O(log(n))-complex algorithm.
    The overlapping boxes are queried using a R-tree, ensuring a log (average case) complexity.

    If `class_ids` is given (one class per object), a class-wise NMS will be applied. This is simply
    done by offsetting the bounding boxes so that bboxes of different classes dont overlap.

    ```python3
    boxes = # array of boxes in format pascal VOC (x0, y0, x1, y1)
    scores = # one-dimensional array of confidence scores

    keep = nms(boxes, scores, iou_threshold=0.5, score_threshold=0.)
    ```
    or
    ```python3
    boxes = # array of boxes in format pascal VOC (x0, y0, x1, y1)
    scores = # one-dimensional array of confidence scores
    class_ids = # one-dimensional array of class indicators (one per object)

    keep = nms(boxes, scores, iou_threshold=0.5, score_threshold=0., class_ids=class_ids)
    ```


    Note that this implementation could be further optimized:
    - Memory management is quite poor: several back and forth list-to-numpy conversions happen
    - Some multi treading could be injected when comparing far appart clusters

    Parameters
    ----------
    boxes : np.array
        Array of boxes, in format (x0, y0, x1, y1) with x1 >= x0, y1 >= y0
    scores : np.array
        One-dimensional array of confidence scores. Note that in the case of multiclass,
        this function must be applied class-wise.
    iou_threshold : float, optional
        Threshold used to consider two boxes to be overlapping, by default 0.5
    score_threshold : float, optional
        Threshold from which boxes are discarded, by default 0.0
    class_ids: np.array, optional
        One-dimensional integer array indicating the respective classes of the bboxes. If this
        is not None, a class-wise NMS will be applied. If None, all boxes are considered of the
        same class.
    rtree_leaf_size: int, optional
        The leaf size parameter of the underlying R-Tree built for box query.

    Returns
    -------
    np.array
        Indices of boxes kept, in decreasing order of confidence score.
    """

    if class_ids is None:
        class_ids = np.zeros(len(boxes), dtype=np.int64)

    # Convert dtype, check shapes, dimensionality, and boundary values.
    boxes, scores, class_ids = check_correct_input(
        boxes, scores, class_ids, iou_threshold=iou_threshold, score_threshold=score_threshold
    )

    # Offset the bounding boxes per class, note that this func is not jitted, so applied here
    boxes = offset_bboxes(boxes, class_ids)

    # Run NMS
    keep = _nms(
        boxes,
        scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        rtree_leaf_size=rtree_leaf_size,
    )

    return keep


@njit(fastmath=True)
def naive_nms(
    boxes: np.array, scores: np.array, iou_threshold: float = 0.5, score_threshold: float = 0.0
) -> np.array:
    """
    Naive nms, for timing and comparisons only.

    Parameters
    ----------
    boxes : np.array
        Array of boxes, in format (x0, y0, x1, y1) with x1 >= x0, y1 >= y0
    scores : np.array
        One-dimensional array of confidence scores. Note that in the case of multiclass,
        this function must be applied class-wise.
    iou_threshold : float, optional
        Threshold used to consider two boxes to be overlapping, by default 0.5
    score_threshold : float, optional
        Threshold from which boxes are discarded, by default 0.0
    cutoff_distance: int, optional
        DEPRECATED, used for compatibility with version 0.1.X.
        Since version 0.2.X, it is useless because overlapping boxes are queried using a R-Tree,
        which is parameter free.
    tree: str, optional
        DEPRECATED, used for compatibility with version 0.1.X.
        Since version 0.2.X, the tree used is a R-Tree.

    Returns
    -------
    np.array
        Indices of boxes kept, in decreasing order of confidence score.
    """
    keep = []

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # n_kept = 0
    suppressed = np.full(len(scores), False)
    order = np.argsort(scores, kind="quicksort")[::-1]
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        current_idx = order[i]

        if scores[current_idx] < score_threshold:
            break

        keep.append(current_idx)

        for j in range(i, len(order), 1):
            if suppressed[j]:
                continue
            inter = intersection(boxes[current_idx], boxes[order[j]])
            sc = inter / (areas[current_idx] + areas[order[j]] - inter)
            suppressed[j] = sc > iou_threshold

    keep = np.array(keep)

    return keep
