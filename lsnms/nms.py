from numba import njit
import numpy as np
from lsnms.balltree import BallTree
from lsnms.boxtree import BoxTree
from lsnms.kdtree import KDTree
from lsnms.util import area, intersection


@njit(cache=True)
def nms(boxes, scores, iou_threshold=0.5):
    """
    Sparse NMS, will perform Non Maximum Suppression by only comparing overlapping boxes.
    This turns the usual O(n**2) complexity of the NMS into a O(log(n))-complex algorithm.
    The overlapping boxes are queried using a R-tree, ensuring a log (average case) complexity.

    Note that this implementation could be further optimized:
    - Memory management is quite poor: several back and forth list-to-numpy conversions happen
    - Some multi treading could be injected when comparing far appart clusters

    Parameters
    ----------
    boxes : np.array
        Array of boxes, in format (x0, y0, x1, y1) with x1 >= x0, y1 >= y0
    scores : np.array
        One-dimensional array of confidence scores.
    iou_threshold : float, optional
        Threshold from which boxes are considered to overlap, and end up aggregated, by default 0.5
        The higher the lower the effect of this operation.

    Returns
    -------
    list
        List of indices to keep, sorted by decreasing score confidence
    """
    keep = []

    # Check that boxes are in correct orientation
    deltas = boxes[:, 2:] - boxes[:, :2]
    if not deltas.min() > 0:
        raise ValueError("Boxes should be encoded [x1, y1, x2, y2] with x1 < x2 & y1 < y2")

    # Build the BallTree
    boxtree = BoxTree(boxes, 32)

    # Compute the areas once and for all: avoid recomputing it at each step
    areas = area(boxes)

    # Order by decreasing confidence
    order = np.argsort(scores)[::-1]
    # Create a mask to keep track of boxes which have alread been visited
    to_consider = np.full(len(boxes), True)
    for current_idx in order:
        # If already visited or discarded
        if not to_consider[current_idx]:
            continue

        boxA = boxes[current_idx]

        # Query the overlapping boxes and return their intersection
        query, query_intersections = boxtree.intersect(boxA, 0.0)

        for k, query_idx in enumerate(query):
            if not to_consider[query_idx]:
                continue
            inter = query_intersections[k]
            sc = inter / (areas[current_idx] + areas[query_idx] - inter)
            to_consider[query_idx] = sc < iou_threshold

        # Add the current box
        keep.append(current_idx)
        to_consider[current_idx] = False

    return np.array(keep)


@njit(fastmath=True)
def naive_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.1):
    """
    Naive NMS, for timing comparisons only.
    """
    # keep = np.empty(len(boxes), dtype=np.int64)
    keep = []

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # n_kept = 0
    suppressed = np.full(len(scores), False)
    order = np.argsort(scores, kind="quicksort")[::-1]
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        current_idx = order[i]

        keep.append(current_idx)

        for j in range(i, len(order), 1):
            if suppressed[j]:
                continue
            inter = intersection(boxes[current_idx], boxes[order[j]])
            sc = inter / (areas[current_idx] + areas[order[j]] - inter)
            suppressed[j] = sc > iou_threshold

    return keep
