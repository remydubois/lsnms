from numba import njit
import numpy as np
from lsnms.balltree import BallTree
from lsnms.kdtree import KDTree
from lsnms.util import area, intersection


@njit
def nms(boxes, scores, iou_threshold=0.5, score_threshold=0.1, cutoff_distance=-1, tree="kdtree"):
    """
    Regular NMS.
    For a given bbox, it will greedily discard all the other overlapping bboxes with lower score.
    Note that this implementation allows some sparsity in the process, by only checking boxes
    distant from less than `cutoff_distance`. See parameters for details.

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
    score_threshold : float, optional
        Score from which to discard instance based on this confidence score, by default 0.1
    cutoff_distance : int, optional
        Distance from which boxes are considered too far appart to be considered, by default -1
        If two boxes are distant of more than this value, the iou score will not even be computed.
        Setting a positive value to this parameter will result in:
            - creating a BallTree indexed on top-left corner of bounding boxes
            - at each trimming step, the tree will be queried to return all the neighbors distant
            from less than `cutoff_distance` from the considered box in a O(log(n)) complexity
            (n=len(boxes))
    tree: str
        Type of tree to use. Either one of "kdtree" or "balltree". No critical difference in
        performances were observed during testing.


    Returns
    -------
    list
        List of indices to keep, sorted by decreasing score confidence
    """
    if tree not in ["kdtree", "balltree"]:
        raise ValueError('`tree` must be either one of "balltree" or "kdtree"')
    keep = []

    # Check that boxes are in correct orientation
    deltas = boxes[:, 2:] - boxes[:, :2]
    assert deltas.min() > 0

    # Build the BallTree
    if cutoff_distance >= 0:
        # Numba can not unify different types for the same variable
        if tree == "kdtree":
            kdtree = KDTree(boxes[:, :2], 32)
        else:
            balltree = BallTree(boxes[:, :2], 32)

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

        # If a cutoff distance was specified, query neighbors within this distance
        if cutoff_distance >= 0:
            if tree == "kdtree":
                query = kdtree.query_radius(boxA[:2], cutoff_distance)
            else:
                query = balltree.query_radius(boxA[:2], cutoff_distance)
        # Else, just review all the boxes
        else:
            query = order

        for query_idx in query:
            if not to_consider[query_idx]:
                continue
            boxB = boxes[query_idx]
            inter = intersection(boxA, boxB)
            sc = inter / (areas[current_idx] + areas[query_idx] - inter)
            to_consider[query_idx] = sc < iou_threshold

        keep.append(current_idx)
        to_consider[current_idx] = False

    return np.array(keep)


@njit
def naive_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.1):
    """
    Naive NMS, for timing comparisons only.
    """
    keep = np.empty(len(boxes), dtype=np.int64)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    n_kept = 0
    order = [i for i in np.argsort(scores, kind="quicksort")[::-1]]
    while len(order):
        current_idx = order.pop(0)

        keep[n_kept] = current_idx
        n_kept += 1

        # Mutate in place the indices list
        n = 0
        for _ in range(len(order)):
            inter = intersection(boxes[current_idx], boxes[order[n]])
            sc = inter / (areas[current_idx] + areas[order[n]] - inter)
            if sc > iou_threshold:
                # If pop, no need to shift the index to the next position
                # Since popping will naturally shift the values
                order.pop(n)
            else:
                # If no pop, then shift to the next box
                n += 1

    return keep[:n_kept]
