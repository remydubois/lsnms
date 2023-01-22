import warnings
from typing import Optional

import numpy as np
from numba import njit

from lsnms.rtree import RTree
from lsnms.util import area, check_correct_input


@njit
def _wbc(
    boxes: np.array,
    scores: np.array,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    iou_reweight: bool = False,
):
    """
    See lsnms.wbc docstring.
    """

    pooled_boxes = []
    pooled_scores = []
    cluster_indices = []

    # Compute box areas once and for all
    areas = area(boxes)

    # Discard boxes below score threshold right now to avoid building the tree on useless boxes
    boxes = boxes[scores > score_threshold]

    # Index boxes
    tree = RTree(boxes, 32)

    # Sort boxes by decreasing confidence scores
    order = np.argsort(scores)[::-1]
    # Create a mask to keep track of boxes which have alread been visited
    to_consider = np.full(len(boxes), True)
    for current_idx in order:
        if not to_consider[current_idx]:
            continue

        # If reached low scoring box, break the algorithm
        if scores[current_idx] < score_threshold:
            break

        boxA = boxes[current_idx]

        # If a cutoff distance was specified, query neighbors within this distance
        query, query_intersections = tree.intersect(boxA)

        # Prepare consensus container
        neighbors = []
        iou_mass = []
        add_self = True
        for query_idx, overlap in zip(query, query_intersections):
            # Ensure that itself is in the cluster, we have no guarantee
            # that it is in the query (if query is specified)
            if query_idx == current_idx:
                add_self = False

            # Or if the box has alread been aggregated, then skip it
            if not to_consider[query_idx]:
                continue

            sc = overlap / (areas[current_idx] + areas[query_idx] - overlap)
            to_consider[query_idx] = sc < iou_threshold
            # If in cluster...
            if sc >= iou_threshold:
                iou_mass.append(sc if iou_reweight else 1.0)
                neighbors.append(query_idx)

        # Add self if needed
        if add_self:
            neighbors.append(current_idx)

        # Compute cluster consensus
        neighbors = np.array(neighbors)
        iou_mass = np.array(iou_mass)
        cluster_score = (scores[neighbors] * iou_mass).sum() / iou_mass.sum()
        cluster_coords = (boxes[neighbors] * np.expand_dims(iou_mass, -1)).sum(0) / iou_mass.sum()

        # Better to explicitely add self index
        cluster_indices.append((current_idx, neighbors))

        # Append to the results
        # Turn into list for further easier array construction
        pooled_boxes.append(list(cluster_coords))
        pooled_scores.append(float(cluster_score))
        to_consider[current_idx] = False

    # Convert to np array
    pooled_boxes_array = np.array(pooled_boxes)
    pooled_scores_array = np.array(pooled_scores)

    return pooled_boxes_array, pooled_scores_array, cluster_indices


def wbc(
    boxes: np.array,
    scores: np.array,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    iou_reweight: bool = False,
    cutoff_distance: Optional[int] = None,
    tree: Optional[str] = None,
) -> np.array:
    """
    Sparse Weighted box clustering loosely inspired from https://arxiv.org/pdf/1811.08661.pdf.
    For a given bbox, instead of discarding all other overlapping bboxes with lower
    score, it aggregates them all (overlapping bboxes which have not already been aggregated) by
    averaging confidence scores and box coordinates.

    It returns a tuple of:
    - the cluster boxes (pooled coordinates per cluster)
    - the cluster score (pooled score per cluster)
    - the cluster indices: for each cluster, a tuple containing:
        - index of the highest scoring box in the cluster
        (the box onto which other boxes were pulled toward)
        - indices of all other boxes (including the highest one) pulled in the cluster


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
    iou_reweight : bool, optional
        Enable IoU reweighting when computing instance consensus, by default False
        If set to `true`, for each box, the higher the IoU with the highest scoring box of the
        cluster, the higher it will contribute to the global (coordinates and scores) consensus.
    cutoff_distance: int, optional
        DEPRECATED, used for compatibility with version 0.1.X.
        Since version 0.2.X, it is useless because overlapping boxes are queried using a R-Tree,
        which is parameter free.
    tree: str, optional
        DEPRECATED, used for compatibility with version 0.1.X.
        Since version 0.2.X, the tree used is a R-Tree.

    Returns
    -------
    Tuple[np.array]
        Pooled boxes, pooled scores, cluster indices
    """
    if cutoff_distance is not None or tree is not None:
        warnings.warn(
            "Both `cutoff_distance` and `tree` are deprecated and effect-less from version"
            "0.2.X, since R-Tree is used by default to query overlapping boxes."
        )

    # Convert dtype, check shapes, dimensionality, and boundary values.
    boxes, scores = check_correct_input(
        boxes, scores, None, iou_threshold=iou_threshold, score_threshold=score_threshold
    )
    # Run WBC
    keep = _wbc(
        boxes,
        scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        iou_reweight=iou_reweight,
    )

    return keep
