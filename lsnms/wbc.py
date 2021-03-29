from numba import njit
import numpy as np
from lsnms.balltree import BallTree
from lsnms.kdtree import KDTree
from lsnms.util import area, intersection


@njit
def wbc(
    boxes,
    scores,
    iou_threshold=0.5,
    score_threshold=0.1,
    cutoff_distance=-1,
    iou_reweight=False,
    tree="kdtree",
):
    """
    Weighted box clustering loosely inspired from https://arxiv.org/pdf/1811.08661.pdf.
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
    cutoff_distance : int, optional
        Distance from which boxes are considered too far appart to be considered, by default -1
        If two boxes are distant of more than this value, the iou score will not even be computed.
        Setting a positive value to this parameter will result in:
            - creating a BallTree indexed on top-left corner of bounding boxes
            - at each trimming step, the tree will be queried to return all the neighbors distant
            from less than `cutoff_distance` from the considered box in a O(log(n)) complexity
            (n=len(boxes))
    iou_reweight : bool, optional
        Enable IoU reweighting when computing instance consensus, by default False
        If set to `true`, for each box, the higher the IoU with the highest scoring box of the
        cluster, the higher it will contribute to the global (coordinates and scores) consensus.
    tree: str
        Type of tree to use. Either one of "kdtree" or "balltree". No critical difference in
        performances were observed during testing.
    Returns
    -------
    Tuple[np.array]
        Pooled boxes, pooled scores, cluster indices
    """
    if tree not in ["kdtree", "balltree"]:
        raise ValueError('`tree` must be either one of "balltree" or "kdtree"')
    pooled_boxes = []
    pooled_scores = []
    cluster_indices = []

    # Check that boxes are in correct orientation
    deltas = boxes[:, 2:] - boxes[:, :2]
    assert deltas.min() > 0

    # Compute box areas once and for all
    areas = area(boxes)

    # Build the BallTree
    if cutoff_distance >= 0:
        # Numba can not unify different types for the same variable
        if tree == "kdtree":
            kdtree = KDTree(boxes[:, :2], 32)
        else:
            balltree = BallTree(boxes[:, :2], 32)

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
        if cutoff_distance >= 0:
            if tree == "kdtree":
                query = kdtree.query_radius(boxA[:2], cutoff_distance)
            else:
                query = balltree.query_radius(boxA[:2], cutoff_distance)
        # Else, just review all the boxes
        else:
            query = order

        # Prepare consensus container
        neighbors = []
        iou_mass = []
        add_self = True
        for query_idx in query:
            # Ensure that itself is in the cluster, we have no guarantee
            # that it is in the query (if query is specified)
            if query_idx == current_idx:
                add_self = False

            # Or if the box has alread been aggregated, then skip it
            if not to_consider[query_idx]:
                continue

            boxB = boxes[query_idx]
            inter = intersection(boxA, boxB)
            sc = inter / (areas[current_idx] + areas[query_idx] - inter)
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
