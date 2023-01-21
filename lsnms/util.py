import math
from typing import Optional

import numpy as np
from numba import njit


@njit(cache=True)
def area(box):
    """
    Computes bbox(es) area: is vectorized.

    Parameters
    ----------
    box : np.array
        Box(es) in format (x0, y0, x1, y1)

    Returns
    -------
    np.array
        area(s)
    """
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])


@njit(cache=True, fastmath=True)
def intersection(boxA, boxB):
    """
    Compute area of intersection of two boxes

    Parameters
    ----------
    boxA : np.array
        First boxes
    boxB : np.array
        Second box

    Returns
    -------
    float64
        Area of intersection
    """
    xA = max(boxA[..., 0], boxB[..., 0])
    xB = min(boxA[..., 2], boxB[..., 2])
    dx = xB - xA
    if dx <= 0:
        return 0.0

    yA = max(boxA[..., 1], boxB[..., 1])
    yB = min(boxA[..., 3], boxB[..., 3])
    dy = yB - yA
    if dy <= 0.0:
        return 0.0

    # compute the area of intersection rectangle
    return dx * dy


@njit(cache=True)
def distance_to_hypersphere(X, centroid, radius):
    """
    Computes the smallest square distance from one point to a sphere defined by its centroid and
    radius.

    Parameters
    ----------
    X : np.array
        Single point
    centroid : np.array
        Sphere centroid coordinates
    radius : float
        Sphere radius

    Returns
    -------
    float
        Distance to the sphere.
    """
    centroid_dist = rdist(X, centroid)
    return max(0, centroid_dist**0.5 - radius**0.5) ** 2


@njit(cache=True)
def rdist(X1, X2):
    """
    Simple square distance between two points.
    """
    dim = X1.shape[-1]
    d_sq = 0.0
    for j in range(dim):
        d_sq += (X1[j] - X2[j]) ** 2
    return d_sq


@njit(cache=True)
def englobing_sphere(data):
    """
    Compute parameters (centroid and radius) of the smallest sphere
    containing all the data points given in `data`.

    Parameters
    ----------
    data : np.array
        Dataset

    Returns
    -------
    Tuple
        centroid, and radius
    """
    centroid = data.sum(0) / len(data)
    max_radius = 0.0
    for x in data:
        radius = rdist(centroid, x)
        max_radius = max(max_radius, radius)
    return centroid, max_radius


@njit(cache=True)
def max_spread_axis(data):
    """
    Returns the axis of maximal spread.

    Parameters
    ----------
    data : np.array
        Dataset

    Returns
    -------
    int
        Axis of maximal spread
    """
    max_spread = 0.0
    splitdim = -1
    for j in range(data.shape[1]):
        spread = data[:, j].max() - data[:, j].min()
        if spread > max_spread:
            max_spread = spread
            splitdim = j
    return splitdim


@njit(cache=True)
def split_along_axis(data, axis):
    """
    Splits the data along axis in two datasets of equal size.
    This method uses an adapted re-implementation of `np.argpartition`

    Parameters
    ----------
    data : np.array
        Dataset
    axis : int
        Axis to split along

    Returns
    -------
    Tuple[np.array]
        Left data point indices, right data point indices
    """
    left, right = median_argsplit(data[:, axis])
    return left, right
    # counts, bins = np.histogram(data[:, axis])
    # bins = (bins[1:] + bins[:-1]) / 2
    # cap = bins[counts.argmin()]
    # mask = data[:, axis] <= cap
    # n_left = mask.sum()
    # # Account for the case where all positions along this axis are equal: split in the middle
    # if n_left == len(data) or n_left == 0:
    #     left = indices[: len(indices) // 2]
    #     right = indices[len(indices) // 2 :]
    # else:
    #     left = indices[mask]
    #     right = indices[np.logical_not(mask)]
    # return left, right


@njit(cache=True)
def distance_to_hyperplan(x, box):
    """
    Computes distance from a point to a hyperplan defined by its bounding box.
    Used to compute distance lower bound between a node and a query point.

    Parameters
    ----------
    x : np.array
        Query point (x0, y0, x1, y1)
    box : np.array
        Bounding box in format  (x0, y0, x1, y1)

    Returns
    -------
    float
        Distance to the bounding box
    """
    d_sq = 0.0
    dim = x.shape[-1]
    for j in range(dim):
        d_sq += max(box[j] - x[j], 0, x[j] - box[j + dim]) ** 2.0
    return d_sq


@njit(cache=True)
def englobing_box(data):
    """
    Computes coordinates of the smallest bounding box containing all
    the data points.

    Parameters
    ----------
    data : np.array
        datapoints

    Returns
    -------
    np.array
        Bounding box in format  (x0, y0, x1, y1)
    """
    bounds = []
    for j in range(data.shape[-1]):
        bounds.insert(j, data[:, j].min())
        bounds.insert(2 * j + 1, data[:, j].max())
    return np.array(bounds)


@njit(cache=True)
def box_englobing_boxes(boxes):
    """
    Computes coordinates of the smallest bounding box containing all
    the boxes.

    Parameters
    ----------
    boxes : np.array
        Boxes

    Returns
    -------
    np.array
        Bounding box in format  (x0, y0, x1, y1)
    """
    dim = boxes.shape[-1]
    bounds = np.empty((dim,))
    for j in range(dim):
        if j < dim // 2:
            bounds[j] = boxes[:, j].min()
        else:
            bounds[j] = boxes[:, j].max()

    return bounds


@njit(cache=True)
def _partition(A, low, high, indices):
    """
    This is straight from numba master:
    https://github.com/numba/numba/blob/b5bd9c618e20985acb0b300d52d57595ef6f5442/numba/np/arraymath.py#L1155
    I modified it so the swaps operate on the indices as well, because I need a argpartition
    """
    mid = (low + high) >> 1
    # NOTE: the pattern of swaps below for the pivot choice and the
    # partitioning gives good results (i.e. regular O(n log n))
    # on sorted, reverse-sorted, and uniform arrays.  Subtle changes
    # risk breaking this property.
    # Use median of three {low, middle, high} as the pivot
    if A[mid] < A[low]:
        A[low], A[mid] = A[mid], A[low]
        indices[low], indices[mid] = indices[mid], indices[low]
    if A[high] < A[mid]:
        A[high], A[mid] = A[mid], A[high]
        indices[high], indices[mid] = indices[mid], indices[high]
    if A[mid] < A[low]:
        A[low], A[mid] = A[mid], A[low]
        indices[low], indices[mid] = indices[mid], indices[low]
    pivot = A[mid]

    A[high], A[mid] = A[mid], A[high]
    indices[high], indices[mid] = indices[mid], indices[high]
    i = low
    j = high - 1
    while True:
        while i < high and A[i] < pivot:
            i += 1
        while j >= low and pivot < A[j]:
            j -= 1
        if i >= j:
            break
        A[i], A[j] = A[j], A[i]
        indices[i], indices[j] = indices[j], indices[i]
        i += 1
        j -= 1
    # Put the pivot back in its final place (all items before `i`
    # are smaller than the pivot, all items at/after `i` are larger)
    # print(A)
    A[i], A[high] = A[high], A[i]
    indices[i], indices[high] = indices[high], indices[i]
    return i


@njit(cache=True)
def _select(arry, k, low, high):
    """
    This is straight from numba master:
    https://github.com/numba/numba/blob/b5bd9c618e20985acb0b300d52d57595ef6f5442/numba/np/arraymath.py#L1155
    Select the k'th smallest element in array[low:high + 1].
    """
    indices = np.arange(len(arry))
    i = _partition(arry, low, high, indices)
    while i != k:
        if i < k:
            low = i + 1
            i = _partition(arry, low, high, indices)
        else:
            high = i - 1
            i = _partition(arry, low, high, indices)
    return indices, i


@njit(cache=True)
def median_argsplit(arry):
    """
    Splits `arry` into two sets of indices, indicating values
    above and below the pivot value. Often, pivot is the median.

    This is approx. three folds faster than computing the median,
    then find indices of values below (left indices) and above (right indices)

    Parameters
    ----------
    arry : np.array
        One dimensional values array

    Returns
    -------
    Tuple[np.array]
        Indices of values below median, indices of values above median
    """
    low = 0
    high = len(arry) - 1
    k = len(arry) >> 1
    tmp_arry = arry.flatten()
    indices, i = _select(tmp_arry, k, low, high)
    left = indices[:k]
    right = indices[k:]
    return left, right


def offset_bboxes(bboxes: np.array, class_ids: np.array):
    """
    Function used to offset bboxes of different classes so that boxes of different
    classes do not overlap. This is the trick used in multiclass NMS standardly.

    The only peculiarity here is that in a regular NMS setup, boxes can be offseted "diagonally".
    i.e. all pushed along a diagonal, something like:
    ```
    max_offset = bboxes[2:].max()
    offset = class_id * max_offset
    ```
    To create a "diagonal per block" aspect.

    Note that this would hurt performances because the underlying RTree that we would build on this
    would be suboptimal: many regions would actually be empty (because RTree builds rectangular
    regions) and the query time would be impacted.

    Instead, here the boxes are offseted forming a "mosaic" of class-wise regions, see figures
    in readme.

    Note that this function is completely dimensionality agnostic.


    Parameters
    ----------
    bboxes : np.array
        Array of bboxes in VOC format
    class_ids : np.array
        One-dimensional array of integers class identifiers.

    Returns
    -------
    np.array
        New array of bounding boxes, offseted class-wise.
    """
    # Compute offset (or class subsector size)
    dimensionality = bboxes.shape[-1] // 2
    # + 2 to avoid any overlap between subregions
    max_offset = bboxes[:, dimensionality:].max() + 2

    # Build the pavement of class-wise subsectors
    # We actually dont care about actual class labels, replace it by their index
    classes, class_ids = np.unique(class_ids, return_inverse=True)
    n_classes = classes.size
    class_indexer = np.arange(n_classes)
    mosaic_width = round(math.ceil(n_classes ** (1.0 / dimensionality)))
    mosaic_shape = (mosaic_width,) * dimensionality
    # Get class subsector positions within the mosaic
    class_subsector_positions = np.unravel_index(class_indexer, mosaic_shape)
    # Make it a vector
    class_subsector_positions = np.stack(class_subsector_positions, axis=1)
    class_offset = class_subsector_positions * max_offset
    # Make it double for both bbox bounds
    class_offset = np.concatenate((class_offset, class_offset), axis=1)

    # Retrieve the offset class wise
    bboxes_offset = class_offset[class_ids]

    return bboxes + bboxes_offset


def check_correct_arrays(boxes: np.array, scores: np.array, class_ids: Optional[np.array]):
    """
    Check arrays characteristics: dtype dimensionality and shape
    """
    # Check dtypes:
    if not boxes.dtype == "float64":
        raise ValueError(f"Boxes should a float64 array. Received {boxes.dtype}")
    if not scores.dtype == "float64":
        raise ValueError(f"Scores should a float64 array. Received {scores.dtype}")
    if class_ids is not None:
        if not np.can_cast(class_ids, np.int64) or class_ids.min() < 0:
            raise ValueError(
                f"Class ids should be a positive integer array. "
                f"Received {class_ids.dtype} with min {class_ids.min()}"
            )

    # Check shapes
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(
            f"Boxes should be of shape (n_boxes, 4). Received object of shape {boxes.shape}."
        )
    if scores.ndim != 1 or len(scores) != len(boxes):
        raise ValueError(
            f"Scores should be a one-dimensional vector of same size as boxes vector. "
            f"Received object of shape {scores.shape} but {len(boxes)} boxes."
        )
    if class_ids is not None:
        if class_ids.ndim != 1 or len(class_ids) != len(boxes):
            raise ValueError(
                f"Class_ids should be a one-dimensional vector of same size as boxes vector. "
                f"Received object of shape {class_ids.shape} but {len(boxes)} boxes."
            )

    # Check that boxes are in correct orientation
    deltas = boxes[:, 2:] - boxes[:, :2]
    if not deltas.min() > 0:
        raise ValueError("Boxes should be encoded [x1, y1, x2, y2] with x1 < x2 & y1 < y2")


def check_correct_input(
    boxes: np.array,
    scores: np.array,
    class_ids: Optional[np.array],
    iou_threshold: float,
    score_threshold: float,
):
    """
    Checks input validity: shape, dtype, dimensionality, and boundary values.
    """

    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if class_ids is not None:
        class_ids = np.asarray(class_ids)

    check_correct_arrays(boxes, scores, class_ids)

    # Check boundary values
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError(f"IoU threshold should be between 0. and 1. Received {iou_threshold}.")
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise ValueError(f"IoU threshold should be between 0. and 1. Received {score_threshold}.")

    if class_ids is not None:
        return boxes, scores, class_ids
    else:
        return boxes, scores
