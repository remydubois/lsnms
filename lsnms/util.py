from numba import njit
import numpy as np


@njit
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


@njit(fastmath=True)
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


@njit
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
    return max(0, centroid_dist ** 0.5 - radius ** 0.5) ** 2


@njit
def rdist(X1, X2):
    """
    Simple square distance between two points.
    """
    dim = X1.shape[-1]
    d_sq = 0.0
    for j in range(dim):
        d_sq += (X1[j] - X2[j]) ** 2
    return d_sq


@njit
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


@njit
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


@njit
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


@njit
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


@njit
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

@njit
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
    bounds = np.empty((dim, ))
    for j in range(dim):
        if j < dim // 2:
            bounds[j] = boxes[:, j].min()
        else:
            bounds[j] = boxes[:, j].max()

    return bounds


@njit
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


@njit
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


@njit
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


def check_correct_arrays(boxes: np.array, scores: np.array):
    """
    Check arrays characteristics: dtype dimensionality and shape
    """
    # Check dtypes:
    if not boxes.dtype == "float64":
        raise ValueError(f"Boxes should a float64 array. Received {boxes.dtype}")
    if not scores.dtype == "float64":
        raise ValueError(f"Scores should a float64 array. Received {scores.dtype}")

    # Check shapes
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(
            f"Boxes should be of shape (n_boxes, 4). Received object of shape {boxes.shape}."
        )
    if scores.ndim != 1:
        raise ValueError(
            f"Scores should be a one-dimensional vector. Received object of shape {scores.shape}."
        )

    # Check that boxes are in correct orientation
    deltas = boxes[:, 2:] - boxes[:, :2]
    if not deltas.min() > 0:
        raise ValueError("Boxes should be encoded [x1, y1, x2, y2] with x1 < x2 & y1 < y2")


def check_correct_input(
    boxes: np.array, scores: np.array, iou_threshold: float, score_threshold: float
):
    """
    Checks input validity: shape, dtype, dimensionality, and boundary values.
    """

    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    check_correct_arrays(boxes, scores)

    # Check boundary values
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError(f"IoU threshold should be between 0. and 1. Received {iou_threshold}.")
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise ValueError(f"IoU threshold should be between 0. and 1. Received {score_threshold}.")

    return boxes, scores
