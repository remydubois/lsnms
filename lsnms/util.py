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


@njit
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
    dx = max(xB - xA, 0.0)
    # if dx <= 0:
    #     return 0.0

    yA = max(boxA[..., 1], boxB[..., 1])
    yB = min(boxA[..., 3], boxB[..., 3])
    dy = max(yB - yA, 0.0)
    # if dy <= 0.0:
    #     return 0.0

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
    Note that this could probably be optimized further, by implementing the median algorithm from
    scratch.

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
    indices = np.arange(len(data))
    cap = np.median(data[:, axis])
    mask = data[:, axis] <= cap
    n_left = mask.sum()
    # Account for the case where all positions along this axis are equal: split in the middle
    if n_left == len(data) or n_left == 0:
        left = indices[: len(indices) // 2]
        right = indices[len(indices) // 2 :]
    else:
        left = indices[mask]
        right = indices[np.logical_not(mask)]
    return left, right


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
    bounds = []
    ndim = boxes.shape[-1] // 2
    for j in range(ndim):
        bounds.insert(j, boxes[:, j].min())
        bounds.insert(2 * j + 1, boxes[:, j + ndim].max())
    return np.array(bounds)
