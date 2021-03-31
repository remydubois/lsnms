import numpy as np
from numba import njit
from collections import OrderedDict
from numba.experimental import jitclass
from numba import deferred_type, optional, int64, float64, boolean
from lsnms.util import (
    intersection,
    split_along_axis,
    rdist,
    distance_to_hyperplan,
    box_englobing_boxes,
    max_spread_axis,
)


specs = OrderedDict()
node_type = deferred_type()
specs["data"] = float64[:, :]
specs["bbox"] = float64[:]
specs["axis"] = int64
specs["dimensionality"] = int64
specs["indices"] = optional(int64[:])
specs["is_leaf"] = boolean
specs["leaf_size"] = int64
specs["left"] = optional(node_type)
specs["right"] = optional(node_type)


@jitclass(specs)
class Node:
    """
    Main object for the node class.

    Note that the tree building process is peculiar:
    Since jit classes methods can not be recursive (node.method can not call node.method), the
    tree building process (recursive node splitting and children instanciation) can not be done
    inside the Node.__init__ method (which is the natural way to do so).
    However, jit classes methods can call recursive functions: hence, the tree building process is
    delegated to an independant function (see `build` function).
    Consequently, this class must be used in the following way:
    ```
    # Instanciate the root node
    node = Node(data)
    # Reccursively attach children to each node
    node.build()
    ```

    For convenience, a wrapper `KDTree` class was implemented, encapsulating this process:
    ```
    tree = KDTree(data)
    tree.query_radius(...)
    ```
    """

    def __init__(self, data, leaf_size=16, axis=0, indices=None):
        # Stores the data
        self.data = data
        self.axis = axis
        assert len(data) > 0, 'Empty dataset'
        assert self.data.shape[-1] % 2 == 0, 'odd dimensionality'
        self.dimensionality = data.shape[-1] // 2

        # Stores indices of each data point
        if indices is None:
            self.indices = np.arange(len(data))
        else:
            self.indices = indices

        self.leaf_size = leaf_size

        # Is it a leaf
        if len(data) <= leaf_size:
            self.is_leaf = True
        else:
            self.is_leaf = False

        # Determine node bounding box
        self.bbox = box_englobing_boxes(self.data)

        # Pre-assign empty children for typing
        self.left = None
        self.right = None

    def split(self):
        """
        Splits a node into two children nodes.

        Returns
        -------
        Tuple[Node]
            Left children and right children
        """
        # Split the boxes using top left corner
        left_indices, right_indices = split_along_axis(
            self.data[:, : self.dimensionality], self.axis
        )
        # Simply reference the data in the children, do not copy arrays
        next_axis = (self.axis + 1) % self.dimensionality
        left_node = Node(
            self.data[left_indices], self.leaf_size, next_axis, self.indices[left_indices]
        )
        right_node = Node(
            self.data[right_indices], self.leaf_size, next_axis, self.indices[right_indices]
        )
        return left_node, right_node

    def assign_left(self, node):
        """
        Assigns the left node.
        Strangely enough, this needs to be delegated to an explicit method.
        """
        self.left = node

    def assign_right(self, node):
        """
        Assigns the right node.
        Strangely enough, this needs to be delegated to an explicit method.
        """
        self.right = node

    def build(self):
        """
        Reccursively build the children.
        Jit methods can not be explicitely recursive:
        `self.build` can not call `self.build`, but it can call a function
        which calls itself, the workaround used here.
        """
        # Reccursively attach children to the parent
        build(self)

    def query_radius(self, X, max_radius):
        """
        Return set of point in the dataset distance from less than `radius` to the query point X.

        Parameters
        ----------
        X : np.array
            Query point (single point only)
        radius : float
            max radius

        Returns
        -------
        np.array
            Indices of points within that radius from the query point

        Raises
        ------
        ValueError
            This function works on single query point only.
        """
        if X.ndim > 1:
            raise ValueError("query_radius only works on single query point.")
        if len(X) != 2:
            raise ValueError("Query point must be two-dimensional")
        # Initialize empty list of int64
        # Needs to be typed
        buffer = [0][:0]

        # Query recursive
        # Fills in-place the neighbors list
        query_radius(self, X, max_radius ** 2, buffer)

        # return np array for convenience
        return np.array(buffer)

    def intersect(self, X, min_area=0.0):

        indices_buffer = [0][:0]
        intersections_buffer = [0.0][:0]

        intersect(
            self,
            X,
            indices_buffer,
            intersections_buffer,
            1.0,
            True,
            min_area,
        )

        indices_buffer = np.array(indices_buffer)
        intersections_buffer = np.array(intersections_buffer)

        return indices_buffer, intersections_buffer


node_type.define(Node.class_type.instance_type)


@njit(fastmath=True)
def build(current):
    """
    Reccursive building process.
    Since jit methods can not be recursive, it has to be a detached function.
    Otherwise, it would just be included inside the Node.__init__ method.

    Parameters
    ----------
    current : Nodetimiings
        Current node to split if needed
    """
    if not current.is_leaf:
        left, right = current.split()
        current.assign_left(left)
        current.assign_right(right)
        build(current.left)
        build(current.right)


@njit(fastmath=True)
def query_radius(node, X, max_radius, buffer, dist_LB=0.0, is_root=True):
    """
    This function should not be used as-is: jitted-class methods can not be recursive.
    The recursive query process is delegated here.

    This is a depth-first search: by ensuring that one chooses the closest
    node first, the algorithm will consequently first go to the node containing the point (if any)
    and then go backward in neighbors node, trimming each node too far from the `max_radius` given.

    Parameters
    ----------
    node: Node
        Currently visited node
    X : np.array
        Query point (one point).
    max_radius : float
        Max radius
    buffer : list
        List of currently-gathered neighbors. Stores in-place the neighbors along the search process
    dist_LB : float, optional
        Distance lower bound: distance from the query point to the currently visited node's
        hyperplan, by default 0.0
    is_root : bool, optional
        Whether the currently visited node is root, by default True
    """
    if is_root:
        # If first call, no lower bound distance has already been computed
        dist_LB = distance_to_hyperplan(X, node.bbox)

    # if query is outside the radius, then trim this node out
    if dist_LB > max_radius:
        return

    # If it's a leaf: check points inside
    elif node.is_leaf:
        for i, y in zip(node.indices, node.data):
            d = rdist(X, y)
            if d <= max_radius:
                buffer.append(i)

    # Else, continue search
    # Going for the closest node first ensures a depth-first search
    else:
        left_LB = distance_to_hyperplan(X, node.left.bbox)
        right_LB = distance_to_hyperplan(X, node.right.bbox)

        if left_LB < right_LB:
            query_radius(node.left, X, max_radius, buffer, left_LB, False)
            query_radius(node.right, X, max_radius, buffer, right_LB, False)
        else:
            query_radius(node.right, X, max_radius, buffer, right_LB, False)
            query_radius(node.left, X, max_radius, buffer, left_LB, False)


@njit(fastmath=True)
def intersect(
    node, X, indices_buffer, intersections_buffer, inter_UB=1.0, is_root=True, min_area=0.0
):
    """
    This function should not be used as-is: jitted-class methods can not be recursive.
    The recursive query process is delegated here.

    This is a depth-first search: by ensuring that one chooses the closest
    node first, the algorithm will consequently first go to the node containing the point (if any)
    and then go backward in neighbors node, trimming each node too far from the `max_radius` given.

    Parameters
    ----------
    node: Node
        Currently visited node
    X : np.array
        Query point (one point).
    max_radius : float
        Max radius
    buffer : list
        List of currently-gathered neighbors. Stores in-place the neighbors along the search process
    dist_LB : float, optional
        Distance lower bound: distance from the query point to the currently visited node's
        hyperplan, by default 0.0
    is_root : bool, optional
        Whether the currently visited node is root, by default True
    """
    if is_root:
        # If first call, no lower bound distance has already been computed
        inter_UB = intersection(X, node.bbox)

    # if query is outside the radius, then trim this node out
    if inter_UB <= min_area:
        return

    # If it's a leaf: check points inside
    elif node.is_leaf:
        for i, y in zip(node.indices, node.data):
            inter = intersection(X, y)
            if inter > min_area:
                # buffer.append((inter, i))
                indices_buffer.append(i)
                intersections_buffer.append(inter)

    # Else, continue search
    # Going for the node with highest intersection first ensures a depth-first search
    else:
        left_UB = intersection(X, node.left.bbox)
        right_UB = intersection(X, node.right.bbox)
        if left_UB > right_UB:
            intersect(node.left, X, indices_buffer, intersections_buffer, left_UB, False, min_area)
            intersect(
                node.right, X, indices_buffer, intersections_buffer, right_UB, False, min_area
            )
        else:
            intersect(
                node.right, X, indices_buffer, intersections_buffer, right_UB, False, min_area
            )
            intersect(node.left, X, indices_buffer, intersections_buffer, left_UB, False, min_area)


specs = OrderedDict()
specs["_root"] = node_type
specs["data"] = float64[:, :]


@jitclass(specs)
class BoxTree:
    """
    Simple wrapper class to build the tree, it encapsulates root node instanciation,
    followed by the recursive building process, which can not be called in node.__init__
    because jitted methods can not be recursive.

    """

    def __init__(self, data, leaf_size=16):
        self.data = data
        axis = max_spread_axis(self.data)
        self._root = Node(data, leaf_size, axis, None)
        self._root.build()

    def query_radius(self, X, radius):
        """
        Return set of point in the dataset distance from less than `radius` to the query point X.

        Parameters
        ----------
        X : np.array
            Query point (single point only)
        radius : float
            max radius

        Returns
        -------
        np.array
            Indices of points within that radius from the query point
        """
        return self._root.query_radius(X, radius)

    def intersect(self, X, min_area=0.0):
        return self._root.intersect(X, min_area)
