from collections import OrderedDict
from typing import List

import numpy as np
from numba import boolean, deferred_type, float64, int64, njit, optional
from numba.experimental import jitclass

from lsnms.util import (
    box_englobing_boxes,
    intersection,
    max_spread_axis,
    split_along_axis,
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
class RNode:
    """
    Main object for the node class.

    Note that the tree building process is peculiar:
    Since jit classes methods can not be recursive (node.method can not call node.method), the
    tree building process (recursive node splitting and children instanciation) can not be done
    inside the RNode.__init__ method (which is the natural way to do so).
    However, jit classes methods can call recursive functions: hence, the tree building process is
    delegated to an independant function (see `build` function).
    Consequently, this class must be used in the following way:
    ```
    # Instanciate the root node
    node = RNode(data)
    # Reccursively attach children to each node
    node.build()
    ```

    For convenience, a wrapper `RTree` class was implemented, encapsulating this process:
    ```
    tree = RTree(data)
    # or
    tree.intersect(box_of_interest)
    ```
    """

    def __init__(self, data, leaf_size=16, axis=0, indices=None):
        # Stores the data
        self.data = data
        self.axis = axis
        # Quick sanity checks
        assert leaf_size > 0, "Leaf size must be strictly positive"
        assert len(data) > 0, "Empty dataset"
        assert self.data.shape[-1] % 2 == 0, "odd dimensionality"
        assert data.ndim == 2, "Boxes to index should be (n_boxes, 4)"

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
        left_node = RNode(
            self.data[left_indices], self.leaf_size, next_axis, self.indices[left_indices]
        )
        right_node = RNode(
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

    def intersect(self, X, min_area=0.0):
        """
        Returns, among the indexed bboxes, the ones intersecting with more than `min_area`
        with the given bbox. The search is depth-first and is of log complexity.

        Parameters
        ----------
        X : np.array
            1-dimensional numpy array of the box to find overlaps with
        min_area : float, optional
            Minimum area to consider overlap significant, by default 0.0

        Returns
        -------
        Tuple[np.array]
            Indices of boxes overlapping with X, and area (in the same order) of the overlaps
        """
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


node_type.define(RNode.class_type.instance_type)


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
def intersect(
    node: RNode,
    X: np.array,
    indices_buffer: List,
    intersections_buffer: List,
    inter_UB: float = 1.0,
    is_root: bool = True,
    min_area: float = 0.0,
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
        Query box (one box).
    indices_buffer : list
        List of currently-gathered neighbors. Stores in-place the neighbor indices along the
        search process
    intersection_buffer : list
        List of currently-gathered neighbor intersection with the query box.
        Since the redundancy criterion is intersection over union, I store it here to avoid
        recomputing it later.
    inter_UB : float, optional
        Intersection upper bound: this is the intersection of X with the current node's bbox. By
        definition, this is the highest intersection a box contained in this node can get with X.
    is_root : bool, optional
        Whether the currently visited node is root, by default True
    """
    if is_root:
        # If first call, no lower bound distance has already been computed
        inter_UB = intersection(X, node.bbox)

    # By definition, each box contained inside this node has an intersection with the current bbox
    # of less than the intersection with the node's englobing bounding box.
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
class RTree:
    """
    Main object for the R-Tree class: used to find in a log time-complexity
    the boxes overlapping with a given box.

    ```
    tree = RTree(existing_boxes)
    # Returns indices of relevant boxes and overlap area
    idxs, overlaps = tree.intersect(new_box)
    ```

    Parameters
    ----------
    X : np.array
        2-dimensional float64 numpy array of boxes to index
    leaf_size: int
        Leaf size of the tree. Region size at which the space stops being sub-divised.
    """

    def __init__(self, data: np.array, leaf_size: int = 16):
        self.data = data
        axis = max_spread_axis(self.data)
        self._root = RNode(data, leaf_size, axis, None)
        self._root.build()

    def intersect(self, X: np.array, min_area: float = 0.0):
        """
        Returns, among the indexed bboxes, the ones intersecting with more than `min_area`
        with the given bbox. The search is depth-first and is of log complexity.

        Parameters
        ----------
        X : np.array
            1-dimensional float64 numpy array of the box to find overlaps with
        min_area : float, optional
            Minimum area to consider overlap significant, by default 0.0

        Returns
        -------
        Tuple[np.array]
            Indices of boxes overlapping with X, and area (in the same order) of the overlaps
        """

        return self._root.intersect(X, min_area)
