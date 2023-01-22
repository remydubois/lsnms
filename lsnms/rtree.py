from collections import OrderedDict

import numpy as np
from numba import boolean, deferred_type, float64, int64, optional, typed
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
specs["_built"] = boolean
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
        self._built = False

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

    def build(self):
        """
        Reccursively build the children.
        Jit methods can not be explicitely recursive:
        `self.build` can not call `self.build`, but it can call a function
        which calls itself, the workaround used here.
        """
        # Reccursively attach children to the parent
        # build(self)
        nodes = typed.List([self])
        while len(nodes):
            current = nodes.pop(-1)
            if not current.is_leaf:
                left, right = current.split()

                # Assign children
                current.left = left
                current.right = right

                # Append children to the list of nodes to split
                nodes.append(left)
                nodes.append(right)
                current._built = True

    def intersect(self, X, min_area=1.0):
        """
        Returns, among the indexed bboxes, the ones intersecting with more than `min_area`
        with the given bbox. The search is depth-first and is of log complexity.

        Parameters
        ----------
        X : np.array
            1-dimensional numpy array of the box to find overlaps with
        min_area : float, optional
            Minimum area to consider overlap significant, by default 1.0 (pixel)

        Returns
        -------
        Tuple[np.array]
            Indices of boxes overlapping with X, and area (in the same order) of the overlaps
        """
        if not self._built:
            raise ValueError("Tree needs to be built before being queried. Call RNode.build first.")

        # Initialize buffers to hold indices of intersects and corresponding areas
        indices_buffer = typed.List.empty_list(int64)
        intersections_buffer = typed.List.empty_list(float64)

        # This list will, by construction, be sorted by increasing area
        # of intersection with the box queried
        to_visit = typed.List([self])
        while len(to_visit):
            # Take the last node in list, which, by design, is the one having the highest
            # intersection with the box queried
            current_node = to_visit.pop(-1)

            # Compute the area of intersection upper bound
            # between this node and the bbox queried
            node_inter_UB = intersection(X, current_node.bbox)

            # If the upper bound is smaller than the minimal requested aread
            # by design all the bboxes contained in this node will
            # have an intersection too small
            if node_inter_UB < min_area:
                continue
            else:
                if not current_node.is_leaf:
                    left_UB = intersection(X, current_node.left.bbox)
                    right_UB = intersection(X, current_node.right.bbox)

                    # Order the children by increasing area of intersection
                    if left_UB > right_UB:
                        to_visit.append(current_node.right)
                        to_visit.append(current_node.left)
                    else:
                        to_visit.append(current_node.left)
                        to_visit.append(current_node.right)
                else:
                    # If it's leaf, simply review all the bboxes contained inside the node
                    for i, y in zip(current_node.indices, current_node.data):
                        inter = intersection(X, y)
                        if inter > min_area:

                            indices_buffer.append(i)
                            intersections_buffer.append(inter)

        return indices_buffer, intersections_buffer


node_type.define(RNode.class_type.instance_type)


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
