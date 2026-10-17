//! Module implementing the rectangular tree and its methods.
//!
//! LSNMS uses rectangular tree to iteratively index bounding boxes
//! and discard redundant boxes, turning the non maximum suppresion
//! from O(n**2) into O(nlog(n)).
use crate::bbox::BoundingBox;
use crate::math::{box_englobing_boxes, max_spread_axis, median_argsplit};
use num_traits::Float;
use std::{format, println};

/// Rectangular tree (R-Tree) node for indexing and querying bounding boxes.
///
/// An `RNode` can either be a leaf holding up to `leaf_size` bounding boxes,
/// or an internal node splitting its space along an alternating axis (0 for x, 1 for y)
/// into left and right subtrees.
///
/// # Example
/// ```
/// use ndarray::array;
/// use lsnms_rs::bbox::BoundingBox;
/// use lsnms_rs::rtree::RNode;
///
/// let mut tree = RNode::new(None, 16, None).unwrap();
/// let bbox = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
/// tree.insert(&bbox).unwrap();
/// assert_eq!(tree.n_bboxes, 1);
/// ```
pub struct RNode<D: Float> {
    pub bboxes: Vec<BoundingBox<D>>,
    pub bounds: BoundingBox<D>,
    pub axis: usize,
    pub leaf_size: usize,
    pub left: Option<Box<RNode<D>>>,
    pub right: Option<Box<RNode<D>>>,
    pub smallest_bbox_area: f64,
    pub n_bboxes: usize,
    // pub split_axis: Option<usize>,
    // pub split_value: Option<D>
}
use std::fmt::Display;

impl<D: Float + Display> RNode<D> {
    /// Create a new R-Tree node.
    ///
    /// Accepts an optional vector of bounding boxes, a split axis (`0` for x, `1` for y),
    /// and the maximum number of elements per leaf (`leaf_size`).
    ///
    /// # Example
    /// ```
    /// use lsnms_rs::rtree::RNode;
    ///
    /// let node: RNode<f64> = RNode::new(None, 16, None).unwrap();
    /// assert_eq!(node.n_bboxes, 0);
    /// assert_eq!(node.axis, 0);
    /// ```
    pub fn new(bboxes: Option<Vec<BoundingBox<D>>>, leaf_size: usize, axis: Option<usize>) -> Result<Self, String> {
        let bboxes = bboxes.unwrap_or(Vec::with_capacity(leaf_size + 1));
        // if axis.is_some() {
        //     if axis.unwrap() > 1 {
        //         return Err(format!("Axis must be 0 or 1, received {}", axis.unwrap()));
        //     };
        // }
        let rnode = Self {
            bounds: box_englobing_boxes(&bboxes)?,
            smallest_bbox_area: bboxes
                .iter()
                .map(|b| b.area)
                .min_by(|a, b| a.total_cmp(b))
                .unwrap_or(f64::INFINITY),
            n_bboxes: bboxes.len(),
            bboxes: bboxes,
            leaf_size: leaf_size,
            left: None,
            right: None,
            axis: axis.unwrap_or(0),
            // split_axis: split_axis,
            // split_value: split_value
        };
        Ok(rnode)
    }

    /// Check whether a candidate bounding box is redundant with any already indexed box.
    ///
    /// Recursively traverses the tree, pruning subtrees whose bounding box cannot produce
    /// an IoU greater than or equal to `iou_threshold` with the candidate box.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use lsnms_rs::bbox::BoundingBox;
    /// use lsnms_rs::rtree::RNode;
    ///
    /// let mut tree = RNode::new(None, 16, None).unwrap();
    /// let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
    /// tree.insert(&bbox1).unwrap();
    ///
    /// let bbox2 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
    /// assert!(tree.is_redundant(&bbox2, 0.5).unwrap());
    ///
    /// let bbox3 = BoundingBox::new(array![10., 10., 12., 12.].view()).unwrap();
    /// assert!(!tree.is_redundant(&bbox3, 0.5).unwrap());
    /// ```
    pub fn is_redundant(&self, bbox: &BoundingBox<D>, iou_threshold: f64) -> Result<bool, String> {
        if self.is_leaf() {
            let redundant = self
                .bboxes
                .iter()
                .any(|existing_bbox| existing_bbox.iou(bbox).unwrap_or(0.) >= iou_threshold);
            // let redundant = (0..self.bboxes.len()).into_par_iter().any(|i| self.bboxes[i].iou(bbox).unwrap_or(0.)>=iou_threshold);
            return Ok(redundant);
        } else {
            // Note: if a node is empty, bounds.intersect will always return 0.
            let left_inter_ub = self.left.as_ref().unwrap().bounds.intersect(bbox)?;
            let right_inter_ub = self.right.as_ref().unwrap().bounds.intersect(bbox)?;
            let left_inter_threshold =
                iou_threshold * (bbox.area + self.left.as_ref().unwrap().smallest_bbox_area) / (1. + iou_threshold);
            let right_inter_threshold =
                iou_threshold * (bbox.area + self.right.as_ref().unwrap().smallest_bbox_area) / (1. + iou_threshold);

            if left_inter_ub >= right_inter_ub {
                // Go left first, then right, if intersection is non null
                if left_inter_ub > left_inter_threshold {
                    if self.left.as_ref().unwrap().is_redundant(bbox, iou_threshold)? {
                        return Ok(true);
                    }
                }
                if right_inter_ub > right_inter_threshold {
                    if self.right.as_ref().unwrap().is_redundant(bbox, iou_threshold)? {
                        return Ok(true);
                    }
                }
            } else {
                if right_inter_ub > right_inter_threshold {
                    if self.right.as_ref().unwrap().is_redundant(bbox, iou_threshold)? {
                        return Ok(true);
                    }
                }
                if left_inter_ub > left_inter_threshold {
                    if self.left.as_ref().unwrap().is_redundant(bbox, iou_threshold)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
    }

    /// Insert a bounding box into the tree.
    ///
    /// If inserting into a leaf that has reached `leaf_size`, the leaf is split into
    /// two children along the node's current axis using the median centroid coordinate.
    /// When inserting into an internal node, the branch with the smaller L1 distance
    /// to the candidate box is chosen.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use lsnms_rs::bbox::BoundingBox;
    /// use lsnms_rs::rtree::RNode;
    ///
    /// let mut tree = RNode::new(None, 2, None).unwrap();
    /// let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
    /// let bbox2 = BoundingBox::new(array![3., 3., 5., 5.].view()).unwrap();
    /// tree.insert(&bbox1).unwrap();
    /// tree.insert(&bbox2).unwrap();
    /// assert_eq!(tree.n_bboxes, 2);
    /// ```
    pub fn insert(&mut self, bbox: &BoundingBox<D>) -> Result<(), String> {
        if self.is_leaf() {
            if self.is_full()? {
                // Push the bbox, split in two, assign children, clear .bboxes
                self.bboxes.push(*bbox);
                // let axis = (self.axis + 1) % 2;
                let axis = max_spread_axis(&self.bboxes)?;
                let positions: Vec<D> = match axis {
                    0 => Ok(self.bboxes.iter().map(|bbox| bbox.centroid.0).collect()),
                    1 => Ok(self.bboxes.iter().map(|bbox| bbox.centroid.1).collect()),
                    _ => Err(format!("Axis should be 0 or 1, received {}", axis)),
                }?;

                // Split in two and attach the children
                let (left_child_idxs, right_child_idxs) = median_argsplit(&positions)?;
                let left_bboxes: Vec<BoundingBox<D>> = left_child_idxs.iter().map(|&i| self.bboxes[i]).collect();
                let right_bboxes: Vec<BoundingBox<D>> = right_child_idxs.iter().map(|&i| self.bboxes[i]).collect();
                self.left = Some(Box::new(RNode::new(Some(left_bboxes), self.leaf_size, Some(axis))?));
                self.right = Some(Box::new(RNode::new(Some(right_bboxes), self.leaf_size, Some(axis))?));

                // Clear the bboxes from the current node to avoid data duplication across the tree
                self.bboxes.clear();
            } else {
                // If is a leaf and still has room, just push
                self.bboxes.push(*bbox);
            }
        } else {
            // Use smallest l1 distance to the children to decide where to go first
            // Note: this choice drives how efficient the pruning is. This is a major
            // performance knob to act on.
            let l1_to_left = self.left.as_ref().unwrap().bounds.l1_distance_to(&bbox)?;
            let l1_to_right = self.right.as_ref().unwrap().bounds.l1_distance_to(&bbox)?;
            if l1_to_left < l1_to_right {
                self.left.as_mut().unwrap().insert(bbox)?;
            } else {
                self.right.as_mut().unwrap().insert(bbox)?;
            }
        }
        // Regardless of leaf or not, this node remains referenced as-is by its parent (splitting
        // only attaches children to it, it does not get replaced), so both bounds and
        // smallest_bbox_area have to be updated **for all the parents on the path**. is_redundant
        // reads a child's cached smallest_bbox_area directly (not a recursive min), so letting it
        // go stale here causes overly aggressive pruning further up the tree.
        // self.bounds = self.bounds.union(bbox)?;
        self.bounds.expand(&bbox)?;
        self.smallest_bbox_area = self.smallest_bbox_area.min(bbox.area);
        self.n_bboxes += 1;
        Ok(())
    }

    /// Return the number of bounding boxes held in this node directly.
    ///
    /// For a leaf node, this is the count of stored boxes. For an internal node,
    /// stored boxes have been moved to children, so this returns 0.
    ///
    /// # Example
    /// ```
    /// use lsnms_rs::rtree::RNode;
    ///
    /// let node: RNode<f64> = RNode::new(None, 16, None).unwrap();
    /// assert_eq!(node.n_elems().unwrap(), 0);
    /// ```
    pub fn n_elems(&self) -> Result<usize, String> {
        Ok(self.bboxes.len())
    }

    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    fn is_full(&self) -> Result<bool, String> {
        if !self.is_leaf() {
            return Err(format!(" "));
        } else {
            return Ok(self.n_elems()? == self.leaf_size);
        }
    }

    /// Return the depth (or height) of the tree.
    ///
    /// If the tree is unbalanced, this is the maximum height.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use lsnms_rs::bbox::BoundingBox;
    /// use lsnms_rs::rtree::RNode;
    ///
    /// let mut tree = RNode::new(None, 2, None).unwrap();
    /// assert_eq!(tree.depth().unwrap(), 1);
    ///
    /// let bbox1 = BoundingBox::from_scalars(0., 0., 2., 2.).unwrap();
    /// let bbox2 = BoundingBox::from_scalars(1., 1., 3., 3.).unwrap();
    /// let bbox3 = BoundingBox::from_scalars(2., 2., 4., 4.).unwrap();
    /// tree.insert(&bbox1).unwrap();
    /// tree.insert(&bbox2).unwrap();
    /// tree.insert(&bbox3).unwrap();
    /// assert_eq!(tree.depth().unwrap(), 2);
    /// ```
    pub fn depth(&self) -> Result<usize, String> {
        if self.left.is_none() && self.right.is_none() {
            return Ok(1);
        }
        let left_depth = self.left.as_ref().unwrap().depth()?;
        let right_depth = self.right.as_ref().unwrap().depth()?;
        Ok(usize::max(left_depth, right_depth) + 1)
    }

    /// Print the size of the leaves across the tree to stdout.
    ///
    /// Recursively prints each leaf's element count.
    ///
    /// # Example
    /// ```
    /// use lsnms_rs::rtree::RNode;
    ///
    /// let tree: RNode<f64> = RNode::new(None, 16, None).unwrap();
    /// tree.print_size().unwrap();
    /// ```
    pub fn print_size(&self) -> Result<(), String> {
        if self.is_leaf() {
            println!("Leaf indexing {} bboxes.", self.n_elems()?);
        } else {
            self.left.as_ref().unwrap().print_size()?;
            self.right.as_ref().unwrap().print_size()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Axis, array, concatenate};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use num_traits::ToPrimitive;

    #[test]
    fn test_new() {
        let node: RNode<f64> = RNode::new(None, 16, None).unwrap();
        assert_eq!(node.n_bboxes, 0);
        // assert_eq!(node.axis, 0);
    }

    #[test]
    fn test_is_redundant() {
        let mut tree = RNode::new(None, 16, None).unwrap();
        let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
        tree.insert(&bbox1).unwrap();

        let bbox2 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
        assert!(tree.is_redundant(&bbox2, 0.5).unwrap());

        let bbox3 = BoundingBox::new(array![10., 10., 12., 12.].view()).unwrap();
        assert!(!tree.is_redundant(&bbox3, 0.5).unwrap());
    }

    #[test]
    fn test_insert() {
        let mut tree = RNode::new(None, 2, None).unwrap();
        let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
        let bbox2 = BoundingBox::new(array![3., 3., 5., 5.].view()).unwrap();
        tree.insert(&bbox1).unwrap();
        tree.insert(&bbox2).unwrap();
        assert_eq!(tree.n_bboxes, 2);
    }

    #[test]
    fn test_n_elems() {
        let node: RNode<f64> = RNode::new(None, 16, None).unwrap();
        assert_eq!(node.n_elems().unwrap(), 0);
    }

    #[test]
    fn test_depth() {
        let mut tree = RNode::new(None, 2, None).unwrap();
        assert_eq!(tree.depth().unwrap(), 1);

        let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
        let bbox2 = BoundingBox::new(array![1., 1., 3., 3.].view()).unwrap();
        let bbox3 = BoundingBox::new(array![2., 2., 4., 4.].view()).unwrap();
        tree.insert(&bbox1).unwrap();
        tree.insert(&bbox2).unwrap();
        tree.insert(&bbox3).unwrap();
        assert_eq!(tree.depth().unwrap(), 2);
    }

    #[test]
    fn test_print_size() {
        let tree: RNode<f64> = RNode::new(None, 16, None).unwrap();
        assert!(tree.print_size().is_ok());
    }

    /// Rust equivalent of the `crowded` bbox-generation fixture in `conftest.py`.
    ///
    /// Builds a `grid_size x grid_size` grid of anchor points spaced by 10 (starting at 5),
    /// repeats each anchor `n_repeats` times, jitters the bottom-left corner with small noise,
    /// and adds a random width/height to obtain the top-right corner. Also returns a random
    /// score per box.
    fn crowded_bboxes(n_objects: usize, n_repeats: usize) -> Result<(Array2<f64>, Array1<f64>), String> {
        // Example: 10 objects -> grid of 3x3 spaced by 10, starting at 5.
        let grid_size = n_objects.to_f64().unwrap().sqrt().floor().to_usize().unwrap();
        let n_total = grid_size.pow(2) * n_repeats;
        let mut bottomleft: Array2<f64> = Array2::zeros((n_total, 2));
        let mut k = 0;
        for i in 0..grid_size {
            for j in 0..grid_size {
                for _ in 0..n_repeats {
                    bottomleft[[k, 0]] = (i * 10) as f64 + 5.;
                    bottomleft[[k, 1]] = (j * 10) as f64 + 5.;
                    k += 1;
                }
            }
        }

        // Now create noise and jitter the bottom-left corners.
        let noise: Array2<f64> = Array2::random((n_total, 2), Uniform::new(0.1, 0.5).unwrap());
        bottomleft.scaled_add(1., &noise);

        // Now create width and height for the top-right corners.
        let wh: Array2<f64> = Array2::random((n_total, 2), Uniform::new(1., 2.).unwrap());
        let topright = &bottomleft + &wh;

        let bboxes = concatenate(Axis(1), &[bottomleft.view(), topright.view()]).unwrap();

        // Now create scores
        let scores: Array1<f64> = Array1::random((n_total,), Uniform::new(0.9, 1.).unwrap());
        Ok((bboxes, scores))
    }

    #[test]
    fn test_optimal_depth() {
        let mut tree: RNode<f64> = RNode::new(None, 4, None).unwrap();
        let (bboxes, scores) = crowded_bboxes(16, 4).unwrap();
        // Insert all the bboxes
        for i in 0..bboxes.dim().0 {
            let bbox: BoundingBox<f64> = BoundingBox::new(bboxes.row(i)).unwrap();
            tree.insert(&bbox).unwrap();
        }
        // With 16 objects all repeated 4 times, the tree should have 16 leaves
        // all containing the 4 repeats, so a depth of 4.
        println!("Tree depth: {}", tree.depth().unwrap());
    }
}
