//! Module implementing the BoundingBox object holding
//! box coordinates and small manipulation routines
//! (intersection area, union, encapsulating box, etc)
use core::fmt;
use std::fmt::{Debug, Display};

use crate::math::compute_area;
use ndarray::{ArrayView1, array};
use num_traits::Float;

/// Base object representing a bounding box.
///
/// Coordinates are in Pascal VOC format. Centroid is computed right away
/// as it is quite likely used for every single BoundingBox created along
/// the large scale NMS process.
///
/// # Example
/// ```
/// use ndarray::array;
/// use lsnms_rs::bbox::BoundingBox;
///
/// let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
/// let bbox2 = BoundingBox::new(array![1., 1., 3., 3.].view()).unwrap();
///
/// // Compute intersection area
/// let inter_area: f64 = bbox1.intersect(&bbox2).unwrap();
/// assert!(inter_area == 1.);
///
/// // Compute union of those two: the bbox encapsulating those two
/// let union_bbox: BoundingBox<f64> = bbox1.union(&bbox2).unwrap();
/// assert!((union_bbox.x0==0.) && (union_bbox.y0==0.) && (union_bbox.x1==3.) && (union_bbox.y1==3.));
///
/// // Compute IoU
/// let iou: f64 = bbox1.iou(&bbox2).unwrap();
/// assert!(iou == 1./7.);
///
/// // Compute l1 distance between two BoundingBoxes
/// let bbox3 = BoundingBox::new(array![4., 4., 6., 6.].view()).unwrap();
/// assert_eq!(bbox3.l1_distance_to(&bbox1).unwrap(), 2.+2.);
/// assert_eq!(bbox3.l1_distance_to(&bbox2).unwrap(), 1.+1.);
/// ```
#[derive(Copy, Clone, Debug)]
pub struct BoundingBox<D: Float> {
    pub x0: D,
    pub y0: D,
    pub x1: D,
    pub y1: D,
    pub area: f64,
    pub centroid: (D, D),
}

impl<D: Float + Display> fmt::Display for BoundingBox<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BoundingBox[{}, {}, {}, {})", self.x0, self.y0, self.x1, self.y1)?;
        Ok(())
    }
}

impl<D: Float + Display> BoundingBox<D> {
    /// Create a new BoundingBox from a viewed ndarray.
    ///
    /// The reason the ndarray is a view is because BoundingBoxes
    /// are created from view of Python's numpy's arrays accessed through pyo3.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use lsnms_rs::bbox::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
    /// ```
    pub fn new(coords: ArrayView1<D>) -> Result<Self, String> {
        let two = D::from(2.).unwrap();
        let bbox = Self {
            x0: coords[0],
            y0: coords[1],
            x1: coords[2],
            y1: coords[3],
            area: compute_area(coords)?,
            centroid: ((coords[0] + coords[2]) / two, (coords[1] + coords[3]) / two),
        };
        Ok(bbox)
    }

    pub fn from_scalars(x0: D, y0: D, x1: D, y1: D) -> Result<Self, String> {
        let area = ((x1 - x0) * (y1 - y0)).to_f64().unwrap_or(0.);
        let two = D::from(2.).unwrap();
        let bbox = Self {
            x0: x0,
            y0: y0,
            x1: x1,
            y1: y1,
            area: area,
            centroid: ((x0 + x1) / two, (y0 + y1) / two),
        };
        Ok(bbox)
    }

    /// Compute the intersection area with another bounding box.
    ///
    /// Returns the area of the overlapping region between two bounding boxes.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use lsnms_rs::bbox::BoundingBox;
    ///
    /// let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
    /// let bbox2 = BoundingBox::new(array![1., 1., 3., 3.].view()).unwrap();
    /// let inter_area: f64 = bbox1.intersect(&bbox2).unwrap();
    /// assert!(inter_area == 1.);
    /// ```
    pub fn intersect(&self, other_bbox: &BoundingBox<D>) -> Result<f64, String> {
        let x_a = self.x0.max(other_bbox.x0);
        let x_b = self.x1.min(other_bbox.x1);
        // Default RNode's bounding box will be (+inf, +inf, -inf, -inf) need to return 0. intersection with everything
        let dx = D::max(x_b - x_a, D::zero());
        if dx <= D::zero() {
            return Ok(0.0);
        }

        let y_a = self.y0.max(other_bbox.y0);
        let y_b = self.y1.min(other_bbox.y1);
        let dy = D::max(y_b - y_a, D::zero());
        if dy <= D::zero() {
            return Ok(0.0);
        }

        let inter_area = (dx * dy).to_f64().unwrap_or_else(|| 0.);
        return Ok(inter_area);
    }

    /// Compute the union bounding box that encapsulates both bounding boxes.
    ///
    /// Returns a new BoundingBox that encompasses both the current and the other bounding box.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use lsnms_rs::bbox::BoundingBox;
    ///
    /// let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
    /// let bbox2 = BoundingBox::new(array![1., 1., 3., 3.].view()).unwrap();
    /// let union_bbox: BoundingBox<f64> = bbox1.union(&bbox2).unwrap();
    /// assert!((union_bbox.x0==0.) && (union_bbox.y0==0.) && (union_bbox.x1==3.) && (union_bbox.y1==3.));
    /// ```
    pub fn union(&self, other: &BoundingBox<D>) -> Result<BoundingBox<D>, String> {
        let union_coords = array![
            self.x0.min(other.x0),
            self.y0.min(other.y0),
            self.x1.max(other.x1),
            self.y1.max(other.y1),
        ];
        BoundingBox::new(union_coords.view())
        // BoundingBox::from_scalars(self.x0.min(other.x0), self.y0.min(other.y0), self.x1.max(other.x1), self.y1.max(other.y1))
    }

    /// Expand a bbox in-place with another bbox.
    ///
    /// This has no clear benefit compared to allocating a new BoundingBox with .union,
    /// except maybe avoiding re-allocation on the heap.
    ///
    /// # Example
    /// ```
    /// use lsnms_rs::bbox::BoundingBox;
    /// let mut bbox1 = BoundingBox::from_scalars(0., 0., 1., 1.).unwrap();
    /// let bbox2 = BoundingBox::from_scalars(1., 1., 2., 2.).unwrap();
    /// bbox1.expand(&bbox2).unwrap();
    /// assert!((bbox1.x1==2.) && (bbox1.y1==2.) && (bbox1.area==4.));
    /// ```
    pub fn expand(&mut self, other: &BoundingBox<D>) -> Result<(), String> {
        self.x0 = self.x0.min(other.x0);
        self.y0 = self.y0.min(other.y0);
        self.x1 = self.x1.max(other.x1);
        self.y1 = self.y1.max(other.y1);
        self.area = ((self.x1 - self.x0) * (self.y1 - self.y0)).to_f64().unwrap_or(0.);
        let two = D::from(2.).unwrap();
        self.centroid = ((self.x0 + self.x1) / two, (self.y0 + self.y1) / two);
        Ok(())
    }

    /// Compute the Intersection over Union (IoU) with another bounding box.
    ///
    /// Returns the ratio of intersection area to union area between two bounding boxes.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use lsnms_rs::bbox::BoundingBox;
    ///
    /// let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
    /// let bbox2 = BoundingBox::new(array![1., 1., 3., 3.].view()).unwrap();
    /// let iou: f64 = bbox1.iou(&bbox2).unwrap();
    /// assert!(iou == 1./7.);
    /// ```
    pub fn iou(&self, other: &BoundingBox<D>) -> Result<f64, String> {
        let inter_area = self.intersect(other)?;
        let union_area = self.area + other.area - inter_area;
        Ok(inter_area / union_area)
    }

    /// Compute the L1 distance to another bounding box.
    ///
    /// Returns the sum of horizontal and vertical distances between two bounding boxes.
    /// If boxes overlap, the distance is 0.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use lsnms_rs::bbox::BoundingBox;
    ///
    /// let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
    /// let bbox3 = BoundingBox::new(array![4., 4., 6., 6.].view()).unwrap();
    /// assert_eq!(bbox3.l1_distance_to(&bbox1).unwrap(), 2.+2.);
    /// ```
    pub fn l1_distance_to(&self, other: &BoundingBox<D>) -> Result<f64, String> {
        let dx = D::max(self.x0 - other.x1, other.x0 - self.x1);
        let dy = D::max(self.y0 - other.y1, other.y0 - self.y1);
        let dx = dx.max(D::zero());
        let dy = dy.max(D::zero());

        let l1_dist = (dx + dy).to_f64().unwrap_or(0.);
        return Ok(l1_dist);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let bbox = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
        assert_eq!(bbox.x0, 0.);
        assert_eq!(bbox.y0, 0.);
        assert_eq!(bbox.x1, 2.);
        assert_eq!(bbox.y1, 2.);
        assert_eq!(bbox.area, 4.);
        assert_eq!(bbox.centroid, (1., 1.));
    }

    #[test]
    fn test_intersect() {
        let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
        let bbox2 = BoundingBox::new(array![1., 1., 3., 3.].view()).unwrap();
        let inter_area: f64 = bbox1.intersect(&bbox2).unwrap();
        assert!(inter_area == 1.);
    }

    #[test]
    fn test_union() {
        let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
        let bbox2 = BoundingBox::new(array![1., 1., 3., 3.].view()).unwrap();
        let union_bbox: BoundingBox<f64> = bbox1.union(&bbox2).unwrap();
        assert!((union_bbox.x0 == 0.) && (union_bbox.y0 == 0.) && (union_bbox.x1 == 3.) && (union_bbox.y1 == 3.));
    }

    #[test]
    fn test_iou() {
        let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
        let bbox2 = BoundingBox::new(array![1., 1., 3., 3.].view()).unwrap();
        let iou: f64 = bbox1.iou(&bbox2).unwrap();
        assert!(iou == 1. / 7.);
    }

    #[test]
    fn test_l1_distance_to() {
        let bbox1 = BoundingBox::new(array![0., 0., 2., 2.].view()).unwrap();
        let bbox3 = BoundingBox::new(array![4., 4., 6., 6.].view()).unwrap();
        assert_eq!(bbox3.l1_distance_to(&bbox1).unwrap(), 2. + 2.);

        let bbox2 = BoundingBox::new(array![1., 1., 3., 3.].view()).unwrap();
        assert_eq!(bbox3.l1_distance_to(&bbox2).unwrap(), 1. + 1.);
    }
}
