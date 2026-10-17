//! Module implementing small maths utilities.
use crate::bbox::BoundingBox;
use ndarray::{ArrayView1, array};
use num_traits::Float;
use std::fmt::Display;
use std::format;

/// Given a set of boxes, compute the coordinates of the smallest box containing them all.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use lsnms_rs::math::box_englobing_boxes;
/// use lsnms_rs::bbox::BoundingBox;
///
/// // Boxes are in (x0, y0, x1, y1) with x1>=x0, y1>=y0 format.
/// let boxes = vec![
///     BoundingBox::from_scalars(0.0, 0.0, 2.0, 2.0).unwrap(),
///     BoundingBox::from_scalars(1.0, 1.0, 3.0, 3.0).unwrap(),
/// ];
///
/// let enclosing = box_englobing_boxes(&boxes).unwrap();
///
/// assert!((enclosing.x0 ==0.) && (enclosing.y0==0.) && (enclosing.x1==3.) && (enclosing.y1==3.));
/// ```
pub fn box_englobing_boxes<D: Float + Display>(bboxes: &Vec<BoundingBox<D>>) -> Result<BoundingBox<D>, String> {
    let bounds = bboxes.iter().fold(
        (D::infinity(), D::infinity(), D::neg_infinity(), D::neg_infinity()),
        |acc, elem| {
            (
                acc.0.min(elem.x0),
                acc.1.min(elem.y0),
                acc.2.max(elem.x1),
                acc.3.max(elem.y1),
            )
        },
    );
    let bbox = BoundingBox::new(array![bounds.0, bounds.1, bounds.2, bounds.3].view())?;
    Ok(bbox)
}

/// Argsplit a list of values in two at the median and return indices of left and right chunk.
///
/// First chunk contains all values < than the median, second chunk contains the median and all values bigger than it.
/// Values in each chunk are not guaranteed to be sorted. This algorithm is not stable either.
///
/// # Example
/// ```
/// use lsnms_rs::math::median_argsplit;
///
/// let values = vec![3., 7., 9., 4., 1.];
/// let (left_chunk_indices, right_chunk_indices) = median_argsplit(&values).unwrap();
///
/// let left_chunk: Vec<f64> = left_chunk_indices.iter().map(|&i| values[i]).collect(); // [3., 1.] or [1., 3.]
/// let right_chunk: Vec<f64> = right_chunk_indices.iter().map(|&i| values[i]).collect(); // [4., 7., 9.] or [4., 9., 7.]
/// ```
pub fn median_argsplit<'py, D: Float>(values: &Vec<D>) -> Result<(Vec<usize>, Vec<usize>), String> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    let k = values.len() >> 1;

    // partial sort indices
    indices.select_nth_unstable_by(k, |&i, &j| {
        values[i].partial_cmp(&values[j]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Split at k to include pivot in right partition
    let (left, right) = indices.split_at(k);
    Ok((left.to_vec(), right.to_vec()))
}

/// Compute the area of a bounding box.
///
/// Any NaN in coordinate will lead to an area of 0.
///
/// # Example
/// ```
/// use ndarray::array;
/// use lsnms_rs::math::compute_area;
///
/// let bbox = array![0., 0., 2., 2.];
///
/// let bbox_area: f64 = compute_area(bbox.view()).unwrap();
/// assert!(bbox_area == 4.);
/// ```
pub fn compute_area<'py, D: Float>(bbox: ArrayView1<'py, D>) -> Result<f64, String> {
    // TODO: check box formatting ?
    let width = bbox[2] - bbox[0];
    let height = bbox[3] - bbox[1];
    let area = (width * height).to_f64().unwrap_or(0.);
    Ok(area)
}

/// Argsort (unstable) values.
///
/// Return a Vec of indices generating ordered values.
///
/// # Example
/// ```
/// use ndarray::array;
/// use lsnms_rs::math::argsort;
///
/// let values = array![2., 1., 4., 5., 3., 7.];
/// let asc_indices = argsort(values.view(), true).unwrap();
/// let desc_indices = argsort(values.view(), false).unwrap();
///
/// // Check that ordering is correct:
/// let is_sorted_asc = asc_indices.windows(2).all(|w| values[w[0]] <= values[w[1]]);
/// let is_sorted_desc = desc_indices.windows(2).all(|w| values[w[1]] <= values[w[0]]);
/// assert!(is_sorted_asc);
/// assert!(is_sorted_desc);
/// ```
pub fn argsort<'py, D: Float>(x: ArrayView1<'py, D>, ascending: bool) -> Result<Vec<usize>, String> {
    let n = x.dim();
    let mut indices: Vec<usize> = (0..n).collect();
    if ascending {
        indices.sort_unstable_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        indices.sort_unstable_by(|&i, &j| x[j].partial_cmp(&x[i]).unwrap_or(std::cmp::Ordering::Equal));
    }
    Ok(indices)
}

/// Compute the axis of maximal spread.
///
/// #Example
/// ```
/// use lsnms_rs::bbox::BoundingBox;
/// use lsnms_rs::math::max_spread_axis;
///
/// let bboxes = vec![
///     BoundingBox::from_scalars(0., 0., 1., 1.).unwrap(),
///     BoundingBox::from_scalars(2., 0., 3., 1.).unwrap()
/// ];
/// let axis = max_spread_axis(&bboxes).unwrap();
/// assert!(axis==0);
/// ```
pub fn max_spread_axis<D: Float>(bboxes: &Vec<BoundingBox<D>>) -> Result<usize, String> {
    if bboxes.len() < 2 {
        return Err(format!("Need at least two bboxes, received {}", bboxes.len()));
    }
    let bounds = bboxes.iter().fold(
        (D::infinity(), D::infinity(), D::neg_infinity(), D::neg_infinity()),
        |acc, elem| {
            (
                acc.0.min(elem.x0),
                acc.1.min(elem.y0),
                acc.2.max(elem.x1),
                acc.3.max(elem.y1),
            )
        },
    );
    let horizontal_spread = bounds.2 - bounds.0;
    let vertical_spread = bounds.3 - bounds.1;
    if horizontal_spread > vertical_spread {
        return Ok(0);
    } else {
        return Ok(1);
    }
}

// pub fn argmin_gini(values: &Vec<D>, idxs: )
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, array};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use std::{assert_eq, format};

    #[test]
    fn test_box_englobing_boxes() -> Result<(), String> {
        let boxes = vec![
            BoundingBox::from_scalars(0.0, 0.0, 2.0, 2.0).unwrap(),
            BoundingBox::from_scalars(1.0, 1.0, 3.0, 3.0).unwrap(),
        ];

        let enclosing = box_englobing_boxes(&boxes).unwrap();

        let mut result = enclosing.x0 == 0.;
        result &= enclosing.y0 == 0.;
        result &= enclosing.x1 == 3.;
        result &= enclosing.y1 == 3.;

        if result {
            Ok(())
        } else {
            Err(format!("Expected [0., 0., 3., 3.], received {:?}", enclosing))
        }
    }

    #[test]
    fn test_median_argsplit() -> Result<(), String> {
        let values = Array::random((11,), Uniform::new(0., 10.).unwrap()).to_vec();

        // Compute median argsplit
        let (left_idxs, right_idxs) = median_argsplit(&values)?;

        // Sort the values in place and take the middle one
        let mut clone = values.clone();
        clone.sort_unstable_by(|&a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = clone[5];

        // test that left part are all smaller
        let mut is_ok = left_idxs.iter().all(|&i| values[i] <= median);
        // test that right part are all bigger
        is_ok &= right_idxs.iter().all(|&i| values[i] >= median);
        // Test that pivot is the first value of the right part
        is_ok &= values[right_idxs[0]] == median;

        if is_ok {
            Ok(())
        } else {
            Err(format!("Expected pivot around median (central) element"))
        }
    }

    #[test]
    fn test_compute_area() -> Result<(), String> {
        let bbox = array![0., 1., 2., 3.];
        let area = compute_area(bbox.view())?;

        if area == 4. {
            Ok(())
        } else {
            Err(format!("Expected area to be 4., received {}", area))
        }
    }

    #[test]
    fn test_argsort() -> Result<(), String> {
        let values = Array::random((10,), Uniform::new(0., 10.).unwrap());
        let ascending_indices = argsort(values.view(), true)?;
        let descending_indices = argsort(values.view(), false)?;

        // Test that is in order
        let is_sorted_ascending =
            (1..ascending_indices.len()).all(|i| values[ascending_indices[i]] >= values[ascending_indices[i - 1]]);
        let is_sorted_descending =
            (1..descending_indices.len()).all(|i| values[descending_indices[i]] <= values[descending_indices[i - 1]]);

        // Test that both are sorted
        if !is_sorted_ascending {
            return Err(format!(
                "argsort indices do not generate sorted values: {:?}",
                ascending_indices.iter().map(|&i| values[i]).collect::<Vec<f64>>()
            ));
        }
        if !is_sorted_descending {
            return Err(format!(
                "argsort indices do not generate reverse sorted values: {:?}",
                descending_indices.iter().map(|&i| values[i]).collect::<Vec<f64>>()
            ));
        }

        Ok(())
    }

    #[test]
    fn test_max_spread_axis() -> Result<(), String> {
        let bboxes = vec![
            BoundingBox::from_scalars(0., 0., 1., 1.)?,
            BoundingBox::from_scalars(2., 0., 3., 1.)?,
        ];
        let axis = max_spread_axis(&bboxes)?;
        assert_eq!(axis, 0, "Max spread axis should be 0, found {}", axis);
        Ok(())
    }
}
