//! Module implementing the RTree-based non maximum suppression.
//!
//! LSNMS uses rectangular tree to iteratively index bounding boxes
//! and discard redundant boxes, turning the non maximum suppresion
//! from O(n**2) into O(nlog(n)).

use std::format;

use crate::bbox::BoundingBox;
use crate::math::argsort;
use crate::rtree::RNode;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use num_traits::Float;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{Bound, PyAny, PyResult, Python};
use std::fmt::Display;

/// Run RTree-accelerated Non Maximum Suppression (NMS).
/// See rtree_nms_rs docstring.
pub fn rtree_nms<'py, Db: Float + Display, Ds: Float + Display>(
    boxes: ArrayView2<'py, Db>,
    scores: ArrayView1<'py, Ds>,
    iou_threshold: f64,
    score_threshold: f64,
    rtree_leaf_size: usize,
) -> Result<Array1<usize>, String> {
    let n_boxes = boxes.dim().0;

    // Filter based on score
    let score_threshold_d: Ds = Ds::from(score_threshold).ok_or_else(|| {
        format!(
            "score_treshold should be integer or float number, received {}",
            score_threshold
        )
    })?;
    // This filters out bboxes with NaN as score (IEEE 754)
    let valid_boxes_indices: Vec<usize> = (0..n_boxes).filter(|&i| scores[i] >= score_threshold_d).collect();
    // let valid_boxes: Vec<BoundingBox<Db>> = valid_boxes_indices.iter().map(|&i| BoundingBox::new(boxes.row(i), Some(i))).collect()?;
    let valid_scores = scores.select(Axis(0), &valid_boxes_indices);

    // Order by decreasing order of score
    let order = argsort(valid_scores.view(), false)?;

    // Initialize the RTree
    let mut rtree = RNode::new(None, rtree_leaf_size, None)?;

    // Now go through boxes one by one
    let mut keep: Vec<usize> = Vec::new();
    for &idx in order.iter() {
        // Retrieve the bbox
        let candidate_idx = valid_boxes_indices[idx];
        let bbox_coo = boxes.row(candidate_idx);
        // Evict bboxes with NaN coordinates
        if bbox_coo.iter().any(|&x| Db::is_nan(x)) {
            eprintln!("{}—th bbox has a NaN coordinate and got evicted.", idx);
            continue;
        }
        let candidate = BoundingBox::new(bbox_coo)?;
        // First see if this bbox is redundant with any other already indexed
        // if rtree.is_redundant_parallel(&candidate, iou_threshold, parallel_depth)? {
        if rtree.is_redundant(&candidate, iou_threshold)? {
            continue;
        } else {
            // If not redundant, insert in the tree and add to the list of boxes to keep
            rtree.insert(&candidate)?;
            keep.push(candidate_idx);
        }
    }
    // println!("Indexed {} boxes in a tree of depth {}", rtree.n_bboxes, rtree.depth()?);
    // rtree.print_size()?;
    Ok(Array1::from_vec(keep))
}

macro_rules! run_rtree_nms {
    ($py:expr, $boxes:expr, $scores: expr, $iou_threshold:expr, $score_threshold:expr, $rtree_leaf_size:expr, $boxes_dtype:ty, $scores_dtype:ty) => {{
        // Here convert the PyAny into a Rust's numpy's PyArray
        let boxes_pyarray = $boxes.extract::<PyReadonlyArray2<'py, $boxes_dtype>>()?;
        // Now convert it into a Rust's ndarray
        let boxes = boxes_pyarray.as_array();
        // Same for scores
        let scores_pyarray = $scores.extract::<PyReadonlyArray1<'py, $scores_dtype>>()?;
        let scores = scores_pyarray.as_array();

        // Can detach python: python objects are not modified during the process
        $py.detach(|| rtree_nms(boxes, scores, $iou_threshold, $score_threshold, $rtree_leaf_size))
            .map_err(PyValueError::new_err)
    }};
}

/// Run RTree-accelerated Non Maximum Suppression (NMS) from the python client.
///
/// Instead of greedily comparing and discarding bboxes like a naive NMS,
/// this NMS uses a rectangular tree to index bboxes and run O(nlog(n)) comparison
/// instead of O(n**2).
///
/// This implementation is peculiarly useful when instances (predicted objects)
/// are numerous and spread apart. On natural images, naive NMS is likely to be faster.
///
/// Parameters
/// ----------
/// boxes: np.ndarray[float32|float64]  
///     The bounding boxes of shape (n_boxes, 4) in Pascal VOC format (x0, y0, x1, y1) with x1>=x0 and y1>=y0.  
/// scores: np.ndarray[float32|float64]  
///     The scores of the corresponding bounding boxes of shape (n_boxes,).  
/// iou_threshold: float64  
///     The intersection over union threshold from which two bounding boxes are considered redundant.  
/// score_threshold: float64, default=0.5  
///     The score from which bounding boxes should be discarded.  
/// rtree_leaf_size: unsigned integer, default=16  
///     Size of the leaves in the rectangular tree. This parameter will not impact the correctness of the result,
///     only the speed of execution.  
///
/// Returns  
/// -------
/// np.ndarray  
///     The indices of boxes to keep, sorted by decreasing order of score.  
///
/// # Example usage
/// ```python
/// import numpy as np
/// from lsnms.lsnms_rs import rtree_nms_rs
///
/// boxes = ... # of shape (n_boxes, 4) in Pascal VOC format
/// scores = ... # of shape (n_boxes, )
///
/// keep = rtree_nms_rs(boxes, scores, iou_threshold=0.5)
/// ```
#[pyfunction]
#[pyo3(signature = (boxes, scores, iou_threshold, score_threshold=0.5, rtree_leaf_size=16))]
pub fn rtree_nms_rs<'py>(
    py: Python<'py>,
    boxes: Bound<'py, PyAny>,
    scores: Bound<'py, PyAny>,
    iou_threshold: f64,
    score_threshold: f64,
    rtree_leaf_size: usize,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    let boxes_dtype: String = boxes.getattr("dtype")?.getattr("str")?.extract()?;
    let scores_dtype: String = scores.getattr("dtype")?.getattr("str")?.extract()?;

    let keep = match (boxes_dtype.as_str(), scores_dtype.as_str()) {
        ("f32" | "<f4", "f32" | "<f4") => run_rtree_nms!(
            py,
            boxes,
            scores,
            iou_threshold,
            score_threshold,
            rtree_leaf_size,
            f32,
            f32
        ),
        ("f32" | "<f4", "f64" | "<f8") => run_rtree_nms!(
            py,
            boxes,
            scores,
            iou_threshold,
            score_threshold,
            rtree_leaf_size,
            f32,
            f64
        ),
        ("f64" | "<f8", "f32" | "<f4") => run_rtree_nms!(
            py,
            boxes,
            scores,
            iou_threshold,
            score_threshold,
            rtree_leaf_size,
            f64,
            f32
        ),
        ("f64" | "<f8", "f64" | "<f8") => run_rtree_nms!(
            py,
            boxes,
            scores,
            iou_threshold,
            score_threshold,
            rtree_leaf_size,
            f64,
            f64
        ),
        _ => Err(PyValueError::new_err(format!(
            "Boxes and scores should be either f32 or f64, received {} and {}",
            boxes_dtype, scores_dtype
        ))),
    }?;

    Ok(PyArray1::from_array(py, &keep))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::compute_area;
    use ndarray::{Array, Array2, Axis, concatenate};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use std::println;
    use std::time::Instant;

    fn naive_nms<'py, Db: Float + Display, Ds: Float>(
        boxes: ArrayView2<'py, Db>,
        scores: ArrayView1<'py, Ds>,
        iou_threshold: f64,
        score_threshold: f64,
    ) -> Result<Array1<usize>, String> {
        let mut to_keep: Vec<usize> = Vec::new();
        let mut sorted_indices = argsort(scores, true)?;
        let score_threshold_d: Ds = Ds::from(score_threshold).ok_or_else(|| {
            format!(
                "score_treshold should be integer or float number, received {}",
                score_threshold
            )
        })?;
        let areas = boxes.map_axis(Axis(1), |row| compute_area(row).unwrap_or(0.));
        let mut is_valid: bool;
        while sorted_indices.len() > 0 {
            is_valid = true;
            let idx = sorted_indices.pop().unwrap();

            // Check score threshold
            if scores[idx] < score_threshold_d {
                break;
            }

            // Compare this bbox to all the bboxes already set aside
            let this_box = BoundingBox::new(boxes.row(idx))?;
            for &other_idx in to_keep.iter() {
                let other_box = BoundingBox::new(boxes.row(other_idx))?;
                let inter_area = this_box.intersect(&other_box)?;
                let iou = inter_area / (areas[idx] + areas[other_idx] - inter_area);
                // If any already set-aside box matches this one, then we discard this one.
                if iou >= iou_threshold {
                    is_valid = false;
                    break;
                }
            }
            if is_valid {
                to_keep.push(idx);
            }
        }

        Ok(Array1::from_vec(to_keep))
    }

    fn create_bboxes(n_boxes: usize) -> Result<Array2<f64>, String> {
        let x0y0 = Array::random((n_boxes, 2), Uniform::new(0., 1000.).unwrap());
        let dimensions = Array::random((n_boxes, 2), Uniform::new(15., 45.).unwrap());
        let x1y1 = &x0y0 + dimensions;
        let bboxes = concatenate(Axis(1), &[x0y0.view(), x1y1.view()]).map_err(|e| e.to_string())?;
        Ok(bboxes)
    }

    #[test]
    // #[ignore = "Timing"]
    fn test_rtree_nms() -> Result<(), String> {
        let n_boxes = 10_000;
        let bboxes = create_bboxes(n_boxes)?;
        // let mut rng = StdRng::seed_from_u64(0);
        let scores = Array::random((n_boxes,), Uniform::new(0., 1.).unwrap());

        let iou_threshold = 0.5;
        let score_threshold = 0.5;
        let target = naive_nms(bboxes.view(), scores.view(), iou_threshold, score_threshold)?;
        let pred = rtree_nms(bboxes.view(), scores.view(), iou_threshold, score_threshold, 16)?;

        if target != pred {
            Err(format!(
                "Predicted and target indices don't match: {} vs {}",
                pred.dim(),
                target.dim()
            ))
        } else {
            Ok(())
        }
    }

    #[test]
    #[ignore = "Timing"]
    fn test_rtree_nms_time() -> Result<(), String> {
        let n_boxes = 10_000;
        let bboxes = create_bboxes(n_boxes)?;
        let scores = Array::random((n_boxes,), Uniform::new(0., 1.).unwrap());

        let iou_threshold = 0.5;
        let score_threshold = 0.5;

        let start = Instant::now();
        let target = naive_nms(bboxes.view(), scores.view(), iou_threshold, score_threshold)?;
        let naive_elapsed = start.elapsed();
        let start = Instant::now();
        let pred = rtree_nms(bboxes.view(), scores.view(), iou_threshold, score_threshold, 4)?;
        let rtree_elapsed = start.elapsed();
        println!(
            "Naive duration: {:?}, RTree duration {:?}",
            naive_elapsed, rtree_elapsed
        );

        if target != pred {
            Err(format!(
                "Predicted and target indices dont match: {:?} vs {:?}",
                pred, target
            ))
        } else {
            Ok(())
        }
    }
}
