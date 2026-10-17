//! Rust backend for the [`LSNMS`] python library.
//!
//! [`LSNMS`]: https://github.com/remydubois/lsnms
use pyo3::prelude::*;

pub mod bbox;
pub mod math;
pub mod nms;
pub mod rtree;

#[doc(hidden)]
#[pymodule]
pub fn lsnms_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(nms::rtree_nms_rs, module)?)?;
    Ok(())
}
