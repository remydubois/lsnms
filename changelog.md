Changelog
=========

Version 0.3.0
------------
- Added multiclass support for NMS

Version 0.2.0
------------
- Discarded BallTree and KDTree which are now replaced by a RTree: as fast and hyperparameter-free
  -> `lsnms.nms` is now twice faster to compile (only one tree to compile)
- `cutoff_distance` and `tree` are now deprecated. Warnings are issued when those are specified.
- cleared the tests structure
- added types conversion and sanity checks for data shape, etc
- cached all the jitted function which could (the non recursive ones) -> saves 2 seconds of compilation time at first use.
-> `lsnms.nms` is now hyperparameter free, runs just as fast as before, and almost three times faster to compile at first use.

Patch 0.1.2
------------
- Fixed typo in BallTree (missing dimensionality argument, not used in algorithm but for sanity check in query)
- Updated Numba to 0.54.1


Version 0.1.0
------------
- First version
- Both BallTree and KDTree are implemented for the sake of exhaustivity
- A cutoff distance needs to be specified to discard boxes to distant one from the other
- Compilation time at first use is quite long: 13 seconds and functions can not be precompiled due to recursivity.


