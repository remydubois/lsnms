Changelog
=========

Version 0.2.0
------------
- Discarded BallTree and KDTree which are now replaced by a RTree: as fast and hyperparameter-free
  -> `lsnms.nms` is now five times faster to compile than before 
- `cutoff_distance` and `tree` are now deprecated. Warnings are issued when those are specified.
- cleared the tests structure
- cached all the jitted function which could (the non recursive ones) -> saves 2 seconds of compilation time at first use.

Patch 0.1.2
------------
- Fixed typo in BallTree (missing dimensionality argument, not used in algorithm but for sanity check in query)
- Updated Numba to 0.54.1


(unreleased)
------------
- - added changelog, upgraded to version 0.1.1. [Rémy Dubois]
- - black. [Rémy Dubois]
- - improved the node splitting method - Updated the runtimes comparison
  versus sklearn - fixed little typos in the readme - fixed typechecks
  in the trees. [Rémy Dubois]
- -readme. [Rémy Dubois]
- Typo + fixed image urls. [Rémy Dubois]
- - poetry. [Rémy Dubois]
- - poetry. [Rémy Dubois]
- - poetry. [Rémy Dubois]
- Initial commit. [Rémy Dubois]


