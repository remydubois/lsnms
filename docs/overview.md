# Overview

![PyPI - Downloads](https://img.shields.io/pypi/dm/lsnms)

`lsnms` speeds up NMS ran on very large images by several folds, using a sparse implementation of the algorithm.

This project becomes useful in the case of very high dimensional image data, when the amount of predicted instances to prune becomes considerable (> 100 objects).

<p align="center">
  <img src="https://raw.githubusercontent.com/remydubois/lsnms/main/assets/images/simple_rtree_timings.png" width="700" alt="Run times on a virtual image of 10kx10k pixels" />
  <br />
  <em>Run times (on a virtual image of 10kx10k pixels)</em>
</p>

## Non Maximum Suppression

A nice introduction to the non maximum suppression algorithm can be found here: [Coursera NMS Lecture](https://www.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH).

Basically, NMS discards redundant boxes in a set of predicted instances. It is an essential — and often unavoidable — step of object detection pipelines.

## Will I benefit from this sparse NMS implementation?

From version 0.5.0, `lsnms` is on par with `torchvision`'s `nms` both on CPU and on GPU, even for small images (and small amount of detections). This does not account for host <-> device data transfer time. So, if you are running a vanilla object detection pipeline (i.e. single image with a GPU-based object detection algorithm): `torchvision` GPU-based NMS will likely be faster.
