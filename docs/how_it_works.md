# How it works

## Scaling up the Non Maximum Suppression process

### Complexity

* In the best case scenario, NMS is a **linear-complex** process ($O(n)$): if all boxes are perfectly overlapping, then one pass of the algorithm discards all the boxes except the highest scoring one.
* In worst case scenario, NMS is a **quadratic-complex** operation (one needs to perform $n \times (n - 1) / 2$ IoU comparisons): if all boxes are perfectly disconnected, each NMS step will discard only one box (the highest scoring one, by decreasing order of score). Hence, one needs to perform $(n-1) + (n-2) + \dots + 1 = n \times (n - 1) / 2$ IoU computations.

### Working with huge images

When working with high-dimensional images (such as satellital or histology images), one often runs object detection inference by patching (with overlap) the input image and applying NMS to independent patches. Because patches do overlap, a final NMS needs to be re-applied afterward.

In that final case, one is close to being in the worst case scenario since each NMS step will discard only a very low amount of candidate instances (actually, pretty much the amount of overlapping passes over each instance, usually $\le 10$). Hence, depending on the size of the input image, computation time can reach several minutes on CPU. If the number of detections is in the hundreds of thousands, it can OOM consumer-grade GPUs.

A more natural way to speed up NMS could be through parallelization, like it is done for GPU-based implementations, but:
1. Efficiently parallelizing NMS is not a straightforward process.
2. If too many instances are predicted, GPU VRAM will often not be sufficient, preventing one from using GPU accelerators.
3. The process remains quadratic, and does not scale well.

### LSNMS

This project offers a way to overcome the aforementioned issues elegantly:
1. Like vanilla NMS, confidence scores are argsorted in decreasing order.
2. A rectangular tree (R-Tree) is initiated (empty).
3. Boxes are reviewed in decreasing order of confidence scores. Every time, the R-Tree is used to check whether this box is redundant with any already-seen bounding box:
    1. If not, the index of this bbox is added to the list of boxes to return, and the box is inserted in the R-Tree.
    2. If redundant, the box is discarded and the algorithm proceeds to the next box.

Insertion and query time complexity both are $O(\log(k))$ with $k$ being the amount of boxes indexed in the R-Tree at insertion/query time. Thus $k$ grows over time (as boxes are eventually inserted in the tree).

> **Note:** An older implementation (< v0.5.0) of `lsnms.nms` used another version of the algorithm, that is still used by the `"numba"` backend. This algorithm works by building the R-Tree on all the bounding boxes at once, then checking whether each bbox is redundant with any other. This older version is approximately 2 to 4 times slower.

An introduction to R-Tree can be found here: [OpenGenus R-Tree](https://iq.opengenus.org/r-tree/).

<p align="center">
  <img src="https://raw.githubusercontent.com/remydubois/lsnms/main/assets/images/simple_rtree_timings.png" width="700" alt="Run times on a virtual image of 10kx10k pixels" />
  <br />
  <em>Run times (on a virtual image of 10kx10k pixels)</em>
</p>

This implementation is based on Numba or Rust. The user has the choice to use either even if Rust is faster. Both backends lead to the exact same results.

> **Concrete example:**  
> Some tests were run considering ~ 40k x 40k pixel images, and detection inference run on 512 x 512 overlapping patches (256-strided). Approximately 300,000 bounding boxes (post patchwise NMS) resulted. Naive NMS ran in approximately 5 minutes on modern CPU, while this implementation ran in 3 seconds, hence offering a close to 120-fold speed up.

### Going further: weighted box clustering

For the sake of completeness, this repo also implements a variant of the Weighted Box Clustering algorithm (from [arXiv:1811.08661](https://arxiv.org/pdf/1811.08661.pdf)). Since NMS can artificially push up confidence scores (by selecting only the highest scoring box per instance), WBC overcomes this by averaging box coordinates and scores of all the overlapping boxes (instead of discarding all the non-maximally scored overlapping boxes).

> **Note:** Only the Numba backend implements WBC. If you need a faster WBC algorithm, please open a GitHub issue.

## Multiclass support

For multiclass support, a peculiar method to offset bounding boxes was used (offsetting bounding boxes class-wise is the standard way to apply NMS class-wise).

Note that the standard way to offset bboxes is to create a "diagonal per block" aspect, with each class' bboxes being positioned along a virtual diagonal.

Note that this would hurt performance because the underlying R-Tree that we would build on this would be suboptimal: many regions would actually be empty (because R-Tree builds rectangular regions) and the query time would be impacted.

Instead, here the boxes are offset forming a "mosaic" of class-wise regions, see figure below.

<p align="center">
  <img src="https://raw.githubusercontent.com/remydubois/lsnms/main/assets/images/multiclass_offset.png" width="700" alt="Illustration of the bbox offsetting process for multiclass support" />
  <br />
  <em>Illustration of the bbox offsetting process for multiclass support</em>
</p>
