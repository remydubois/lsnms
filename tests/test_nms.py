import re

import numpy as np
import pytest
import torch
from torchvision.ops import boxes as box_ops

from lsnms import nms
from lsnms.nms import naive_nms


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numba", marks=pytest.mark.inter_framework),
        pytest.param("rust", marks=pytest.mark.intra_framework),
        pytest.param("torch", marks=pytest.mark.inter_framework),
    ],
)
def test_rtree_nms(instances, iou_threshold, score_threshold, benchmark, backend, request):
    boxes, scores = instances
    params = re.match(r".*\[(.*)\]", request.node.name).group(1).split("-")
    group_params = [p for i, p in enumerate(params) if i in [0, 1]]
    benchmark.group = "-".join(group_params)

    # Manually filter instances based on scores because torch's NMS does not do it
    scores_mask = scores > score_threshold
    torch_boxes = boxes[scores_mask]
    torch_scores = scores[scores_mask]

    # Compare against torch
    _k1 = box_ops.nms(torch.tensor(torch_boxes), torch.tensor(torch_scores), iou_threshold).numpy()
    # argwhere returns array of shape (n, 1)
    k1 = np.argwhere(scores_mask)[_k1, 0]

    def torch_nms(boxes, scores, iou_threshold):
        scores_mask = scores > score_threshold
        torch_boxes = boxes[scores_mask]
        torch_scores = scores[scores_mask]
        idxs = box_ops.nms(torch_boxes, torch_scores, iou_threshold)
        return torch.argwhere(scores_mask)[idxs, 0]

    # Compare sparse NMS
    if backend == "torch":
        tboxes = torch.tensor(torch_boxes)
        tscores = torch.tensor(torch_scores)
        k2 = benchmark(torch_nms, tboxes, tscores, iou_threshold)
    else:
        k2 = benchmark(
            nms, boxes, scores, iou_threshold, score_threshold, rtree_leaf_size=32, backend=backend
        )

    assert np.allclose(k1, k2)


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_empty_nms(instances, backend):
    boxes, scores = instances
    # Put all the scores to zero artificially
    keep = nms(boxes, scores * 0.0, score_threshold=0.5, backend=backend)

    assert keep.size == 0


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numba", marks=pytest.mark.inter_framework),
        pytest.param("rust", marks=pytest.mark.intra_framework),
    ],
)
def test_rtree_multiclass_nms(instances, score_threshold, benchmark, backend):

    boxes, scores = instances
    class_ids = np.random.randint(0, 50, size=len(boxes))

    # Manually filter instances based on scores because torch's NMS does not do it
    score_mask = scores > score_threshold
    torch_boxes = boxes[score_mask]
    torch_scores = scores[score_mask]
    torch_class_ids = class_ids[score_mask]

    # Compare against torch
    _k1 = box_ops.batched_nms(
        torch.tensor(torch_boxes), torch.tensor(torch_scores), torch.tensor(torch_class_ids), 0.5
    ).numpy()
    k1 = np.argwhere(score_mask)[_k1, 0]

    # Compare sparse NMS
    k2 = benchmark(nms, boxes, scores, 0.5, score_threshold, class_ids=class_ids, backend=backend)

    assert np.allclose(k1, k2)


def test_naive_nms(instances):
    boxes, scores = instances

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()

    # Compare naive NMS
    k2 = naive_nms(boxes, scores, 0.5, 0.0)

    assert np.allclose(k1, k2)


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_boxes_shape(instances, backend):
    boxes, scores = instances
    boxes = boxes[None]
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1, backend=backend)


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_scores_shape(instances, backend):
    boxes, scores = instances
    scores = scores[..., None]
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1, backend=backend)


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_box_encoding(backend):
    boxes, scores = np.array([[1.0, 0.0, 0.0, 1.0]]), np.array([1.0])
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1, backend=backend)


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_nan_coords_evicted(backend):
    boxes, scores = np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([1.0])
    boxes[0, 0] = np.nan
    k = nms(boxes, scores, 0.5, 0.1, backend=backend)
    assert k.size == 0


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_nan_scores_evicted(backend):
    boxes, scores = np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([1.0])
    scores[0] = np.nan
    k = nms(boxes, scores, 0.5, 0.1, backend=backend)
    assert k.size == 0


def test_non_supported_backend():
    boxes, scores = np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([1.0])
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1, backend="cpp")
