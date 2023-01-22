from multiprocessing import Process
from time import perf_counter

import numpy as np
import pytest
import torch
from torchvision.ops import boxes as box_ops

from lsnms import nms
from lsnms.nms import naive_nms
from lsnms.util import clear_cache


def test_rtree_nms(instances, score_threshold):

    boxes, scores = instances

    # Manually filter instances based on scores because torch's NMS does not do it
    torch_boxes = boxes[scores > score_threshold]
    torch_scores = scores[scores > score_threshold]

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(torch_boxes), torch.tensor(torch_scores), 0.5).numpy()

    # Compare sparse NMS
    k2 = nms(boxes, scores, 0.5, score_threshold)

    assert np.allclose(k1, k2)


def test_rtree_nms_verbose(instances_subset, score_threshold):

    boxes, scores = instances_subset

    # Manually filter instances based on scores because torch's NMS does not do it
    torch_boxes = boxes[scores > score_threshold]
    torch_scores = scores[scores > score_threshold]

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(torch_boxes), torch.tensor(torch_scores), 0.5).numpy()

    # Compare sparse NMS
    k2 = nms(boxes, scores, 0.5, score_threshold, rtree_verbosity_level=100)

    assert np.allclose(k1, k2)


def test_empty_nms(instances):
    boxes, scores = instances
    # Put all the scores to zero artificially
    keep = nms(boxes, scores * 0.0)

    assert keep.size == 0


def test_rtree_multiclass_nms(instances, score_threshold):

    boxes, scores = instances
    class_ids = np.random.randint(0, 50, size=len(boxes))

    # Manually filter instances based on scores because torch's NMS does not do it
    torch_boxes = boxes[scores > score_threshold]
    torch_scores = scores[scores > score_threshold]
    torch_class_ids = class_ids[scores > score_threshold]

    # Compare against torch
    k1 = box_ops.batched_nms(
        torch.tensor(torch_boxes), torch.tensor(torch_scores), torch.tensor(torch_class_ids), 0.5
    ).numpy()

    # Compare sparse NMS
    k2 = nms(boxes, scores, 0.5, score_threshold, class_ids=class_ids)

    assert np.allclose(k1, k2)


def test_naive_nms(instances):

    boxes, scores = instances

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()

    # Compare naive NMS
    k2 = naive_nms(boxes, scores, 0.5, 0.0)

    assert np.allclose(k1, k2)


def test_boxes_shape(instances):
    boxes, scores = instances
    boxes = boxes[None]
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1)


def test_scores_shape(instances):
    boxes, scores = instances
    scores = scores[..., None]
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1)


def test_box_encoding(instances):
    boxes, scores = instances
    # Make the first box odd: x1 > x2
    boxes[0, 0] = boxes[0, 2] + 1
    with pytest.raises(ValueError):
        nms(boxes, scores, 0.5, 0.1)


def routine(boxes, scores):
    _ = nms(boxes, scores, 0.5, score_threshold=0.0)
    return


def test_caching(instances):

    clear_cache()
    process = Process(target=routine, args=(instances))
    process2 = Process(target=routine, args=(instances))

    st = perf_counter()
    process.start()
    process.join()
    delta_0 = perf_counter() - st

    st = perf_counter()
    process2.start()
    process2.join()
    delta_1 = perf_counter() - st

    assert (
        delta_1 * 3 < delta_0
    ), f"Compilation did not cache. First call: {delta_0:.2f}s, second call: {delta_1:.2f}s"
