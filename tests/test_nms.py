import json
from multiprocessing import Process

import numpy as np
import pytest
import torch
from torchvision.ops import boxes as box_ops

from lsnms import nms
from lsnms.nms import _nms, naive_nms
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


def cached_routine(boxes, scores):
    _ = nms(boxes, scores, 0.5, score_threshold=0.0)

    stats = {"cache_hits": {str(k): v for k, v in _nms.stats.cache_hits.items()}}
    stats["cache_misses"] = {str(k): v for k, v in _nms.stats.cache_misses.items()}

    with open("/tmp/cached_stats.json", "w+") as outfile:
        json.dump(stats, outfile)

    return


def uncached_routine(boxes, scores):
    _ = nms(boxes, scores, 0.5, score_threshold=0.0)

    stats = {"cache_hits": {str(k): v for k, v in _nms.stats.cache_hits.items()}}
    stats["cache_misses"] = {str(k): v for k, v in _nms.stats.cache_misses.items()}

    with open("/tmp/uncached_stats.json", "w+") as outfile:
        json.dump(stats, outfile)

    return


def test_caching_hits(instances):
    """
    Very manul cache testing:
    1 - First, cache is cleared
    2 - A first process calls nms, cache should be empty here and should miss
    3 - Another process then calls nms, cache should now hit
    """
    clear_cache()
    process = Process(target=uncached_routine, args=(instances))
    process2 = Process(target=cached_routine, args=(instances))

    process.start()
    process.join()

    process2.start()
    process2.join()

    sig = "(array(float64, 2d, C), array(float64, 1d, C), float64, float64, int64, int64)"

    with open("/tmp/uncached_stats.json", "r") as infile:
        stats = json.load(infile)
        n_misses = stats["cache_misses"][sig]
        n_hits = stats["cache_hits"].get(sig, 0)
        assert (
            n_misses > 0
        ), f"Cache clearing malfunctioned, no miss to report at first call: {n_misses}"
        assert n_hits == 0, f"Cache clearing malfunctioned, number of hits is non null: {n_hits}"

    with open("/tmp/cached_stats.json", "r") as infile:
        stats = json.load(infile)
        n_misses = stats["cache_misses"].get(sig, 0)
        n_hits = stats["cache_hits"].get(sig, 0)

        assert n_misses == 0, f"Caching malfunctioned, misses to report at second call: {n_misses}"
        assert n_hits > 0, f"Caching malfunctioned, number of hits is null: {n_hits}"
