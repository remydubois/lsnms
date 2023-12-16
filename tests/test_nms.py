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
    scores_mask = scores > score_threshold
    torch_boxes = boxes[scores_mask]
    torch_scores = scores[scores_mask]

    # Compare against torch
    _k1 = box_ops.nms(torch.tensor(torch_boxes), torch.tensor(torch_scores), 0.5).numpy()
    # argwhere returns array of shape (n, 1)
    k1 = np.argwhere(scores_mask)[_k1, 0]

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


def cached_routine(boxes, scores, tmp_path):
    _ = nms(boxes, scores, 0.5, score_threshold=0.0)

    stats = {"cache_hits": {str(k): v for k, v in _nms.stats.cache_hits.items()}}
    stats["cache_misses"] = {str(k): v for k, v in _nms.stats.cache_misses.items()}

    with open(tmp_path / "cached_stats.json", "w+") as outfile:
        json.dump(stats, outfile)

    return


def uncached_routine(boxes, scores, tmp_path):
    _ = nms(boxes, scores, 0.5, score_threshold=0.0)

    stats = {"cache_hits": {str(k): v for k, v in _nms.stats.cache_hits.items()}}
    stats["cache_misses"] = {str(k): v for k, v in _nms.stats.cache_misses.items()}

    with open(tmp_path / "uncached_stats.json", "w+") as outfile:
        json.dump(stats, outfile)

    return


@pytest.mark.skip(reason="To discard")
def test_caching_hits(instances, tmp_path, nms_signature):
    """
    Very manul cache testing:
    1 - First, cache is cleared
    2 - A first process calls nms, cache should be empty here and should miss
    3 - Another process then calls nms, cache should now hit
    """
    clear_cache()
    process = Process(target=uncached_routine, args=(*instances, tmp_path))
    process2 = Process(target=cached_routine, args=(*instances, tmp_path))

    process.start()
    process.join()

    process2.start()
    process2.join()

    with open(tmp_path / "uncached_stats.json", "r") as infile:
        stats = json.load(infile)
        n_misses = stats["cache_misses"][nms_signature]
        n_hits = stats["cache_hits"].get(nms_signature, 0)
        assert (
            n_misses
            > 0
            # ), f"Cache clearing malfunctioned, no miss to report at first call: {n_misses}"
        ), print(stats)
        # assert n_hits == 0, f"Cache clearing malfunctioned, number of hits is non null: {n_hits}"
        assert n_hits == 0, print(stats)

    with open(tmp_path / "cached_stats.json", "r") as infile:
        stats = json.load(infile)
        n_misses = stats["cache_misses"].get(nms_signature, 0)
        n_hits = stats["cache_hits"].get(nms_signature, 0)

        # assert n_misses == 0, f"Caching malfunctioned, misses to report at second call:{n_misses}"
        # assert n_hits > 0, f"Caching malfunctioned, number of hits is null: {n_hits}"
        assert n_misses == 0, print(stats)
        assert n_hits > 0, print(stats)
