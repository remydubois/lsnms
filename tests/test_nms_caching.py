import json
import os
from multiprocessing import Process

import pytest

from lsnms import nms
from lsnms.nms import _nms
from lsnms.util import clear_cache

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_caching_hits(instances, tmp_path, nms_signature):
    """
    Very manual cache testing:
    1 - First, cache is cleared
    2 - A first process calls nms, cache should be empty here and should miss
    3 - Another process then calls nms, cache should now hit
    """
    # Empty eventually existing cache
    clear_cache()

    # First call must not happen in current process otherwise second call would
    # call the currently compiled _nms.
    process = Process(target=uncached_routine, args=(*instances, tmp_path))
    process.start()
    process.join()

    # The second call can happen in current process however
    cached_routine(*instances, tmp_path)

    with open(tmp_path / "uncached_stats.json", "r") as infile:
        stats = json.load(infile)
        n_misses = stats["cache_misses"].get(nms_signature, 0)
        n_hits = stats["cache_hits"].get(nms_signature, 0)
        assert (
            n_misses > 0
        ), f"Cache clearing malfunctioned, no miss to report at first call: {n_misses}"
        assert n_hits == 0, f"Cache clearing malfunctioned, number of hits is non null: {n_hits}"

    with open(tmp_path / "cached_stats.json", "r") as infile:
        stats = json.load(infile)
        n_misses = stats["cache_misses"].get(nms_signature, 0)
        n_hits = stats["cache_hits"].get(nms_signature, 0)

        assert n_misses == 0, f"Caching malfunctioned, misses to report at second call:{n_misses}"
        assert n_hits > 0, f"Caching malfunctioned, number of hits is null: {n_hits}"
