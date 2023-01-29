import json
from multiprocessing import Process

from lsnms import nms
from lsnms.nms import _nms
from lsnms.util import clear_cache


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
