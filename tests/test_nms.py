import numpy as np
import pytest
import torch
from lsnms import nms
from lsnms.nms import naive_nms
from torchvision.ops import boxes as box_ops


def test_rtree_nms(instances):

    boxes, scores = instances

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()

    # Compare sparse NMS
    k2 = nms(boxes, scores, 0.5, 0.0)

    assert np.allclose(k1, k2)


def test_rtree_nms_non_null_threshold(instances, score_threshold):

    boxes, scores = instances

    # Manually filter instances based on scores because torch's NMS does not do it
    torch_boxes = boxes[scores > score_threshold]
    torch_scores = scores[scores > score_threshold]

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(torch_boxes), torch.tensor(torch_scores), 0.5).numpy()

    # Compare sparse NMS
    k2 = nms(boxes, scores, 0.5, score_threshold)

    assert np.allclose(k1, k2)


def test_empty_nms(instances):
    boxes, scores = instances
    # Put all the scores to zero artificially
    keep = nms(boxes, scores * 0.0)

    assert keep.size == 0


def test_rtree_multiclass_nms(instances):

    boxes, scores = instances
    class_ids = np.random.randint(0, 50, size=len(boxes))

    # Compare against torch
    k1 = box_ops.batched_nms(
        torch.tensor(boxes), torch.tensor(scores), torch.tensor(class_ids), 0.5
    ).numpy()

    # Compare sparse NMS
    k2 = nms(boxes, scores, 0.5, 0.0, class_ids=class_ids)

    assert np.allclose(k1, k2)


def test_rtree_multiclass_nms_non_null_threshold(instances, score_threshold):

    boxes, scores = instances
    class_ids = np.random.randint(0, 50, size=len(boxes))

    # Manually filter instances based on scores because torch's NMS does not do it
    torch_boxes = boxes[scores > score_threshold]
    torch_scores = scores[scores > score_threshold]

    # Compare against torch
    k1 = box_ops.batched_nms(
        torch.tensor(torch_boxes), torch.tensor(torch_scores), torch.tensor(class_ids), 0.5
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


def test_warning_dist_arg(instances):
    boxes, scores = instances
    with pytest.warns(None):
        nms(boxes, scores, 0.5, 0.1, cutoff_distance=64)


def test_warning_tree_arg(instances):
    boxes, scores = instances
    with pytest.warns(None):
        nms(boxes, scores, 0.5, 0.1, tree="faketree")


def test_issue_12():
    # From https://github.com/remydubois/lsnms/issues/12
    import numpy as np
    from lsnms import nms  # v0.3.1

    nms_test_dict_fail = {
        "scores": np.array([0.03128776, 0.15489164, 0.05489164]),
        "boxes": np.array(
            [
                [623.47991943, 391.94015503, 675.83850098, 445.0836792],
                [11.48574257, 15.99506855, 1053.84313965, 1074.78381348],
                [11.48574257, 15.99506855, 1053.84313965, 1074.78381348],
            ]
        ),
        "class_labels": np.array([1, 27, 23]),
        "score_thresh": 0.1,
    }

    keep = nms(
        boxes=nms_test_dict_fail["boxes"],
        scores=nms_test_dict_fail["scores"],
        score_threshold=nms_test_dict_fail["score_thresh"],
        class_ids=nms_test_dict_fail["class_labels"],
    )
    assert len(keep) == 1
