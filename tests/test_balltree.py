import numpy as np
import torch
from lsnms import nms, wbc
from lsnms.balltree import BallTree
from torchvision.ops import boxes as box_ops


def test_query_radius_tree():

    data = np.random.uniform(0, 100, (10000, 2))
    tree = BallTree(data, 16)

    radius = 20
    queries = np.array([50.0, 50.0])
    indices = tree.query_radius(queries, radius)
    distances = np.power(data - queries, 2.0).sum(-1) ** 0.5

    distances_in = distances[indices]
    distances_out = distances[[i for i in range(len(data)) if i not in indices]]

    np.testing.assert_array_less(distances_in, radius)
    np.testing.assert_array_less(radius, distances_out)


def test_nms():

    topleft = np.random.uniform(0.0, high=1_000, size=(10_000, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)
    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = np.random.uniform(0.1, 1.0, size=len(topleft))

    # Compare against torch
    k1 = box_ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5).numpy()

    # Compare naive NMS
    k2 = nms(boxes, scores, 0.5, 0.1)
    # Compare naive NMS
    k3 = nms(boxes, scores, 0.5, 0.1, 64, tree="balltree")

    assert np.allclose(k1, k2) and np.allclose(k1, k3)


def test_wbc():

    boxes = []
    scores = []
    # First cluster at 10, 10
    # Cluster location is a rough localization, not centroid
    cluster_locations = np.array([[10, 10]])
    # Randomly place clusters on the diagonal, 100 pixels apart one from another to avoid overlaps
    cluster_locations = cluster_locations + np.arange(100, 1100, 100)[:, None]

    final_cluster_boxes = []
    final_cluster_scores = []
    # Build 10 clusters
    for i in range(len(cluster_locations)):
        # Random number of boxes per cluster
        n_boxes = np.random.randint(5, 15)

        # Grossly align  the boxes of this cluster on the anchor
        cluster_location = cluster_locations[i]
        noise = np.random.normal(loc=0.0, scale=1, size=(n_boxes, 2))
        # Slightly shift the boxes of this cluster
        topleft = cluster_location + noise
        wh = np.random.normal(32, 2, size=topleft.shape)
        wh = np.clip(wh, 16, 48)
        cluster_boxes = np.concatenate([topleft, topleft + wh], axis=1)
        cluster_scores = np.random.uniform(0.1, 1.0, size=len(topleft))

        # Compute this cluster box and scores
        final_cluster_boxes.append(cluster_boxes.mean(0))
        final_cluster_scores.append(cluster_scores.mean())

        boxes.append(cluster_boxes)
        scores.append(cluster_scores)

    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    final_cluster_boxes = np.array(final_cluster_boxes)
    final_cluster_scores = np.array(final_cluster_scores)

    # Apply WBC
    wbc_boxes, wbc_scores, _ = wbc(
        boxes, scores, iou_threshold=0.5, score_threshold=0.1, cutoff_distance=64, tree="balltree"
    )

    # put the boxes in the same order to be able to match to the the pre-computed centroids
    sorted_indices = np.argsort(wbc_boxes[:, 0])
    wbc_boxes = wbc_boxes[sorted_indices]
    wbc_scores = wbc_scores[sorted_indices]

    # Compare
    np.testing.assert_allclose(wbc_boxes, final_cluster_boxes)
    np.testing.assert_allclose(wbc_scores, final_cluster_scores)
