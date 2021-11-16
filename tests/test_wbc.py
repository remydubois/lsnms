import numpy as np
from lsnms import wbc


def generate_data():
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

    return boxes, scores, final_cluster_boxes, final_cluster_scores


def test_wbc():

    boxes, scores, final_cluster_boxes, final_cluster_scores = generate_data()

    # Apply WBC
    wbc_boxes, wbc_scores, _ = wbc(
        boxes,
        scores,
        iou_threshold=0.5,
        iou_reweight=False
    )

    # put the boxes in the same order to be able to match to the the pre-computed centroids
    sorted_indices = np.argsort(wbc_boxes[:, 0])
    wbc_boxes = wbc_boxes[sorted_indices]
    wbc_scores = wbc_scores[sorted_indices]

    # Compare
    np.testing.assert_allclose(wbc_boxes, final_cluster_boxes)
    np.testing.assert_allclose(wbc_scores, final_cluster_scores)