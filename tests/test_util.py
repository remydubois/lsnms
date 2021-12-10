import numpy as np
from numba import njit
from lsnms.util import offset_bboxes
from lsnms.rtree import RTree


def datagen(n=10_000):
    topleft = np.random.uniform(0.0, high=1_000, size=(n, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)

    boxes = np.concatenate([topleft, topleft + wh], axis=1)
    scores = np.random.uniform(0.1, 1.0, size=len(topleft))

    return boxes, scores

@njit
def intersect_many(tree, boxes):
    intersections = []
    for box in boxes:
        indices, _ = tree.intersect(box)
        intersections.extend([i for i in indices])
    
    return intersections

def test_offset_bboxes():
    boxes, _ = datagen()
    rng = np.random.default_rng(0)
    class_ids = rng.integers(0, 50, size=len(boxes))

    new_boxes = offset_bboxes(boxes, class_ids)

    # Assert that no boxes for class a intersect with class b
    tree = RTree(new_boxes)
    unique_class_ids, class_index = np.unique(class_ids, return_inverse=True)
    for i in range(len(unique_class_ids) - 1):
        for j in range(i + 1, len(unique_class_ids)):
            class_i = unique_class_ids[i]
            class_j = unique_class_ids[j]
            boxes_i = 
