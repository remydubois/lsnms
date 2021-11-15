import numpy as np
from lsnms.balltree import BallTree


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
