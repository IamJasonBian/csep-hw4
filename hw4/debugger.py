from typing import Callable
from unittest import TestCase

import numpy as np
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.k_means.k_means import calculate_centers, calculate_error, cluster_data

X = np.array(
    [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5], ]
)
centers = np.array([[0.5, 0.0, 0.5], [0.25, 0.5, 0.25], ])
expected = np.array([0, 0, 1, 1])

actual = cluster_data(X, centers)
np.testing.assert_almost_equal(actual, expected)

# Generate data
X = np.array(
    [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5], ]
)
centers = np.array([[0.5, 0.0, 0.5], [0.25, 0.5, 0.25], ])
expected = 0.4183

actual = calculate_error(X, centers)
np.testing.assert_almost_equal(actual, expected, decimal=4)