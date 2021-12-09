import numpy as np

from utils import problem


@problem.tag("hw4-A")
def calculate_centers(
    data: np.ndarray, classifications: np.ndarray, num_centers: int
) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.
            Might be usefull for pre-allocating numpy array (Faster that appending to list).

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.
    """
    new_centroids = np.zeros((num_centers, data.shape[1]))
    for j in range(0, num_centers):

        # compute centroids
        J = np.where(classifications == j)
        data_C = data[J]
        new_centroids[j] = data_C.mean(axis = 0)

    return new_centroids

@problem.tag("hw4-A")
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to centers given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """
    n = data.shape[0]
    k = centers.shape[0]

    distances = np.zeros((n, k), dtype=float)

    for d, data_v in enumerate(data):
        for centroid, centroid_v in enumerate(centers):
            distance = np.linalg.norm(data_v - centroid_v)**2
            distances[d, centroid] = distance

    #Determine nearest cluster
    nearest_center = np.argmin(distances, axis=1)

    return nearest_center


@problem.tag("hw4-A")
def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """
    n = data.shape[0]
    k = centers.shape[0]

    sse = 0
    for i, data_v in enumerate(data):
        for j, centroid_v in enumerate(centers):
            sse += np.sum(np.power(data[i, :] - centers[j], 2))

    mse = sse/(k * n)

    return mse


@problem.tag("hw4-A")
def lloyd_algorithm(
    data: np.ndarray, num_centers: int, epsilon: float = 10e-3
) -> np.ndarray:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing trained centers.

    Note:
        - For initializing centers please use the first `num_centers` data points.
    """

    I = np.random.choice(data.shape[0], num_centers)
    centroids = data[I, :]
    classifications = np.zeros(data.shape[0], dtype=np.int64)

    loss = 0
    max_iter = 300
    loss = 0
    tolerance = 10e-3

    for m in range(0, max_iter):
        # Compute the classifications

        classifications = cluster_data(data, centroids)
        new_centroids = calculate_centers(data, classifications, num_centers)
        new_loss = calculate_error(data, new_centroids)

        # Stopping criterion
        if np.abs(loss - new_loss) < tolerance:
            return new_centroids, classifications, new_loss

        centroids = new_centroids
        loss = new_loss

        print(loss)

    print("Failed to converge!")

    return centroids, classifications, loss
