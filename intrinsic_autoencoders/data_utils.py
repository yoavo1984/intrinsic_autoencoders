"""
Provides utility functions to generate data.
"""
import numpy as np


def low_dim_data(ambient_dim: int = 100, intrinsic_dim: int = 10,
                 num_pts: int = 10000, seed: int = 1001) -> np.ndarray:
    """
    Generates data according to the following scheme:
        M = Draws basis vectors of size intrinsic_dim x ambient_dim
        C = Draws coeffs of size num_pts x intrinsic_dim
        D = C x M (matrix multiplication).
    """
    np.random.seed(seed)

    # Generates data according to the input parameters
    basis_vectors = np.random.uniform(0, 50, (intrinsic_dim, ambient_dim))

    # Generate random coefficients
    coeffs = np.random.uniform(0, 50, (num_pts, intrinsic_dim))

    # Generate Data.
    data = coeffs @ basis_vectors  # → [num_pts, ambient_dim]

    return data


def low_dim_data_transformed(ambient_dim: int = 100, intrinsic_dim: int = 10,
                             num_pts: int = 10000, seed: int = 1001) -> np.ndarray:
    """
    Generates data according to the following scheme:
        M = Draws basis vectors of size intrinsic_dim x ambient_dim
        C = Draws coeffs of size num_pts x intrinsic_dim
        Set C[:, -1] = √ C[:, :-1] ** 2
        D = C x M (matrix multiplication).
    """
    np.random.seed(seed)

    # Generates data according to the input parameters
    basis_vectors = np.random.uniform(0, 50, (intrinsic_dim, ambient_dim))

    # Generate random coefficients
    coeffs = np.random.uniform(0, 50, (num_pts, intrinsic_dim))

    # For each coefficients vector set x_10 = sin(x_1)
    coeffs[:, -1] = np.sqrt(np.sum(coeffs[:, :-1]**2, axis=1))

    # Generate Data.
    data = coeffs @ basis_vectors  # → [num_pts, ambient_dim]

    return data


def low_dim_data_with_noise(ambient_dim: int = 100, intrinsic_dim: int = 10,
                            num_pts: int = 10000, seed: int = 1001) -> np.ndarray:
    """
    Generates data according to the following scheme:
        M = Draws basis vectors of size intrinsic_dim x ambient_dim
        C = Draws coeffs of size num_pts x intrinsic_dim
        D = C x M (matrix multiplication) + noise
    """
    clean_data = low_dim_data(ambient_dim, intrinsic_dim, num_pts, seed)

    np.random.seed(seed)

    # old way...
    # noise = np.random.normal(loc=0, scale=1, size=clean_data.shape)
    # noisy_data = clean_data + noise

    # new way... ???
    sigma = 0.5
    print(sigma)
    noise = np.random.lognormal(mean=1, sigma=sigma, size=clean_data.shape)
    noisy_data = clean_data * noise

    return noisy_data


if __name__ == '__main__':
    data = low_dim_data()
    data = (data - data.mean()) / data.std()
    print(np.abs(data - data.mean()).mean())


