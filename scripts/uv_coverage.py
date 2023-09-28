import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations 

def sim_uv_cov(obs_length):
    np.random.seed(7)

    integration_time = 5.0 / 3600.0  # in hours
    n_samples = int(np.ceil(obs_length / integration_time))
    obs_length = n_samples * integration_time

    obs_start_time = (np.random.random() - 0.5) * 5.0
    n_antenna = int(np.ceil(10 + np.random.random() * 50))
    max_baseline_length = 1000 + np.random.random() * 2000  # in meters
    do = -1 * np.random.random() * np.pi / 2

    ENU = np.vstack([np.random.random((2, n_antenna)) * max_baseline_length, np.zeros(n_antenna)])

    lat = -23.02 * np.pi / 180  # latitude of ALMA
    ENU_to_xyz = np.array([[0, -np.sin(lat), np.cos(lat)],
                           [1, 0, 0],
                           [0, np.cos(lat), np.sin(lat)]])

    obs_length = obs_length * 2 * np.pi / 24
    obs_start_time = obs_start_time * 2 * np.pi / 24

    HourAngle = np.linspace(obs_start_time, obs_start_time + obs_length, n_samples)

    n_baselines = n_antenna * (n_antenna - 1) // 2
    antennas = np.array(list(combinations(range(1, n_antenna + 1), 2)))

    xyz = np.dot(ENU_to_xyz, ENU)
    B = xyz[:, antennas[:, 1] - 1] - xyz[:, antennas[:, 0] - 1]

    u = np.zeros((n_samples, n_baselines))
    v = np.zeros((n_samples, n_baselines))

    for i in range(n_samples):
        ho = HourAngle[i]
        Bto_uvw = np.array([[np.sin(ho), np.cos(ho), 0],
                            [-np.sin(do) * np.cos(ho), np.sin(do) * np.sin(ho), np.cos(do)],
                            [np.cos(do) * np.cos(ho), -np.cos(do) * np.sin(ho), np.sin(do)]])

        uvw = np.dot(Bto_uvw, B)
        u[i, :] = uvw[0, :]
        v[i, :] = uvw[1, :]

    UVGRID, _, _ = np.histogram2d(u.ravel(), v.ravel(), bins=[np.linspace(-1000, 1000, 128), np.linspace(-1000, 1000, 128)])
    UVGRID = UVGRID.astype(float)
    noise_rms = 1.0 / (np.sqrt(UVGRID) + 1e-12)

    ants1 = np.tile(antennas[:, 0], (n_samples, 1)).flatten()
    ants2 = np.tile(antennas[:, 1], (n_samples, 1)).flatten()

    u = u.flatten()
    v = v.flatten()

    return UVGRID, noise_rms, u, v, ants1, ants2

# # Example usage:
# obs_length = 2.0  # in hours
# grid, noise_rms, u, v, ants1, ants2 = sim_uv_cov(obs_length)
