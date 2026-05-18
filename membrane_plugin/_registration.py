import math
import os
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import least_squares
from scipy.ndimage import affine_transform, gaussian_filter, zoom
from scipy.ndimage import minimum_filter, maximum_filter
from skimage.measure import label as sk_label
import skimage.morphology as morphology


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def euler_angles_to_rotation_matrix(angles_rad):
    alpha, beta, gamma = angles_rad
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(alpha), -math.sin(alpha)],
                    [0, math.sin(alpha),  math.cos(alpha)]])
    R_y = np.array([[ math.cos(beta), 0, math.sin(beta)],
                    [0,               1, 0             ],
                    [-math.sin(beta), 0, math.cos(beta)]])
    R_z = np.array([[math.cos(gamma), -math.sin(gamma), 0],
                    [math.sin(gamma),  math.cos(gamma), 0],
                    [0,               0,                1]])
    return R_z @ R_y @ R_x


# ---------------------------------------------------------------------------
# Thread-safe shared best — lets parallel starts skip the expensive polish
# ---------------------------------------------------------------------------

class _SharedBest:
    def __init__(self):
        self._cost = np.inf
        self._lock = threading.Lock()

    @property
    def cost(self):
        with self._lock:
            return self._cost

    def update(self, cost):
        with self._lock:
            if cost < self._cost:
                self._cost = cost

    def is_hopeless(self, cost, factor=10.0):
        """True if cost is more than `factor` × best seen so far."""
        with self._lock:
            return self._cost < np.inf and cost > factor * self._cost


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------

def _residuals(parameters, x_data, y_data, mask):
    R = euler_angles_to_rotation_matrix(parameters[:3])
    t = parameters[3:]
    warped = affine_transform(x_data, R, t, order=1, prefilter=False)
    return np.ravel(warped[mask] - y_data[mask])


# ---------------------------------------------------------------------------
# Single pyramid run — original iteration schedule, abort only before polish
# ---------------------------------------------------------------------------

def _run_pyramid(x_bnd, y_bnd, p0, sigmas, zoom_factors, max_nfev,
                 shared_best=None):
    params = p0.copy()
    n = len(sigmas)

    for i, (sigma, zf) in enumerate(zip(sigmas, zoom_factors)):
        # original schedule: budget grows toward the finest level
        level_nfev = max(100, max_nfev // (2 ** (n - 1 - i)))

        if zf < 1.0:
            xz = zoom(x_bnd, zf, order=1).astype(np.float32)
            yz = zoom(y_bnd, zf, order=1).astype(np.float32)
            sigma_z = sigma * zf
        else:
            xz, yz = x_bnd, y_bnd
            sigma_z = sigma
        xb = gaussian_filter(xz, sigma_z) + 0.1 * xz
        yb = gaussian_filter(yz, sigma_z) + 0.1 * yz
        mask = yb > 1e-6
        pz = params.copy(); pz[3:] *= zf
        res = least_squares(_residuals, pz, method='dogbox', max_nfev=level_nfev,
                            args=(xb, yb, mask))
        params = res.x.copy(); params[3:] = res.x[3:] / zf
        if shared_best is not None:
            shared_best.update(res.cost)
        print(f'    sigma={sigma:.2f} zoom={zf:.2f}: cost={res.cost:.6f}', flush=True)

    # skip expensive full-resolution polish if clearly worse than best so far
    if shared_best is not None and shared_best.is_hopeless(res.cost):
        print(f'    polish skipped (cost={res.cost:.6f} >> best={shared_best.cost:.6f})',
              flush=True)
        return params, res.cost

    # full-resolution polish
    mask = y_bnd > 1e-6
    res = least_squares(_residuals, params, method='dogbox', max_nfev=max_nfev,
                        args=(x_bnd.astype(np.float32), y_bnd.astype(np.float32), mask))
    params = res.x.copy()
    if shared_best is not None:
        shared_best.update(res.cost)
    print(f'    polish zoom=1.00: cost={res.cost:.6f}', flush=True)
    return params, res.cost


# ---------------------------------------------------------------------------
# Multi-start with parallel execution
# ---------------------------------------------------------------------------

def register_multiscale(x_bnd, y_bnd, initial_params,
                        max_nfev=2000, n_random_starts=12,
                        angle_perturb=0.1, translation_perturb=5.0, seed=0):
    min_dim = min(x_bnd.shape)
    s = max(2.0, min_dim / 4.0)
    sigmas = []
    while s >= 1.0:
        sigmas.append(round(s, 2)); s /= 2.0
    n = len(sigmas)
    zoom_factors = [max(0.25, 0.5 ** (n - 1 - i)) for i in range(n)]

    rng = np.random.default_rng(seed)
    starts = [('warm start', initial_params.copy()), ('identity', np.zeros(6))]
    for i in range(n_random_starts):
        noise = np.zeros(6)
        noise[:3] = rng.normal(0, angle_perturb, 3)
        noise[3:] = rng.normal(0, translation_perturb, 3)
        starts.append((f'random {i+1}', initial_params + noise))

    x_f = x_bnd.astype(np.float32)
    y_f = y_bnd.astype(np.float32)
    shared_best = _SharedBest()

    def _run_one(name_p0):
        name, p0 = name_p0
        print(f'\n  --- start: {name} ---', flush=True)
        params, cost = _run_pyramid(x_f, y_f, p0, sigmas, zoom_factors, max_nfev,
                                    shared_best=shared_best)
        print(f'  --- {name} done  cost={cost:.6f} ---', flush=True)
        return params, cost

    with ThreadPoolExecutor() as pool:
        results = list(pool.map(_run_one, starts))

    best_params, best_cost = min(results, key=lambda r: r[1])
    print(f'\n  best cost={best_cost:.6f}  params={np.round(best_params, 4)}', flush=True)
    return best_params


# ---------------------------------------------------------------------------
# Prep (array-based — no file I/O)
# ---------------------------------------------------------------------------

def prep_array(lab_img, reg_res, min_size=800):
    """Isotropize, remove small objects, compute boundary image from a numpy array."""
    orig = lab_img.copy()
    zoomed = zoom(lab_img, (reg_res, reg_res / 10.0, reg_res / 10.0), order=0)

    cc = sk_label(zoomed, connectivity=zoomed.ndim)
    cc = morphology.remove_small_objects(cc, min_size=min_size)
    clean = zoomed.copy()
    clean[cc == 0] = 0

    lab_int   = clean.astype(np.int32)
    local_min = minimum_filter(lab_int, size=3)
    local_max = maximum_filter(lab_int, size=3)
    boundary  = (local_min != local_max).astype(np.uint8)

    return boundary, orig, clean


# ---------------------------------------------------------------------------
# Warp back to original anisotropic space  (parallel over Z chunks)
# ---------------------------------------------------------------------------

def warp_to_original_space(src_orig, rot_matrix, translation, reg_res, ref_shape):
    s_vec = np.array([reg_res, reg_res / 10.0, reg_res / 10.0])
    S     = np.diag(s_vec)
    S_inv = np.diag(1.0 / s_vec)
    R_orig = S_inv @ rot_matrix @ S
    t_orig = S_inv @ np.array(translation)

    src = src_orig.astype(np.float32)
    n_z = ref_shape[0]
    n_workers = min(n_z, os.cpu_count() or 4)
    chunk_size = max(1, (n_z + n_workers - 1) // n_workers)

    out = np.empty(ref_shape, dtype=src_orig.dtype)

    def _warp_chunk(z_start):
        z_end = min(z_start + chunk_size, n_z)
        chunk_shape = (z_end - z_start,) + ref_shape[1:]
        # Shift translation so chunk-local z=0 maps to global z=z_start
        t_chunk = t_orig + R_orig @ np.array([z_start, 0.0, 0.0])
        chunk = affine_transform(src, R_orig, t_chunk,
                                 output_shape=chunk_shape,
                                 order=0, prefilter=False)
        out[z_start:z_end] = chunk

    z_starts = list(range(0, n_z, chunk_size))
    print(f'  Warping {n_z} Z-slices across {len(z_starts)} parallel chunks…', flush=True)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_warp_chunk, z_starts))

    return out
