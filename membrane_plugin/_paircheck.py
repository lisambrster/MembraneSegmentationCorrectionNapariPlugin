import numpy as np
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Inlined matching utilities (from stardist/csbdeep matching.py)
# csbdeep removed as a dependency; numba is optional
# ---------------------------------------------------------------------------

def _raise(err):
    raise err


def is_array_of_integers(y):
    return isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.integer)


def label_are_sequential(y):
    labels = np.unique(y)
    return (set(labels) - {0}) == set(range(1, 1 + labels.max()))


def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError("{label} must be an array of {integers}.".format(
        label='labels' if name is None else name,
        integers=('sequential ' if check_sequential else '') + 'non-negative integers',
    ))
    is_array_of_integers(y) or _raise(err)
    if len(y) == 0:
        return True
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True


def _label_overlap_numpy(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + int(x.max()), 1 + int(y.max())), dtype=np.uint64)
    np.add.at(overlap, (x, y), 1)
    return overlap


try:
    from numba import jit as _numba_jit

    @_numba_jit(nopython=True)
    def _label_overlap_numba(x, y):
        x = x.ravel()
        y = y.ravel()
        overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint64)
        for i in range(len(x)):
            overlap[x[i], y[i]] += 1
        return overlap

    _label_overlap = _label_overlap_numba
except Exception:
    _label_overlap = _label_overlap_numpy


def label_overlap(x, y, check=True):
    if check:
        _check_label_array(x, 'x', True)
        _check_label_array(y, 'y', True)
        x.shape == y.shape or _raise(ValueError("x and y must have the same shape"))
    return _label_overlap(x, y)


def _safe_divide(x, y, eps=1e-10):
    if np.isscalar(x) and np.isscalar(y):
        return x / y if np.abs(y) > eps else 0.0
    else:
        out = np.zeros(np.broadcast(x, y).shape, np.float32)
        np.divide(x, y, out=out, where=np.abs(y) > eps)
        return out


def intersection_over_union(overlap):
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, (n_pixels_pred + n_pixels_true - overlap))


def relabel_sequential(label_field, offset=1):
    offset = int(offset)
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if np.min(label_field) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    max_label = int(label_field.max())
    if not np.issubdtype(label_field.dtype, np.integer):
        new_type = np.min_scalar_type(max_label)
        label_field = label_field.astype(new_type)
    labels = np.unique(label_field)
    labels0 = labels[labels != 0]
    new_max_label = offset - 1 + len(labels0)
    new_labels0 = np.arange(offset, new_max_label + 1)
    output_type = label_field.dtype
    required_type = np.min_scalar_type(new_max_label)
    if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
        output_type = required_type
    forward_map = np.zeros(max_label + 1, dtype=output_type)
    forward_map[labels0] = new_labels0
    inverse_map = np.zeros(new_max_label + 1, dtype=output_type)
    inverse_map[offset:] = labels0
    relabeled = forward_map[label_field]
    return relabeled, forward_map, inverse_map


def _matching(y_true, y_pred, thresh=0.5, criterion='iou'):
    """Returns (scores, map_rev_true, map_rev_pred) — the IoU table and label maps."""
    _check_label_array(y_true, 'y_true')
    _check_label_array(y_pred, 'y_pred')
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred have different shapes")
    if thresh is None:
        thresh = 0

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = intersection_over_union(overlap)
    return scores, map_rev_true, map_rev_pred


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def check_frame_pair(img_prev, img_curr, iou_thresh=0.75):
    """
    Compare two label frames and find segmentation inconsistencies.

    Parameters
    ----------
    img_prev : (Z,Y,X) uint label — registered previous frame
    img_curr : (Z,Y,X) uint label — current frame
    iou_thresh : IoU threshold passed to matching (default 0.75)

    Returns
    -------
    dict with keys:
        n_prev, n_curr              — cell counts
        daughter_with_two_mothers   — list of [daughter_id, [[mother_id, iou], ...]]
        cells_with_no_daughter      — list of cell ids
        cells_with_two_daughters    — list of cell ids
    """
    IoUTable, lab1, lab2 = _matching(img_prev, img_curr, thresh=iou_thresh)

    cell_list = list(lab1)
    next_list = list(lab2)

    DaughterToMother   = {}
    cells_no_daughter  = []
    cells_two_daughter = []

    for ilab in range(1, len(lab1)):          # skip 0 (background)
        ilab_id = cell_list[ilab]
        row     = IoUTable[ilab]

        if np.max(row) < 0.2:
            cells_no_daughter.append(int(ilab_id))
            continue

        daughter_id = int(next_list[np.argmax(row)])
        iou         = float(np.max(row))
        DaughterToMother.setdefault(daughter_id, []).append((int(ilab_id), iou))

        if np.sum(row[1:]) > 1.2:
            cells_two_daughter.append(int(ilab_id))

    daughter_with_two_mothers = [
        [d, mothers]
        for d, mothers in DaughterToMother.items()
        if len(mothers) > 1
    ]

    return {
        'n_prev':                   len(lab1) - 1,
        'n_curr':                   len(lab2) - 1,
        'daughter_with_two_mothers': daughter_with_two_mothers,
        'cells_with_no_daughter':   cells_no_daughter,
        'cells_with_two_daughters': cells_two_daughter,
    }


def summary_text(result: dict) -> str:
    lines = [
        f"Prev cells : {result['n_prev']}",
        f"Curr cells : {result['n_curr']}",
        f"Daughters with 2 mothers : {len(result['daughter_with_two_mothers'])}",
        f"Cells with no daughter   : {len(result['cells_with_no_daughter'])}",
        f"Cells with 2 daughters   : {len(result['cells_with_two_daughters'])}",
    ]
    if result['daughter_with_two_mothers']:
        lines.append("")
        lines.append("Daughters with 2 mothers:")
        for d_id, mothers in result['daughter_with_two_mothers']:
            mothers_str = ', '.join(f'{m}(iou={v:.2f})' for m, v in mothers)
            lines.append(f"  daughter {d_id} <- [{mothers_str}]")
    if result['cells_with_no_daughter']:
        lines.append(f"No daughter: {result['cells_with_no_daughter']}")
    if result['cells_with_two_daughters']:
        lines.append(f"Two daughters: {result['cells_with_two_daughters']}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Fix: smooth Gaussian-field split
# ---------------------------------------------------------------------------

def _smooth_split(lab_mother, mother_ids, d_idx, image_shape, sigma=10):
    pad   = int(3 * sigma) + 1
    z_min = max(0,              d_idx[0].min() - pad)
    z_max = min(image_shape[0], d_idx[0].max() + pad + 1)
    y_min = max(0,              d_idx[1].min() - pad)
    y_max = min(image_shape[1], d_idx[1].max() + pad + 1)
    x_min = max(0,              d_idx[2].min() - pad)
    x_max = min(image_shape[2], d_idx[2].max() + pad + 1)

    crop = lab_mother[z_min:z_max, y_min:y_max, x_min:x_max]
    blurred = np.stack([
        gaussian_filter((crop == mid).astype(np.float32), sigma=sigma)
        for mid in mother_ids
    ], axis=-1)

    lz = d_idx[0] - z_min
    ly = d_idx[1] - y_min
    lx = d_idx[2] - x_min
    return np.argmax(blurred[lz, ly, lx], axis=1)


def fix_seg_errors(lab_mother, lab_daughter, check_result, sigma=10):
    """
    Split daughters that have two mothers using Gaussian-blurred influence fields.

    Parameters
    ----------
    lab_mother    : (Z,Y,X) registered previous-frame labels
    lab_daughter  : (Z,Y,X) current-frame labels  (will be copied)
    check_result  : dict returned by check_frame_pair
    sigma         : Gaussian blur sigma for smooth boundary (default 10)

    Returns
    -------
    lab_daughter_fixed : (Z,Y,X) uint, same dtype as lab_daughter
    n_fixed : number of daughters that were split
    """
    lab_daughter = lab_daughter.copy()
    next_label   = int(lab_daughter.max()) + 1
    n_fixed      = 0

    for daughter_id, mothers_with_iou in check_result['daughter_with_two_mothers']:
        mother_ids = [m[0] for m in mothers_with_iou]
        d_idx = np.where(lab_daughter == daughter_id)
        if len(d_idx[0]) == 0:
            print(f'  daughter {daughter_id} not found — skipping')
            continue

        nearest = _smooth_split(lab_mother, mother_ids, d_idx,
                                lab_mother.shape, sigma=sigma)

        new_labels = [daughter_id] + list(range(next_label, next_label + len(mother_ids) - 1))
        next_label += len(mother_ids) - 1

        for k, new_lbl in enumerate(new_labels):
            vox = nearest == k
            lab_daughter[d_idx[0][vox], d_idx[1][vox], d_idx[2][vox]] = new_lbl
            print(f'  daughter {daughter_id} -> label {new_lbl}: {int(vox.sum())} voxels '
                  f'(mother {mother_ids[k]})')
        n_fixed += 1

    return lab_daughter, n_fixed
