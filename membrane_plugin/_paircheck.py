import numpy as np
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------
#
# Every cell on both sides of the pair is explained by one of:
#   prev (mother) side: exactly one significant daughter (normal), no
#     significant daughter at all (cells_with_no_daughter), or 2+ significant
#     daughters (a split -- see below for how that's further classified)
#   curr (daughter) side: exactly one significant mother (normal), no
#     significant mother at all (new_cells -- unexplained/new), or 2+
#     significant mothers (daughter_with_two_mothers -- a merge)
#
# "Significant" here means Io2 (intersection-over-prediction, i.e.
# intersection / that label's OWN volume) >= io2_thresh -- same metric this
# project's Io2Table/ComputeIo2TableFromTransforms.py use elsewhere, chosen
# over plain IoU because it correctly separates "this daughter is mostly
# explained by this mother" from "this mother happens to be huge", which
# plain IoU conflates.
#
# A split (mother with 2+ significant daughters) is further split into:
#   bad_splits -- one daughter's volume is close to the mother's own volume
#     (BAD_SPLIT_DOMINANT_RANGE) while every other daughter is either much
#     smaller (BAD_SPLIT_WEAK_VOL_RATIO) or weakly attached
#     (BAD_SPLIT_WEAK_IO2) -- the signature of a spurious fragment getting
#     cut off a cell that didn't actually divide, not a real division.
#   splits -- everything else (daughters comparably sized/attached -- reads
#     as a genuine division candidate).

IO2_THRESH               = 0.15
BAD_SPLIT_DOMINANT_RANGE = (0.8, 1.25)
BAD_SPLIT_WEAK_VOL_RATIO = 0.5
BAD_SPLIT_WEAK_IO2       = 0.35


def _label_overlap_matrix(img_prev, img_curr):
    """overlap[m, d] = voxel count where img_prev==m and img_curr==d (0 included)."""
    flat_prev = img_prev.ravel().astype(np.int64)
    flat_curr = img_curr.ravel().astype(np.int64)
    max_prev = int(flat_prev.max())
    max_curr = int(flat_curr.max())
    combined = flat_prev * (max_curr + 1) + flat_curr
    counts = np.bincount(combined, minlength=(max_prev + 1) * (max_curr + 1))
    return counts.reshape(max_prev + 1, max_curr + 1)


def check_frame_pair(img_prev, img_curr, io2_thresh=IO2_THRESH):
    """
    Compare two label frames and explain every cell on both sides.

    Parameters
    ----------
    img_prev : (Z,Y,X) uint label — registered previous frame
    img_curr : (Z,Y,X) uint label — current frame
    io2_thresh : minimum Io2 (intersection / that label's own volume) to
        count as a significant match (default 0.15)

    Returns
    -------
    dict with keys:
        n_prev, n_curr              — cell counts
        cells_with_no_daughter      — mother ids with no significant daughter
        new_cells                   — daughter ids with no significant mother
        daughter_with_two_mothers   — [daughter_id, [(mother_id, io2, mother_vol), ...]]
            (a merge: two prev cells both explain this one daughter)
        splits                      — [mother_id, [(daughter_id, io2, daughter_vol), ...]]
            (mother has 2+ significant daughters, roughly balanced — likely
            a real division)
        bad_splits                  — same shape as splits, but one daughter
            ~ the mother's own size and the rest are small/weak — likely a
            spurious fragment, not a real division
    """
    labels_prev = sorted(int(l) for l in np.unique(img_prev) if l != 0)
    labels_curr = sorted(int(l) for l in np.unique(img_curr) if l != 0)

    overlap = _label_overlap_matrix(img_prev, img_curr)

    vol_prev = {m: int(overlap[m, :].sum()) for m in labels_prev}
    vol_curr = {d: int(overlap[:, d].sum()) for d in labels_curr}

    # mother -> significant daughters: Io2 = overlap / daughter's own volume
    mother_daughters = {}
    for m in labels_prev:
        sig = []
        for d in labels_curr:
            ov = int(overlap[m, d])
            if ov == 0:
                continue
            io2 = ov / vol_curr[d]
            if io2 >= io2_thresh:
                sig.append((d, io2, vol_curr[d]))
        mother_daughters[m] = sorted(sig, key=lambda x: -x[1])

    # daughter -> significant mothers: Io2 = overlap / mother's own volume
    daughter_mothers = {}
    for d in labels_curr:
        sig = []
        for m in labels_prev:
            ov = int(overlap[m, d])
            if ov == 0:
                continue
            io2 = ov / vol_prev[m]
            if io2 >= io2_thresh:
                sig.append((m, io2, vol_prev[m]))
        daughter_mothers[d] = sorted(sig, key=lambda x: -x[1])

    cells_no_daughter = [m for m in labels_prev if not mother_daughters[m]]
    new_cells         = [d for d in labels_curr if not daughter_mothers[d]]

    daughter_with_two_mothers = [
        [d, mothers] for d, mothers in daughter_mothers.items() if len(mothers) > 1
    ]

    splits, bad_splits = [], []
    for m, daughters in mother_daughters.items():
        if len(daughters) < 2:
            continue
        mvol = vol_prev[m]
        dominant_ratio = (daughters[0][2] / mvol) if mvol else 0.0
        rest_weak = all(
            (dvol / mvol < BAD_SPLIT_WEAK_VOL_RATIO) or (io2 < BAD_SPLIT_WEAK_IO2)
            for _, io2, dvol in daughters[1:]
        )
        entry = [m, daughters]
        if BAD_SPLIT_DOMINANT_RANGE[0] <= dominant_ratio <= BAD_SPLIT_DOMINANT_RANGE[1] and rest_weak:
            bad_splits.append(entry)
        else:
            splits.append(entry)

    return {
        'n_prev':                   len(labels_prev),
        'n_curr':                   len(labels_curr),
        'cells_with_no_daughter':   cells_no_daughter,
        'new_cells':                new_cells,
        'daughter_with_two_mothers': daughter_with_two_mothers,
        'splits':                   splits,
        'bad_splits':               bad_splits,
    }


def summary_text(result: dict) -> str:
    lines = [
        f"Prev cells : {result['n_prev']}",
        f"Curr cells : {result['n_curr']}",
        f"No daughter (cell disappeared) : {len(result['cells_with_no_daughter'])}",
        f"New/unexplained cells          : {len(result['new_cells'])}",
        f"Merges (daughter <- 2+ mothers) : {len(result['daughter_with_two_mothers'])}",
        f"Splits (likely real division)   : {len(result['splits'])}",
        f"BAD splits (likely seg error)   : {len(result['bad_splits'])}",
    ]
    if result['bad_splits']:
        lines.append("")
        lines.append("BAD splits:")
        for m, daughters in result['bad_splits']:
            dstr = ', '.join(f'{d}(io2={io2:.2f},vol={vol})' for d, io2, vol in daughters)
            lines.append(f"  mother {m} -> {dstr}")
    if result['splits']:
        lines.append("")
        lines.append("Splits:")
        for m, daughters in result['splits']:
            dstr = ', '.join(f'{d}(io2={io2:.2f},vol={vol})' for d, io2, vol in daughters)
            lines.append(f"  mother {m} -> {dstr}")
    if result['daughter_with_two_mothers']:
        lines.append("")
        lines.append("Merges:")
        for d_id, mothers in result['daughter_with_two_mothers']:
            mothers_str = ', '.join(f'{m}(io2={io2:.2f},vol={vol})' for m, io2, vol in mothers)
            lines.append(f"  daughter {d_id} <- [{mothers_str}]")
    if result['cells_with_no_daughter']:
        lines.append(f"No daughter: {result['cells_with_no_daughter']}")
    if result['new_cells']:
        lines.append(f"New/unexplained: {result['new_cells']}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Fix: smooth Gaussian-field split (merges -- daughter_with_two_mothers)
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

    for daughter_id, mothers in check_result['daughter_with_two_mothers']:
        mother_ids = [m[0] for m in mothers]
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
