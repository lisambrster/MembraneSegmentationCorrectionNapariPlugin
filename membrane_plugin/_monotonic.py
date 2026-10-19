"""Cleaning constrained so the cell count never decreases along a sequence.

clean_volume() judges each frame on its own, and two of its filters remove
objects that the sequence says are real:

  * the fragment cut drops anything below min_cell_frac of the frame's median
    cell, and
  * the flatness filter drops anything too thin in z to be a cell.

The polar body trips both, inconsistently. In 260612_stack1 it measures
500-4,100 um^3 against ~95,000 um^3 blastomeres, and one or two z-slices
thick its flatness swings between 0.10 and 0.99 frame to frame -- so the same
object reads as a cell in one frame and a background slab in the next, and no
per-frame threshold separates them. The sequence can: an embryo only gains
cells, so a decrease is a cleaning artefact, and the object that holds the
count level is the one to put back.

Nothing is invented. Candidates are admitted in order of how much we trust
them, and never below an absolute volume floor -- without that floor the
constraint will happily admit a 1 um^3 speck (11 voxels) to hold a count of
30. A frame that still cannot reach the required count is FLAGGED: the cell
is missing from the segmentation itself, and no threshold brings it back.
"""

import numpy as np

SHAPE_MIN_FLATNESS = 0.35
SHAPE_MAX_XYFRAC = 0.45


def monotonic_keep_sets(frames, load, mod, start_count=3, min_cell_frac=0.10,
                        min_admit_um3=100.0, progress=None):
    """Decide which labels to keep in each frame, count never decreasing.

    `frames` are frame numbers in order, `load(frame)` returns that frame's
    label volume, `mod` is the CleanEmbryoSegmentation module. `progress`, if
    given, is called with each row as it is computed.

    Returns {frame: row}, where row['keep'] is the set of label ids to keep.
    """
    running = start_count
    out = {}
    for n in frames:
        lab = load(n)
        if lab is None:
            continue
        raw = [int(x) for x in np.unique(lab) if x]

        # one pass with the fragment cut disabled: everything that passed
        # shape AND connectivity, which is the pool to choose from
        _, rep = mod.clean_volume(lab, min_cell_frac=0.0)
        pool = rep.get('kept_labels', []) if rep['status'] == 'ok' else []
        need = max(start_count, running)

        row = dict(frame=n, raw=len(raw), pool=len(pool), need=need,
                   status=rep['status'], relaxed=False, flagged=False,
                   admitted=[], note='',
                   vol_um3=rep.get('embryo_vol_um3') or 0.0,
                   fill=rep.get('median_fill'))

        if rep['status'] != 'ok':
            # no embryo found: carry the frame through rather than blank it,
            # because clean_volume returns an EMPTY volume in this case
            keep = set(raw)
            row.update(flagged=True, note='no embryo - frame left as-is')
        else:
            feats = mod.label_features(lab, lab.shape)
            vol = lambda l: feats[l]['vol_um3']
            med = float(np.median([vol(l) for l in pool])) if pool else 0.0
            default = [l for l in pool if vol(l) >= min_cell_frac * med]

            # tiers: 1 in the embryo group but under the fragment cut,
            #        2 passes shape but outside the group,
            #        3 rejected by the shape filter as well (the polar body)
            shape_ok = [l for l, f in feats.items()
                        if f['flatness'] >= SHAPE_MIN_FLATNESS
                        and f['xyfrac'] <= SHAPE_MAX_XYFRAC]
            tier = {}
            for l in pool:
                if l not in default:
                    tier[l] = 1
            for l in shape_ok:
                if l not in pool:
                    tier[l] = 2
            for l in feats:
                if l not in default and l not in tier:
                    tier[l] = 3

            keep = list(default)
            for l in sorted(tier, key=lambda l: (tier[l], -vol(l))):
                if len(keep) >= need:
                    break
                if vol(l) < min_admit_um3:
                    break          # holding the count is not worth a fake cell
                keep.append(l)
                row['admitted'].append(dict(label=int(l), tier=tier[l],
                                            vol_um3=vol(l),
                                            flatness=feats[l]['flatness']))
            if row['admitted']:
                row['relaxed'] = True
                row['note'] = (f"{len(default)} -> {len(keep)} by admitting "
                               f"{len(row['admitted'])}")
            if len(keep) < need:
                row['flagged'] = True
                row['note'] = (f'only {len(keep)} of {need} - a cell is missing '
                               f'from the segmentation')
            keep = set(keep)

        row['keep'] = set(int(x) for x in keep)
        row['kept'] = len(row['keep'])
        running = max(running, row['kept'])
        row['running'] = running
        out[n] = row
        if progress is not None:
            progress(row)
    return out


def apply_keep(lab, keep):
    """Zero every label not in `keep`, leaving the surviving ids unchanged."""
    if not keep:
        return np.zeros_like(lab)
    lut = np.zeros(int(lab.max()) + 1, dtype=lab.dtype)
    idx = np.asarray(sorted(int(k) for k in keep), dtype=np.intp)
    idx = idx[idx <= int(lab.max())]
    lut[idx] = idx.astype(lab.dtype)
    return lut[lab]


def summarise(rows):
    """One-line summary plus a per-frame table for the results box."""
    order = [rows[k] for k in sorted(rows)]
    counts = [r['kept'] for r in order]
    drops = [(order[i]['frame'], counts[i - 1], counts[i])
             for i in range(1, len(counts)) if counts[i] < counts[i - 1]]
    relaxed = [r['frame'] for r in order if r['relaxed']]
    flagged = [r['frame'] for r in order if r['flagged']]

    lines = [f"{'frame':>6} {'raw':>4} {'pool':>5} {'kept':>5}  note"]
    for r in order:
        mark = ' FLAG' if r['flagged'] else ''
        lines.append(f"{r['frame']:>6} {r['raw']:>4} {r['pool']:>5} "
                     f"{r['kept']:>5}  {r['note']}{mark}")
    head = (f"{len(order)} frames: {counts[0]} -> {counts[-1]} cells, "
            f"{len(relaxed)} relaxed, {len(flagged)} flagged")
    if drops:
        head += f"\nstill decreasing at {drops} - those frames are short"
    else:
        head += "\nmonotonically non-decreasing throughout"
    return head, '\n'.join(lines), dict(drops=drops, relaxed=relaxed, flagged=flagged)
