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

Two counts are held, not one. Keeping only the total monotonic lets the
polar body fall out the moment a cell divides: at frame 45 of 260612_stack1
the clean keeps 2 cells + the polar body = 3, and at frame 46 a division
makes 3 cells, which already meets a floor of 3, so nothing is relaxed and
the polar body is dropped. The total never decreased but its composition
flipped. So the cells (objects above the fragment cut) and the extras
(anything admitted below it) are each required not to decrease.

Nothing is invented. Candidates are admitted in order of how much we trust
them, and never below an absolute volume floor -- without that floor the
constraint will happily admit a 1 um^3 speck (11 voxels) to hold a count of
30. A frame that still cannot reach the required count is FLAGGED: the cell
is missing from the segmentation itself, and no threshold brings it back.
"""

import numpy as np

SHAPE_MIN_FLATNESS = 0.35
SHAPE_MAX_XYFRAC = 0.45


def monotonic_keep_sets(frames, load, mod, start_cells=2, min_cell_frac=0.10,
                        min_admit_um3=100.0, pb_max_frac=0.25, progress=None):
    """Decide which labels to keep in each frame.

    Two separate rules, because the two kinds of object behave differently:

      * CELLS -- objects above the fragment cut. Their count may not decrease;
        an embryo only gains cells, so a drop is a cleaning artefact. When a
        frame falls short, candidates are admitted largest-first.
      * The POLAR BODY -- an object far smaller than the median cell
        (pb_max_frac of it). It is kept whenever one is present, whatever the
        counts say, and nothing is expected when none is.

    The polar body is identified by SIZE, in each frame on its own. Following
    it from frame to frame would need the two frames registered, which is not
    available here. The cut is relative to the frame's own median cell, so it
    follows the cells down as they divide.

    The second rule is what stops the polar body vanishing at a division: at
    frame 45 of 260612_stack1 the clean keeps 2 cells + polar body = 3, and at
    frame 46 a division makes 3 cells, which already meets a floor of 3, so a
    rule that only counted the total would drop the polar body and call the
    sequence monotonic. It is also why the rule does not ratchet: this embryo
    has no polar-body-sized object at all after about frame 54, and demanding
    one would flag every later frame.

    Nothing is invented. Admission stops at min_admit_um3 -- without that
    floor the count constraint will admit a 1 um^3 speck (11 voxels) to hold
    a count of 30. A frame that still falls short is FLAGGED: the cell is
    missing from the segmentation itself.
    """
    running_cells = start_cells
    out = {}
    for n in frames:
        lab = load(n)
        if lab is None:
            continue
        raw = [int(x) for x in np.unique(lab) if x]

        _, rep = mod.clean_volume(lab, min_cell_frac=0.0)
        pool = rep.get('kept_labels', []) if rep['status'] == 'ok' else []
        need = max(start_cells, running_cells)

        row = dict(frame=n, raw=len(raw), pool=len(pool), need=need,
                   status=rep['status'], relaxed=False, flagged=False,
                   admitted=[], note='', cells=0, extras=0,
                   vol_um3=rep.get('embryo_vol_um3') or 0.0,
                   fill=rep.get('median_fill'))

        if rep['status'] != 'ok':
            keep = set(raw)
            row.update(flagged=True, note='no embryo - frame left as-is',
                       cells=len(raw))
        else:
            feats = mod.label_features(lab, lab.shape)
            vol = lambda l: feats[l]['vol_um3']
            med = float(np.median([vol(l) for l in pool])) if pool else 0.0
            default = [l for l in pool if vol(l) >= min_cell_frac * med]

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

            # 1. cells: hold the floor, admitting the least-doubtful extras
            cells = list(default)
            for l in sorted(tier, key=lambda l: (tier[l], -vol(l))):
                if len(cells) >= need:
                    break
                if vol(l) < min_admit_um3:
                    break
                cells.append(l)
                row['admitted'].append(dict(label=int(l), tier=tier[l],
                                            vol_um3=vol(l),
                                            flatness=feats[l]['flatness'],
                                            why='cell count'))
            if len(cells) < need:
                row['flagged'] = True
                row['note'] = (f'only {len(cells)} cells of {need} - one is missing '
                               f'from the segmentation')

            # 2. the polar body: keep any object far smaller than a cell
            med_cell = float(np.median([vol(l) for l in cells])) if cells else 0.0
            pb_cut = pb_max_frac * med_cell
            row['pb_cut_um3'] = pb_cut
            pb = [l for l in feats
                  if l not in cells and min_admit_um3 <= vol(l) < pb_cut]
            pb = sorted(pb, key=lambda l: -vol(l))[:1]     # at most one
            for l in pb:
                row['admitted'].append(dict(label=int(l), tier=tier.get(l, 0),
                                            vol_um3=vol(l),
                                            flatness=feats[l]['flatness'],
                                            why='polar body'))

            keep = set(cells) | set(pb)
            row['cells'] = len(cells)
            row['extras'] = len(pb)
            row['relaxed'] = bool(row['admitted'])
            if row['relaxed'] and not row['flagged']:
                why = ', '.join(sorted({a['why'] for a in row['admitted']}))
                row['note'] = f"{len(default)} -> {len(keep)} ({why})"

        row['keep'] = set(int(x) for x in keep)
        row['kept'] = len(row['keep'])
        running_cells = max(running_cells, row['cells'])
        row['running_cells'] = running_cells
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
    counts = [r['cells'] for r in order]
    drops = [(order[i]['frame'], counts[i - 1], counts[i])
             for i in range(1, len(counts)) if counts[i] < counts[i - 1]]
    relaxed = [r['frame'] for r in order if r['relaxed']]
    flagged = [r['frame'] for r in order if r['flagged']]

    lines = [f"{'frame':>6} {'raw':>4} {'cells':>6} {'extra':>6} {'kept':>5}  note"]
    for r in order:
        mark = ' FLAG' if r['flagged'] else ''
        lines.append(f"{r['frame']:>6} {r['raw']:>4} {r.get('cells','-'):>6} "
                     f"{r.get('extras','-'):>6} {r['kept']:>5}  {r['note']}{mark}")
    n_pb = sum(1 for r in order if r['extras'])
    head = (f"{len(order)} frames: {counts[0]} -> {counts[-1]} cells "
            f"(+polar body in {n_pb}), {len(relaxed)} relaxed, {len(flagged)} flagged")
    if drops:
        head += f"\nstill decreasing at {drops} - those frames are short"
    else:
        head += "\ncell count monotonically non-decreasing throughout"
    return head, '\n'.join(lines), dict(drops=drops, relaxed=relaxed, flagged=flagged)
