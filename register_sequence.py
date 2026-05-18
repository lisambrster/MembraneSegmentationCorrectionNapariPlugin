"""
register_sequence.py
--------------------
Optionally clean, then register each previous frame to its successor
across an entire membrane_segmentation directory.

For every consecutive pair (frame N-1, frame N) the script:
  1. Loads both label tiffs (preferring *_corrected* variants).
  2. Optionally runs prep (remove small objects) on each frame.
  3. Optionally runs remove_bad_slices on each frame.
  4. Skips the pair if <prev_stem>_registered.tif already exists.
  5. Runs multi-start rigid registration (prev → curr).
  6. Saves the warped previous frame as <prev_stem>_registered.tif
     in the same membrane_segmentation directory.

Usage
-----
python register_sequence.py /path/to/experiment [options]

The script expects:
    <path>/membrane_segmentation/   — label tiffs

Options
-------
  --start N            first frame number to process (default: auto)
  --end N              last frame number to process  (default: auto)
  --skip-existing      skip pairs whose registered file already exists (default)
  --no-skip-existing   re-run registration even if output file exists

  Cleaning (applied to every frame before registration):
  --min-volume N       remove connected components smaller than N voxels
  --min-slice N        zero out z-slices below N
  --max-slice N        zero out z-slices above N

  Registration:
  --max-nfev N         optimizer iteration budget per start (default: 2000)
  --n-starts N         number of random restarts (default: 12)
  --angle-perturb F    std-dev of rotation noise in radians (default: 0.1)
  --trans-perturb F    std-dev of translation noise in pixels (default: 5.0)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

from membrane_plugin._cleaning import prep, remove_bad_slices
from membrane_plugin._registration import (
    prep_array,
    register_multiscale,
    euler_angles_to_rotation_matrix,
    warp_to_original_space,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_frame(directory: Path, frame_num: int, prefer_corrected: bool = True) -> Path | None:
    candidates = []
    for f in sorted(directory.iterdir()):
        if f.suffix not in ('.tif', '.tiff'):
            continue
        if 'registered_to_' in f.name:
            continue
        nums = [int(m.group()) for m in re.finditer(r'\d+', f.name)]
        if frame_num in nums:
            candidates.append(f)
    if not candidates:
        return None
    corrected   = [f for f in candidates if 'corrected' in f.name.lower()]
    uncorrected = [f for f in candidates if 'corrected' not in f.name.lower()]
    if prefer_corrected:
        return corrected[0] if corrected else candidates[0]
    return uncorrected[0] if uncorrected else candidates[0]


def _all_frame_numbers(seg_dir: Path) -> list[int]:
    nums = set()
    for f in seg_dir.iterdir():
        if f.suffix not in ('.tif', '.tiff'):
            continue
        if 'registered_to_' in f.name:
            continue
        for m in re.finditer(r'\d+', f.name):
            nums.add(int(m.group()))
    return sorted(nums)


def _clean(img: np.ndarray, min_volume: int | None,
           min_slice: int | None, max_slice: int | None) -> np.ndarray:
    if min_volume is not None:
        img, n_ccs = prep(img, min_volume)
        print(f"    prep: {n_ccs} CCs found, removed objects < {min_volume} voxels",
              flush=True)
    if min_slice is not None or max_slice is not None:
        lo = min_slice if min_slice is not None else 0
        hi = max_slice if max_slice is not None else img.shape[0] - 1
        img = remove_bad_slices(img, lo, hi)
        print(f"    remove_bad_slices: kept z {lo}–{hi}", flush=True)
    return img


# ---------------------------------------------------------------------------
# Per-pair registration
# ---------------------------------------------------------------------------

def register_pair(prev_path: Path, curr_path: Path,
                  prev_orig_path: Path | None = None,
                  min_volume: int | None = None,
                  min_slice: int | None = None,
                  max_slice: int | None = None,
                  reg_res: int = 2,
                  max_nfev: int = 2000,
                  n_starts: int = 12,
                  angle_perturb: float = 0.1,
                  trans_perturb: float = 5.0) -> tuple[np.ndarray, np.ndarray | None]:
    """Register *prev_path* (corrected) to *curr_path* and return warped arrays.

    Returns (warped_corrected, warped_original_or_None).  If *prev_orig_path* is
    supplied and differs from *prev_path*, the same transform is also applied to
    the original (non-corrected) frame.
    """
    print(f"  Loading {prev_path.name}", flush=True)
    prev = tifffile.imread(str(prev_path)).astype(np.uint32)
    print(f"  Loading {curr_path.name}", flush=True)
    curr = tifffile.imread(str(curr_path)).astype(np.uint32)

    prev = _clean(prev, min_volume, min_slice, max_slice)
    curr = _clean(curr, min_volume, min_slice, max_slice)

    print("  Preparing boundary images…", flush=True)
    bnd_prev, orig_prev, _ = prep_array(prev, reg_res)
    bnd_curr, orig_curr, _ = prep_array(curr, reg_res)

    print("  Registering…", flush=True)
    params = register_multiscale(
        bnd_prev.astype(np.float32), bnd_curr.astype(np.float32), np.zeros(6),
        max_nfev=max_nfev,
        n_random_starts=n_starts,
        angle_perturb=angle_perturb,
        translation_perturb=trans_perturb,
    )
    R = euler_angles_to_rotation_matrix(params[:3])
    t = params[3:]
    print("  Warping corrected prev to original space…", flush=True)
    warped = warp_to_original_space(orig_prev, R, t, reg_res, orig_curr.shape)

    warped_orig = None
    if prev_orig_path is not None and prev_orig_path != prev_path:
        print(f"  Warping original (non-corrected) prev: {prev_orig_path.name}…", flush=True)
        orig_data = tifffile.imread(str(prev_orig_path)).astype(np.uint32)
        orig_data = _clean(orig_data, min_volume, min_slice, max_slice)
        _, orig_prev_uncorr, _ = prep_array(orig_data, reg_res)
        warped_orig = warp_to_original_space(orig_prev_uncorr, R, t, reg_res, orig_curr.shape)

    return warped, warped_orig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clean and register consecutive membrane segmentation frames."
    )
    parser.add_argument("path", type=Path,
                        help="Root experiment directory (parent of membrane_segmentation/)")
    parser.add_argument("--start", type=int, default=None,
                        help="First frame number to process")
    parser.add_argument("--end", type=int, default=None,
                        help="Last frame number to process")
    parser.add_argument("--skip-existing", dest="skip_existing",
                        action="store_true", default=True,
                        help="Skip pairs whose registered file already exists (default)")
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false",
                        help="Re-run registration even if output already exists")

    clean = parser.add_argument_group("cleaning (applied before registration)")
    clean.add_argument("--min-volume", type=int, default=None, metavar="N",
                       help="Remove connected components smaller than N voxels")
    clean.add_argument("--min-slice", type=int, default=None, metavar="N",
                       help="Zero out z-slices below N")
    clean.add_argument("--max-slice", type=int, default=None, metavar="N",
                       help="Zero out z-slices above N")

    reg = parser.add_argument_group("registration")
    reg.add_argument("--max-nfev", type=int, default=2000, metavar="N")
    reg.add_argument("--n-starts", type=int, default=12, metavar="N")
    reg.add_argument("--angle-perturb", type=float, default=0.1, metavar="F")
    reg.add_argument("--trans-perturb", type=float, default=5.0, metavar="F")

    args = parser.parse_args()

    seg_dir = args.path / "membrane_segmentation"
    if not seg_dir.is_dir():
        sys.exit(f"ERROR: {seg_dir} does not exist")

    all_frames = _all_frame_numbers(seg_dir)
    if not all_frames:
        sys.exit(f"ERROR: no tiff files found in {seg_dir}")

    start = args.start if args.start is not None else all_frames[0]
    end   = args.end   if args.end   is not None else all_frames[-1]
    frames = [n for n in all_frames if start <= n <= end]

    print(f"Found {len(all_frames)} frames in {seg_dir}")
    print(f"Processing frames {start}–{end}  ({len(frames)} frames, "
          f"{max(0, len(frames)-1)} pairs)")
    if args.min_volume:
        print(f"  prep: min_volume={args.min_volume}")
    if args.min_slice is not None or args.max_slice is not None:
        print(f"  remove_bad_slices: min={args.min_slice}  max={args.max_slice}")
    print()

    n_done = n_skipped = n_missing = 0

    for i in range(1, len(frames)):
        curr_num = frames[i]
        prev_num = frames[i - 1]

        curr_path = _find_frame(seg_dir, curr_num)
        prev_path = _find_frame(seg_dir, prev_num)          # corrected preferred
        prev_orig_path = _find_frame(seg_dir, prev_num, prefer_corrected=False)
        if prev_orig_path == prev_path:
            prev_orig_path = None  # no separate original; skip dual warp

        if curr_path is None or prev_path is None:
            print(f"[SKIP] frame {prev_num}→{curr_num}: file not found", flush=True)
            n_missing += 1
            continue

        out_path      = seg_dir / f"registered_to_{curr_num}{prev_path.suffix}"
        orig_out_path = seg_dir / f"original_registered_to_{curr_num}{prev_path.suffix}"

        if args.skip_existing and out_path.exists():
            print(f"[SKIP] frame {prev_num}→{curr_num}: {out_path.name} already exists",
                  flush=True)
            n_skipped += 1
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"Registering frame {prev_num} → frame {curr_num}", flush=True)
        if prev_orig_path:
            print(f"  (will also warp original: {prev_orig_path.name})", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            warped, warped_orig = register_pair(
                prev_path, curr_path,
                prev_orig_path=prev_orig_path,
                min_volume=args.min_volume,
                min_slice=args.min_slice,
                max_slice=args.max_slice,
                max_nfev=args.max_nfev,
                n_starts=args.n_starts,
                angle_perturb=args.angle_perturb,
                trans_perturb=args.trans_perturb,
            )
            tifffile.imwrite(str(out_path), warped)
            print(f"  Saved → {out_path.name}", flush=True)
            if warped_orig is not None:
                tifffile.imwrite(str(orig_out_path), warped_orig)
                print(f"  Saved → {orig_out_path.name}", flush=True)
            n_done += 1
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}", flush=True)
            traceback.print_exc()

    print(f"\nDone.  registered={n_done}  skipped={n_skipped}  missing={n_missing}")


if __name__ == "__main__":
    main()
