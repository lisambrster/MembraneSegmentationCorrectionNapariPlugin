# Membrane Segmentation Correction

A napari plugin for correcting 3D membrane segmentation label images across time frames.

## Features

- Load a frame and its previous frame from a structured experiment directory
- Rigid 3D registration of the previous frame to the current frame (multi-start optimisation)
- Clean label images: remove small objects, remove bad Z-slices
- Check membrane pairs for segmentation inconsistencies (cells with two mothers / no daughter)
- Fix segmentation errors using Gaussian-blurred influence fields
- Saves registered frames to disk for reuse across sessions
- Batch registration script for processing entire sequences

## Expected directory structure

```
<experiment>/
    membrane_segmentation/     # label tiffs (e.g. seg_146.tif, seg_146_corrected.tif)
    Raw_data/
        MemNucCombo/           # intensity tiffs (preferred if present)
        membrane/              # fallback intensity tiffs
```

## Installation

### Step 1 — install napari with Qt bindings

napari requires a Qt backend (PyQt6 recommended).  Choose **one** of:

```bash
# conda / miniforge (recommended)
conda install -c conda-forge napari pyqt6

# pip
pip install "napari[pyqt6]"
```

### Step 2 — install this plugin

```bash
git clone <repo-url>
cd MembraneSegmentationCorrectionNapariPlugin
pip install -e .
```

To update an existing install after `git pull`, always re-run `pip install -e .` — entry points are registered at install time and won't update otherwise.

Then launch napari and find **Membrane Segmentation Correction** under *Plugins*.

### Optional: faster label overlap (recommended)

```bash
pip install numba
```

## Usage in napari

1. Enter the experiment root path and frame number, click **Load**
2. Click **Remove Bad Slices** (manual or auto) and **Prep** to clean labels if needed
3. To delete one bad cell, click **Pick** (this puts the layer into napari's pick mode), then click the cell in the viewer — its id appears in the **Remove label id** box — then click **Remove**. You can also type an id directly. **Undo remove** restores the last one. Surviving labels keep their ids.
4. **Prep ALL frames** applies the Min volume threshold to every frame on disk, writing each result as a **new version beside its original** (`Corrected` → `Corrected1`) so **Load** opens the cleaned frames straight away while the originals survive. The whole sequence lands on one suffix, so a single `membrane_label_path` pattern still matches. Tick **overwrite originals** to replace in place instead (not undoable). Progress is printed per frame.
5. **Save frame (new version)** writes the active layer beside its source with the version bumped — `Corrected` → `Corrected1` → `Corrected2`, `corrected_4` → `corrected_5`. The original is never overwritten, and the frame number is never touched.
6. **Save ALL frames (new version)** versions the whole sequence at once, every frame landing on the *same* suffix so a single `membrane_label_path` pattern still matches. The loaded frame is written from memory (including edits); the rest are carried across from disk.
7. **Clean ALL frames (shape + connectivity)** runs `CleanEmbryoSegmentation.clean_volume` over every frame — drops flat/sprawling objects, keeps the largest connected group — and writes the result exactly as Prep ALL does, so Load opens the cleaned frames. A frame with no identifiable embryo is **carried through unchanged** rather than blanked or skipped, and listed in the report; the per-frame table shows labels in/out, embryo volume and median fill.
8. **Clean ALL (monotonic, no save)** cleans every frame with the cell count constrained so it never decreases (starting at the **start at** value). The polar body trips both the fragment cut and the flatness filter inconsistently — one or two z-slices thick, its flatness swings between 0.10 and 0.99 frame to frame — so no per-frame threshold separates it, but the sequence can: when the count would drop, the object that holds it level is put back. Candidates are admitted largest-first and never below 100 µm³, so the constraint cannot invent a cell; a frame that still falls short is FLAGGED, meaning the cell is missing from the segmentation itself. The decision is cached per frame, so loading a frame afterwards shows the constrained result. Reads the files on disk, writes nothing.
9. Click **Cells per frame** to count labels in every frame and flag any drop (cell count should only increase).
10. Click **Load Final Registration Transforms** to load an exported `RegisteredTransforms_*.json`, then **Register frame N-1 → frame N** to warp the previous frame onto the current one using it. Registration is only ever applied here, never searched for — the transforms are computed on the cluster, and a second implementation in the plugin could only disagree with them. (`register_sequence.py` still has the optimiser if you need it.)
11. **Update Io2 table (this frame)** recomputes just the pairs the loaded frame takes part in — correcting frame F invalidates pairs F-1 and F and nothing else — and merges them into `Io2Table_<transforms name>.json` beside the transforms, keeping a timestamped backup. The current frame is read from the viewer, so unsaved edits count.
12. Click **Check Membrane Pairs** to find inconsistencies
13. Click **Fix Seg Errors** if errors were found

## Batch registration

Register all consecutive frame pairs in a sequence:

```bash
register-sequence /path/to/experiment

# with cleaning options:
register-sequence /path/to/experiment --min-volume 500 --min-slice 5 --max-slice 60

# specific frame range:
register-sequence /path/to/experiment --start 140 --end 160
```

## Dependencies

- [napari](https://napari.org)
- numpy, scipy, scikit-image, tifffile, qtpy
- numba *(optional)*
