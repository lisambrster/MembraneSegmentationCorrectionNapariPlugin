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
3. Click **Register frame N-1 → frame N** to align the previous frame
4. Click **Check Membrane Pairs** to find inconsistencies
5. Click **Fix Seg Errors** if errors were found

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
