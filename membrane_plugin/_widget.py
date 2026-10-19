import json
import os
import shutil
import re
import numpy as np
import tifffile
from pathlib import Path

import threading
from qtpy.QtCore import Qt, QObject, Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QSpinBox, QToolButton, QSizePolicy,
    QDoubleSpinBox, QPlainTextEdit, QLineEdit, QCheckBox, QScrollArea,
)
import napari
from napari.qt import thread_worker
from membrane_plugin._cleaning import prep, remove_bad_slices, auto_slice_range
from membrane_plugin._registration import warp_to_original_space
from membrane_plugin._paircheck import check_frame_pair, fix_seg_errors, summary_text
from membrane_plugin._monotonic import monotonic_keep_sets, apply_keep, summarise

_CONFIG = Path.home() / ".membrane_plugin.json"

def _load_config() -> dict:
    try:
        return json.loads(_CONFIG.read_text())
    except Exception:
        return {}

def _save_config(data: dict):
    _CONFIG.write_text(json.dumps(data))


def _version_of(path: Path) -> int:
    """Trailing version number of a name: Corrected1 -> 1, Corrected -> 0."""
    m = re.match(r'^.*?(\d+)$', path.stem.split('.')[-1])
    return int(m.group(1)) if m else 0


def _frame_token(path: Path):
    """The FIRST integer in the name - the Cam_long_%05d frame token."""
    m = re.search(r'\d+', path.name)
    return int(m.group()) if m else None


def _find_frame(directory: Path, frame_num: int, prefer_corrected: bool = True) -> Path | None:
    """Return the best tiff in *directory* whose name contains *frame_num*.

    Looks inside the 'membrane_segmentation' subdirectory of *directory* if it
    exists, otherwise uses *directory* directly.  When *prefer_corrected* is
    True (default), a 'corrected' variant is returned over a plain one;
    when False, the plain (non-corrected) variant is preferred.

    The frame is matched against the FIRST integer in the name, not any
    integer anywhere in it. Matching any integer made a suffix collide with a
    frame number: with 'crop_seg_corrected_4' names, frame 4 resolved to
    Cam_long_00000's file, and the versioned names this plugin now writes
    (Corrected1, Corrected2, ...) would collide the same way for frames 1, 2.

    Among the candidates for one frame, Load takes the MOST RECENTLY CORRECTED
    one, ranked as:

      1. fewer dot-separated parts -- keeps side files like
         '...corrected_4.pre20261004.tif' from beating the segmentation itself;
      2. newest mtime -- "most recently corrected", and the only signal that
         works across inconsistent suffixes (Corrected1 vs corrected_4 are not
         comparable as version numbers);
      3. highest version -- the tiebreak when mtimes match, which is the normal
         case for a bulk transfer where every file lands in the same second;
      4. name, so the choice is deterministic.
    """
    seg_dir = directory / "membrane_segmentation"
    search_dir = seg_dir if seg_dir.is_dir() else directory

    def _usable(f):
        return f.suffix in ('.tif', '.tiff') and 'registered_to_' not in f.name

    files = [f for f in sorted(search_dir.iterdir()) if _usable(f)]
    candidates = [f for f in files if _frame_token(f) == frame_num]
    if not candidates:
        # permissive fallback for names that do not lead with the frame token
        candidates = [f for f in files
                      if frame_num in [int(m.group()) for m in re.finditer(r'\d+', f.name)]]
    if not candidates:
        return None

    def rank(f):
        try:
            mtime = int(f.stat().st_mtime)      # whole seconds: a bulk copy
        except OSError:                          # lands the set in one bucket,
            mtime = 0                            # leaving version to decide
        return (len(f.stem.split('.')), -mtime, -_version_of(f), f.name)

    corrected   = sorted((f for f in candidates if 'corrected' in f.name.lower()), key=rank)
    uncorrected = sorted((f for f in candidates if 'corrected' not in f.name.lower()), key=rank)
    ordered     = sorted(candidates, key=rank)
    if prefer_corrected:
        return corrected[0] if corrected else ordered[0]
    return uncorrected[0] if uncorrected else ordered[0]


# Native (raw tiff) voxel pitch, in microns, numpy (Z, Y, X) order -- must
# match the pitch the exported final-transforms JSON and this plugin's own
# reg_res=2 search grid were both built from (see BuildNuclearSequenceWith
# MembraneFallback.py / _registration.py's prep_array/warp_to_original_space).
_RES_XY          = 0.208
_RES_Z_MEMBRANE  = 2.0
_PLUGIN_REG_RES  = 2  # matches the fixed _REG_RES used by _run_registration


def _final_transform_pitch_zyx(reg_res=_PLUGIN_REG_RES):
    """Physical size (microns) of one pixel in the plugin's own reg_res-zoomed
    search grid -- prep_array zooms the native array by (reg_res, reg_res/10,
    reg_res/10), so one zoomed-grid step covers 1/zoom_factor native pixels,
    i.e. (RES_Z_MEMBRANE/reg_res, RES_XY/(reg_res/10), RES_XY/(reg_res/10))."""
    return np.array([
        _RES_Z_MEMBRANE / reg_res,
        _RES_XY / (reg_res / 10.0),
        _RES_XY / (reg_res / 10.0),
    ])


def pairwise_transform_from_final(final_transforms: dict, prev_num: int, frame_num: int,
                                   reg_res=_PLUGIN_REG_RES):
    """Convert the exported cumulative (anchor-relative) transforms for frame
    *prev_num* and *frame_num* -- both expressed on the shared isotropic grid
    at final_transforms['resolution_um'] -- into a single pairwise (matrix,
    offset) that operates directly on the plugin's own reg_res-zoomed grid
    units, ready to hand to warp_to_original_space() so the result lands
    directly on frame_num's native anisotropic array (no isotropic
    resampling in the output).

    Returns None if either frame is missing from final_transforms.
    """
    frames = final_transforms.get('frames', {})
    fp, fc = frames.get(str(prev_num)), frames.get(str(frame_num))
    if fp is None or fc is None:
        return None

    iso_res = float(final_transforms.get('resolution_um', 0.832))
    pitch_reg = _final_transform_pitch_zyx(reg_res)
    D     = np.diag(pitch_reg / iso_res)   # reg-grid coord -> iso-grid coord
    D_inv = np.diag(iso_res / pitch_reg)   # iso-grid coord -> reg-grid coord

    matrix_prev = np.array(fp['matrix'], dtype=np.float64)
    offset_prev = np.array(fp['offset'], dtype=np.float64)
    matrix_curr = np.array(fc['matrix'], dtype=np.float64)
    offset_curr = np.array(fc['offset'], dtype=np.float64)

    # Both frames' cumulative transforms map their own iso-grid coords into
    # the shared anchor's iso-grid coords (x_f = matrix_f @ y_anchor +
    # offset_f); compose curr's inverse with prev's forward transform to get
    # the direct prev<-curr mapping, still in iso-grid units.
    matrix_curr_inv = np.linalg.inv(matrix_curr)
    M_iso = matrix_prev @ matrix_curr_inv
    b_iso = offset_prev - M_iso @ offset_curr

    M_reg = D_inv @ M_iso @ D
    b_reg = D_inv @ b_iso
    return M_reg, b_reg


# CleanEmbryoSegmentation.py is the shape+connectivity cleaner used for the
# cluster runs (flatness, xy-fraction, largest connected group, fragment drop).
# It lives outside this plugin package, so it is loaded from a file rather than
# imported, keeping ONE copy of the algorithm instead of a vendored fork that
# drifts. Its PZ/PXY (2.0, 0.208) match the native segmentations this plugin
# loads, so its thresholds apply unchanged.
_CLEAN_EMBRYO = None


def _clean_embryo_module():
    """Import CleanEmbryoSegmentation.py from the first place it is found."""
    global _CLEAN_EMBRYO
    if _CLEAN_EMBRYO is not None:
        return _CLEAN_EMBRYO

    import importlib.util, os
    here = Path(__file__).resolve()
    candidates = [
        os.environ.get('CLEAN_EMBRYO_PATH'),
        here.parent / 'CleanEmbryoSegmentation.py',
        here.parents[3] / 'Membrane' / 'CleanEmbryoSegmentation.py',
        Path.home() / 'Documents/projects/Mouse/Membrane/CleanEmbryoSegmentation.py',
    ]
    for c in candidates:
        if not c:
            continue
        c = Path(c)
        if not c.is_file():
            continue
        spec = importlib.util.spec_from_file_location('CleanEmbryoSegmentation', c)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # safe: its main() is __main__-guarded
        _CLEAN_EMBRYO = mod
        return mod
    return None


def _write_label_tiff(path, data, verify=True):
    """Write a label volume the way the rest of this project writes them.

    deflate-compressed, uint8 with the glasbey palette so FIJI shows colour
    and napari reads indices. Compression is not cosmetic here: these volumes
    are (101, 816, 788) uint32 = 260 MB raw against ~400 KB deflated, so an
    uncompressed sequence is ~39 GB and fills the disk. When it did fill,
    every later imwrite left an 8-byte TIFF header behind, and because the
    newest name wins at Load those stubs then shadowed the good files.

    Hence also the temp-file-then-rename: a failed or truncated write never
    replaces or shadows anything.

    Label VALUES are preserved whenever they fit in uint8. The Io2 table and
    the transforms are keyed by label id, so renumbering silently invalidates
    them; it happens only when ids exceed 255 while at most 255 are present,
    and is reported.
    """
    data = np.asarray(data)
    ids  = [int(v) for v in np.unique(data) if v]
    mx   = max(ids) if ids else 0
    note = ''

    mod = _clean_embryo_module()
    lut = mod.glasbey_dark_lut(256) if mod is not None else None

    if lut is not None and mx <= 255:
        out, palette = data.astype(np.uint8), True
    elif lut is not None and len(ids) <= 255:
        out = np.zeros(data.shape, np.uint8)
        for new, old in enumerate(ids, start=1):
            out[data == old] = new
        palette = True
        note = f' (relabelled 1..{len(ids)}; ids reached {mx} > 255)'
    else:
        dt = np.uint16 if mx <= 65535 else np.uint32
        out, palette = data.astype(dt), False
        note = (' (no palette: >255 labels)' if lut is not None
                else ' (no palette: CleanEmbryoSegmentation.py not found)')

    path = Path(path)
    tmp  = path.with_name(path.name + '.part')
    try:
        if palette:
            tifffile.imwrite(str(tmp), out, photometric='palette',
                             colormap=(lut.astype(np.uint16) * 257),
                             compression='deflate')
        else:
            tifffile.imwrite(str(tmp), out, compression='deflate')
        if verify:
            back = tifffile.imread(str(tmp))
            if back.shape != out.shape or not np.array_equal(back, out):
                raise RuntimeError(f'verification failed for {path.name}')
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return f'{len(ids)} labels, {out.dtype}, deflate{note}'


def _version_split(path: Path):
    """Break *path* into (leading parts, base, next version number).

    The target name is '.'.join(parts + [f'{base}{n}']) + suffix. Only the
    LAST dot-separated component of the stem is versioned, so the frame number
    in `Cam_long_00005` is never bumped -- that would rename a frame onto its
    neighbour. A name with no such component gains a 'Corrected' one.
    """
    parts, suffix = path.stem.split('.'), path.suffix
    if len(parts) == 1:
        return parts, 'Corrected', 1
    m = re.match(r'^(.*?)(\d+)$', parts[-1])
    if m:
        return parts[:-1], m.group(1), int(m.group(2)) + 1
    return parts[:-1], parts[-1], 1


def _version_name(path: Path, parts, base, n) -> Path:
    return path.with_name('.'.join(list(parts) + [f'{base}{n}']) + path.suffix)


def _next_version_path(path: Path) -> Path:
    """Next free versioned name for *path*, in the same directory.

        Cam_long_00005.crop_seg_Corrected.tif   -> ...crop_seg_Corrected1.tif
        Cam_long_00005.crop_seg_Corrected1.tif  -> ...crop_seg_Corrected2.tif
        Cam_long_00005.crop_seg_corrected_4.tif -> ...crop_seg_corrected_5.tif

    Bumps again while the target exists, so nothing is overwritten.
    """
    parts, base, n = _version_split(path)
    cand = _version_name(path, parts, base, n)
    while cand.exists():
        n += 1
        cand = _version_name(path, parts, base, n)
    return cand


def _prev_file(path: Path) -> Path | None:
    """Find a tiff in the same directory whose name contains the last number in path.name - 1."""
    matches = list(re.finditer(r'\d+', path.name))
    if not matches:
        return None
    m = matches[-1]
    prev_num = int(m.group()) - 1
    if prev_num < 0:
        return None
    return _find_frame(path.parent, prev_num)


class _WorkerSignals(QObject):
    """Carries done/error signals from a plain thread back to the Qt main thread."""
    done  = Signal(object)
    error = Signal(object)


class _CollapsiblePanel(QWidget):
    """A toggle-button header that shows/hides a content widget."""

    def __init__(self, title: str, html: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toggle = QToolButton()
        self._toggle.setText(f"▶  {title}")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle.setStyleSheet("QToolButton { text-align: left; font-weight: bold; }")
        self._toggle.toggled.connect(self._on_toggle)

        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(12, 4, 4, 4)
        lbl = QLabel(html)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.RichText)
        body_layout.addWidget(lbl)
        self._body.setVisible(False)

        layout.addWidget(self._toggle)
        layout.addWidget(self._body)

    def _on_toggle(self, checked: bool):
        self._toggle.setText(
            f"{'▼' if checked else '▶'}  {self._toggle.text()[2:]}"
        )
        self._body.setVisible(checked)


class TiffReaderWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self._config = _load_config()

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        self.label = QLabel("No frame loaded")
        self.label.setWordWrap(True)

        # --- Frame loading ---
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path:"))
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("/path/to/label/directory")
        self._path_edit.setText(self._config.get("last_dir", ""))
        path_row.addWidget(self._path_edit)
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(28)
        btn_browse.clicked.connect(self._browse)
        path_row.addWidget(btn_browse)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 99999)
        self._frame_spin.setValue(self._config.get("last_frame", 0))
        frame_row.addWidget(self._frame_spin)
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self._load_frame)
        frame_row.addWidget(btn_load)

        btn_count = QPushButton("Count labels")
        btn_count.clicked.connect(self._count_labels)
        frame_row.addWidget(btn_count)

        # --- Final registration TRANSFORMS (used by Register button below) ---
        final_xform_row = QHBoxLayout()
        final_xform_row.addWidget(QLabel("Final transforms JSON:"))
        self._final_xform_path_edit = QLineEdit()
        self._final_xform_path_edit.setPlaceholderText("/path/to/RegisteredTransforms_iso0p832um_0to173.json")
        self._final_xform_path_edit.setText(self._config.get(
            "final_xform_path",
            "/Users/lbrown/Documents/projects/Mouse/Membrane/260206_stack2/RegFrameSeg/Registration/RegisteredTransforms_iso0p832um_0to173.json",
        ))
        final_xform_row.addWidget(self._final_xform_path_edit)
        btn_final_xform_browse = QPushButton("…")
        btn_final_xform_browse.setFixedWidth(28)
        btn_final_xform_browse.clicked.connect(self._browse_final_xform)
        final_xform_row.addWidget(btn_final_xform_browse)

        btn_load_final_xform = QPushButton("Load Final Registration Transforms")
        btn_load_final_xform.clicked.connect(self._load_final_transforms)

        self._btn_res = QPushButton("Set resolution (10, 1, 1)")
        self._btn_res.clicked.connect(self._toggle_resolution)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

        # --- Cleaning ---
        min_vol_row = QHBoxLayout()
        min_vol_row.addWidget(QLabel("Min volume:"))
        self._min_vol = QSpinBox()
        self._min_vol.setRange(1, 10_000_000)
        self._min_vol.setValue(500)
        self._min_vol.setSingleStep(100)
        min_vol_row.addWidget(self._min_vol)

        btn_prep = QPushButton("Prep  (remove small objects)")
        btn_prep.clicked.connect(self._run_prep)

        # Cleaning is applied WHEN A FRAME IS LOADED, in memory. Nothing is
        # written until Save is pressed: a batch that cleans every frame to
        # disk writes a whole sequence nobody asked for (and, uncompressed,
        # filled the disk once already).
        prep_all_row = QHBoxLayout()
        prep_all_row.addWidget(QLabel("On load:"))
        self._chk_prep_on_load = QCheckBox("prep")
        self._chk_prep_on_load.setToolTip(
            "Apply Prep (remove objects under Min volume) to each frame as it\n"
            "is loaded. In memory only - press Save to keep it."
        )
        prep_all_row.addWidget(self._chk_prep_on_load)
        self._chk_clean_on_load = QCheckBox("clean embryo")
        self._chk_clean_on_load.setToolTip(
            "Apply CleanEmbryoSegmentation.clean_volume (shape + connectivity)\n"
            "to each frame as it is loaded. In memory only - press Save to keep it.\n"
            "A frame with no identifiable embryo is left as-is, never blanked."
        )
        prep_all_row.addWidget(self._chk_clean_on_load)
        self._chk_prep_all_overwrite = QCheckBox("Save overwrites originals")
        self._chk_prep_all_overwrite.setToolTip(
            "Off: Save writes a new version beside the original (Corrected ->\n"
            "Corrected1), which is what Load then opens.\n"
            "On: Save overwrites in place - cannot be undone."
        )
        prep_all_row.addWidget(self._chk_prep_all_overwrite)

        # Save a new version rather than overwriting: Corrected -> Corrected1
        # -> Corrected2. Save ALL versions the whole sequence in one step, so
        # the config's single membrane_label_path pattern still matches every
        # frame (a per-frame suffix would break it).
        save_row = QHBoxLayout()
        self._btn_save_curr = QPushButton("Save frame (new version)")
        self._btn_save_curr.clicked.connect(self._save_current_version)
        save_row.addWidget(self._btn_save_curr)
        self._btn_save_all = QPushButton("Save ALL (latest, full label images)")
        self._btn_save_all.setToolTip(
            "Write every frame as a new version: 8-bit, glasbey palette,\n"
            "deflate-compressed, label ids preserved.\n"
            "Takes the LATEST state of each frame - unsaved edits from any\n"
            "open layer, and the On-load cleaning for the rest."
        )
        self._btn_save_all.clicked.connect(self._save_all_versions)
        save_row.addWidget(self._btn_save_all)

        tools_row = QHBoxLayout()
        self._btn_clean_all = QPushButton("Clean ALL (monotonic, no save)")
        self._btn_clean_all.setToolTip(
            "Clean every frame with the count constrained so it never drops.\n"
            "Decides each frame once and caches it, so loading a frame after\n"
            "this shows the constrained result. Nothing is written."
        )
        self._btn_clean_all.clicked.connect(self._run_clean_all_monotonic)
        tools_row.addWidget(self._btn_clean_all)
        tools_row.addWidget(QLabel("start at"))
        self._start_count = QSpinBox()
        self._start_count.setRange(1, 1000)
        self._start_count.setValue(3)
        self._start_count.setFixedWidth(55)
        self._start_count.setToolTip("Cell count the sequence must start at")
        tools_row.addWidget(self._start_count)
        self._btn_prep_all = QPushButton("Prep ALL (no save)")
        self._btn_prep_all.setToolTip(
            "Apply Prep to every frame and report the per-frame cell counts.\n"
            "Nothing is written: the cleaning is applied in memory, and stays\n"
            "on for every frame you load afterwards. Save when you want it kept."
        )
        self._btn_prep_all.clicked.connect(self._run_prep_all_preview)
        tools_row.addWidget(self._btn_prep_all)
        self._btn_cell_counts = QPushButton("Cells per frame")
        self._btn_cell_counts.setToolTip(
            "Count labels in every frame and flag where the count drops.\n"
            "Applies the same On-load cleaning, so the numbers match what\n"
            "you see - nothing is written."
        )
        self._btn_cell_counts.clicked.connect(self._run_cell_counts)
        tools_row.addWidget(self._btn_cell_counts)

        # Remove one label by id. "Pick" reads the id the user clicked in the
        # viewer (napari's picker sets layer.selected_label), so the usual
        # workflow is: click the bad cell, Pick, Remove.
        remove_id_row = QHBoxLayout()
        remove_id_row.addWidget(QLabel("Remove label id:"))
        self._remove_id = QSpinBox()
        self._remove_id.setRange(0, 10_000_000)
        self._remove_id.setValue(0)
        remove_id_row.addWidget(self._remove_id)
        btn_pick_id = QPushButton("Pick")
        btn_pick_id.setFixedWidth(50)
        btn_pick_id.setToolTip("Copy the active Labels layer's selected label into the box")
        btn_pick_id.clicked.connect(self._pick_label_id)
        remove_id_row.addWidget(btn_pick_id)
        btn_remove_id = QPushButton("Remove")
        btn_remove_id.clicked.connect(self._run_remove_label)
        remove_id_row.addWidget(btn_remove_id)
        self._btn_undo_remove = QPushButton("Undo remove")
        self._btn_undo_remove.setEnabled(False)
        self._btn_undo_remove.clicked.connect(self._undo_remove_label)
        remove_id_row.addWidget(self._btn_undo_remove)

        slice_row = QHBoxLayout()
        self._rbs_manual = QPushButton("Manual")
        self._rbs_manual.setCheckable(True)
        self._rbs_manual.setChecked(True)
        self._rbs_manual.setFixedWidth(60)
        self._rbs_manual.toggled.connect(self._on_rbs_manual_toggled)
        slice_row.addWidget(self._rbs_manual)
        slice_row.addWidget(QLabel("Min:"))
        self._min_slice = QSpinBox()
        self._min_slice.setRange(0, 100_000)
        self._min_slice.setValue(self._config.get("min_slice", 0))
        slice_row.addWidget(self._min_slice)
        slice_row.addWidget(QLabel("Max:"))
        self._max_slice = QSpinBox()
        self._max_slice.setRange(0, 100_000)
        self._max_slice.setValue(self._config.get("max_slice", 100))
        slice_row.addWidget(self._max_slice)

        self._btn_rbs = QPushButton("Remove Bad Slices")
        self._btn_rbs.clicked.connect(self._run_remove_bad_slices)

        # --- Registration ---
        from qtpy.QtWidgets import QDoubleSpinBox

        _REG_RES = 2  # fixed internally

        layout.addLayout(path_row)
        layout.addLayout(frame_row)
        layout.addLayout(final_xform_row)
        layout.addWidget(btn_load_final_xform)
        layout.addWidget(self._btn_res)
        layout.addLayout(min_vol_row)
        layout.addWidget(btn_prep)
        layout.addLayout(prep_all_row)
        layout.addLayout(save_row)
        layout.addLayout(tools_row)
        layout.addLayout(remove_id_row)
        layout.addLayout(slice_row)
        layout.addWidget(self._btn_rbs)

        self._btn_reg = QPushButton("")
        self._btn_reg.clicked.connect(self._run_registration)
        layout.addWidget(self._btn_reg)

        reg_all_row = QHBoxLayout()
        self._btn_reg_all = QPushButton("Register All")
        self._btn_reg_all.clicked.connect(self._run_register_all)
        self._chk_reg_to_last = QCheckBox("to last frame")
        self._chk_reg_to_last.setChecked(False)
        reg_all_row.addWidget(self._btn_reg_all)
        reg_all_row.addWidget(self._chk_reg_to_last)
        layout.addLayout(reg_all_row)

        # --- Check / Fix membrane pairs ---
        btn_check = QPushButton("Check Membrane Pairs")
        btn_check.clicked.connect(self._run_check)

        sigma_row = QHBoxLayout()
        sigma_row.addWidget(QLabel("Split sigma:"))
        self._sigma = QDoubleSpinBox()
        self._sigma.setRange(1.0, 100.0)
        self._sigma.setValue(10.0)
        self._sigma.setSingleStep(1.0)
        self._sigma.setDecimals(1)
        sigma_row.addWidget(self._sigma)

        self._btn_fix = QPushButton("Fix Seg Errors")
        self._btn_fix.clicked.connect(self._run_fix)
        self._btn_fix.setEnabled(False)

        self._results = QPlainTextEdit()
        self._results.setReadOnly(True)
        self._results.setMaximumHeight(140)
        self._results.setPlaceholderText("Check results will appear here…")

        io2_row = QHBoxLayout()
        self._btn_io2 = QPushButton("Update Io2 table (this frame)")
        self._btn_io2.setToolTip(
            "Recompute only the pairs this frame takes part in and merge them\n"
            "into the Io2 table beside the loaded transforms. Correcting frame\n"
            "F invalidates pairs F-1 and F and nothing else."
        )
        self._btn_io2.clicked.connect(self._run_update_io2)
        io2_row.addWidget(self._btn_io2)
        io2_row.addWidget(QLabel("min size"))
        self._io2_min_size = QSpinBox()
        self._io2_min_size.setRange(1, 100000)
        self._io2_min_size.setValue(1)
        self._io2_min_size.setFixedWidth(70)
        self._io2_min_size.setToolTip(
            "Small-object filter applied before matching. 1 = none, which is\n"
            "right when the segmentation has already been cleaned."
        )
        io2_row.addWidget(self._io2_min_size)

        layout.addLayout(io2_row)
        layout.addWidget(btn_check)
        layout.addLayout(sigma_row)
        layout.addWidget(self._btn_fix)
        layout.addWidget(self._results)
        layout.addWidget(self.label)
        layout.addStretch()
        self._check_result       = None   # stores last check output
        self._layer_prev         = None   # Labels layer for frame N-1
        self._layer_curr         = None   # Labels layer for frame N
        self._layer_reg          = None   # registered corrected prev frame
        self._layer_reg_orig     = None   # registered original (non-corrected) prev frame
        self._prev_orig_path     = None   # path to non-corrected prev frame
        self._reg_save_path      = None
        self._orig_reg_save_path = None
        self._original_scales    = {}     # layer → original scale (for resolution toggle)
        self._final_transforms   = None   # loaded RegisteredTransforms_...json (frame-indexed), or None
        self._curr_frame_num     = None   # frame number self._layer_curr currently holds
        self._last_removed       = None   # (layer, volume before last Remove, label id)
        self._picked_layer       = None   # Labels layer whose selected_label the box mirrors
        self._mono_keep          = {}     # frame -> row from the monotonic pass

        # after the UI and the state above exist, not before
        self._track_selected_label()
        # The panel has more rows than a napari dock is usually given, and a
        # dock cannot scroll a plain widget: the children's minimum heights
        # become the widget's minimum, so the bottom rows end up off-screen
        # and the dock refuses to shrink. Putting everything inside a
        # QScrollArea decouples the two - the dock can be any size, and the
        # panel scrolls within it.
        inner = QWidget()
        inner.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setLayout(outer)

        # let the dock be dragged narrow/short; the scroll area takes over
        self.setMinimumSize(1, 1)
        inner.setMinimumWidth(320)

    def _load_as_labels(self, path: Path):
        img = tifffile.imread(str(path))
        layer = self.viewer.add_labels(img.astype(np.uint32), name=path.name)
        # the viewer needs uint32, but the file should go back out at its own
        # width; both are recorded so Save knows where this layer came from
        layer.metadata['source_path']  = path
        layer.metadata['source_dtype'] = img.dtype
        return layer, img.shape, img.dtype

    def _load_as_image(self, path: Path):
        img = tifffile.imread(str(path))
        self.viewer.add_image(img, name=path.name)
        return img.shape, img.dtype

    def _browse(self):
        start = self._path_edit.text() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select directory", start)
        if chosen:
            self._path_edit.setText(chosen)

    def _browse_final_xform(self):
        start = self._final_xform_path_edit.text() or str(Path.home())
        chosen, _ = QFileDialog.getOpenFileName(self, "Select final transforms JSON", start, "JSON (*.json)")
        if chosen:
            self._final_xform_path_edit.setText(chosen)

    def _load_final_transforms(self):
        """Load the exported per-frame cumulative transforms JSON (see
        ExportRegisteredTransforms.py). Once loaded, clicking the 'Register
        frame N-1 -> N' button below uses the precomputed transform for the
        currently loaded frame pair instead of running register_multiscale.
        """
        path = Path(self._final_xform_path_edit.text().strip())
        if not path.is_file():
            self.label.setText(f"Not a file: {path}")
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as exc:
            self.label.setText(f"Failed to load transforms JSON: {exc}")
            return

        n_frames = len(data.get('frames', {}))
        if n_frames == 0:
            self.label.setText(f"No 'frames' entries found in {path.name}")
            return

        self._final_transforms = data
        self._config["final_xform_path"] = str(path)
        _save_config(self._config)
        self.label.setText(
            f"Final registration transforms loaded: {path.name}\n"
            f"  {n_frames} frames, resolution_um={data.get('resolution_um')}\n"
            f"  Register button will now use these transforms directly."
        )

    def _load_frame(self):
        directory = Path(self._path_edit.text().strip())
        frame_num = self._frame_spin.value()

        if not directory.is_dir():
            self.label.setText(f"Not a directory: {directory}")
            return

        path = _find_frame(directory, frame_num)
        if path is None:
            self.label.setText(f"No tiff found in {directory} for frame {frame_num}")
            return

        self._config["last_dir"] = str(directory)
        self._config["last_frame"] = frame_num
        _save_config(self._config)

        prev_num = frame_num - 1
        msgs = []

        # intensity image for current frame only: prefer MemNucCombo, fall back to membrane
        raw_base = directory / "Raw_data"
        raw_dir = (raw_base / "MemNucCombo") if (raw_base / "MemNucCombo").is_dir() \
                  else (raw_base / "membrane")
        raw_path = _find_frame(raw_dir, frame_num) if raw_dir.is_dir() else None
        if raw_path:
            shape, dtype = self._load_as_image(raw_path)
            msgs.append(f"Raw frame {frame_num}: {raw_path.name}  {shape}  {dtype}")
        else:
            msgs.append(f"(no raw image found for frame {frame_num})")

        # Loading a frame loads ONLY that frame's own labels -- no automatic
        # previous-frame load and no automatic pickup of stale cached
        # registration tiffs from disk. If the frame just loaded as "current"
        # was frame_num - 1, it's shifted into "previous" here instead of
        # re-reading from disk, so any in-memory edits (manual paint fixes,
        # Fix Seg Errors output) carry forward rather than being discarded.
        self._layer_reg          = None
        self._layer_reg_orig     = None
        self._reg_save_path      = None
        self._orig_reg_save_path = None
        self._prev_orig_path     = None
        self._curr_path = path

        if self._layer_curr is not None and self._curr_frame_num == prev_num:
            self._layer_prev = self._layer_curr
            msgs.append(f"Previous frame {prev_num}: carried over from last-loaded '{self._layer_prev.name}'")
        else:
            self._layer_prev = None
            if frame_num > 0:
                msgs.append(f"(no frame {prev_num} in memory — load it first to enable registration)")

        if self._layer_prev is not None:
            prev_path = _find_frame(directory, prev_num)
            if prev_path is not None:
                # original (non-corrected) prev frame — used for dual warp
                prev_orig = _find_frame(prev_path.parent, prev_num, prefer_corrected=False)
                self._prev_orig_path = prev_orig if prev_orig != prev_path else None
                self._reg_save_path      = prev_path.parent / f"registered_to_{frame_num}{prev_path.suffix}"
                self._orig_reg_save_path = prev_path.parent / f"original_registered_to_{frame_num}{prev_path.suffix}"

        self._layer_curr, shape, dtype = self._load_as_labels(path)
        self._curr_frame_num = frame_num
        msgs.append(f"Labels frame {frame_num}: {path.name}  {shape}  {dtype}")

        # cleaning is applied to what you SEE, not to what is on disk
        cleaned, notes = self._apply_on_load(np.asarray(self._layer_curr.data), frame_num)
        if notes:
            self._layer_curr.data = cleaned
            msgs.append("on load: " + ", ".join(notes) + "  (not saved)")

        self.label.setText("\n".join(msgs))

    def _update_reg_label(self, frame_num: int):
        prev = frame_num - 1
        self._btn_reg.setText(f"Register frame {prev} → frame {frame_num}")

    def _count_labels(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not isinstance(layer, napari.layers.Labels):
            self.label.setText("Select a Labels layer first")
            return
        n = int(np.count_nonzero(np.unique(layer.data)))
        self.label.setText(f"'{layer.name}': {n} labels (excluding 0)")

    def _active_labels(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not isinstance(layer, napari.layers.Labels):
            self.label.setText("No active Labels layer selected")
            return None
        return layer

    def _apply_on_load(self, data, frame_num=None):
        """Apply the ticked On-load cleaning to `data`. Returns (data, notes).

        In memory only. This is the one place the cleaning is defined, so the
        viewer, Save ALL and the per-frame counts all show the same numbers.
        """
        notes = []
        if self._chk_prep_on_load.isChecked():
            before = int(np.count_nonzero(np.unique(data)))
            data, _ = prep(data, self._min_vol.value())
            after = int(np.count_nonzero(np.unique(data)))
            notes.append(f"prep {before}->{after}")

        if self._chk_clean_on_load.isChecked():
            before = int(np.count_nonzero(np.unique(data)))
            row = self._mono_keep.get(frame_num)
            if row is not None:
                # the constrained decision for this frame, made with the whole
                # sequence in view; a per-frame clean cannot see the count
                data = apply_keep(data, row['keep'])
                note = f"clean {before}->{row['kept']} (monotonic, need {row['need']})"
                if row['relaxed']:
                    note += f" +{len(row['admitted'])} admitted"
                if row['flagged']:
                    note += "  FLAGGED: " + row['note']
                notes.append(note)
            else:
                mod = _clean_embryo_module()
                if mod is None:
                    notes.append("clean SKIPPED (CleanEmbryoSegmentation.py not found)")
                else:
                    cleaned, rep = mod.clean_volume(data)
                    if rep['status'] == 'ok':
                        data = cleaned
                        notes.append(
                            f"clean {before}->{int(np.count_nonzero(np.unique(data)))}"
                            f" (per-frame; run Clean ALL for the monotonic version)")
                    else:
                        # clean_volume returns an EMPTY volume when it finds no
                        # embryo - never show that as if it were a result
                        notes.append(f"clean kept as-is ({rep.get('reason','no embryo')})")
        return data, notes

    def _run_prep(self):
        layer = self._active_labels()
        if layer is None:
            return
        cleaned, n_ccs = prep(layer.data, self._min_vol.value(), verbose=True)
        layer.data = cleaned
        self.label.setText(
            f"Prep done on '{layer.name}'  —  {n_ccs} CCs found, "
            f"min_volume={self._min_vol.value()}"
        )

    def _layer_source_path(self, layer):
        """Where *layer* was read from, or None if it was not loaded from disk."""
        src = layer.metadata.get('source_path') if hasattr(layer, 'metadata') else None
        if src is None and layer is self._layer_curr:
            src = self._curr_path
        return Path(src) if src else None

    @staticmethod
    def _narrow(data, dtype):
        """Back to the file's own integer width when the labels still fit."""
        if dtype is None:
            return data
        try:
            if data.max() <= np.iinfo(dtype).max:
                return data.astype(dtype)
        except (ValueError, TypeError):
            pass
        return data

    def _save_current_version(self):
        """Write the active Labels layer beside its source, version bumped."""
        layer = self._active_labels()
        if layer is None:
            return
        src = self._layer_source_path(layer)
        if src is None:
            self.label.setText(
                f"'{layer.name}' was not loaded from a file - nothing to version"
            )
            return

        dst = _next_version_path(src)
        data = self._narrow(np.asarray(layer.data),
                            layer.metadata.get('source_dtype'))
        info = _write_label_tiff(dst, data)

        # the layer now represents the new file, so a second Save bumps again
        layer.metadata['source_path'] = dst
        layer.name = dst.name
        if layer is self._layer_curr:
            self._curr_path = dst
        self.label.setText(f"Saved {dst.name}  ({info})  in {dst.parent}")

    def _save_all_versions(self):
        """Version every frame in the sequence, not just the loaded one.

        Frames carry their on-disk content across unchanged; the frame that is
        currently loaded is written from the layer instead, so in-memory edits
        are included. Keeping the whole sequence on one suffix is the point --
        the config has a single membrane_label_path pattern.
        """
        directory = Path(self._path_edit.text().strip())
        if not directory.is_dir():
            self.label.setText(f"Not a directory: {directory}")
            return

        seg_dir = directory / "membrane_segmentation"
        search_dir = seg_dir if seg_dir.is_dir() else directory
        all_frames = self._all_frame_numbers(search_dir)
        if not all_frames:
            self.label.setText(f"No frames found in {search_dir}")
            return

        # EVERY unsaved layer, not just the current one: _load_frame keeps the
        # previous frame as a live layer so its edits survive, and writing
        # that frame back from disk would silently drop them.
        in_mem = {f: d.copy() for f, d in self._in_memory_frames().items()}

        self._btn_save_all.setEnabled(False)
        self.label.setText(f"Save ALL: versioning {len(all_frames)} frames…")

        signals = _WorkerSignals()

        def _on_done(result):
            n_done, n_failed, suffixes, last = result
            msg = f"Save ALL done — {n_done} frames written, {n_failed} failed"
            if last:
                msg += f"\ne.g. {last}"
            if len(suffixes) > 1:
                msg += ("\nWARNING: frames did not all land on one suffix "
                        f"({', '.join(sorted(suffixes))}) - a single "
                        "membrane_label_path pattern will not match them all")
            self.label.setText(msg)
            self._btn_save_all.setEnabled(True)

        def _on_error(exc):
            self.label.setText(f"Save ALL failed: {exc}")
            self._btn_save_all.setEnabled(True)

        signals.done.connect(_on_done)
        signals.error.connect(_on_error)

        def _run():
            try:
                # Resolve every source first, then pick ONE version number
                # for all of them: bumping each file independently leaves the
                # sequence on mixed suffixes (a frame saved twice is already a
                # version ahead), and the config has a single
                # membrane_label_path pattern that must match every frame.
                srcs = {}
                for frame_num in all_frames:
                    src = _find_frame(directory, frame_num)
                    if src is None:
                        print(f"[MISS] frame {frame_num}: file not found", flush=True)
                        continue
                    srcs[frame_num] = src

                n_failed = len(all_frames) - len(srcs)
                if not srcs:
                    signals.done.emit((0, n_failed, set(), None))
                    return

                split   = {f: _version_split(p) for f, p in srcs.items()}
                version = max(n for _, _, n in split.values())
                while any(_version_name(srcs[f], *split[f][:2], version).exists()
                          for f in srcs):
                    version += 1
                print(f"[INFO] writing every frame at version {version}", flush=True)

                n_done = 0
                suffixes, last = set(), None
                for frame_num, src in srcs.items():
                    try:
                        if frame_num in in_mem:
                            data = in_mem[frame_num]
                            note = " (from the open layer, with edits)"
                        else:
                            data = tifffile.imread(str(src))
                            data, _n = self._apply_on_load(data, frame_num)
                            note = " (+on-load cleaning)" if _n else ""
                        parts, base, _ = split[frame_num]
                        dst = _version_name(src, parts, base, version)
                        _write_label_tiff(dst, data)
                        suffixes.add(dst.stem.split('.')[-1])
                        last = dst.name
                        n_done += 1
                        print(f"[OK] frame {frame_num}: {src.name} -> {dst.name}{note}",
                              flush=True)
                    except Exception as e:
                        print(f"[FAIL] frame {frame_num}: {e}", flush=True)
                        n_failed += 1
                signals.done.emit((n_done, n_failed, suffixes, last))
            except Exception as e:
                signals.error.emit(e)

        threading.Thread(target=_run, daemon=True).start()

    def _in_memory_frames(self):
        """{frame number: data} for every Labels layer open in the viewer.

        Not just the current frame: _load_frame carries the previous frame
        forward as a LAYER precisely so its edits survive, and any other
        loaded frame is equally live. Reading those from disk would report
        the pre-edit version.

        Warped copies ('registered_to_') are skipped - they duplicate a frame
        rather than being one, and would otherwise double-count.
        """
        out = {}
        for layer in self.viewer.layers:
            if not isinstance(layer, napari.layers.Labels):
                continue
            if 'registered_to_' in layer.name:
                continue
            src = self._layer_source_path(layer)
            if src is None:
                continue
            tok = _frame_token(src)
            if tok is not None:
                out[tok] = np.asarray(layer.data)
        if self._layer_curr is not None and self._curr_frame_num is not None:
            out[self._curr_frame_num] = np.asarray(self._layer_curr.data)
        return out

    def _pick_label_id(self):
        """Arm napari's picker: the next click in the viewer fills the box.

        Clicking a cell only changes layer.selected_label when the layer is in
        'pick' mode -- in the default pan/zoom mode a click sets nothing, so
        there is nothing for the box to mirror. Pressing Pick therefore puts
        the layer into pick mode rather than just reading the current value,
        which is the order people actually use it in: Pick, then click.
        """
        layer = self._active_labels()
        if layer is None:
            return

        self._track_selected_label()      # make sure the box is mirroring THIS layer
        picked = int(layer.selected_label)
        if picked > 0:
            self._remove_id.setValue(picked)

        try:
            layer.mode = 'pick'
        except Exception as e:
            self.label.setText(
                f"Could not switch '{layer.name}' to pick mode ({e}) - use "
                f"napari's picker button, or type the id"
            )
            return

        was = f"  (box currently {picked})" if picked > 0 else ""
        self.label.setText(
            f"Pick mode on '{layer.name}': click a cell in the viewer and its "
            f"id appears in the box.{was}"
        )

    def _run_remove_label(self):
        """Zero every voxel of one label id in the active Labels layer.

        In-memory only, like Prep and Remove Bad Slices - nothing is written
        until the layer is saved. The previous volume is kept so a mis-click
        can be undone without reloading the frame and losing the other edits.
        """
        layer = self._active_labels()
        if layer is None:
            return
        lab_id = self._remove_id.value()
        if lab_id == 0:
            self.label.setText("Label 0 is background - pick a label to remove")
            return

        data = layer.data
        n_vox = int(np.count_nonzero(data == lab_id))
        if n_vox == 0:
            self.label.setText(f"Label {lab_id} is not present in '{layer.name}'")
            return

        self._last_removed = (layer, data.copy(), lab_id)
        self._btn_undo_remove.setEnabled(True)

        cleaned = data.copy()
        cleaned[cleaned == lab_id] = 0
        layer.data = cleaned

        n_left = int(np.count_nonzero(np.unique(cleaned)))
        self.label.setText(
            f"Removed label {lab_id} ({n_vox} voxels) from '{layer.name}'  -  "
            f"{n_left} labels left"
        )

    def _undo_remove_label(self):
        """Restore the volume as it was before the last Remove."""
        if self._last_removed is None:
            return
        layer, data, lab_id = self._last_removed
        # lab_id is an int for a single-label Remove, or 'clean' when the last
        # destructive step was Clean embryo - both restore the same way
        what = f"label {lab_id}" if isinstance(lab_id, int) else "the pre-clean labels"
        if layer not in self.viewer.layers:
            self.label.setText(f"Layer holding {what} is gone - cannot undo")
            self._last_removed = None
            self._btn_undo_remove.setEnabled(False)
            return
        layer.data = data
        self._last_removed = None
        self._btn_undo_remove.setEnabled(False)
        self.label.setText(f"Restored {what} in '{layer.name}'")

    def _track_selected_label(self):
        """Mirror the active Labels layer's selected_label into the Remove box.

        napari's label picker (and clicking with the brush) sets
        selected_label on the layer, so clicking the bad cell in the viewer
        fills the box directly - the Pick button is only a manual fallback.
        Re-pointed whenever the active layer changes, and disconnected from
        the old one so a background layer cannot keep writing to the box.
        """
        layer = self.viewer.layers.selection.active
        if self._picked_layer is not None and self._picked_layer is not layer:
            try:
                self._picked_layer.events.selected_label.disconnect(
                    self._on_selected_label)
            except Exception:
                pass          # layer already closed, or never connected
            self._picked_layer = None

        if isinstance(layer, napari.layers.Labels) and layer is not self._picked_layer:
            layer.events.selected_label.connect(self._on_selected_label)
            self._picked_layer = layer
            self._on_selected_label()

    def _on_selected_label(self, event=None):
        """Copy the mirrored layer's selected label into the box."""
        if self._picked_layer is None:
            return
        picked = int(self._picked_layer.selected_label)
        if picked > 0:                     # 0 is background: leave the box alone
            self._remove_id.setValue(picked)

    def _run_clean_all_monotonic(self):
        """Clean every frame with the count constrained, writing nothing.

        The constraint is a property of the SEQUENCE, so it cannot be decided
        at load time one frame at a time: the floor for frame N depends on
        every frame before it. This pass decides all of them once and caches
        the keep-set per frame; Load then applies the cached decision, which
        is why the clean-on-load tick-box shows the constrained result
        afterwards and a plain per-frame clean before.
        """
        directory = Path(self._path_edit.text().strip())
        if not directory.is_dir():
            self.label.setText(f"Not a directory: {directory}")
            return
        mod = _clean_embryo_module()
        if mod is None:
            self.label.setText(
                "CleanEmbryoSegmentation.py not found - set CLEAN_EMBRYO_PATH "
                "to it, or put it beside this plugin")
            return

        seg_dir = directory / "membrane_segmentation"
        search_dir = seg_dir if seg_dir.is_dir() else directory
        all_frames = self._all_frame_numbers(search_dir)
        if not all_frames:
            self.label.setText(f"No frames found in {search_dir}")
            return

        start_count = self._start_count.value()
        self._btn_clean_all.setEnabled(False)
        self.label.setText(
            f"Clean ALL (monotonic, start {start_count}): {len(all_frames)} frames "
            f"from disk, ~1s each, writing nothing…")

        signals = _WorkerSignals()

        def _on_done(result):
            rows, head, table = result
            self._mono_keep = rows
            if not self._chk_clean_on_load.isChecked():
                self._chk_clean_on_load.setChecked(True)
            self._results.setPlainText(table)
            self.label.setText(head + "\n[in memory - Save when you want it kept]")
            self._btn_clean_all.setEnabled(True)

        def _on_error(exc):
            self.label.setText(f"Clean ALL failed: {exc}")
            self._btn_clean_all.setEnabled(True)

        signals.done.connect(_on_done)
        signals.error.connect(_on_error)

        def _load(n):
            # Deliberately the file, never the open layer. A loaded layer has
            # already had the on-load cleaning applied to it, so feeding it
            # back in would clean the cleaned result and the pass could never
            # recover an object the per-frame filters had removed. Save edits
            # first if they should count.
            src = _find_frame(directory, n)
            if src is None:
                print(f"[MISS] frame {n}: file not found", flush=True)
                return None
            return tifffile.imread(str(src))

        def _run():
            try:
                def _progress(row):
                    print(f"  frame {row['frame']:>4}: raw {row['raw']:>3} "
                          f"pool {row['pool']:>3} -> {row['kept']:>3} "
                          f"(need {row['need']}) {row['note']}", flush=True)
                rows = monotonic_keep_sets(all_frames, _load, mod,
                                           start_count=start_count,
                                           min_cell_frac=self._min_vol_frac(),
                                           progress=_progress)
                head, table, _ = summarise(rows)
                signals.done.emit((rows, head, table))
            except Exception as e:
                signals.error.emit(e)

        threading.Thread(target=_run, daemon=True).start()

    @staticmethod
    def _min_vol_frac():
        """Fragment cut used by the monotonic pass (clean_volume's default)."""
        return 0.10

    def _run_prep_all_preview(self):
        """Prep every frame and show the counts - without writing anything.

        Turns the on-load prep on (so every frame opened from here is
        prepped in the viewer) and runs the per-frame count, which applies
        the same pipeline. The prepped volumes are not cached: prep is ~0.4s
        on a real frame, and holding a whole sequence uncompressed in memory
        would be tens of GB.
        """
        if not self._chk_prep_on_load.isChecked():
            self._chk_prep_on_load.setChecked(True)
        self.label.setText(
            f"Prep ALL (min volume {self._min_vol.value()}) - counting every "
            f"frame, writing nothing…"
        )
        self._run_cell_counts()

    def _run_cell_counts(self):
        """Label count per frame, flagging every drop.

        An embryo only ever gains cells, so a decrease is a segmentation
        problem, not biology - usually a cell lost to a threshold, or the
        polar body flickering in and out between frames.
        """
        directory = Path(self._path_edit.text().strip())
        if not directory.is_dir():
            self.label.setText(f"Not a directory: {directory}")
            return

        seg_dir = directory / "membrane_segmentation"
        search_dir = seg_dir if seg_dir.is_dir() else directory
        all_frames = self._all_frame_numbers(search_dir)
        if not all_frames:
            self.label.setText(f"No frames found in {search_dir}")
            return

        in_mem = {f: d.copy() for f, d in self._in_memory_frames().items()}

        self._btn_cell_counts.setEnabled(False)
        self.label.setText(f"Counting cells in {len(all_frames)} frames…")

        signals = _WorkerSignals()

        def _on_done(result):
            counts, drops, missing = result
            lines = [f"{'frame':>6}  {'cells':>5}   {'delta':>5}  {'from':>4}"]
            prev = None
            for f, n, where in counts:
                d = '' if prev is None else f'{n - prev:+d}'
                flag = '  <-- DROP' if (prev is not None and n < prev) else ''
                lines.append(f"{f:>6}  {n:>5}   {d:>5}  {where:>4}{flag}")
                prev = n
            n_mem = sum(1 for _, _, wsrc in counts if wsrc == 'mem')
            n_mem_txt = (f" ({n_mem} from unsaved layer{'s' if n_mem > 1 else ''})"
                         if n_mem else '')
            if missing:
                lines.append(f"\nunreadable frames: {missing}")
            self._results.setPlainText("\n".join(lines))

            prepping = (self._chk_prep_on_load.isChecked()
                        or self._chk_clean_on_load.isChecked())
            if prepping:
                n_mem_txt += "  [cleaning applied in memory - nothing saved]"
            if drops:
                shown = ', '.join(f"{f} ({a}->{b})" for f, a, b in drops[:6])
                more = f" +{len(drops) - 6} more" if len(drops) > 6 else ''
                self.label.setText(
                    f"Cells per frame: {len(counts)} frames, "
                    f"{counts[0][1]} -> {counts[-1][1]} cells{n_mem_txt}.\n"
                    f"NOT monotonic - {len(drops)} drop(s): {shown}{more}"
                )
            else:
                self.label.setText(
                    f"Cells per frame: {len(counts)} frames, "
                    f"{counts[0][1]} -> {counts[-1][1]} cells{n_mem_txt}, "
                    f"monotonically non-decreasing throughout"
                )
            self._btn_cell_counts.setEnabled(True)

        def _on_error(exc):
            self.label.setText(f"Cells per frame failed: {exc}")
            self._btn_cell_counts.setEnabled(True)

        signals.done.connect(_on_done)
        signals.error.connect(_on_error)

        def _run():
            try:
                counts, drops, missing = [], [], []
                prev = None
                for frame_num in all_frames:
                    if frame_num in in_mem:
                        img, where = in_mem[frame_num], 'mem'   # unsaved edits included
                    else:
                        where = 'disk'
                        src = _find_frame(directory, frame_num)
                        if src is None:
                            missing.append(frame_num)
                            continue
                        try:
                            img = tifffile.imread(str(src))
                            img, _n = self._apply_on_load(img, frame_num)
                            if _n:
                                where = 'clean'
                        except Exception as e:
                            print(f"[FAIL] frame {frame_num}: {e}", flush=True)
                            missing.append(frame_num)
                            continue
                    n = int(np.count_nonzero(np.unique(img)))
                    if prev is not None and n < prev:
                        drops.append((frame_num, prev, n))
                    counts.append((frame_num, n, where))
                    print(f"frame {frame_num}: {n} cells ({where})", flush=True)
                    prev = n
                signals.done.emit((counts, drops, missing))
            except Exception as e:
                signals.error.emit(e)

        threading.Thread(target=_run, daemon=True).start()

    def _on_rbs_manual_toggled(self, checked: bool):
        self._min_slice.setEnabled(checked)
        self._max_slice.setEnabled(checked)
        self._btn_rbs.setText("Remove Bad Slices" if checked else "Remove Bad Slices (auto)")

    def _run_remove_bad_slices(self):
        layer = self._active_labels()
        if layer is None:
            return

        if self._rbs_manual.isChecked():
            min_s = self._min_slice.value()
            max_s = self._max_slice.value()
        else:
            # auto-detect: Z extent of the largest connected component
            result = auto_slice_range(layer.data)
            if result is None:
                self.label.setText("No labels found in any slice")
                return
            min_s, max_s = result
            # update spinboxes so the user can see what was computed
            self._min_slice.setValue(min_s)
            self._max_slice.setValue(max_s)

        cleaned = remove_bad_slices(layer.data, min_s, max_s)
        layer.data = cleaned
        self._config["min_slice"] = min_s
        self._config["max_slice"] = max_s
        _save_config(self._config)
        mode = "manual" if self._rbs_manual.isChecked() else "auto"
        self.label.setText(
            f"Remove Bad Slices ({mode}) on '{layer.name}'  "
            f"(kept z {min_s}–{max_s})"
        )

    def _run_registration(self):
        """Warp frame N-1 onto frame N using the loaded final transforms.

        The from-scratch search (register_multiscale) has been removed: the
        transforms are computed on the cluster and exported, and a second
        implementation here only invited the two to disagree. Load a
        RegisteredTransforms_*.json first.
        """
        frame_num = self._frame_spin.value()
        prev_num  = frame_num - 1

        layer_curr = self._layer_curr
        if layer_curr is None:
            self.label.setText("Load the current frame first")
            return

        if self._final_transforms is None:
            self.label.setText(
                "No registration transforms loaded - choose a "
                "RegisteredTransforms_*.json above and press 'Load Final "
                "Registration Transforms'."
            )
            return

        reg_res   = 2
        data_curr = layer_curr.data.copy()
        precomputed = pairwise_transform_from_final(
            self._final_transforms, prev_num, frame_num, reg_res)
        if precomputed is None:
            self.label.setText(
                f"The loaded transforms do not cover frame {prev_num} or {frame_num}."
            )
            return

        # Prefer the previous frame as loaded (it may carry manual edits);
        # otherwise read it straight from disk - the transform is known, so
        # it needs no interactive prep.
        if self._layer_prev is not None:
            data_prev = self._layer_prev.data.copy()
        else:
            directory = Path(self._path_edit.text().strip())
            prev_path = _find_frame(directory, prev_num) if directory.is_dir() else None
            if prev_path is None:
                self.label.setText(f"No tiff found for frame {prev_num} in {directory}")
                return
            data_prev = tifffile.imread(str(prev_path)).astype(np.uint32)
            prev_orig = _find_frame(prev_path.parent, prev_num, prefer_corrected=False)
            self._prev_orig_path     = prev_orig if prev_orig != prev_path else None
            self._reg_save_path      = prev_path.parent / f"registered_to_{frame_num}{prev_path.suffix}"
            self._orig_reg_save_path = prev_path.parent / f"original_registered_to_{frame_num}{prev_path.suffix}"

        xform_name = Path(self._final_xform_path_edit.text().strip()).name
        self._run_registration_from_transform(precomputed, frame_num, prev_num,
                                               data_prev, data_curr, reg_res, xform_name)

    def _run_registration_from_transform(self, precomputed, frame_num, prev_num,
                                          data_prev, data_curr, reg_res, xform_name):
        """Warp the previous frame directly using a precomputed (matrix, offset)
        pair from the loaded final-transforms JSON, converted to the plugin's
        reg_res grid convention by pairwise_transform_from_final(). Skips
        prep_array/register_multiscale entirely -- the transform is already
        known -- and warps straight from the native anisotropic arrays to a
        result on frame_num's own native anisotropic grid, via the same
        warp_to_original_space() used by the search-based path.

        Layer names start with the (target) frame number and carry an
        '_xform' tag plus the loaded transforms file's own name, so it's
        unambiguous both which frame pair this is and that it came from the
        explicitly-chosen final-transforms JSON rather than a fresh search.
        """
        M_reg, b_reg = precomputed
        name_out = f"{prev_num}_registered_to_{frame_num}_xform_{xform_name}"

        self._btn_reg.setEnabled(False)
        self.label.setText(
            f"Registering pair {prev_num}→{frame_num} using loaded final transforms "
            f"'{xform_name}'…"
        )

        signals = _WorkerSignals()

        def _on_done(result):
            warped      = result['corrected']
            warped_orig = result['original']
            self._layer_reg = self.viewer.add_labels(warped, name=name_out)
            msg = f"Registration (from final transforms '{xform_name}') done — '{name_out}' loaded"
            if warped_orig is not None:
                orig_name_out = f"{name_out}_orig"
                self._layer_reg_orig = self.viewer.add_labels(warped_orig, name=orig_name_out)
                msg += f", '{orig_name_out}' loaded"
            self.label.setText(msg)
            self._btn_reg.setEnabled(True)

        def _on_error(exc):
            import traceback
            msg = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            print(f"Registration (from final transforms) error:\n{msg}", flush=True)
            self.label.setText(f"Registration failed: {exc}")
            self._btn_reg.setEnabled(True)

        signals.done.connect(_on_done)
        signals.error.connect(_on_error)

        save_path      = self._reg_save_path
        orig_save_path = self._orig_reg_save_path
        prev_orig_path = self._prev_orig_path

        def _run():
            try:
                warped = warp_to_original_space(data_prev, M_reg, b_reg, reg_res, data_curr.shape)

                warped_orig = None
                if prev_orig_path is not None:
                    print(f"Warping original (non-corrected) prev frame: {prev_orig_path.name}…",
                          flush=True)
                    data_orig = tifffile.imread(str(prev_orig_path)).astype(np.uint32)
                    warped_orig = warp_to_original_space(data_orig, M_reg, b_reg, reg_res, data_curr.shape)

                if save_path is not None:
                    print(f"Saving registered frame to {save_path}…", flush=True)
                    tifffile.imwrite(str(save_path), warped)
                    print(f"Saved → {save_path.name}", flush=True)
                if warped_orig is not None and orig_save_path is not None:
                    print(f"Saving original registered frame to {orig_save_path}…", flush=True)
                    tifffile.imwrite(str(orig_save_path), warped_orig)
                    print(f"Saved → {orig_save_path.name}", flush=True)

                signals.done.emit({'corrected': warped, 'original': warped_orig})
            except Exception as exc:
                signals.error.emit(exc)

        threading.Thread(target=_run, daemon=True).start()

    @staticmethod
    def _all_frame_numbers(seg_dir: Path) -> list[int]:
        """Frame numbers present in *seg_dir*.

        Only the FIRST integer of each name counts, for the same reason
        _find_frame does it that way: every other integer belongs to a suffix.
        Taking all of them invented frames out of version numbers and dated
        backups ('...pre20261004.tif' became frame 20261004).
        """
        nums = set()
        for f in seg_dir.iterdir():
            if f.suffix not in ('.tif', '.tiff'):
                continue
            if 'registered_to_' in f.name:
                continue
            tok = _frame_token(f)
            if tok is not None:
                nums.add(tok)
        return sorted(nums)

    def _run_register_all(self):
        """Warp every frame onto one reference using the loaded transforms.

        Same source of truth as the single-pair button: no search, so a run
        here cannot disagree with the exported registration.
        """
        directory = Path(self._path_edit.text().strip())
        if not directory.is_dir():
            self.label.setText(f"Not a directory: {directory}")
            return
        if self._final_transforms is None:
            self.label.setText(
                "No registration transforms loaded - choose a "
                "RegisteredTransforms_*.json above and press 'Load Final "
                "Registration Transforms'."
            )
            return

        seg_dir = directory / "membrane_segmentation"
        search_dir = seg_dir if seg_dir.is_dir() else directory
        all_frames = self._all_frame_numbers(search_dir)
        if len(all_frames) < 2:
            self.label.setText("Need at least 2 frames for Register All")
            return

        reg_res = 2
        to_last = self._chk_reg_to_last.isChecked()
        ref_num = all_frames[-1] if to_last else all_frames[0]
        moving  = [n for n in all_frames if n != ref_num]
        xform_name = Path(self._final_xform_path_edit.text().strip()).name

        self._btn_reg_all.setEnabled(False)
        self.label.setText(
            f"Register All: {len(moving)} frames -> reference {ref_num} "
            f"using '{xform_name}'…")

        signals = _WorkerSignals()

        def _on_done(result):
            n_done, n_skipped, n_failed = result
            self.label.setText(
                f"Register All done - {n_done} registered, {n_skipped} skipped, "
                f"{n_failed} failed")
            self._btn_reg_all.setEnabled(True)

        def _on_error(exc):
            self.label.setText(f"Register All failed: {exc}")
            self._btn_reg_all.setEnabled(True)

        signals.done.connect(_on_done)
        signals.error.connect(_on_error)
        transforms = self._final_transforms

        def _run():
            try:
                ref_path = _find_frame(directory, ref_num)
                if ref_path is None:
                    signals.error.emit(Exception(f"Reference frame {ref_num} not found"))
                    return
                ref_shape = tifffile.imread(str(ref_path)).shape

                n_done = n_skipped = n_failed = 0
                for n in moving:
                    out_path = search_dir / f"{n}_registered_to_{ref_num}{ref_path.suffix}"
                    if out_path.exists():
                        print(f"[SKIP] frame {n}: {out_path.name} exists", flush=True)
                        n_skipped += 1
                        continue
                    pre = pairwise_transform_from_final(transforms, n, ref_num, reg_res)
                    if pre is None:
                        print(f"[MISS] frame {n}: not covered by the transforms", flush=True)
                        n_failed += 1
                        continue
                    src = _find_frame(directory, n)
                    if src is None:
                        print(f"[MISS] frame {n}: file not found", flush=True)
                        n_failed += 1
                        continue
                    try:
                        data = tifffile.imread(str(src)).astype(np.uint32)
                        M_reg, b_reg = pre
                        warped = warp_to_original_space(data, M_reg, b_reg, reg_res, ref_shape)
                        _write_label_tiff(out_path, warped)
                        n_done += 1
                        print(f"[OK] frame {n} -> {out_path.name}", flush=True)
                    except Exception as e:
                        print(f"[FAIL] frame {n}: {e}", flush=True)
                        n_failed += 1
                signals.done.emit((n_done, n_skipped, n_failed))
            except Exception as e:
                signals.error.emit(e)

        threading.Thread(target=_run, daemon=True).start()

    def _run_update_io2(self):
        """Recompute this frame's Io2 pairs and merge them into the table.

        A pair (f, f+1) is built from exactly two frames, so correcting frame
        F invalidates pairs F-1 and F and nothing else: three frames are
        warped instead of the whole sequence. The table is the one beside the
        loaded transforms JSON, and the current frame is taken from the
        viewer so unsaved edits count.
        """
        if self._final_transforms is None:
            self.label.setText(
                "Load the registration transforms first - the Io2 table is "
                "built from them and is written beside them.")
            return
        if self._layer_curr is None or self._curr_frame_num is None:
            self.label.setText("Load a frame first")
            return

        frame_num = self._curr_frame_num
        xform_path = Path(self._final_xform_path_edit.text().strip())
        table_path = xform_path.parent / f"Io2Table_{xform_path.stem}.json"
        if not table_path.exists():
            self.label.setText(
                f"No table to update at {table_path.name} - build it once with "
                f"ComputeIo2TableFromTransforms.py, then this button keeps it current.")
            return

        begin = self._final_transforms['begin_frame']
        end   = self._final_transforms['end_frame']
        resolution = self._final_transforms['resolution_um']
        pairs = [f for f in (frame_num - 1, frame_num) if begin <= f <= end - 1]
        if not pairs:
            self.label.setText(f"Frame {frame_num} takes part in no pair in {begin}..{end}")
            return

        directory = Path(self._path_edit.text().strip())
        min_size = self._io2_min_size.value()
        curr_data = np.asarray(self._layer_curr.data).copy()
        self._btn_io2.setEnabled(False)
        self.label.setText(
            f"Updating Io2 pairs {pairs} in {table_path.name} "
            f"(frame {frame_num} from the viewer)…")

        signals = _WorkerSignals()

        def _on_done(result):
            lines, backup = result
            self._results.setPlainText("\n".join(lines))
            self.label.setText(
                f"Io2 table updated: pairs {pairs} in {table_path.name}\n"
                f"previous table kept as {backup.name}")
            self._btn_io2.setEnabled(True)

        def _on_error(exc):
            self.label.setText(f"Io2 update failed: {exc}")
            self._btn_io2.setEnabled(True)

        signals.done.connect(_on_done)
        signals.error.connect(_on_error)

        def _run():
            try:
                from RegisterLabelPairIo2 import compute_io2_table
            except ImportError as e:
                signals.error.emit(Exception(
                    f"{e} - RegisterLabelPairIo2.py must be importable "
                    f"(it lives with the lineage code)"))
                return
            try:
                from scipy.ndimage import affine_transform, zoom
                from skimage.measure import label as sk_label
                import skimage.morphology as morphology

                def warp(f):
                    if f == frame_num:
                        img = curr_data
                    else:
                        src = _find_frame(directory, f)
                        if src is None:
                            raise FileNotFoundError(f'frame {f} not found in {directory}')
                        img = tifffile.imread(str(src))
                    z = (_RES_Z_MEMBRANE / resolution, _RES_XY / resolution,
                         _RES_XY / resolution)
                    img = zoom(img, z, order=0)
                    if min_size > 1:
                        cc = sk_label(img, connectivity=img.ndim)
                        cc = morphology.remove_small_objects(cc, min_size=min_size)
                        img = img.copy()
                        img[cc == 0] = 0
                    e = self._final_transforms['frames'][str(f)]
                    return affine_transform(img.astype(np.float32),
                                            np.array(e['matrix']), np.array(e['offset']),
                                            order=0).astype(img.dtype)

                with open(table_path) as fh:
                    table = json.load(fh)
                from datetime import datetime
                backup = table_path.with_name(
                    f'{table_path.stem}.pre{datetime.now():%Y%m%d_%H%M%S}.json')
                shutil.copy2(table_path, backup)

                cache, lines = {}, []
                for f in pairs:
                    for g in (f, f + 1):
                        if g not in cache:
                            cache[g] = warp(g)
                            print(f'  warped frame {g}', flush=True)
                    io2 = compute_io2_table(cache[f], cache[f + 1])
                    before = table.get(str(f))
                    table[str(f)] = io2
                    new = [v for row in io2.values() for v in row.values() if v > 0]
                    old = ([v for row in before.values() for v in row.values() if v > 0]
                           if before else [])
                    lines.append(
                        f"pair {f}->{f+1}: mean {np.mean(new):.4f} (n={len(new)})"
                        + (f"   was {np.mean(old):.4f} (n={len(old)})" if old else ""))
                    print(' ', lines[-1], flush=True)

                with open(table_path, 'w') as fh:
                    json.dump(table, fh)
                signals.done.emit((lines, backup))
            except Exception as e:
                signals.error.emit(e)

        threading.Thread(target=_run, daemon=True).start()

    def _run_check(self):
        if self._layer_reg is None:
            self._results.setPlainText(
                "Register the previous frame first (or load a frame that has a saved registration)"
            )
            return
        if self._layer_curr is None:
            self._results.setPlainText("Load a frame first")
            return

        layer_reg  = self._layer_reg
        layer_curr = self._layer_curr
        self._results.setPlainText(
            f"Checking:\n  prev (registered): {layer_reg.name}\n  curr: {layer_curr.name}\n…"
        )
        self._btn_fix.setEnabled(False)
        data1 = layer_reg.data.copy()
        data2 = layer_curr.data.copy()

        @thread_worker
        def _work():
            return check_frame_pair(data1, data2)

        def _on_done(result):
            self._check_result = result
            self._results.setPlainText(summary_text(result))
            has_errors = len(result['daughter_with_two_mothers']) > 0
            self._btn_fix.setEnabled(has_errors)
            self.label.setText("Check done" + (" — errors found" if has_errors else " — no errors"))

        def _on_error(exc):
            err = exc[1] if isinstance(exc, tuple) else exc
            self._results.setPlainText(f"Check failed: {err}")

        worker = _work()
        worker.returned.connect(_on_done)
        worker.errored.connect(_on_error)
        worker.start()

    def _run_fix(self):
        if self._check_result is None or self._layer_reg is None or self._layer_curr is None:
            return

        sigma     = self._sigma.value()
        data1     = self._layer_reg.data.copy()
        data2     = self._layer_curr.data.copy()
        curr_name = self._layer_curr.name
        result    = self._check_result

        self.label.setText("Fixing seg errors…")
        self._btn_fix.setEnabled(False)

        @thread_worker
        def _work():
            return fix_seg_errors(data1, data2, result, sigma=sigma)

        def _on_done(out):
            fixed, n_fixed = out
            self.viewer.add_labels(fixed, name=f"{curr_name}_fixed")
            self.label.setText(f"Fixed {n_fixed} daughter(s) → '{curr_name}_fixed' loaded")
            self._btn_fix.setEnabled(True)

        def _on_error(exc):
            self.label.setText(f"Fix failed: {exc[1]}")
            self._btn_fix.setEnabled(True)

        worker = _work()
        worker.returned.connect(_on_done)
        worker.errored.connect(_on_error)
        worker.start()

    def _on_active_layer_changed(self, event=None):
        layer = self.viewer.layers.selection.active
        self._track_selected_label()
        if layer in self._original_scales:
            self._btn_res.setText(f"Revert resolution {self._original_scales[layer]}")
        else:
            self._btn_res.setText("Set resolution (10, 1, 1)")

    def _toggle_resolution(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, 'scale'):
            self.label.setText("No active layer selected")
            return
        if layer not in self._original_scales:
            self._original_scales[layer] = tuple(layer.scale)
            layer.scale = (10, 1, 1)
            self._btn_res.setText(f"Revert resolution {self._original_scales[layer]}")
            self.label.setText(f"Scale set to (10, 1, 1) on '{layer.name}'")
        else:
            orig = self._original_scales.pop(layer)
            layer.scale = orig
            self.label.setText(f"Scale reverted to {orig} on '{layer.name}'")
            # update button label based on whether any layer is still scaled
            active = self.viewer.layers.selection.active
            if active in self._original_scales:
                self._btn_res.setText(f"Revert resolution {self._original_scales[active]}")
            else:
                self._btn_res.setText("Set resolution (10, 1, 1)")
