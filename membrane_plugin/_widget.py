import json
import re
import numpy as np
import tifffile
from pathlib import Path

import threading
from qtpy.QtCore import Qt, QObject, Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QSpinBox, QToolButton, QSizePolicy,
    QDoubleSpinBox, QPlainTextEdit, QLineEdit, QCheckBox,
)
import napari
from napari.qt import thread_worker
from membrane_plugin._cleaning import prep, remove_bad_slices, auto_slice_range
from membrane_plugin._registration import (
    prep_array, register_multiscale,
    euler_angles_to_rotation_matrix, warp_to_original_space,
)
from membrane_plugin._paircheck import check_frame_pair, fix_seg_errors, summary_text

_CONFIG = Path.home() / ".membrane_plugin.json"

def _load_config() -> dict:
    try:
        return json.loads(_CONFIG.read_text())
    except Exception:
        return {}

def _save_config(data: dict):
    _CONFIG.write_text(json.dumps(data))


def _find_frame(directory: Path, frame_num: int, prefer_corrected: bool = True) -> Path | None:
    """Return the best tiff in *directory* whose name contains *frame_num*.

    Looks inside the 'membrane_segmentation' subdirectory of *directory* if it
    exists, otherwise uses *directory* directly.  When *prefer_corrected* is
    True (default), a 'corrected' variant is returned over a plain one;
    when False, the plain (non-corrected) variant is preferred.
    """
    seg_dir = directory / "membrane_segmentation"
    search_dir = seg_dir if seg_dir.is_dir() else directory

    candidates = []
    for f in sorted(search_dir.iterdir()):
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
        self._btn_unload_final_xform = QPushButton("Unload (use search-based registration)")
        self._btn_unload_final_xform.clicked.connect(self._unload_final_transforms)
        self._btn_unload_final_xform.setEnabled(False)

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

        starts_row = QHBoxLayout()
        starts_row.addWidget(QLabel("N random starts:"))
        self._n_starts = QSpinBox()
        self._n_starts.setRange(0, 50)
        self._n_starts.setValue(12)
        starts_row.addWidget(self._n_starts)

        nfev_row = QHBoxLayout()
        nfev_row.addWidget(QLabel("Max iters:"))
        self._max_nfev = QSpinBox()
        self._max_nfev.setRange(100, 50000)
        self._max_nfev.setValue(2000)
        self._max_nfev.setSingleStep(500)
        nfev_row.addWidget(self._max_nfev)

        perturb_row = QHBoxLayout()
        perturb_row.addWidget(QLabel("Angle perturb (rad):"))
        self._angle_perturb = QDoubleSpinBox()
        self._angle_perturb.setRange(0.01, 3.14)
        self._angle_perturb.setValue(0.1)
        self._angle_perturb.setSingleStep(0.05)
        self._angle_perturb.setDecimals(2)
        perturb_row.addWidget(self._angle_perturb)

        trans_row = QHBoxLayout()
        trans_row.addWidget(QLabel("Trans perturb (px):"))
        self._trans_perturb = QDoubleSpinBox()
        self._trans_perturb.setRange(0.0, 500.0)
        self._trans_perturb.setValue(5.0)
        self._trans_perturb.setSingleStep(5.0)
        self._trans_perturb.setDecimals(1)
        trans_row.addWidget(self._trans_perturb)

        tips = _CollapsiblePanel("Registration guide", """
<b>Load Final Registration Transforms</b> — load a precomputed per-frame<br>
transforms JSON (e.g. RegisteredTransforms_iso0p832um_...json). While<br>
loaded, the Register button below uses the transform for the current<br>
pair directly instead of searching, and the previous frame is read<br>
straight from disk if it isn't already loaded.<br><br>
<b>Unload (use search-based registration)</b> — discard the loaded<br>
transforms and go back to letting Register run its own from-scratch<br>
optimization (register_multiscale) using the parameters below.<br><br>
<b>Register frame N-1 → N</b> — register the previous frame onto the<br>
currently loaded frame: uses the loaded transforms if present and they<br>
cover this pair, otherwise runs the search below.<br><br>
<b>Register All</b> — batch-register every frame found in the current<br>
Path directory to a single reference frame (first or last, via the<br>
"to last frame" checkbox), always via the from-scratch search — it<br>
does not use loaded final transforms.<br><br>
<hr>
<b>N random starts</b> — increase first if registration looks wrong.<br>
Try 8–16 for difficult cases.<br><br>
<b>Angle perturb (rad)</b> — widen if there is large rotation between<br>
frames. Try 0.3–0.5 rad (~17–29°).<br><br>
<b>Trans perturb (px)</b> — widen if there is large translation between<br>
frames. Try 20–50 px.<br><br>
<b>Max iters</b> — increase if the console shows the optimizer hitting<br>
the limit before converging (cost still dropping at last iteration).
""")

        self._btn_reg = QPushButton("")
        self._btn_reg.clicked.connect(self._run_registration)
        self._frame_spin.valueChanged.connect(self._update_reg_label)
        self._update_reg_label(self._frame_spin.value())

        layout.addLayout(path_row)
        layout.addLayout(frame_row)
        layout.addLayout(final_xform_row)
        layout.addWidget(btn_load_final_xform)
        layout.addWidget(self._btn_unload_final_xform)
        layout.addWidget(self._btn_res)
        layout.addLayout(min_vol_row)
        layout.addWidget(btn_prep)
        layout.addLayout(slice_row)
        layout.addWidget(self._btn_rbs)

        layout.addLayout(starts_row)
        layout.addLayout(nfev_row)
        layout.addLayout(perturb_row)
        layout.addLayout(trans_row)
        layout.addWidget(tips)
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
        self.setLayout(layout)

    def _load_as_labels(self, path: Path):
        img = tifffile.imread(str(path))
        layer = self.viewer.add_labels(img.astype(np.uint32), name=path.name)
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
        self._btn_unload_final_xform.setEnabled(True)
        self.label.setText(
            f"Final registration transforms loaded: {path.name}\n"
            f"  {n_frames} frames, resolution_um={data.get('resolution_um')}\n"
            f"  Register button will now use these transforms directly."
        )

    def _unload_final_transforms(self):
        self._final_transforms = None
        self._btn_unload_final_xform.setEnabled(False)
        self.label.setText("Final registration transforms unloaded — Register button will search from scratch again")

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
        frame_num = self._frame_spin.value()
        prev_num  = frame_num - 1

        layer_curr = self._layer_curr
        if layer_curr is None:
            self.label.setText("Load the current frame first")
            return

        reg_res  = 2
        name_out = f"{prev_num}_registered_to_{frame_num}"
        data_curr = layer_curr.data.copy()

        # If final registration transforms are loaded and cover this exact
        # frame pair, use the precomputed transform directly instead of
        # running register_multiscale's from-scratch search. The transforms
        # file must be explicitly chosen (via "Load Final Registration
        # Transforms") -- self._final_transforms stays None otherwise, so
        # this never silently kicks in.
        precomputed = None
        if self._final_transforms is not None:
            precomputed = pairwise_transform_from_final(self._final_transforms, prev_num, frame_num, reg_res)
            if precomputed is None:
                self.label.setText(
                    f"Final transforms loaded but frame {prev_num} or {frame_num} not found in them — "
                    f"falling back to search-based registration…"
                )

        # The previous frame's labels: prefer whatever is already loaded/
        # carried-over as "previous" (may carry manual edits), but when a
        # precomputed transform is available, fall back to reading the
        # previous frame's raw label file straight from disk -- the
        # transform-based path needs no interactive prep on it, so it
        # doesn't need to have been loaded as a viewer layer at all.
        if self._layer_prev is not None:
            data_prev = self._layer_prev.data.copy()
        elif precomputed is not None:
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
        else:
            self.label.setText(
                f"Need frame {prev_num}'s labels — load it first (so it carries over as "
                f"'previous'), or load final registration transforms covering this pair "
                f"(then it's read straight from disk)."
            )
            return

        if precomputed is not None:
            xform_name = Path(self._final_xform_path_edit.text().strip()).name
            self._run_registration_from_transform(precomputed, frame_num, prev_num,
                                                    data_prev, data_curr, reg_res, xform_name)
            return

        self._btn_reg.setEnabled(False)
        self.label.setText(
            f"Registering '{self._layer_prev.name}' → '{layer_curr.name}'  (reg_res={reg_res})…"
        )

        reg_params = {
            'max_nfev':      self._max_nfev.value(),
            'n_starts':      self._n_starts.value(),
            'angle_perturb': self._angle_perturb.value(),
            'trans_perturb': self._trans_perturb.value(),
        }

        signals = _WorkerSignals()

        def _on_done(result):
            warped      = result['corrected']
            warped_orig = result['original']
            self._layer_reg = self.viewer.add_labels(warped, name=name_out)
            msg = f"Registration done — '{name_out}' loaded"
            if warped_orig is not None:
                orig_name_out = f"{name_out}_orig"
                self._layer_reg_orig = self.viewer.add_labels(warped_orig, name=orig_name_out)
                msg += f", '{orig_name_out}' loaded"
            self.label.setText(msg)
            self._btn_reg.setEnabled(True)

        def _on_error(exc):
            import traceback
            msg = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            print(f"Registration error:\n{msg}", flush=True)
            self.label.setText(f"Registration failed: {exc}")
            self._btn_reg.setEnabled(True)

        signals.done.connect(_on_done)
        signals.error.connect(_on_error)

        save_path      = self._reg_save_path        # capture before entering thread
        orig_save_path = self._orig_reg_save_path
        prev_orig_path = self._prev_orig_path
        prev_name      = self._layer_prev.name

        def _run():
            try:
                print(f"Prep prev frame: {prev_name}", flush=True)
                bnd_prev, orig_prev, _ = prep_array(data_prev, reg_res)
                print(f"Prep curr frame: {layer_curr.name}", flush=True)
                bnd_curr, orig_curr, _ = prep_array(data_curr, reg_res)
                params = register_multiscale(
                    bnd_prev.astype(np.float32), bnd_curr.astype(np.float32), np.zeros(6),
                    max_nfev=reg_params['max_nfev'],
                    n_random_starts=reg_params['n_starts'],
                    angle_perturb=reg_params['angle_perturb'],
                    translation_perturb=reg_params['trans_perturb'],
                )
                R = euler_angles_to_rotation_matrix(params[:3])
                t = params[3:]
                warped = warp_to_original_space(orig_prev, R, t, reg_res, orig_curr.shape)

                # also warp the original (non-corrected) prev frame with the same transform
                warped_orig = None
                if prev_orig_path is not None:
                    print(f"Warping original (non-corrected) prev frame: {prev_orig_path.name}…",
                          flush=True)
                    data_orig = tifffile.imread(str(prev_orig_path)).astype(np.uint32)
                    _, orig_prev_uncorr, _ = prep_array(data_orig, reg_res)
                    warped_orig = warp_to_original_space(
                        orig_prev_uncorr, R, t, reg_res, orig_curr.shape
                    )

                # save to disk in the background thread — not on the Qt main thread
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
        nums = set()
        for f in seg_dir.iterdir():
            if f.suffix not in ('.tif', '.tiff'):
                continue
            if 'registered_to_' in f.name:
                continue
            for m in re.finditer(r'\d+', f.name):
                nums.add(int(m.group()))
        return sorted(nums)

    def _run_register_all(self):
        directory = Path(self._path_edit.text().strip())
        if not directory.is_dir():
            self.label.setText(f"Not a directory: {directory}")
            return

        seg_dir = directory / "membrane_segmentation"
        search_dir = seg_dir if seg_dir.is_dir() else directory

        all_frames = self._all_frame_numbers(search_dir)
        if len(all_frames) < 2:
            self.label.setText("Need at least 2 frames for Register All")
            return

        to_last = self._chk_reg_to_last.isChecked()
        ref_num = all_frames[-1] if to_last else all_frames[0]
        moving_frames = [n for n in all_frames if n != ref_num]

        self._btn_reg_all.setEnabled(False)
        self.label.setText(
            f"Register All: {len(moving_frames)} frames → reference frame {ref_num}…"
        )

        signals = _WorkerSignals()

        def _on_done(result):
            n_done, n_skipped, n_failed = result
            self.label.setText(
                f"Register All done — {n_done} registered, "
                f"{n_skipped} skipped, {n_failed} failed"
            )
            self._btn_reg_all.setEnabled(True)

        def _on_error(exc):
            self.label.setText(f"Register All failed: {exc}")
            self._btn_reg_all.setEnabled(True)

        signals.done.connect(_on_done)
        signals.error.connect(_on_error)

        reg_params = {
            'max_nfev':      self._max_nfev.value(),
            'n_starts':      self._n_starts.value(),
            'angle_perturb': self._angle_perturb.value(),
            'trans_perturb': self._trans_perturb.value(),
        }

        def _run():
            try:
                ref_path = _find_frame(directory, ref_num)
                if ref_path is None:
                    signals.error.emit(Exception(f"Reference frame {ref_num} not found"))
                    return
                ref_img = tifffile.imread(str(ref_path)).astype(np.uint32)
                bnd_ref, orig_ref, _ = prep_array(ref_img, 2)

                n_done = n_skipped = n_failed = 0
                for frame_num in moving_frames:
                    out_path = search_dir / f"{frame_num}_registered_to_{ref_num}{ref_path.suffix}"
                    if out_path.exists():
                        print(f"[SKIP] frame {frame_num}: {out_path.name} already exists",
                              flush=True)
                        n_skipped += 1
                        continue
                    frame_path = _find_frame(directory, frame_num)
                    if frame_path is None:
                        print(f"[MISS] frame {frame_num}: file not found", flush=True)
                        n_failed += 1
                        continue
                    try:
                        frame_img = tifffile.imread(str(frame_path)).astype(np.uint32)
                        bnd_frame, orig_frame, _ = prep_array(frame_img, 2)
                        params = register_multiscale(
                            bnd_frame.astype(np.float32), bnd_ref.astype(np.float32),
                            np.zeros(6),
                            max_nfev=reg_params['max_nfev'],
                            n_random_starts=reg_params['n_starts'],
                            angle_perturb=reg_params['angle_perturb'],
                            translation_perturb=reg_params['trans_perturb'],
                        )
                        R = euler_angles_to_rotation_matrix(params[:3])
                        t = params[3:]
                        warped = warp_to_original_space(orig_frame, R, t, 2, orig_ref.shape)
                        tifffile.imwrite(str(out_path), warped)
                        print(f"  Saved → {out_path.name}", flush=True)
                        n_done += 1
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()
                        print(f"[FAIL] frame {frame_num}: {exc}", flush=True)
                        n_failed += 1

                signals.done.emit((n_done, n_skipped, n_failed))
            except Exception as exc:
                signals.error.emit(exc)

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
