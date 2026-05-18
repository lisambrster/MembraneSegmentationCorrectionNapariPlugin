import numpy as np
from skimage.measure import label, regionprops


def prep(lab_img, min_volume, verbose=False):
    """Remove connected components smaller than min_volume from a label image.
    Returns (cleaned_label_img, n_connected_components)."""
    lab_img = lab_img.copy()
    cc_img = label(lab_img, connectivity=lab_img.ndim)
    props  = regionprops(cc_img)
    n_ccs  = len(props)

    for prop in props:
        if 0 < prop.area < min_volume:
            cz, cy, cx = int(prop.centroid[0]), int(prop.centroid[1]), int(prop.centroid[2])
            orig_label = lab_img[cz, cy, cx]
            if verbose:
                print(f'removing label {orig_label} (cc {prop.label}) volume={prop.area}')
            lab_img[cc_img == prop.label] = 0

    return lab_img, n_ccs


def remove_bad_slices(lab_img, min_slice, max_slice):
    """Zero out all z-slices outside [min_slice, max_slice]."""
    lab_img = lab_img.copy()
    lab_img[:min_slice, :, :] = 0
    lab_img[max_slice + 1:, :, :] = 0
    return lab_img


def auto_slice_range(lab_img: np.ndarray) -> tuple[int, int] | None:
    """Return (min_z, max_z) of the Z extent of the largest connected component.

    Operates on the binary foreground (any label > 0).  Returns None if the
    image is empty.
    """
    cc_img = label(lab_img > 0, connectivity=3)
    props = regionprops(cc_img)
    if not props:
        return None
    largest = max(props, key=lambda p: p.area)
    # regionprops bbox is (min_z, min_y, min_x, max_z_excl, max_y_excl, max_x_excl)
    min_z = largest.bbox[0]
    max_z = largest.bbox[3] - 1
    return min_z, max_z
