"""Build the DLP map bundle (maps/DLP/) from the CC0 recordings alone.

The DLP toolkit ships a drawn drivable-area image, but it is GPL-3; this repo
is MIT and the recordings are CC0, so the map is derived from the data:

    paved lot = moving-vehicle centre traces, drawn half a car width (1 m) wide
              + parked-car footprints (obstacles, and agents while stopped),
                which mark the stalls
              + small enclosed holes filled (empty stalls between parked cars)
    aisles    = paved lot minus the stall bands, where a stall band is the
                parked footprints grown to a full stall and merged along the row

Semantic channels, following tools/map_tools/process_nuscenes.py (values in
[-1, 1], -1 = nothing):

    0  drivable        0.5 on the chosen area. nuScenes writes
                       max(lane, road_segment, drivable_area) * 0.75 + ped_crossing * 0.25
                       and rescales [0, 1] -> [-1, 1], so plain drivable surface is 0.5
                       there too. DLP has no crossings, so nothing is 1.0.
    1  road_divider    -1 everywhere: a parking lot has none.
    2  lane_divider    -1 everywhere: stall lines are not lane dividers, and
                       the data does not contain them anyway.

The raster is 1 px per metre over the MAPPED area (dlp.MAPPED_X_M x
dlp.MAPPED_Y_M: the lot plus the approach road and pavements where agents
are also tracked), with a -1 margin of MARGIN_M on every side.
ContextVAEPredictor.crop_map cuts +/- ext = ceil(56 * sqrt(13)) = 202 px
around the pivot BEFORE rotating, and zero-pads anything past the raster
edge. 0 is not "nothing" in this encoding (it is "half drivable"), so the
margin must keep every crop inside the raster for any pivot in the mapped
area. nuScenes maps get the same guarantee from a 300 m margin.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .dlp import LOT_HEIGHT_M, LOT_WIDTH_M, MAPPED_X_M, MAPPED_Y_M, SOURCE, VEHICLE_LABELS, RawRecording

CROP_EXT_PX = 202            # ContextVAEPredictor.ext for map_size 224 at 1 px/m
MARGIN_M = 210               # >= CROP_EXT_PX, plus slack for points just outside the mapped area
DRIVABLE = 0.5
NOTHING = -1.0
FINE_RES_M = 0.25            # the masks are drawn at 4 px/m, then averaged to 1 px/m
HALF_CAR_WIDTH_M = 1.0
MOVING_MIN_SPEED = 0.3       # m/s; slower samples are parked or creeping, drawn as footprints
STOPPED_MAX_SPEED = 0.05
PARKED_MIN_S = 30.0          # a stop this long is parking; shorter ones are waiting in the aisle
MAX_HOLE_M2 = 40.0           # enclosed holes up to ~3 stalls are filled
STALL_LENGTH_M, STALL_WIDTH_M = 5.2, 2.6
STALL_MERGE_M = 1.5          # closes the gap of one empty stall between parked cars
VARIANTS = ("lot", "aisles")


# -- geometry helpers ----------------------------------------------------
def box_corners(x: float, y: float, heading: float, length: float, width: float) -> np.ndarray:
    """(4, 2) corners of a heading-aligned box centred on (x, y)."""
    c, s = np.cos(heading), np.sin(heading)
    local = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]]) * [length, width]
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def to_fine_px(xy: np.ndarray, res: float = FINE_RES_M) -> np.ndarray:
    """World (x, y) -> fine-raster (col, row) pixel coordinates over the
    mapped area, y flipped because raster rows grow downwards."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([(xy[..., 0] - MAPPED_X_M[0]) / res, (MAPPED_Y_M[1] - xy[..., 1]) / res], axis=-1)


def fine_shape(res: float = FINE_RES_M) -> Tuple[int, int]:
    return (int(round((MAPPED_Y_M[1] - MAPPED_Y_M[0]) / res)),
            int(round((MAPPED_X_M[1] - MAPPED_X_M[0]) / res)))


def lot_to_mapped(mask_lot: np.ndarray) -> np.ndarray:
    """(80, 140) mask of the lot itself -> (H, W) mask of the mapped area at 1 px/m."""
    h, w = fine_shape(1.0)
    out = np.zeros((h, w), dtype=bool)
    r0, c0 = int(MAPPED_Y_M[1] - LOT_HEIGHT_M), int(-MAPPED_X_M[0])
    out[r0:r0 + mask_lot.shape[0], c0:c0 + mask_lot.shape[1]] = mask_lot
    return out


def disk(radius_px: float) -> np.ndarray:
    r = int(np.ceil(radius_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy <= radius_px * radius_px).astype(np.uint8)


def fill_small_holes(mask: np.ndarray, max_area_px: int) -> np.ndarray:
    """Fill enclosed background components no larger than max_area_px.
    Components touching the border are never holes."""
    lab, n = ndimage.label(~mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    small = [i + 1 for i, s in enumerate(sizes) if s <= max_area_px and (i + 1) not in border]
    return mask | np.isin(lab, small)


def downsample(mask_fine: np.ndarray, res: float = FINE_RES_M) -> np.ndarray:
    """Fine mask -> 1 px/m mask: a metre cell is set when at least half of it is."""
    f = int(round(1.0 / res))
    h, w = mask_fine.shape
    return mask_fine.reshape(h // f, f, w // f, f).mean(axis=(1, 3)) >= 0.5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


# -- the two layers ------------------------------------------------------
def parked_boxes(recordings: Iterable[RawRecording]) -> List[np.ndarray]:
    """Footprints of every parked vehicle: DLP obstacles, plus vehicle agents
    while stopped for at least PARKED_MIN_S (they park or leave a stall during
    the recording). Shorter stops are cars waiting in an aisle (for a
    pedestrian, a car backing out); counting them would turn aisles into stalls."""
    boxes = []
    for rec in recordings:
        for o in rec.obstacles:
            if o.label in VEHICLE_LABELS:
                boxes.append(box_corners(o.x, o.y, o.heading, o.length_m, o.width_m))
        for tr in rec.tracks:
            if tr.label not in VEHICLE_LABELS:
                continue
            stopped = (tr.speed < STOPPED_MAX_SPEED).astype(int)
            edges = np.flatnonzero(np.diff(np.concatenate([[0], stopped, [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                if tr.t[b - 1] - tr.t[a] < PARKED_MIN_S:
                    continue
                for i in range(a, b, 25):  # one footprint per second is plenty
                    boxes.append(box_corners(tr.x[i], tr.y[i], tr.heading[i], *tr.size_m))
    return boxes


def trace_mask(recordings: Iterable[RawRecording], res: float = FINE_RES_M,
               half_width_m: float = HALF_CAR_WIDTH_M) -> np.ndarray:
    """Moving vehicle centre-lines, drawn 2 x half_width_m wide."""
    img = np.zeros(fine_shape(res), dtype=np.uint8)
    thick = max(1, int(round(2 * half_width_m / res)))
    for rec in recordings:
        for tr in rec.tracks:
            if tr.label not in VEHICLE_LABELS:
                continue
            px = np.round(to_fine_px(np.stack([tr.x, tr.y], axis=1), res)).astype(np.int32)
            moving = tr.speed >= MOVING_MIN_SPEED
            # Draw each run of consecutive moving samples as one polyline.
            edges = np.flatnonzero(np.diff(np.concatenate([[0], moving.astype(int), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a >= 2:
                    cv2.polylines(img, [px[a:b]], False, 1, thickness=thick)
                elif b - a == 1:
                    cv2.circle(img, tuple(px[a]), thick // 2, 1, -1)
    return img.astype(bool)


def boxes_mask(boxes: List[np.ndarray], res: float = FINE_RES_M) -> np.ndarray:
    img = np.zeros(fine_shape(res), dtype=np.uint8)
    for b in boxes:
        cv2.fillPoly(img, [np.round(to_fine_px(b, res)).astype(np.int32)], 1)
    return img.astype(bool)


def lot_mask(recordings: List[RawRecording], res: float = FINE_RES_M) -> np.ndarray:
    m = trace_mask(recordings, res) | boxes_mask(parked_boxes(recordings), res)
    m = ndimage.binary_closing(m, structure=disk(0.5 / res))
    return fill_small_holes(m, int(MAX_HOLE_M2 / res ** 2))


def stall_band_mask(recordings: List[RawRecording], res: float = FINE_RES_M) -> np.ndarray:
    """Parked footprints grown to a full stall and merged along each row."""
    grown = []
    for b in parked_boxes(recordings):
        centre = b.mean(axis=0)
        d = b[0] - b[3]                                  # front-to-back edge = length direction
        heading = np.arctan2(d[1], d[0])
        length = max(np.hypot(*(b[0] - b[3])), STALL_LENGTH_M)
        width = max(np.hypot(*(b[0] - b[1])), STALL_WIDTH_M)
        grown.append(box_corners(centre[0], centre[1], heading, length, width))
    m = boxes_mask(grown, res)
    return ndimage.binary_closing(m, structure=disk(STALL_MERGE_M / res))


def area_mask(recordings: List[RawRecording], variant: str = "lot", res: float = FINE_RES_M) -> np.ndarray:
    """-> (125, 180) bool at 1 px/m over the mapped area, row 0 = its top edge."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    lot = lot_mask(recordings, res)
    if variant == "lot":
        return downsample(lot, res)
    aisles = lot & ~stall_band_mask(recordings, res)
    aisles = ndimage.binary_opening(aisles, structure=disk(1.0 / res))
    return downsample(aisles, res)


# -- bundle --------------------------------------------------------------
def semantic_from_mask(mask_1m: np.ndarray, margin_m: int = MARGIN_M) -> Tuple[np.ndarray, np.ndarray]:
    """(125, 180) mapped-area mask -> ((3, 125 + 2m, 180 + 2m) float32, H)
    where H maps world (x, y, 1) -> raster (row, col), like the nuScenes maps."""
    h, w = mask_1m.shape
    sem = np.full((3, h + 2 * margin_m, w + 2 * margin_m), NOTHING, dtype=np.float32)
    sem[0, margin_m:margin_m + h, margin_m:margin_m + w][mask_1m] = DRIVABLE
    H = np.array([[0.0, -1.0, MAPPED_Y_M[1] + margin_m],
                  [1.0, 0.0, margin_m - MAPPED_X_M[0]],
                  [0.0, 0.0, 1.0]])
    return sem, H


def bundle_meta(recordings: List[RawRecording], variant: str, fov_x: Tuple[float, float],
                fov_x_per_recording: Optional[dict] = None, fov_y_per_recording: Optional[dict] = None,
                provenance: Optional[dict] = None) -> dict:
    # The dashboard view is 800 x 600 px and scales x and y independently, so
    # the declared extent is the mapped area widened to 4:3 (180 x 135 m,
    # centred) to keep metres square on screen.
    width = MAPPED_X_M[1] - MAPPED_X_M[0]
    height = width * 3 / 4
    y_mid = (MAPPED_Y_M[0] + MAPPED_Y_M[1]) / 2
    return {
        "world_width_m": width,
        "world_height_m": height,
        "world_origin_x": MAPPED_X_M[0],
        "world_origin_y": y_mid - height / 2,
        "pixels_per_metre": 800 / width,
        "lot_m": {"x": [0.0, LOT_WIDTH_M], "y": [0.0, LOT_HEIGHT_M]},
        "mapped_area_m": {"x": list(MAPPED_X_M), "y": list(MAPPED_Y_M)},
        "road_category": None,
        "source": SOURCE,
        "built_from": [r.name for r in recordings],
        **(provenance or {}),
        "variant": variant,
        "semantic_pixels_per_metre": 1.0,
        "semantic_margin_m": MARGIN_M,
        "semantic_channels": {"0": f"drivable = {DRIVABLE} on the {variant} area, -1 elsewhere",
                              "1": "road_divider: -1 everywhere (none in a parking lot)",
                              "2": "lane_divider: -1 everywhere"},
        "fov_x_m": [round(fov_x[0], 2), round(fov_x[1], 2)],
        "fov_x_m_per_recording": {k: [round(a, 2), round(b, 2)] for k, (a, b) in (fov_x_per_recording or {}).items()},
        "fov_y_m_per_recording": {k: [round(a, 2), round(b, 2)] for k, (a, b) in (fov_y_per_recording or {}).items()},
        "approximations": [
            "Derived from the CC0 recordings only, not from the GPL-3 toolkit map: the drivable area is where "
            "vehicles were seen, so parts of the lot outside the drone's view (fov_*_per_recording) or never driven "
            "or parked in during the recordings listed in built_from are missing.",
            f"Moving-vehicle centre traces (speed >= {MOVING_MIN_SPEED} m/s) drawn {2 * HALF_CAR_WIDTH_M} m wide; "
            f"parked footprints are DLP obstacles plus vehicle agents stopped >= {PARKED_MIN_S:.0f} s; enclosed holes "
            f"<= {MAX_HOLE_M2} m2 filled.",
            "aisles variant: stall bands = parked footprints grown to "
            f"{STALL_LENGTH_M} x {STALL_WIDTH_M} m and closed with a {STALL_MERGE_M} m disk; rows with no parked "
            "car in any recording stay drivable.",
            "world_* describe a 4:3 dashboard view around mapped_area_m, not the lot itself (lot_m).",
        ],
        "_comment": "fov_x_m is the drone's x-range estimated as the extent of all agent positions (union over "
                    "built_from); the drone was not in the same place for every recording, so the per-recording "
                    "ranges differ and are what censoring uses. Tracks that start or end within 5 m of their "
                    "recording's range were cut off by the camera, not real entries/exits; see the per-agent "
                    "censored_start / censored_end flags in data/dlp/<recording>.meta.json.",
    }


def write_bundle(out_dir: str, sem: np.ndarray, H: np.ndarray, meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "semantic_map.pkl"), "wb") as f:
        pickle.dump((sem, H), f, protocol=4)
    with open(os.path.join(out_dir, "map_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    with open(os.path.join(out_dir, "waypoints.json"), "w") as f:
        # The simulator's route graph; DLP is replayed, never simulated.
        json.dump({"waypoints": []}, f)
        f.write("\n")


def build_bundle(recordings: List[RawRecording], out_dir: str, variant: str = "lot",
                 fov_x: Optional[Tuple[float, float]] = None, provenance: Optional[dict] = None) -> np.ndarray:
    """Write maps/<name>/ and return the (80, 140) mask it was built from.
    ``provenance`` (e.g. which split the recordings came from) is copied into
    map_meta.json, because a map built from vehicle traces must never include
    the recordings it is evaluated on."""
    from .dlp import fov_x_range, fov_y_range
    mask = area_mask(recordings, variant)
    sem, H = semantic_from_mask(mask)
    per_x = {r.name: fov_x_range(r) for r in recordings}
    per_y = {r.name: fov_y_range(r) for r in recordings}
    if fov_x is None:
        fov_x = (min(a for a, _ in per_x.values()), max(b for _, b in per_x.values()))
    write_bundle(out_dir, sem, H, bundle_meta(recordings, variant, fov_x, per_x, per_y, provenance))
    return mask


__all__ = ["area_mask", "semantic_from_mask", "lot_to_mapped", "build_bundle", "write_bundle", "bundle_meta", "iou",
           "box_corners", "fill_small_holes", "MARGIN_M", "CROP_EXT_PX", "DRIVABLE", "VARIANTS"]
