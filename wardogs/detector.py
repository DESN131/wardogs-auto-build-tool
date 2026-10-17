"""Detector for the build-marker X in the game screenshots.

The marker is a yellow, hand-painted X projected onto the ground.  It turns
white when the crosshair is on it.  The detector intentionally uses a narrow
colour range: the marker is an unlit overlay with an almost constant colour
(measured RGB ~= 255,250,82), while nearby yellow scenery (the curb in
3.png/4.png) is much less saturated and has more blue.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    centroid: tuple[float, float]
    area: int
    fill: float
    aspect: float
    hsv_median: tuple[float, float, float]


@dataclass
class Detection:
    width: int
    height: int
    yellow: list[Candidate]
    white: list[Candidate]
    best: Optional[Candidate]
    locked: bool
    center_v: float
    center_s: float
    ring_v: float
    white_relaxed: list[Candidate] = field(default_factory=list)

    @property
    def screen_center(self) -> tuple[float, float]:
        return (self.width / 2.0, self.height / 2.0)

    def error(self) -> Optional[tuple[float, float]]:
        if self.best is None:
            return None
        cx, cy = self.screen_center
        return (cx - self.best.center[0], cy - self.best.center[1])


@dataclass
class WhiteConfig:
    v_min: int = 185
    relaxed_v: int = 170
    relaxed_s_max: int = 55
    dv_min: float = 35.0
    s_max: int = 50
    min_area: int = 600
    max_area: int = 15000
    min_w: int = 35
    max_w: int = 400
    min_h: int = 20
    max_h: int = 300
    fill_min: float = 0.15
    fill_max: float = 0.65
    aspect_min: float = 0.4
    aspect_max: float = 3.5
    click_radius_px: float = 120.0


@dataclass
class DetectorConfig:
    hsv_low: tuple[int, int, int] = (22, 130, 110)
    hsv_high: tuple[int, int, int] = (36, 255, 255)
    max_b_over_g: float = 0.55
    max_rg_over_g: float = 0.35
    min_area: int = 600
    max_area: int = 12000
    min_w: int = 40
    max_w: int = 400
    min_h: int = 30
    max_h: int = 300
    fill_min: float = 0.15
    fill_max: float = 0.65
    aspect_min: float = 0.5
    aspect_max: float = 3.0
    morph_kernel: int = 3
    center_ui_radius: int = 20
    center_ui_max_size: int = 60
    fast_mode: bool = True
    white: WhiteConfig = field(default_factory=WhiteConfig)

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "DetectorConfig":
        data = data or {}
        white_data = data.get("white") or {}
        white = WhiteConfig(**{k: v for k, v in white_data.items()
                               if k in WhiteConfig.__dataclass_fields__})
        values = {k: v for k, v in data.items()
                  if k in cls.__dataclass_fields__ and k != "white"}
        if "hsv_low" in values:
            values["hsv_low"] = tuple(int(x) for x in values["hsv_low"])
        if "hsv_high" in values:
            values["hsv_high"] = tuple(int(x) for x in values["hsv_high"])
        return cls(white=white, **values)


def _candidates_from_components(
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    hsv: np.ndarray,
    *,
    min_area: int,
    max_area: int,
    min_w: int,
    max_w: int,
    min_h: int,
    max_h: int,
    fill_min: float,
    fill_max: float,
    aspect_min: float,
    aspect_max: float,
    center: tuple[float, float],
    include_hsv: bool = True,
) -> list[Candidate]:
    count = len(stats)
    out: list[Candidate] = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if not (min_area <= area <= max_area):
            continue
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            continue
        fill = area / float(max(w * h, 1))
        aspect = w / float(max(h, 1))
        if not (fill_min <= fill <= fill_max and aspect_min <= aspect <= aspect_max):
            continue
        med = (0.0, 0.0, 0.0)
        if include_hsv:
            pixels = hsv[labels == label]
            if len(pixels):
                med = tuple(float(v) for v in np.median(pixels, axis=0))
        out.append(
            Candidate(
                bbox=(x, y, w, h),
                center=(x + w / 2.0, y + h / 2.0),
                centroid=(float(centroids[label][0]), float(centroids[label][1])),
                area=area,
                fill=fill,
                aspect=aspect,
                hsv_median=med,  # type: ignore[arg-type]
            )
        )
    out.sort(key=lambda c: (c.center[0] - center[0]) ** 2 + (c.center[1] - center[1]) ** 2)
    return out


class MarkerDetector:
    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        self._kernel = np.ones(
            (max(1, int(self.cfg.morph_kernel)), max(1, int(self.cfg.morph_kernel))),
            np.uint8,
        )

    def yellow_mask(self, frame_bgr: np.ndarray, hsv: Optional[np.ndarray] = None) -> np.ndarray:
        if hsv is None:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        cfg = self.cfg
        mask = cv2.inRange(hsv, np.array(cfg.hsv_low, np.uint8),
                           np.array(cfg.hsv_high, np.uint8))
        # Chromaticity gates are invariant to overall display brightness.
        b, g, r = cv2.split(frame_bgr.astype(np.float32))
        g_safe = np.maximum(g, 1.0)
        bad = (b > cfg.max_b_over_g * g_safe) | (np.abs(r - g) > cfg.max_rg_over_g * g_safe)
        mask[bad] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    def white_mask(self, hsv: np.ndarray, relaxed: bool = False) -> np.ndarray:
        w = self.cfg.white
        low_v = w.relaxed_v if relaxed else w.v_min
        s_max = w.relaxed_s_max if relaxed else w.s_max
        mask = cv2.inRange(hsv, np.array([0, 0, low_v], np.uint8),
                           np.array([180, s_max, 255], np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    @staticmethod
    def _ring_v(hsv: np.ndarray, cx: int, cy: int) -> float:
        h, w = hsv.shape[:2]
        values: list[float] = []
        for radius in (24, 32, 40):
            for degrees in range(0, 360, 30):
                angle = np.deg2rad(degrees)
                x = int(round(cx + radius * np.cos(angle)))
                y = int(round(cy + radius * np.sin(angle)))
                if 0 <= x < w and 0 <= y < h:
                    values.append(float(hsv[y, x, 2]))
        return float(np.mean(values)) if values else 0.0

    def _white_component_ok(self, stats: np.ndarray, label: int) -> bool:
        if label <= 0 or label >= len(stats):
            return False
        wcfg = self.cfg.white
        x, y, cw, ch, area = (int(v) for v in stats[label])
        if not (wcfg.min_area <= area <= wcfg.max_area):
            return False
        if not (wcfg.min_w <= cw <= wcfg.max_w and wcfg.min_h <= ch <= wcfg.max_h):
            return False
        fill = area / float(max(cw * ch, 1))
        aspect = cw / float(max(ch, 1))
        return (wcfg.fill_min <= fill <= wcfg.fill_max
                and wcfg.aspect_min <= aspect <= wcfg.aspect_max)

    def detect(
        self,
        frame_bgr: np.ndarray,
        center: Optional[tuple[float, float]] = None,
    ) -> Detection:
        """Fast two-stage detection with the full detector as fallback."""
        if (self.cfg.fast_mode and frame_bgr is not None
                and frame_bgr.size
                and min(frame_bgr.shape[:2]) >= 400):
            fast = self._detect_fast(frame_bgr, center)
            if fast is not None and (fast.best is not None or fast.locked):
                return fast
        return self._detect_full(frame_bgr, center)

    def _detect_full(
        self,
        frame_bgr: np.ndarray,
        center: Optional[tuple[float, float]] = None,
    ) -> Detection:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("empty frame")
        height, width = frame_bgr.shape[:2]
        if center is None:
            center = (width / 2.0, height / 2.0)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        cfg = self.cfg
        yellow_mask = self.yellow_mask(frame_bgr, hsv)
        _, y_labels, y_stats, y_centroids = cv2.connectedComponentsWithStats(yellow_mask, 8)
        yellow = _candidates_from_components(
            y_labels, y_stats, y_centroids, hsv,
            min_area=cfg.min_area, max_area=cfg.max_area,
            min_w=cfg.min_w, max_w=cfg.max_w,
            min_h=cfg.min_h, max_h=cfg.max_h,
            fill_min=cfg.fill_min, fill_max=cfg.fill_max,
            aspect_min=cfg.aspect_min, aspect_max=cfg.aspect_max,
            center=center,
        )
        # A small yellow build-UI/ghost often sits exactly on the crosshair
        # after a build.  It is closer than the real X but is not a buildable
        # marker; ignoring it avoids aiming at the crosshair itself.
        yellow = [
            c for c in yellow
            if not (
                abs(c.center[0] - center[0]) <= cfg.center_ui_radius
                and abs(c.center[1] - center[1]) <= cfg.center_ui_radius
                and c.bbox[2] < cfg.center_ui_max_size
            )
        ]

        wcfg = cfg.white
        cx = int(np.clip(round(center[0]), 0, width - 1))
        cy = int(np.clip(round(center[1]), 0, height - 1))
        patch = hsv[max(0, cy - 4):min(height, cy + 5),
                    max(0, cx - 4):min(width, cx + 5)]
        if patch.size:
            center_v = float(patch[:, :, 2].mean())
            center_s = float(patch[:, :, 1].mean())
            patch_white_frac = float(
                ((patch[:, :, 2] >= wcfg.v_min) & (patch[:, :, 1] <= wcfg.s_max)).mean()
            )
            patch_white_frac_relaxed = float(
                ((patch[:, :, 2] >= wcfg.relaxed_v) & (patch[:, :, 1] <= wcfg.s_max)).mean()
            )
        else:
            center_v = center_s = patch_white_frac = patch_white_frac_relaxed = 0.0
        ring_v = self._ring_v(hsv, cx, cy)

        best = yellow[0] if yellow else None
        # When the centre is clearly bright, the X may already be white even
        # if the yellow mask still sees a ghost/outline nearby.  Do not skip
        # the white mask in that case.
        if best is not None and center_v < wcfg.relaxed_v:
            return Detection(
                width=width, height=height, yellow=yellow, white=[],
                best=best, locked=False,
                center_v=center_v, center_s=center_s, ring_v=ring_v,
            )

        white_strict = self.white_mask(hsv, relaxed=False)
        _, w_labels, w_stats, w_centroids = cv2.connectedComponentsWithStats(white_strict, 8)
        white = _candidates_from_components(
            w_labels, w_stats, w_centroids, hsv,
            min_area=wcfg.min_area, max_area=wcfg.max_area,
            min_w=wcfg.min_w, max_w=wcfg.max_w,
            min_h=wcfg.min_h, max_h=wcfg.max_h,
            fill_min=wcfg.fill_min, fill_max=wcfg.fill_max,
            aspect_min=wcfg.aspect_min, aspect_max=wcfg.aspect_max,
            center=center,
            include_hsv=False,
        )

        # The marker's crossing point is directly under the crosshair.  Requiring
        # the centre pixel to belong to the white component is more robust than
        # requiring the white bbox centre to be within N pixels of the screen
        # centre: touching an arm tip is enough to make the game buildable.
        label = int(w_labels[cy, cx])
        comp_ok = self._white_component_ok(w_stats, label)
        bright = patch_white_frac >= 0.45
        if not comp_ok:
            if (center_v >= wcfg.relaxed_v and center_s <= wcfg.s_max
                    and center_v - ring_v >= wcfg.dv_min
                    and patch_white_frac_relaxed >= 0.45):
                white_relaxed = self.white_mask(hsv, relaxed=True)
                _, r_labels, r_stats, _ = cv2.connectedComponentsWithStats(white_relaxed, 8)
                relaxed_label = int(r_labels[cy, cx])
                comp_ok = self._white_component_ok(r_stats, relaxed_label)
                bright = comp_ok

        primary_lock = bool(comp_ok and bright and center_s <= wcfg.s_max
                            and (center_v >= wcfg.relaxed_v
                                 or center_v - ring_v >= wcfg.dv_min))
        # If the crosshair is drawn over the white marker, the exact centre
        # pixel is not white.  A valid white component whose bbox contains the
        # crosshair still means "the X under the crosshair is white".
        bbox_contains_centre = any(
            x <= center[0] <= x + cw and y <= center[1] <= y + ch
            for x, y, cw, ch in (c.bbox for c in white)
        )
        white_near_centre = any(
            math.hypot(c.center[0] - center[0], c.center[1] - center[1])
            <= wcfg.click_radius_px
            for c in white
        )
        white_relaxed: list[Candidate] = []
        if not primary_lock and not bbox_contains_centre:
            relaxed_mask = self.white_mask(hsv, relaxed=True)
            _, r_labels, r_stats, r_centroids = cv2.connectedComponentsWithStats(
                relaxed_mask, 8
            )
            white_relaxed = _candidates_from_components(
                r_labels, r_stats, r_centroids, hsv,
                min_area=wcfg.min_area, max_area=wcfg.max_area,
                min_w=wcfg.min_w, max_w=wcfg.max_w,
                min_h=wcfg.min_h, max_h=wcfg.max_h,
                fill_min=wcfg.fill_min, fill_max=wcfg.fill_max,
                aspect_min=wcfg.aspect_min, aspect_max=wcfg.aspect_max,
                center=center,
                include_hsv=False,
            )
            bbox_contains_centre = any(
                x <= center[0] <= x + cw and y <= center[1] <= y + ch
                for x, y, cw, ch in (c.bbox for c in white_relaxed)
            )
            white_near_centre = any(
                math.hypot(c.center[0] - center[0], c.center[1] - center[1])
                <= wcfg.click_radius_px
                for c in white_relaxed
            )
        # A white lock is only valid when the yellow marker is gone.  The
        # build surface itself may be white (WARDOGS containers), so allowing
        # a white background to lock while a yellow X is still visible would
        # click far away from the marker.
        # The centre-white test is strong enough to coexist with a yellow
        # ghost: the crosshair is on the white X.  Off-centre white candidates
        # remain guarded by "no yellow" to avoid clicking white background.
        locked = bool(primary_lock or (
            (bbox_contains_centre or white_near_centre) and best is None
        ))

        return Detection(
            width=width,
            height=height,
            yellow=yellow,
            white=white,
            best=best,
            locked=locked,
            center_v=center_v,
            center_s=center_s,
            ring_v=ring_v,
            white_relaxed=white_relaxed,
        )

    def _refine_fast_candidate(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        expected: tuple[float, float],
        center: tuple[float, float],
    ) -> Optional[Candidate]:
        """Confirm one downscaled candidate on the full-resolution image."""
        cfg = self.cfg
        height, width = frame_bgr.shape[:2]
        x, y, w, h = bbox
        margin = max(10, int(max(w, h) * 0.18))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(width, x + w + margin)
        y1 = min(height, y + h + margin)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        crop = frame_bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(cfg.hsv_low, np.uint8),
                           np.array(cfg.hsv_high, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        local_center = (expected[0] - x0, expected[1] - y0)
        candidates = _candidates_from_components(
            labels, stats, centroids, hsv,
            min_area=cfg.min_area, max_area=cfg.max_area,
            min_w=cfg.min_w, max_w=cfg.max_w,
            min_h=cfg.min_h, max_h=cfg.max_h,
            fill_min=cfg.fill_min, fill_max=cfg.fill_max,
            aspect_min=cfg.aspect_min, aspect_max=cfg.aspect_max,
            center=local_center,
        )
        for candidate in candidates:
            bx, by, bw, bh = candidate.bbox
            submask = mask[by:by + bh, bx:bx + bw]
            if not submask.size or not np.any(submask):
                continue
            pixels = crop[by:by + bh, bx:bx + bw][submask > 0]
            if not len(pixels):
                continue
            b, g, r = (float(v) for v in np.median(pixels, axis=0))
            g_safe = max(g, 1.0)
            if (b > cfg.max_b_over_g * g_safe
                    or abs(r - g) > cfg.max_rg_over_g * g_safe):
                continue
            return replace(
                candidate,
                bbox=(bx + x0, by + y0, bw, bh),
                center=(candidate.center[0] + x0,
                        candidate.center[1] + y0),
                centroid=(candidate.centroid[0] + x0,
                          candidate.centroid[1] + y0),
            )
        return None

    def _fast_white_info(
        self,
        frame_bgr: np.ndarray,
        center: tuple[float, float],
        best: Optional[Candidate],
    ) -> tuple[bool, list[Candidate], list[Candidate], float, float, float]:
        """White-lock check on a small centre crop, mapped back to full frame."""
        height, width = frame_bgr.shape[:2]
        wcfg = self.cfg.white
        cx = int(np.clip(round(center[0]), 0, width - 1))
        cy = int(np.clip(round(center[1]), 0, height - 1))
        half = 160
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(width, cx + half), min(height, cy + half)
        crop = frame_bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        ch, cw = hsv.shape[:2]
        lcx = int(np.clip(cx - x0, 0, cw - 1))
        lcy = int(np.clip(cy - y0, 0, ch - 1))
        patch = hsv[max(0, lcy - 4):min(ch, lcy + 5),
                    max(0, lcx - 4):min(cw, lcx + 5)]
        if patch.size:
            center_v = float(patch[:, :, 2].mean())
            center_s = float(patch[:, :, 1].mean())
            patch_white_frac = float(
                ((patch[:, :, 2] >= wcfg.v_min)
                 & (patch[:, :, 1] <= wcfg.s_max)).mean()
            )
            patch_white_frac_relaxed = float(
                ((patch[:, :, 2] >= wcfg.relaxed_v)
                 & (patch[:, :, 1] <= wcfg.s_max)).mean()
            )
        else:
            center_v = center_s = 0.0
            patch_white_frac = patch_white_frac_relaxed = 0.0
        ring_v = self._ring_v(hsv, lcx, lcy)
        white_mask = self.white_mask(hsv, relaxed=False)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(
            white_mask, 8
        )
        white = _candidates_from_components(
            labels, stats, centroids, hsv,
            min_area=wcfg.min_area, max_area=wcfg.max_area,
            min_w=wcfg.min_w, max_w=wcfg.max_w,
            min_h=wcfg.min_h, max_h=wcfg.max_h,
            fill_min=wcfg.fill_min, fill_max=wcfg.fill_max,
            aspect_min=wcfg.aspect_min, aspect_max=wcfg.aspect_max,
            center=(lcx, lcy), include_hsv=False,
        )
        label = int(labels[lcy, lcx])
        comp_ok = self._white_component_ok(stats, label)
        bright = patch_white_frac >= 0.45
        if (not comp_ok
                and center_v >= wcfg.relaxed_v
                and center_s <= wcfg.s_max
                and center_v - ring_v >= wcfg.dv_min
                and patch_white_frac_relaxed >= 0.45):
            relaxed_mask = self.white_mask(hsv, relaxed=True)
            _, r_labels, r_stats, _ = cv2.connectedComponentsWithStats(
                relaxed_mask, 8
            )
            comp_ok = self._white_component_ok(
                r_stats, int(r_labels[lcy, lcx])
            )
            bright = comp_ok
        primary_lock = bool(
            comp_ok and bright and center_s <= wcfg.s_max
            and (center_v >= wcfg.relaxed_v
                 or center_v - ring_v >= wcfg.dv_min)
        )
        bbox_contains = any(
            x <= lcx <= x + cw and y <= lcy <= y + ch
            for x, y, cw, ch in (c.bbox for c in white)
        )
        white_near = any(
            math.hypot(c.center[0] - lcx, c.center[1] - lcy)
            <= wcfg.click_radius_px
            for c in white
        )
        white_relaxed: list[Candidate] = []
        if not primary_lock and not bbox_contains:
            relaxed_mask = self.white_mask(hsv, relaxed=True)
            _, r_labels, r_stats, r_centroids = (
                cv2.connectedComponentsWithStats(relaxed_mask, 8)
            )
            white_relaxed = _candidates_from_components(
                r_labels, r_stats, r_centroids, hsv,
                min_area=wcfg.min_area, max_area=wcfg.max_area,
                min_w=wcfg.min_w, max_w=wcfg.max_w,
                min_h=wcfg.min_h, max_h=wcfg.max_h,
                fill_min=wcfg.fill_min, fill_max=wcfg.fill_max,
                aspect_min=wcfg.aspect_min, aspect_max=wcfg.aspect_max,
                center=(lcx, lcy), include_hsv=False,
            )
            bbox_contains = any(
                x <= lcx <= x + cw and y <= lcy <= y + ch
                for x, y, cw, ch in (c.bbox for c in white_relaxed)
            )
            white_near = any(
                math.hypot(c.center[0] - lcx, c.center[1] - lcy)
                <= wcfg.click_radius_px
                for c in white_relaxed
            )
        locked = bool(
            primary_lock
            or ((bbox_contains or white_near) and best is None)
        )

        def shift(candidate: Candidate) -> Candidate:
            x, y, sw, sh = candidate.bbox
            return replace(
                candidate,
                bbox=(x + x0, y + y0, sw, sh),
                center=(candidate.center[0] + x0,
                        candidate.center[1] + y0),
                centroid=(candidate.centroid[0] + x0,
                          candidate.centroid[1] + y0),
            )

        return (
            locked,
            [shift(c) for c in white],
            [shift(c) for c in white_relaxed],
            center_v,
            center_s,
            ring_v,
        )

    def _detect_fast(
        self,
        frame_bgr: np.ndarray,
        center: Optional[tuple[float, float]] = None,
    ) -> Optional[Detection]:
        """Downscale the search, then refine candidates at full resolution."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("empty frame")
        height, width = frame_bgr.shape[:2]
        if center is None:
            center = (width / 2.0, height / 2.0)
        cfg = self.cfg
        wcfg = cfg.white
        cx = int(np.clip(round(center[0]), 0, width - 1))
        cy = int(np.clip(round(center[1]), 0, height - 1))
        # Only the centre neighbourhood is needed for V/S/ring; converting the
        # whole full-resolution frame to HSV here would negate the fast path.
        r = 48
        px0, py0 = max(0, cx - r), max(0, cy - r)
        px1, py1 = min(width, cx + r + 1), min(height, cy + r + 1)
        centre_hsv = cv2.cvtColor(frame_bgr[py0:py1, px0:px1],
                                  cv2.COLOR_BGR2HSV)
        lcx, lcy = cx - px0, cy - py0
        patch = centre_hsv[max(0, lcy - 4):min(centre_hsv.shape[0], lcy + 5),
                           max(0, lcx - 4):min(centre_hsv.shape[1], lcx + 5)]
        if patch.size:
            center_v = float(patch[:, :, 2].mean())
            center_s = float(patch[:, :, 1].mean())
        else:
            center_v = center_s = 0.0
        ring_v = self._ring_v(centre_hsv, lcx, lcy)
        scale = 2
        sw, sh = max(1, width // scale), max(1, height // scale)
        small = cv2.resize(frame_bgr, (sw, sh),
                           interpolation=cv2.INTER_AREA)
        hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_small, np.array(cfg.hsv_low, np.uint8),
                           np.array(cfg.hsv_high, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        coarse: list[tuple[float, tuple[int, int, int, int],
                           tuple[float, float]]] = []
        min_area_s = max(20.0, cfg.min_area / (scale * scale) * 0.5)
        max_area_s = cfg.max_area / (scale * scale) * 1.6
        min_w_s = max(4.0, cfg.min_w / scale * 0.5)
        max_w_s = cfg.max_w / scale * 1.6
        min_h_s = max(4.0, cfg.min_h / scale * 0.5)
        max_h_s = cfg.max_h / scale * 1.6
        for label in range(1, len(stats)):
            x, y, cw, ch, area = (int(v) for v in stats[label])
            if not (min_area_s <= area <= max_area_s):
                continue
            if not (min_w_s <= cw <= max_w_s
                    and min_h_s <= ch <= max_h_s):
                continue
            fill = area / float(max(cw * ch, 1))
            aspect = cw / float(max(ch, 1))
            if not (cfg.fill_min * 0.5 <= fill <= cfg.fill_max * 1.4
                    and cfg.aspect_min * 0.7 <= aspect <= cfg.aspect_max * 1.4):
                continue
            ex = (x + cw / 2.0) * scale
            ey = (y + ch / 2.0) * scale
            coarse.append((
                math.hypot(ex - center[0], ey - center[1]),
                (x * scale, y * scale, cw * scale, ch * scale),
                (ex, ey),
            ))
        coarse.sort(key=lambda item: item[0])
        yellow: list[Candidate] = []
        seen: set[tuple[int, int]] = set()
        for _, bbox, expected in coarse[:6]:
            candidate = self._refine_fast_candidate(
                frame_bgr, bbox, expected, center
            )
            if candidate is None:
                continue
            key = (round(candidate.center[0] / 6),
                   round(candidate.center[1] / 6))
            if key in seen:
                continue
            seen.add(key)
            yellow.append(candidate)
        # The small yellow build-UI/ghost on the crosshair is not a target.
        yellow = [
            c for c in yellow
            if not (
                abs(c.center[0] - center[0]) <= cfg.center_ui_radius
                and abs(c.center[1] - center[1]) <= cfg.center_ui_radius
                and c.bbox[2] < cfg.center_ui_max_size
            )
        ]
        yellow.sort(key=lambda c: (c.center[0] - center[0]) ** 2
                    + (c.center[1] - center[1]) ** 2)
        best = yellow[0] if yellow else None
        white: list[Candidate] = []
        white_relaxed: list[Candidate] = []
        locked = False
        if (best is None
                or center_v >= wcfg.relaxed_v - 20.0
                or center_v - ring_v >= wcfg.dv_min):
            (locked, white, white_relaxed,
             center_v, center_s, ring_v) = self._fast_white_info(
                frame_bgr, center, best
            )
        if best is None and not locked:
            return None
        return Detection(
            width=width,
            height=height,
            yellow=yellow,
            white=white,
            best=best,
            locked=locked,
            center_v=center_v,
            center_s=center_s,
            ring_v=ring_v,
            white_relaxed=white_relaxed,
        )


def annotate(frame_bgr: np.ndarray, det: Detection) -> np.ndarray:
    """Return a copy with detection overlays; useful for dry-run debugging."""
    out = frame_bgr.copy()
    cx, cy = (int(round(v)) for v in det.screen_center)
    for i, c in enumerate(det.yellow):
        x, y, w, h = c.bbox
        color = (0, 255, 0) if i else (0, 200, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2 if i == 0 else 1)
        cv2.drawMarker(out, (int(round(c.center[0])), int(round(c.center[1]))),
                       (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
    for c in det.white:
        x, y, w, h = c.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    status = "WHITE LOCK" if det.locked else ("YELLOW" if det.best else "SEARCH")
    if det.best is not None:
        ex, ey = det.error() or (0.0, 0.0)
        text = f"{status} e=({ex:+.0f},{ey:+.0f}) box={det.best.bbox}"
    else:
        text = f"{status} V={det.center_v:.0f} S={det.center_s:.0f}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if det.locked or det.best else (0, 0, 255), 1, cv2.LINE_AA)
    return out


def shift_detection(
    det: Detection,
    dx: int,
    dy: int,
    width: int,
    height: int,
) -> Detection:
    """Translate a ROI-local Detection back to full-frame coordinates."""
    def shift(c: Optional[Candidate]) -> Optional[Candidate]:
        if c is None:
            return None
        x, y, w, h = c.bbox
        return replace(
            c,
            bbox=(x + dx, y + dy, w, h),
            center=(c.center[0] + dx, c.center[1] + dy),
            centroid=(c.centroid[0] + dx, c.centroid[1] + dy),
        )

    return Detection(
        width=width,
        height=height,
        yellow=[shift(c) for c in det.yellow],
        white=[shift(c) for c in det.white],
        best=shift(det.best),
        locked=det.locked,
        center_v=det.center_v,
        center_s=det.center_s,
        ring_v=det.ring_v,
        white_relaxed=[shift(c) for c in det.white_relaxed],
    )
