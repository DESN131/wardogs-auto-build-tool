"""Automatic build loop for the yellow-X build marker.

Flow implemented here:
  yellow X -> human-like mouse movement -> white lock -> LMB down 510 ms ->
  release -> wait for the X to move -> repeat.

Every N successful aimed builds, the bot performs one intentional click while
the crosshair is *not* on the X (the "non-aim click" requested by the user).

Use F8 to start/pause/resume and F9 to stop.  Use --dry-run first.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import mss
import numpy as np
import yaml

from .detector import (
    Candidate,
    Detection,
    DetectorConfig,
    MarkerDetector,
    annotate,
    shift_detection,
)
from .win_input import (
    HumanConfig,
    Hotkeys,
    MouseMover,
    client_region,
    cursor_pos,
    enable_dpi_awareness,
    ensure_cursor_in_region,
    find_window,
    is_foreground,
    list_windows,
    send_left_down,
    send_left_up,
    window_title,
    window_alive,
)

log = logging.getLogger("auto_build")

# mss 10.x emits a runtime DeprecationWarning for mss.mss(); PowerShell treats
# stderr output from a native command as an error when Tee-Object is used.
warnings.filterwarnings(
    "ignore",
    message=r"mss\.mss is deprecated.*",
    category=DeprecationWarning,
)


class Aborted(Exception):
    pass


def cfg_get(data: dict[str, Any], path: str, default: Any) -> Any:
    node: Any = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


class Capture:
    """Small mss/dxcam wrapper returning BGR frames for a client region."""

    def __init__(self, backend: str = "mss", monitor_index: int = 1):
        self.backend = backend
        self.monitor_index = monitor_index
        mss_factory = getattr(mss, "MSS", None)
        self._mss = mss_factory() if mss_factory is not None else mss.mss()
        self._dxcam = None
        if backend == "dxcam":
            try:
                import dxcam  # type: ignore
                self._dxcam = dxcam.create(output_idx=max(0, monitor_index - 1),
                                          output_color="BGR")
            except Exception as exc:
                log.warning("dxcam 不可用，回退 mss: %s", exc)
                self.backend = "mss"

    def monitor_region(self) -> tuple[int, int, int, int]:
        monitors = self._mss.monitors
        index = min(max(1, self.monitor_index), len(monitors) - 1)
        mon = monitors[index]
        return (int(mon["left"]), int(mon["top"]), int(mon["width"]), int(mon["height"]))

    def grab(self, region: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        left, top, width, height = region
        if self.backend == "dxcam" and self._dxcam is not None:
            frame = self._dxcam.grab(region=(left, top, left + width, top + height))
            return frame if frame is not None else None
        shot = self._mss.grab({"left": left, "top": top,
                               "width": width, "height": height})
        frame = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(
            shot.height, shot.width, 4
        )
        return frame[:, :, :3]

    def close(self) -> None:
        try:
            self._mss.close()
        except Exception:
            pass
        if self._dxcam is not None:
            try:
                self._dxcam.release()
            except Exception:
                pass


@dataclass
class Stats:
    aimed_builds: int = 0
    non_aim_clicks: int = 0
    failed_clicks: int = 0
    swings: int = 0
    aborted_swings: int = 0
    no_target_swings: int = 0
    first_swing_at: Optional[float] = None
    last_swing_at: Optional[float] = None
    started_at: float = field(default_factory=time.monotonic)


class BuildBot:
    def __init__(
        self,
        config: dict[str, Any],
        detector: MarkerDetector,
        capture: Capture,
        mover: MouseMover,
        hotkeys: Hotkeys,
        *,
        hwnd: Optional[int],
        fixed_region: Optional[tuple[int, int, int, int]],
        dry_run: bool,
        max_builds: int,
        debug_dir: Optional[Path],
        debug_trace: bool = False,
    ):
        self.config = config
        self.detector = detector
        self.capture = capture
        self.mover = mover
        self.hotkeys = hotkeys
        self.hwnd = hwnd
        self.fixed_region = fixed_region
        self.dry_run = dry_run
        self.max_builds = max_builds
        self.debug_dir = debug_dir
        self.roi_half = int(cfg_get(config, "detector.roi_half", 500))
        self.debug_trace = debug_trace
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        aim = config.get("aim") or {}
        self.gain_x = float(aim.get("gain_x", -0.55))
        self.gain_y = float(aim.get("gain_y", -0.55))
        self.gain_min = float(aim.get("gain_min", 0.05))
        self.gain_max = float(aim.get("gain_max", 5.0))
        self.gain_ema = float(aim.get("gain_ema", 0.35))
        self.alpha_far = float(aim.get("alpha_far", 0.88))
        self.alpha_near = float(aim.get("alpha_near", 0.76))
        self.far_px = float(aim.get("far_px", 70.0))
        self.max_step = float(aim.get("max_step_counts", 360.0))
        self.max_step_px = float(aim.get("max_step_px", 520.0))
        self.min_step = float(aim.get("min_step_counts", 6.0))
        self.white_frames = int(aim.get("white_frames", 1))
        self.stable_frames = int(aim.get("stable_frames", 1))
        self.stable_px = float(aim.get("stable_px", 8.0))
        self.settle_ms = tuple(float(v) for v in aim.get("settle_ms", [25, 50]))
        self.max_iterations = int(aim.get("max_iterations", 8))
        self.lost_timeout = float(aim.get("lost_timeout_s", 2.5))
        self.calibrate_min_error = float(aim.get("calibrate_min_error_px", 150.0))
        self.calibrate_probe = int(aim.get("calibrate_probe_counts", 12))
        self.white_recover_radius = float(aim.get("white_recover_radius_px", 450.0))
        self.white_recover_max = int(aim.get("white_recover_max_attempts", 2))

        swing = config.get("swing") or {}
        self.swing_enabled = bool(swing.get("enabled", True))
        self.chain_enabled = bool(swing.get("chain", False))
        self.swing_strike_guard = float(swing.get("strike_guard_ms", 60.0))
        self.swing_max_start = float(swing.get("max_start_error_px", 420.0))
        self.swing_tolerance = float(swing.get("aim_tolerance_px", 12.0))
        self.swing_max_moves = int(swing.get("max_aim_moves", 2))
        self.chain_min_move = float(swing.get("chain_min_move_px", 40.0))
        self.chain_probe = float(swing.get("chain_probe_ms", 200.0)) / 1000.0
        # After a successful build the old white X lingers at the crosshair
        # while the next yellow X is already visible nearby.  These values
        # let us tell that stale white lock apart from a real lock.
        self.stale_lock_min_px = float(swing.get("stale_lock_min_px", 55.0))
        self.stale_lock_hint_px = float(swing.get("stale_lock_hint_px", 180.0))
        self.stale_lock_max_age = float(swing.get("stale_lock_max_age_s", 2.0))

        click = config.get("click") or {}
        self.click_hold_ms = float(click.get("hold_ms", 510.0))
        self.post_release_ms = tuple(float(v) for v in click.get("post_release_ms", [8, 20]))
        self.build_wait = float(click.get("build_wait_s", 2.0))
        self.poll_ms = float(click.get("poll_ms", 10.0))
        self.max_failed = int(click.get("max_failed_clicks", 3))
        self.retry_delay_ms = float(click.get("retry_delay_ms", 250.0))
        self.min_build_move = float(click.get("min_move_px", 12.0))
        self.ensure_cursor_in_window = bool(click.get("ensure_cursor_in_window", True))
        self.max_swings = int(click.get("max_swings", 0))

        miss = config.get("miss") or {}
        self.miss_every = int(miss.get("every_n_builds", 4))
        self.miss_min_distance = float(miss.get("min_distance_px", 120.0))
        self.miss_look_away_tries = int(miss.get("look_away_max_tries", 2))
        self.miss_hold_ms = float(miss.get("hold_ms", self.click_hold_ms))

        self.human = mover.cfg
        self.allow_cold_white = bool(cfg_get(config, "game.allow_cold_white", True))
        self.random = random.Random()

        self.stats = Stats()
        self.mode = "IDLE"
        self.active = False
        self.hard_pause = False
        self.stop = False
        self._white_hits = 0
        self._stable_center: Optional[tuple[float, float]] = None
        self._stable_hits = 0
        self._iterations = 0
        self._close_checks = 0
        self._calibrated = False
        self._last_seen = time.monotonic()
        self._ever_seen = False
        self._last_yellow_time: Optional[float] = None
        self._focus_warned = False
        self._last_focus_warn = 0.0
        self._grab_failures = 0
        self._loop_errors = 0
        self._last_debug_save = 0.0
        self._last_det: Optional[Detection] = None
        self._last_det_time = 0.0
        self._last_yellow: Optional[Candidate] = None
        self._pending_target: Optional[tuple[float, float]] = None
        self._pending_target_time = 0.0
        self._click_target: Optional[Candidate] = None
        self._swing_counts = (0.0, 0.0)
        self._chain_pending: Optional[dict[str, Any]] = None
        self._chain_stop_after = False
        self._chain_pause_reason: Optional[str] = None
        self._chain_abort_swing = False
        self._white_recover_attempts = 0
        self._trace_start = time.perf_counter()
        self._trace_last = self._trace_start
        self._build_start: Optional[float] = None
        self._last_grab_ms = 0.0
        self._last_detect_ms = 0.0
        self._aim_since_miss = 0
        self._pending_miss = False
        self._build_streak = 0
        self._last_wait_log = 0.0

    # ------------------------------------------------------------- utilities
    def _trace(self, tag: str, message: str = "", **fields: Any) -> None:
        if not self.debug_trace:
            return
        now = time.perf_counter()
        delta = (now - self._trace_last) * 1000.0
        self._trace_last = now
        text = f"TRACE {tag:<7} {message}".rstrip()
        if fields:
            text += " " + " ".join(f"{key}={value}" for key, value in fields.items())
        log.info("%s (dt=%.1fms, t=%.2fs)", text, delta, now - self._trace_start)

    def _set_mode(self, mode: str) -> None:
        if mode != self.mode:
            if mode == "AIM":
                # A new AIM attempt may start directly from a post-click poll
                # frame; never carry the previous attempt's iteration budget.
                self._iterations = 0
                self._close_checks = 0
                self._white_recover_attempts = 0
            self._trace("STATE", f"{self.mode}->{mode}")
            log.debug("状态 %s -> %s", self.mode, mode)
            self.mode = mode

    def _abort(self) -> bool:
        return self.hotkeys.down("f9")

    def _sleep(self, ms: float) -> None:
        end = time.perf_counter() + ms / 1000.0
        while True:
            if self._abort():
                raise Aborted
            remaining = end - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(0.01, remaining))

    def _region(self) -> Optional[tuple[int, int, int, int]]:
        if self.fixed_region is not None:
            return self.fixed_region
        if self.hwnd is not None:
            return client_region(self.hwnd)
        return None

    def _focus_ok(self) -> bool:
        if self.dry_run or self.hwnd is None:
            return True
        if not is_foreground(self.hwnd):
            now = time.monotonic()
            if not self._focus_warned or now - self._last_focus_warn >= 5.0:
                log.warning("游戏窗口不在前台，暂停输入（切回游戏窗口后自动继续）")
                self._focus_warned = True
                self._last_focus_warn = now
            return False
        if self._focus_warned:
            log.info("游戏窗口回到前台，继续")
            self._focus_warned = False
        return True

    def _prepare_click(self) -> None:
        """Make sure the click is routed to the game window, not the desktop."""
        if not self.ensure_cursor_in_window:
            return
        region = self._region()
        if region is None:
            return
        try:
            if ensure_cursor_in_region(region):
                log.debug("点击前光标不在游戏客户区，已移回客户区中心")
        except Exception:
            log.debug("点击前光标归位失败", exc_info=True)

    def _grab(self, region: Optional[tuple[int, int, int, int]] = None
              ) -> Optional[np.ndarray]:
        if region is None:
            region = self._region()
        if region is None:
            return None
        left, top, width, height = region
        if width < 80 or height < 80:
            return None
        try:
            return self.capture.grab(region)
        except Exception as exc:
            log.debug("抓屏失败: %s", exc)
            return None

    def _detect(self, full: bool = False,
                allow_fallback: bool = True) -> Optional[Detection]:
        full_region = self._region()
        if full_region is None:
            return None
        grab_region = full_region
        roi: Optional[tuple[int, int, int, int]] = None
        if not full and self.roi_half > 0:
            left, top, width, height = full_region
            cx, cy = width // 2, height // 2
            half = self.roi_half
            rx0, ry0 = max(0, cx - half), max(0, cy - half)
            rx1, ry1 = min(width, cx + half), min(height, cy + half)
            if rx1 - rx0 >= 200 and ry1 - ry0 >= 200:
                roi = (rx0, ry0, rx1 - rx0, ry1 - ry0)
                grab_region = (left + rx0, top + ry0,
                               rx1 - rx0, ry1 - ry0)
        grab_start = time.perf_counter()
        frame = self._grab(grab_region)
        grab_done = time.perf_counter()
        if frame is None:
            self._grab_failures += 1
            self._trace("DET", "no frame", failures=self._grab_failures)
            return None
        self._grab_failures = 0
        detect_start = time.perf_counter()
        det = self.detector.detect(frame)
        if roi is not None:
            x0, y0, _, _ = roi
            det = shift_detection(det, x0, y0,
                                  full_region[2], full_region[3])
            if allow_fallback and not det.yellow and not det.locked:
                # Marker is outside the center ROI; pay for one full-frame
                # search rather than losing the target.
                frame = self._grab(full_region)
                if frame is None:
                    return None
                det = self.detector.detect(frame)
                roi = None
        detect_done = time.perf_counter()
        self._last_grab_ms = (grab_done - grab_start) * 1000.0
        self._last_detect_ms = (detect_done - detect_start) * 1000.0
        now = time.monotonic()
        self._last_det = det
        self._last_det_time = now
        if det.best is not None:
            self._ever_seen = True
            self._last_seen = now
            self._last_yellow_time = now
            self._last_yellow = det.best
        if det.locked:
            self._ever_seen = True
            self._last_seen = now
        best = det.best
        err = det.error()
        self._trace(
            "DET",
            "frame",
            grab_ms=round(self._last_grab_ms, 1),
            detect_ms=round(self._last_detect_ms, 1),
            yellow=len(det.yellow),
            yellow_bboxes=[c.bbox for c in det.yellow[:3]],
            white=len(det.white),
            relaxed=len(det.white_relaxed),
            white_bboxes=[c.bbox for c in det.white[:2]],
            relaxed_bboxes=[c.bbox for c in det.white_relaxed[:2]],
            lock=det.locked,
            V=round(det.center_v),
            S=round(det.center_s),
            roi=roi,
            bbox=(best.bbox if best else None),
            err=((round(err[0]), round(err[1])) if err else None),
        )
        if self.debug_dir is not None and now - self._last_debug_save >= 0.5:
            self._last_debug_save = now
            try:
                debug_det = det
                if roi is not None:
                    x0, y0, _, _ = roi
                    debug_det = shift_detection(
                        det, -x0, -y0, frame.shape[1], frame.shape[0]
                    )
                cv2.imwrite(str(self.debug_dir / "latest.jpg"),
                            annotate(frame, debug_det),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
            except Exception:
                pass
        return det

    # ------------------------------------------------------------- state code
    def run(self) -> None:
        log.info("F8 开始/暂停，F9 退出。先把准星附近留出可见的黄叉/白叉。")
        while not self.stop:
            if self.hotkeys.pressed("f9"):
                break
            if self.hotkeys.pressed("f8"):
                if self.hard_pause:
                    self.hard_pause = False
                    self.active = True
                    self._chain_pending = None
                    self._chain_stop_after = False
                    self._chain_pause_reason = None
                    self.stats.failed_clicks = 0
                    self._set_mode("SEARCH")
                    log.info("已恢复")
                else:
                    self.active = not self.active
                    if not self.active:
                        # A pending chain would otherwise start a stale swing
                        # immediately after resuming.
                        self._chain_pending = None
                        self._chain_stop_after = False
                        self._chain_pause_reason = None
                    self._set_mode("SEARCH" if self.active else "IDLE")
                    log.info("开始" if self.active else "暂停")
            if not self.active or self.hard_pause:
                time.sleep(0.05)
                continue
            if self.hwnd is not None and not window_alive(self.hwnd):
                log.error("游戏窗口已关闭，退出")
                break
            if self.hwnd is not None and not is_foreground(self.hwnd):
                self._focus_ok()
                time.sleep(0.2)
                continue
            try:
                self._step()
            except Aborted:
                log.info("收到 F9，退出")
                break
            except Exception:
                self._loop_errors += 1
                log.exception("循环异常（%d）", self._loop_errors)
                if self._loop_errors >= 8:
                    log.error("连续异常过多，退出")
                    break
                time.sleep(0.4)
            else:
                self._loop_errors = 0
        self.mover.release_left()
        elapsed = time.monotonic() - self.stats.started_at
        if (self.stats.first_swing_at is not None
                and self.stats.last_swing_at is not None):
            phase = self.stats.last_swing_at - self.stats.first_swing_at
        else:
            phase = 0.0
        per_build = (phase * 1000.0 / self.stats.aimed_builds
                     if self.stats.aimed_builds > 0 else 0.0)
        per_swing = (phase * 1000.0 / self.stats.swings
                     if self.stats.swings > 0 else 0.0)
        log.info(
            "统计：瞄准建造 %d 次，非瞄准点击 %d 次，失败 %d 次，"
            "真实落锤 %d 次（提前松开 %d），建造阶段 %.2f s，"
            "%.0f ms/件，%.0f ms/锤，总用时 %.1f s",
            self.stats.aimed_builds, self.stats.non_aim_clicks,
            self.stats.failed_clicks, self.stats.swings,
            self.stats.aborted_swings, phase, per_build, per_swing, elapsed,
        )

    def _step(self) -> None:
        if self._pending_miss and self.mode == "SEARCH":
            self._do_miss_click()
            return
        if self.mode in ("IDLE", "PAUSED"):
            self._set_mode("SEARCH")
            return
        if self.mode == "SEARCH":
            self._search_step()
        elif self.mode == "AIM":
            self._aim_step()
        elif self.mode == "CLICK":
            if self._chain_pending is not None:
                self._chain_step()
            else:
                self._click_step()
        else:
            self._set_mode("SEARCH")

    def _search_step(self) -> None:
        det = self._detect(full=True)
        if det is None:
            self._capture_problem()
            return
        if det.locked:
            if self._stale_lock_target(det) is not None:
                self._trace("SEARCH", "stale white -> AIM")
                self._last_det = None
                self._set_mode("AIM")
                return
            if self._build_start is None:
                self._build_start = time.perf_counter()
            if self._white_confirm(det):
                return
            return
        self._white_hits = 0
        if det.best is not None:
            center = det.best.center
            if (self._stable_center is not None
                    and math.dist(center, self._stable_center) <= self.stable_px):
                self._stable_hits += 1
            else:
                self._stable_center = center
                self._stable_hits = 1
            if self._stable_hits >= self.stable_frames:
                if self._build_start is None:
                    self._build_start = time.perf_counter()
                self._iterations = 0
                self._close_checks = 0
                self._set_mode("AIM")
                self._trace("SEARCH", "yellow stable -> AIM",
                            bbox=det.best.bbox,
                            err=tuple(round(v) for v in (det.error() or (0, 0))))
                log.debug("发现黄叉 %s，误差 %s", det.best.bbox, det.error())
            else:
                self._sleep(35)
            return
        self._stable_hits = 0
        self._stable_center = None
        now = time.monotonic()
        if now - self._last_wait_log > 5.0:
            self._last_wait_log = now
            missing = now - self._last_seen if self._ever_seen else 0.0
            if self._ever_seen and missing >= self.lost_timeout:
                log.warning("黄叉已消失 %.1f s，继续检测（按 F9 手动停止）", missing)
            else:
                log.info("等待黄叉（请确认建造菜单已打开且目标在画面内）")
        self._sleep(60)

    def _white_confirm(self, det: Detection) -> bool:
        if not self.allow_cold_white and self._last_yellow_time is None:
            log.info("检测到白叉但尚未跟踪过黄叉，按安全策略等待")
            self._sleep(80)
            return False
        self._white_hits += 1
        self._trace("WHITE", "hit", frames=self._white_hits,
                    need=self.white_frames, V=round(det.center_v),
                    S=round(det.center_s), white=len(det.white),
                    relaxed=len(det.white_relaxed))
        if self._white_hits >= self.white_frames:
            self._set_mode("CLICK")
            self._trace("WHITE", "locked -> CLICK",
                        frames=self._white_hits, V=round(det.center_v),
                        S=round(det.center_s))
            return True
        self._sleep(40)
        return False

    def _stale_lock_target(self, det: Detection) -> Optional[Candidate]:
        """Return the real next yellow X while an old white X is still fading.

        WARDOGS keeps the just-built white X under the crosshair for a short
        while and already shows the next yellow X nearby.  Treating that
        leftover white as a lock makes the bot swing in place.  Only accept a
        far yellow candidate that matches the target confirmed by the last
        successful build poll, so unrelated yellow scenery is ignored.
        """
        if not det.locked or self._pending_target is None:
            return None
        if time.monotonic() - self._pending_target_time > self.stale_lock_max_age:
            self._pending_target = None
            return None
        cx, cy = det.screen_center
        nearby: list[tuple[float, Candidate]] = []
        for candidate in det.yellow:
            centre_dist = math.hypot(candidate.center[0] - cx,
                                     candidate.center[1] - cy)
            if centre_dist < self.stale_lock_min_px:
                continue
            hint_dist = math.hypot(candidate.center[0] - self._pending_target[0],
                                   candidate.center[1] - self._pending_target[1])
            if hint_dist <= self.stale_lock_hint_px:
                nearby.append((hint_dist, candidate))
        if not nearby:
            return None
        nearby.sort(key=lambda item: item[0])
        return nearby[0][1]

    def _aim_step(self) -> None:
        # Reuse the yellow detection that just promoted SEARCH -> AIM, or the
        # new-yellow frame returned by the post-click poll.  Re-capturing the
        # same frame costs 100-150 ms and was a visible pause.
        if (self._last_det is not None
                and time.monotonic() - self._last_det_time <= 0.25
                and self._last_det.best is not None
                and not self._last_det.locked):
            det = self._last_det
            self._trace("AIM", "consume cached yellow",
                        bbox=det.best.bbox)
        else:
            det = self._detect()
        if det is None:
            self._capture_problem()
            return
        if det.locked:
            stale = self._stale_lock_target(det)
            if stale is None:
                self._pending_target = None
                if self._build_start is None:
                    self._build_start = time.perf_counter()
                if self._white_confirm(det):
                    return
                return
            # The crosshair is still on the previous build's fading white X.
            # Aim at the next yellow X instead of confirming the stale lock.
            det.best = stale
            self._trace("AIM", "stale lock -> real target",
                        bbox=stale.bbox,
                        target=(round(stale.center[0]), round(stale.center[1])),
                        hint=self._pending_target)
        self._white_hits = 0
        if det.best is None:
            if self._recover_white(det):
                return
            self._trace("AIM", "no yellow", V=round(det.center_v),
                        S=round(det.center_s), white=len(det.white),
                        relaxed=len(det.white_relaxed))
            log.debug("AIM 无黄叉: lock=%s V=%.0f S=%.0f white=%d relaxed=%d",
                      det.locked, det.center_v, det.center_s,
                      len(det.white), len(det.white_relaxed))
            self._set_mode("SEARCH")
            return
        center = det.best.center
        err = det.error()
        if err is None:
            self._set_mode("SEARCH")
            return
        ex, ey = err
        distance = math.hypot(ex, ey)
        if distance < 4.0:
            if self._close_checks < 2:
                self._close_checks += 1
                self._sleep(50)
                return
            # The marker is yellow although the crosshair is on its bbox
            # centre.  A tiny nudge often puts it on the painted crossing.
            if 0 < self._close_checks < 4:
                self._close_checks += 1
                self._nudge_towards(ex, ey, 6)
                return
            log.warning("准星已到叉中心但仍是黄色，回到搜索")
            self._close_checks = 0
            self._set_mode("SEARCH")
            return

        self._iterations += 1
        if self._iterations > self.max_iterations:
            self._hard_pause("瞄准迭代超过 %d 次" % self.max_iterations)
            return

        if not self._calibrated and distance > self.calibrate_min_error:
            if self._calibrate(det):
                return
            det = self._detect()
            if det is None or det.best is None:
                return
            center = det.best.center
            err = det.error() or (0.0, 0.0)
            ex, ey = err
            distance = math.hypot(ex, ey)
        if not self._calibrated:
            self._calibrated = True

        if self.swing_enabled and distance <= self.swing_max_start:
            self._trace("SWING", "start", err=(round(ex), round(ey)),
                        tolerance=self.swing_max_start)
            # Keep the verified frame for the CLICK check (avoids a second
            # 40-70 ms capture/detect before pressing) and pass its target to
            # the swing so a lingering white X at the crosshair is ignored.
            self._click_target = det.best
            self._set_mode("CLICK")
            return

        # A real aim move makes the cached frame stale.
        self._last_det = None
        self._click_target = None
        plan_start = time.perf_counter()
        mx, my = self._planned_move(ex, ey)
        plan_ms = (time.perf_counter() - plan_start) * 1000.0
        if mx == 0 and my == 0:
            self._set_mode("SEARCH")
            return
        log.debug("瞄准：e=(%+.0f,%+.0f) counts=(%+.0f,%+.0f) g=(%.2f,%.2f)",
                  ex, ey, mx, my, self.gain_x, self.gain_y)
        before = center
        move_start = time.perf_counter()
        if not self.mover.move(int(round(mx)), int(round(my)), abort=self._abort):
            raise Aborted
        move_ms = (time.perf_counter() - move_start) * 1000.0
        settle_start = time.perf_counter()
        self._sleep(self.random.uniform(*self.settle_ms))
        settle_ms = (time.perf_counter() - settle_start) * 1000.0
        post_start = time.perf_counter()
        after_det = self._detect()
        post_ms = (time.perf_counter() - post_start) * 1000.0
        self._trace("AIM", "move",
                    e=(round(ex), round(ey)),
                    counts=(int(round(mx)), int(round(my))),
                    g=(round(self.gain_x, 3), round(self.gain_y, 3)),
                    plan_ms=round(plan_ms, 1),
                    move_ms=round(move_ms, 1),
                    settle_ms=round(settle_ms, 1),
                    post_ms=round(post_ms, 1),
                    iteration=self._iterations,
                    after=("lock" if after_det and after_det.locked else
                           ("yellow" if after_det and after_det.best else "lost")))
        if after_det is not None and after_det.locked:
            self._trace("AIM", "post-move lock -> CLICK")
            if self._white_confirm(after_det):
                return
        if after_det is not None and after_det.best is not None and not after_det.locked:
            self._update_gain((mx, my), before, after_det.best.center,
                              after_det.screen_center)
        self._close_checks = 0

    def _recover_white(self, det: Detection) -> bool:
        """If the X turned white off-centre, aim at that white component."""
        if self._white_recover_attempts >= self.white_recover_max:
            return False
        candidates = list(det.white) + list(det.white_relaxed)
        if not candidates:
            return False
        center = det.screen_center
        expected = self._last_yellow
        ranked: list[tuple[float, Candidate]] = []
        for candidate in candidates:
            distance = math.hypot(candidate.center[0] - center[0],
                                  candidate.center[1] - center[1])
            if distance > self.white_recover_radius or candidate.bbox[2] < 40:
                continue
            penalty = 0.0
            if expected is not None:
                ratio = candidate.area / float(max(expected.area, 1))
                if ratio < 0.35 or ratio > 3.0:
                    continue
                penalty = abs(math.log(max(ratio, 1e-6))) * 80.0
            ranked.append((distance + penalty, candidate))
        if not ranked:
            return False
        ranked.sort(key=lambda item: item[0])
        target = ranked[0][1]
        err = (center[0] - target.center[0], center[1] - target.center[1])
        mx, my = self._planned_move(err[0], err[1])
        if mx == 0 and my == 0:
            return False
        self._white_recover_attempts += 1
        self._trace("WHITE", "recover aim", bbox=target.bbox,
                    err=(round(err[0]), round(err[1])),
                    attempts=self._white_recover_attempts)
        if not self.mover.move(int(round(mx)), int(round(my)), abort=self._abort):
            raise Aborted
        self._sleep(self.random.uniform(*self.settle_ms))
        return True

    def _nudge_towards(self, ex: float, ey: float, counts: float = 6.0) -> None:
        """Small human-like nudge toward the marker (or a random direction)."""
        gx = self.gain_x if abs(self.gain_x) >= 1e-4 else -0.55
        gy = self.gain_y if abs(self.gain_y) >= 1e-4 else -0.55
        ux, uy = ex / gx, ey / gy
        norm = math.hypot(ux, uy)
        if norm < 1e-6:
            angle = self.random.uniform(0.0, math.tau)
            ux, uy = math.cos(angle), math.sin(angle)
            norm = 1.0
        mx = int(round(ux / norm * counts))
        my = int(round(uy / norm * counts))
        if mx == 0 and my == 0:
            mx = 1
        if not self.mover.move(mx, my, abort=self._abort):
            raise Aborted
        self._sleep(self.random.uniform(*self.settle_ms))

    # ------------------------------------------------------------- planning
    def _planned_move(self, ex: float, ey: float) -> tuple[float, float]:
        distance = math.hypot(ex, ey)
        alpha = self.alpha_far if distance > self.far_px else self.alpha_near
        gx = self.gain_x if abs(self.gain_x) >= 1e-4 else -0.55
        gy = self.gain_y if abs(self.gain_y) >= 1e-4 else -0.55
        mx = alpha * ex / gx
        my = alpha * ey / gy
        length = math.hypot(mx, my)
        if length > self.max_step:
            scale = self.max_step / length
            mx *= scale
            my *= scale
        elif length < self.min_step:
            ux, uy = ex / gx, ey / gy
            norm = math.hypot(ux, uy)
            if norm < 1e-6:
                return (0.0, 0.0)
            mx, my = ux / norm * self.min_step, uy / norm * self.min_step
        # Limit the *screen* step as well as the mouse counts.  On a very
        # sensitive setup 180 counts can move the marker hundreds of pixels.
        screen_x, screen_y = mx * gx, my * gy
        screen_length = math.hypot(screen_x, screen_y)
        if screen_length > self.max_step_px:
            scale = self.max_step_px / screen_length
            mx *= scale
            my *= scale
        return (round(mx), round(my))

    def _update_gain(self, counts: tuple[float, float],
                     before: tuple[float, float],
                     after: tuple[float, float],
                     center: tuple[float, float]) -> None:
        for axis, current in enumerate((self.gain_x, self.gain_y)):
            dm = counts[axis]
            dc = after[axis] - before[axis]
            if abs(dm) < 8 or abs(dc) < 2:
                continue
            observed = dc / dm
            if not (self.gain_min <= abs(observed) <= self.gain_max):
                continue
            err_before = center[axis] - before[axis]
            err_after = center[axis] - after[axis]
            crossed = err_before * err_after < 0.0
            if crossed:
                # The marker crossed the crosshair: the linear gain estimate
                # was too low.  Trust the measured gain much more strongly.
                updated = 0.80 * observed + 0.20 * current
            elif abs(observed) > abs(current) * 1.5:
                updated = 0.55 * observed + 0.45 * current
            else:
                updated = (1.0 - self.gain_ema) * current + self.gain_ema * observed
            updated = max(-self.gain_max, min(self.gain_max, updated))
            if axis == 0:
                self.gain_x = updated
            else:
                self.gain_y = updated

    def _calibrate(self, det: Detection) -> bool:
        """Small, far-away probe to learn sign and magnitude of each axis."""
        if det.best is None:
            return False
        center = det.best.center
        err = det.error() or (0.0, 0.0)
        self._trace("CALIB", "start", err=(round(err[0]), round(err[1])))
        log.info("校准鼠标增益（黄叉较远，安全距离 %.0f px）", math.hypot(*err))
        for axis in (0, 1):
            if abs(err[axis]) < 100.0:
                continue
            probe = self.calibrate_probe
            counts = (probe, 0) if axis == 0 else (0, probe)
            probe_start = time.perf_counter()
            if not self.mover.move(*counts, abort=self._abort):
                raise Aborted
            self._sleep(110)
            after = self._detect()
            if after is None:
                continue
            if after.locked:
                self._set_mode("CLICK")
                return True
            if after.best is None:
                continue
            measured = (after.best.center[axis] - center[axis]) / probe
            if self.gain_min <= abs(measured) <= self.gain_max:
                if axis == 0:
                    self.gain_x = measured
                else:
                    self.gain_y = measured
                log.info("校准 axis=%d 增益 %.3f px/count", axis, measured)
            self._trace("CALIB", "axis", axis=axis, counts=counts,
                        measured=round(measured, 3),
                        elapsed_ms=round((time.perf_counter() - probe_start) * 1000.0, 1))
            center = after.best.center
        self._calibrated = True
        return False

    # ------------------------------------------------------------- clicking
    def _click_step(self) -> None:
        # The white lock was established in the previous state; reuse that
        # frame instead of spending another 100-150 ms on a fresh capture.
        used_cache = False
        if (self._last_det is not None
                and time.monotonic() - self._last_det_time <= 0.25):
            det = self._last_det
            used_cache = True
        else:
            det = self._detect()
        if det is None:
            self._capture_problem()
            return
        stale = self._stale_lock_target(det)
        preferred = stale
        if (preferred is None and used_cache
                and self._click_target is not None and det.best is not None
                and math.dist(self._click_target.center,
                              det.best.center) <= 20.0):
            # Reuse the freshly verified yellow X; the swing will ignore the
            # old white X if it reappears under the crosshair.
            preferred = self._click_target
        start_ok = det.locked
        if not start_ok and self.swing_enabled and det.best is not None:
            err = det.error()
            start_ok = bool(err and math.hypot(*err) <= self.swing_max_start)
        self._trace("CLICK", "check", cached=used_cache,
                    locked=det.locked, start_ok=start_ok,
                    stale=(stale is not None),
                    preferred=(preferred is not None),
                    V=round(det.center_v), S=round(det.center_s))
        if not start_ok:
            self._white_hits = 0
            self._click_target = None
            self._set_mode("AIM")
            return
        if stale is not None:
            if not self.swing_enabled:
                # Without swing-time aiming, clicking here would hit the old
                # spot again.  Aim first, then let the normal AIM loop lock.
                self._set_mode("AIM")
                return
            self._trace("CLICK", "stale white ignored",
                        target=(round(stale.center[0]), round(stale.center[1])))
        if not self._focus_ok():
            self._click_target = None
            self._set_mode("SEARCH")
            return
        if self.dry_run:
            log.info("[dry-run] 白叉锁定，模拟按下左键 %.0f ms", self.click_hold_ms)
            self._record_build_success(None)
            return
        self._prepare_click()
        start_det = det
        # Where the current X sits before the swing; after a successful build
        # the new X must be somewhere else than this (predicted) position.
        if stale is not None:
            pre_target = stale.center
        elif preferred is not None:
            pre_target = preferred.center
        elif det.best is not None:
            pre_target = det.best.center
        else:
            pre_target = det.screen_center
        log.info("起锤 %.0f ms，期间继续瞄准（落锤前 %.0f ms 停手）",
                 self.click_hold_ms, self.swing_strike_guard)
        click_start = time.perf_counter()
        self._swing_counts = (0.0, 0.0)
        aim_state = {"ok": False}
        if self.swing_enabled:
            def aim_callback(deadline: float) -> bool:
                aim_state["ok"] = self._swing_aim(deadline, start_det, preferred)
                return aim_state["ok"]

            ok = self.mover.click_with_aim(
                self.click_hold_ms,
                aim_callback,
                self.swing_strike_guard,
                abort=self._abort,
            )
        else:
            ok = self.mover.click(self.click_hold_ms, abort=self._abort)
        self._click_target = None
        timing = getattr(self.mover, "last_click_timing", (0.0, 0.0, 0.0))
        click_ms = (time.perf_counter() - click_start) * 1000.0
        self._trace("CLICK", "button", pre_ms=round(timing[0], 1),
                    hold_ms=round(timing[1], 1), post_ms=round(timing[2], 1),
                    call_ms=round(click_ms, 1),
                    swing_ok=(aim_state["ok"] if self.swing_enabled else None))
        if self._count_swing(timing[1]):
            log.info("达到 --max-swings=%d，停止", self.max_swings)
            return
        # The swing's net mouse counts tell us where the old X ended up on
        # screen.  Comparing the next yellow X against that position prevents
        # a missed/partial swing from re-counting the same, unmoved X as a
        # successful build.
        sx, sy = self._swing_counts
        start_center = (pre_target[0] + sx * self.gain_x,
                        pre_target[1] + sy * self.gain_y)
        swing_px = math.hypot(sx * self.gain_x, sy * self.gain_y)
        verify_move = max(self.min_build_move,
                          min(35.0, swing_px * 0.15))
        if self.swing_enabled and not aim_state["ok"]:
            # The hammer may still land by luck, but require a clearly moved X
            # so a partially moved / unmoved old X is never counted as a build.
            verify_move = max(verify_move, 60.0)
        if self.chain_enabled and self.swing_enabled:
            # Do not wait for the new X here.  The next swing starts at once;
            # its aim callback waits for the new yellow X and aims during the
            # 510 ms hammer animation.  The previous swing is confirmed by
            # the new X's position relative to this predicted old position.
            self._chain_pending = {
                "expected_old": start_center,
                "aim_ok": aim_state["ok"],
                "build_start": (self._build_start
                                if self._build_start is not None
                                else click_start),
                "pre_target": pre_target,
                "verify_move": verify_move,
            }
            self._pending_target = None
            self._last_det = None
            self._trace("CHAIN", "next swing immediately",
                        expected_old=(round(start_center[0]),
                                      round(start_center[1])),
                        aim_ok=aim_state["ok"])
            self._set_mode("CLICK")
            return
        if stale is None:
            # A genuine lock belongs to the current target; the next build
            # success will publish a new pending target.
            self._pending_target = None
        if not ok:
            raise Aborted
        release_start = time.perf_counter()
        self._sleep(self.random.uniform(*self.post_release_ms))
        self._trace("CLICK", "release-wait",
                    ms=round((time.perf_counter() - release_start) * 1000.0, 1))

        deadline = time.monotonic() + self.build_wait
        new_det: Optional[Detection] = None
        last_locked = bool(det.locked)
        poll_start = time.perf_counter()
        polls = 0
        while time.monotonic() < deadline:
            polls += 1
            # The new X appears within a few hundred pixels of the crosshair.
            # A full-frame fallback costs ~250 ms and made the first
            # post-release poll look like a pause; keep ROI polling cheap and
            # only fall back if the marker is still missing after 350 ms.
            probe = self._detect(
                allow_fallback=(time.perf_counter() - poll_start) > 0.35
            )
            if probe is None:
                continue
            self._trace("VERIFY", "poll", n=polls,
                        ms=round((time.perf_counter() - poll_start) * 1000.0, 1),
                        yellow=(probe.best is not None),
                        lock=probe.locked)
            if probe.best is not None and not probe.locked:
                moved = math.hypot(probe.best.center[0] - start_center[0],
                                   probe.best.center[1] - start_center[1])
                self._trace("VERIFY", "candidate", moved_px=round(moved, 1),
                            need_px=round(verify_move, 1))
                if moved >= verify_move:
                    new_det = probe
                    break
            last_locked = probe.locked
            self._sleep(self.poll_ms)
        if new_det is not None:
            self._trace("VERIFY", "result", ok=True, polls=polls,
                        ms=round((time.perf_counter() - poll_start) * 1000.0, 1))
            self._record_build_success(new_det)
            return

        self.stats.failed_clicks += 1
        self._trace("VERIFY", "result", ok=False, polls=polls,
                    ms=round((time.perf_counter() - poll_start) * 1000.0, 1),
                    last_locked=last_locked)
        if last_locked:
            log.warning("点击后仍是白叉（可能是物资不足），失败 %d/%d",
                        self.stats.failed_clicks, self.max_failed)
        else:
            log.warning("点击后 %.1f s 内没有找到新的黄叉，失败 %d/%d",
                        self.build_wait, self.stats.failed_clicks, self.max_failed)
        if self.stats.failed_clicks >= self.max_failed:
            self._hard_pause("连续建造失败，可能物资不足或预览无效")
        else:
            self._white_hits = 0
            self._set_mode("AIM")
            self._sleep(self.retry_delay_ms)

    # ------------------------------------------------------- chained swings
    def _count_swing(self, hold_ms: float) -> bool:
        """Track real hammer strikes; early chain releases are not strikes."""
        if hold_ms >= self.click_hold_ms * 0.9:
            self.stats.swings += 1
            now = time.perf_counter()
            if self.stats.first_swing_at is None:
                self.stats.first_swing_at = now
            self.stats.last_swing_at = now
        else:
            self.stats.aborted_swings += 1
        if self.max_swings > 0 and self.stats.swings >= self.max_swings:
            self.stop = True
            return True
        return False

    def _chain_target_from(self, det: Detection) -> Optional[Candidate]:
        """Pick the new yellow X, ignoring the old white/ghost at the crosshair."""
        if not det.yellow:
            return None
        cx, cy = det.screen_center
        # The newly spawned X can be close to the old one but never identical.
        # Ignore only the very small centre ghost/fringe.
        limit = max(30.0, self.stale_lock_min_px * 0.6)
        far = [
            c for c in det.yellow
            if math.hypot(c.center[0] - cx, c.center[1] - cy) >= limit
        ]
        if not far:
            return None
        far.sort(key=lambda c: math.hypot(c.center[0] - cx,
                                          c.center[1] - cy))
        return far[0]

    def _chain_judge_previous(self, target: Candidate,
                              prev: dict[str, Any]) -> bool:
        expected = prev["expected_old"]
        moved = math.hypot(target.center[0] - expected[0],
                           target.center[1] - expected[1])
        need = max(self.chain_min_move, float(prev.get("verify_move", 0.0)))
        self._trace("CHAIN", "new target", moved_px=round(moved, 1),
                    need_px=round(need, 1), prev_aim=prev["aim_ok"])
        if moved < need:
            if prev["aim_ok"] and moved < 8.0:
                self.stats.failed_clicks += 1
                self._trace("CHAIN", "previous X did not move",
                            failed=self.stats.failed_clicks)
                if self.stats.failed_clicks >= self.max_failed:
                    self._chain_stop_after = True
                    self._chain_pause_reason = "链式挥锤连续未推动建造"
            elif prev["aim_ok"]:
                # A small but non-zero shift may be a close next X; do not
                # count it as a success or a failure, just keep aiming.
                self._trace("CHAIN", "ambiguous move",
                            moved_px=round(moved, 1))
            else:
                self._trace("CHAIN", "partial hit (X stayed)",
                            moved_px=round(moved, 1))
            return False
        # The old X is knocked away only when the build completes.  A late
        # aim_ok=False correction can still land before the strike, so the
        # position change is the authoritative success signal.
        if not prev["aim_ok"]:
            self._trace("CHAIN", "late aim but built",
                        moved_px=round(moved, 1))
        self.stats.aimed_builds += 1
        self.stats.failed_clicks = 0
        self._build_streak += 1
        self._aim_since_miss += 1
        total_ms = (time.perf_counter() - prev["build_start"]) * 1000.0
        self._trace("BUILD", "OK chain", n=self.stats.aimed_builds,
                    total_ms=round(total_ms, 1), moved_px=round(moved, 1))
        log.info("建造 #%d 完成，新黄叉中心 %s", self.stats.aimed_builds,
                 tuple(round(v) for v in target.center))
        self._pending_target = target.center
        self._pending_target_time = time.monotonic()
        if self.max_builds > 0 and self.stats.aimed_builds >= self.max_builds:
            self._chain_stop_after = True
        if self.miss_every > 0 and self._aim_since_miss >= self.miss_every:
            self._pending_miss = True
            self._aim_since_miss = 0
        return True

    def _chain_wait_target(self, deadline: float, prev: dict[str, Any],
                           state: dict[str, Any]) -> Optional[Candidate]:
        """Wait for the next X that appears when the previous one is knocked away."""
        probe_deadline = min(deadline,
                             time.perf_counter() + self.chain_probe)
        while time.perf_counter() < deadline:
            if self._abort():
                raise Aborted
            det = self._detect(allow_fallback=False)
            if det is not None:
                target = self._chain_target_from(det)
                if target is not None:
                    state["target"] = target
                    state["start_det"] = det
                    self._chain_judge_previous(target, prev)
                    return target
            if time.perf_counter() >= probe_deadline:
                self._chain_abort_swing = True
                break
            time.sleep(0.01)
        return None

    def _chain_step(self) -> None:
        """One immediate next swing; the new X is awaited during the hold."""
        prev = self._chain_pending
        self._chain_pending = None
        if prev is None:
            self._set_mode("SEARCH")
            return
        if self._chain_stop_after:
            self._chain_pending = None
            if self._chain_pause_reason is not None:
                reason = self._chain_pause_reason
                self._chain_pause_reason = None
                self._hard_pause(reason)
            else:
                log.info("链式模式达到 --max-builds=%d，停止", self.max_builds)
                self.stop = True
            return
        if not self._focus_ok():
            self._chain_pending = None
            self._set_mode("SEARCH")
            return
        if self.dry_run:
            log.info("[dry-run] 链式挥锤（不发送输入）")
            self._sleep(30)
            self._set_mode("SEARCH")
            return
        self._prepare_click()
        self._swing_counts = (0.0, 0.0)
        self._chain_abort_swing = False
        state: dict[str, Any] = {
            "target": None, "start_det": None, "aim_ok": False,
        }
        swing_start = time.perf_counter()
        self._build_start = swing_start

        def aim_callback(deadline: float) -> bool:
            target = self._chain_wait_target(deadline, prev, state)
            if target is None:
                return False
            state["aim_ok"] = self._swing_aim(
                deadline, state["start_det"], target
            )
            return state["aim_ok"]

        ok = self.mover.click_with_aim(
            self.click_hold_ms,
            aim_callback,
            self.swing_strike_guard,
            abort=self._abort,
            stop_check=lambda: (self._chain_stop_after
                                or self._chain_abort_swing),
        )
        if not ok:
            raise Aborted
        self._chain_abort_swing = False
        timing = getattr(self.mover, "last_click_timing", (0.0, 0.0, 0.0))
        self._trace("CLICK", "chain button",
                    pre_ms=round(timing[0], 1),
                    hold_ms=round(timing[1], 1),
                    post_ms=round(timing[2], 1),
                    aim_ok=state["aim_ok"])
        pre_strike_abort = timing[1] < (self.click_hold_ms
                                        - self.swing_strike_guard)
        if self._count_swing(timing[1]):
            self._chain_pending = None
            log.info("达到 --max-swings=%d，停止", self.max_swings)
            return

        target = state["target"]
        if target is None:
            self.stats.failed_clicks += 1
            self._trace("CHAIN", "no yellow during swing",
                        failed=self.stats.failed_clicks,
                        prev_aim=prev["aim_ok"],
                        pre_strike_abort=pre_strike_abort)
            log.warning("链式挥锤期间没有看到黄叉%s，回到搜索（失败 %d/%d）",
                        ("（落锤前松开，0 进度）" if pre_strike_abort
                         else "（已落锤但未瞄准，约 1/5 进度）"),
                        self.stats.failed_clicks, self.max_failed)
            if self.stats.failed_clicks >= self.max_failed:
                self._hard_pause("链式挥锤连续找不到黄叉")
            else:
                self._set_mode("AIM")
            return

        pre_target = target.center
        sx, sy = self._swing_counts
        expected_old = (pre_target[0] + sx * self.gain_x,
                        pre_target[1] + sy * self.gain_y)
        swing_px = math.hypot(sx * self.gain_x, sy * self.gain_y)
        verify_move = max(self.chain_min_move,
                          min(60.0, swing_px * 0.15))
        self._chain_pending = {
            "expected_old": expected_old,
            "aim_ok": state["aim_ok"],
            "build_start": swing_start,
            "pre_target": pre_target,
            "verify_move": verify_move,
        }
        self._trace("CHAIN", "next swing", target=(
            round(pre_target[0]), round(pre_target[1])),
            aim_ok=state["aim_ok"])
        if self._chain_stop_after:
            self._chain_pending = None
            if self._chain_pause_reason is not None:
                reason = self._chain_pause_reason
                self._chain_pause_reason = None
                self._hard_pause(reason)
            else:
                log.info("达到 --max-builds=%d，停止", self.max_builds)
                self.stop = True
            return
        if self._pending_miss:
            self._chain_pending = None
            self._set_mode("SEARCH")
            return
        self._set_mode("CLICK")

    def _swing_aim(self, deadline: float, start_det: Detection,
                   preferred_target: Optional[Candidate] = None) -> bool:
        """Aim while LMB is held; stop before the strike guard."""
        if start_det.locked and preferred_target is None:
            self._trace("SWING", "already locked")
            return True
        center = start_det.screen_center
        last_target: Optional[tuple[float, float]] = None
        if preferred_target is not None:
            last_target = preferred_target.center
        elif start_det.best is not None:
            last_target = start_det.best.center
        else:
            whites = list(start_det.white) + list(start_det.white_relaxed)
            if whites:
                nearest = min(whites, key=lambda c: math.hypot(
                    c.center[0] - center[0], c.center[1] - center[1]))
                last_target = nearest.center
        moves = 0
        self._trace("SWING", "start", target=last_target,
                    stale=(preferred_target is not None))
        while time.perf_counter() < deadline and moves < self.swing_max_moves:
            det = self._detect(allow_fallback=False)
            if det is None:
                time.sleep(0.02)
                continue
            # At the start the "lock" may only be the previous build's fading
            # white X.  When a next target is known, always make one move
            # before trusting a white lock; the old X then leaves the
            # crosshair, so a later lock is the target we just aimed at.
            if det.locked and (preferred_target is None or moves > 0):
                self._trace("SWING", "locked", moves=moves)
                return True
            target = None
            if preferred_target is not None and last_target is not None:
                candidates = [
                    c for c in det.yellow
                    if (moves > 0 or math.hypot(c.center[0] - center[0],
                                                c.center[1] - center[1])
                        >= self.stale_lock_min_px)
                    and math.hypot(c.center[0] - last_target[0],
                                   c.center[1] - last_target[1]) <= 220.0
                ]
                if candidates:
                    candidates.sort(key=lambda c: math.hypot(
                        c.center[0] - last_target[0],
                        c.center[1] - last_target[1]))
                    target = candidates[0].center
                elif target is None:
                    # The new X may be hidden by the hammer animation; keep
                    # moving toward its last known screen position.
                    target = last_target
            elif preferred_target is None:
                if det.best is not None:
                    target = det.best.center
                else:
                    whites = list(det.white) + list(det.white_relaxed)
                    if whites:
                        nearest = min(whites, key=lambda c: math.hypot(
                            c.center[0] - center[0], c.center[1] - center[1]))
                        target = nearest.center
            if target is None:
                target = last_target
            if target is None:
                time.sleep(0.02)
                continue
            last_target = target
            err = (center[0] - target[0], center[1] - target[1])
            distance = math.hypot(*err)
            if distance <= self.swing_tolerance:
                self._trace("SWING", "aimed", moves=moves,
                            remaining_ms=round((deadline - time.perf_counter()) * 1000.0, 1))
                return True
            remaining = deadline - time.perf_counter()
            if remaining < 0.04:
                break
            mx, my = self._planned_move(err[0], err[1])
            if mx == 0 and my == 0:
                break
            fast = remaining < 0.22
            self._trace("SWING", "move", counts=(int(round(mx)), int(round(my))),
                        err=(round(err[0]), round(err[1])), moves=moves + 1,
                        fast=fast)
            if fast:
                if not self.mover.move_fast(
                        int(round(mx)), int(round(my)), abort=self._abort):
                    raise Aborted
            else:
                if not self.mover.move(
                        int(round(mx)), int(round(my)), abort=self._abort):
                    raise Aborted
            self._swing_counts = (self._swing_counts[0] + mx,
                                  self._swing_counts[1] + my)
            moves += 1
            if fast:
                self._sleep(self.random.uniform(6.0, 12.0))
            else:
                self._sleep(self.random.uniform(*self.settle_ms))
            # Predict where the X moved on screen while it may be hidden by
            # the hammer animation.
            last_target = (last_target[0] + mx * self.gain_x,
                           last_target[1] + my * self.gain_y)
        self._trace("SWING", "window ended", moves=moves)
        return False

    def _record_build_success(self, det: Optional[Detection]) -> None:
        total_ms = ((time.perf_counter() - self._build_start) * 1000.0
                    if self._build_start is not None else 0.0)
        self.stats.aimed_builds += 1
        self._build_streak += 1
        self.stats.failed_clicks = 0
        self._aim_since_miss += 1
        self._set_mode("SEARCH")
        self._white_hits = 0
        self._trace("BUILD", "OK", n=self.stats.aimed_builds,
                    total_ms=round(total_ms, 1),
                    aimed=self.stats.aimed_builds,
                    non_aim=self.stats.non_aim_clicks,
                    failed=self.stats.failed_clicks)
        self._build_start = None
        if det is not None and det.best is not None:
            log.info("建造 #%d 完成，新黄叉中心 %s", self.stats.aimed_builds,
                     tuple(round(v) for v in det.best.center))
            # Remember the newly verified X.  While the just-built white X is
            # still fading at the crosshair, this hint lets AIM/CLICK prefer
            # the real next target instead of swinging in place.
            self._pending_target = det.best.center
            self._pending_target_time = time.monotonic()
        else:
            log.info("建造 #%d 完成", self.stats.aimed_builds)
            self._pending_target = None
        if self.max_builds > 0 and self.stats.aimed_builds >= self.max_builds:
            log.info(
                "已达到 --max-builds=%d，脚本按参数退出；"
                "需要连续建造请去掉该参数（或设为 0）",
                self.max_builds,
            )
            self.stop = True
            return
        if self.miss_every > 0 and self._aim_since_miss >= self.miss_every:
            self._pending_miss = True
            self._aim_since_miss = 0
        if self._pending_miss:
            self._between_builds()
            self._set_mode("SEARCH")
        elif det is not None and det.best is not None:
            # The new X is already confirmed: start the next swing at once
            # instead of adding a human-style pause between hammers.
            self._between_builds(fast=True)
            # The post-click poll already found the new yellow X.  Keep that
            # frame and start AIM directly instead of searching/capturing it
            # again.
            self._last_det = det
            self._last_det_time = time.monotonic()
            self._build_start = time.perf_counter()
            self._stable_center = det.best.center
            self._stable_hits = 1
            self._set_mode("AIM")
            self._trace("BUILD", "next AIM from poll", bbox=det.best.bbox)
        else:
            self._between_builds()
            self._set_mode("SEARCH")

    def _between_builds(self, fast: bool = False) -> None:
        if fast:
            wait_ms = self.random.uniform(0.0, 8.0)
        else:
            wait_ms = self.random.uniform(*self.human.between_builds_ms)
        self._sleep(wait_ms)
        self._trace("WAIT", "between-builds", ms=round(wait_ms, 1))
        every = int(self.human.long_break_every)
        if not fast and every > 0 and self._build_streak % every == 0:
            pause = self.random.uniform(*self.human.long_break_ms)
            log.debug("人类化长停顿 %.0f ms", pause)
            self._sleep(pause)
            self._trace("WAIT", "long-break", ms=round(pause, 1))

    def _do_miss_click(self) -> None:
        self._white_hits = 0
        det = self._detect()
        if det is None:
            self._capture_problem()
            return
        self._trace("MISS", "check",
                    yellow=(det.best is not None), lock=det.locked)
        if det.best is None:
            self._pending_miss = False
            log.info("非瞄准点击跳过：当前没有黄色目标")
            self._set_mode("SEARCH")
            return
        if det.locked:
            self._pending_miss = False
            log.info("非瞄准点击跳过：黄叉正处于白叉锁定状态")
            self._set_mode("SEARCH")
            return
        err = det.error() or (0.0, 0.0)
        distance = math.hypot(*err)
        tries = 0
        while distance < self.miss_min_distance and tries < self.miss_look_away_tries:
            # Turn the camera away from the marker: use the negative of the
            # normal aiming correction, clipped to a safe step.
            mx, my = self._planned_move(-err[0] * 1.2, -err[1] * 1.2)
            if mx == 0 and my == 0:
                break
            if not self.mover.move(int(round(mx)), int(round(my)), abort=self._abort):
                raise Aborted
            self._sleep(self.random.uniform(*self.settle_ms))
            self._trace("MISS", "look-away",
                        counts=(int(round(mx)), int(round(my))),
                        distance_before=round(distance, 1))
            det = self._detect()
            if det is None or det.best is None or det.locked:
                break
            err = det.error() or (0.0, 0.0)
            distance = math.hypot(*err)
            tries += 1
        if det is None or det.best is None or det.locked or distance < self.miss_min_distance:
            self._pending_miss = False
            log.info("非瞄准点击跳过：黄叉仍太靠近准星（%.0f px）", distance)
            self._set_mode("SEARCH")
            return
        log.info("执行一次非瞄准点击（黄叉距离 %.0f px）", distance)
        if not self._focus_ok():
            self._pending_miss = True
            self._set_mode("SEARCH")
            return
        if not self.dry_run:
            self._prepare_click()
            click_start = time.perf_counter()
            if not self.mover.click(self.miss_hold_ms, abort=self._abort):
                raise Aborted
            self._trace("MISS", "click", ms=round((time.perf_counter() - click_start) * 1000.0, 1),
                        hold_ms=self.miss_hold_ms, distance=round(distance, 1))
            self._sleep(self.random.uniform(*self.post_release_ms))
        else:
            log.info("[dry-run] 非瞄准点击（不发送输入）")
        self.stats.non_aim_clicks += 1
        self._pending_miss = False
        self._between_builds()
        self._set_mode("SEARCH")

    # ------------------------------------------------------------- failures
    def _capture_problem(self) -> None:
        if self._grab_failures >= 20:
            self._hard_pause("连续抓屏失败，请检查窗口/显示器/独占全屏")
        else:
            time.sleep(0.05)

    def _hard_pause(self, reason: str) -> None:
        self.hard_pause = True
        self._grab_failures = 0
        self._chain_pending = None
        self._set_mode("PAUSED")
        log.warning("已暂停：%s。按 F8 恢复，F9 退出。", reason)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_test_images(directories: list[str], recursive: bool) -> list[str]:
    """Collect image files from user-selected directories."""
    paths: list[Path] = []
    for directory in directories:
        base = Path(directory)
        if not base.is_dir():
            log.error("测试目录不存在: %s", base)
            continue
        iterator = base.rglob("*") if recursive else base.glob("*")
        paths.extend(
            path for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return [str(path) for path in sorted(set(paths))]


def run_test_images(paths: list[str], detector: MarkerDetector,
                    save_dir: Optional[Path], expect: str = "any") -> int:
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    if not paths:
        log.error("没有找到可验证的图片（支持 png/jpg/jpeg/bmp/webp）")
        return 1
    failures = 0
    yellow_count = white_count = empty_count = 0
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            log.error("无法读取图片: %s", path)
            failures += 1
            continue
        try:
            display = str(Path(path).resolve().relative_to(Path.cwd()))
        except ValueError:
            display = str(path)
        start = time.perf_counter()
        det = detector.detect(frame)
        elapsed = (time.perf_counter() - start) * 1000.0
        has_yellow = len(det.yellow) > 0
        has_white = det.locked
        if has_yellow:
            yellow_count += 1
        elif has_white:
            white_count += 1
        else:
            empty_count += 1
        if expect == "yellow":
            ok = has_yellow
        elif expect == "white":
            ok = has_white
        else:
            ok = True
        if not ok:
            failures += 1
        best = det.best
        best_text = "-"
        if best is not None:
            best_text = (f"box={best.bbox} center=({best.center[0]:.0f},{best.center[1]:.0f}) "
                         f"asp={best.aspect:.2f} area={best.area}")
        status = "PASS" if ok else "FAIL"
        print(f"{status:4s} {display} yellow={len(det.yellow)} lock={det.locked!s:5s} "
              f"{best_text} V={det.center_v:.0f} S={det.center_s:.0f} {elapsed:.1f}ms")
        if save_dir is not None:
            cv2.imwrite(str(save_dir / f"detected_{Path(path).stem}.jpg"),
                        annotate(frame, det), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(
        f"Summary: images={len(paths)} yellow={yellow_count} "
        f"white={white_count} empty={empty_count} failed={failures} "
        f"expect={expect}"
    )
    return 0 if failures == 0 else 1


def run_click_test(seconds: float) -> int:
    """Send exactly one left click to whatever window is foreground."""
    enable_dpi_awareness()
    log.info("将在 %.1f 秒后向当前前台窗口发送一次左键；请先切换到目标窗口。", seconds)
    end = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < end:
        time.sleep(min(0.05, max(0.0, end - time.monotonic())))
    hwnd = find_window("")
    if hwnd is None:
        log.error("没有前台窗口")
        return 1
    region = client_region(hwnd)
    log.info("前台窗口: %s %r 客户区: %s 光标: %s",
             hex(hwnd), window_title(hwnd), region, cursor_pos())
    if region is not None:
        try:
            moved = ensure_cursor_in_region(region)
            log.info("光标检查: %s", "不在客户区，已移回中心" if moved else "已在客户区")
        except Exception:
            log.debug("光标检查失败", exc_info=True)
    try:
        send_left_down()
        time.sleep(0.51)
        send_left_up()
        log.info("SendInput LEFT DOWN/UP 调用成功（返回值正常）")
        return 0
    except Exception:
        log.exception("SendInput LEFT DOWN/UP 失败")
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="黄叉自动建造脚本")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
    )
    parser.add_argument("--test-images", nargs="+", metavar="IMAGE",
                        help="只跑离线检测，不抓屏、不发送输入")
    parser.add_argument("--test-dir", nargs="+", metavar="DIR",
                        help="批量测试一个或多个目录中的图片（png/jpg/jpeg/bmp/webp）")
    parser.add_argument("--recursive", action="store_true",
                        help="--test-dir 时递归搜索子目录")
    parser.add_argument("--expect", choices=("any", "yellow", "white"),
                        default="any",
                        help="离线验证期望：any=只报告，yellow=必须有黄叉，white=必须白叉锁定")
    parser.add_argument("--save-debug", metavar="DIR", help="保存标注图到此目录")
    parser.add_argument("--list-windows", action="store_true", help="列出可见窗口标题")
    parser.add_argument("--test-click", nargs="?", const=3.0, type=float, default=None,
                        metavar="SECONDS",
                        help="倒计时后向前台窗口发送一次左键，用于排查点击失效")
    parser.add_argument("--window-title", default=None, help="覆盖配置中的窗口标题子串")
    parser.add_argument("--region", default=None, help="固定截屏区域 left,top,width,height")
    parser.add_argument("--dry-run", action="store_true", help="只检测/规划，不发送鼠标输入")
    parser.add_argument("--max-builds", type=int, default=None, help="瞄准建造多少件后停止；0=不限")
    parser.add_argument("--max-swings", type=int, default=None,
                        help="完成多少次真实落锤后停止（提前松开不计）；0=不限")
    parser.add_argument("--no-chain", action="store_true",
                        help="覆盖配置，关闭链式挥锤（先确认新黄叉再起锤）")
    parser.add_argument("--chain", action="store_true",
                        help="覆盖配置，启用链式挥锤（落锤后立即起下一锤）")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="连续建造时逐步打印检测/瞄准/白叉/点击/等待/建造耗时")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if (args.verbose or args.debug) else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    if args.test_click is not None:
        return run_click_test(args.test_click)

    if args.list_windows:
        for hwnd, title, cls in list_windows():
            print(f"{hwnd:#x}  {title}  [{cls}]")
        return 0

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("配置文件不存在: %s", config_path)
        return 1
    try:
        config = load_yaml(config_path)
    except Exception:
        log.exception("读取配置失败")
        return 1
    if args.no_chain:
        config.setdefault("swing", {})["chain"] = False
    if args.chain:
        config.setdefault("swing", {})["chain"] = True
    if args.max_swings is not None:
        config.setdefault("click", {})["max_swings"] = args.max_swings

    detector = MarkerDetector(DetectorConfig.from_dict(config.get("detector") or {}))
    test_paths = list(args.test_images or [])
    if args.test_dir:
        test_paths.extend(collect_test_images(args.test_dir, args.recursive))
    if args.test_images or args.test_dir:
        if not test_paths:
            log.error("没有找到可验证的图片（支持 png/jpg/jpeg/bmp/webp）")
            return 1
        save_dir = Path(args.save_debug) if args.save_debug else None
        return run_test_images(test_paths, detector, save_dir,
                               expect=args.expect)

    enable_dpi_awareness()
    backend = str(cfg_get(config, "game.capture_backend", "mss"))
    monitor_index = int(cfg_get(config, "game.monitor_index", 1))
    try:
        capture = Capture(backend=backend, monitor_index=monitor_index)
    except Exception:
        log.exception("初始化截屏后端失败")
        return 1

    fixed_region = None
    if args.region:
        try:
            parts = [int(v.strip()) for v in args.region.split(",")]
            if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
                raise ValueError
            fixed_region = tuple(parts)  # type: ignore[assignment]
        except ValueError:
            log.error("--region 需要 left,top,width,height 四个整数")
            capture.close()
            return 1

    title = args.window_title if args.window_title is not None else str(
        cfg_get(config, "game.window_title", "WARDOGS")
    )
    if not title.strip() and fixed_region is None and not args.dry_run:
        log.error("game.window_title 为空；请填写窗口标题子串，或用 --window-title 指定")
        capture.close()
        return 1
    hwnd = None
    if fixed_region is None and title.strip():
        hwnd = find_window(title)
    elif fixed_region is None and args.dry_run:
        fixed_region = capture.monitor_region()
        log.warning("窗口标题为空，dry-run 使用主显示器区域")
    if hwnd is None and fixed_region is None:
        if args.dry_run:
            log.warning("没有绑定窗口，dry-run 使用主显示器区域")
            fixed_region = capture.monitor_region()
        else:
            log.error("找不到窗口 %r。用 --list-windows 查看标题，或设置 game.window_title。", title)
            for _, t, _ in list_windows()[:20]:
                log.error("  可见窗口: %s", t)
            capture.close()
            return 2
    if hwnd is not None:
        log.info("绑定窗口: %s", window_title(hwnd))
        log.info("客户区: %s", client_region(hwnd))
    elif fixed_region is not None and not args.dry_run:
        log.warning("使用固定区域模式，不检查前台窗口；请确保游戏在前台")

    human_cfg = HumanConfig(**{
        k: v for k, v in (config.get("human") or {}).items()
        if k in HumanConfig.__dataclass_fields__
    })
    mover = MouseMover(human_cfg, dry_run=args.dry_run)
    max_builds = (args.max_builds if args.max_builds is not None else
                  int(cfg_get(config, "click.max_builds", 0)))
    if max_builds > 0:
        log.info("运行模式：完成 %d 次瞄准建造后自动退出（连续运行请去掉 --max-builds）",
                 max_builds)
    else:
        log.info("运行模式：连续建造，按 F8 暂停/恢复，按 F9 停止")
    max_swings = int(cfg_get(config, "click.max_swings", 0))
    if max_swings > 0:
        log.info("测试限制：完成 %d 次真实落锤后停止（提前松开不计）", max_swings)
    bot = BuildBot(
        config, detector, capture, mover, Hotkeys(),
        hwnd=hwnd, fixed_region=fixed_region, dry_run=args.dry_run,
        max_builds=max_builds,
        debug_dir=Path(args.save_debug) if args.save_debug else None,
        debug_trace=args.debug,
    )
    try:
        bot.run()
    except KeyboardInterrupt:
        log.info("Ctrl+C，退出")
    finally:
        mover.close()
        capture.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
