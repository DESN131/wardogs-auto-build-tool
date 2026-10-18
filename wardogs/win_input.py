"""Windows focus, capture-region, hotkey and human-like mouse utilities."""

from __future__ import annotations

import ctypes
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import win32gui
from ctypes import wintypes


# ---------------------------------------------------------------- DPI / timer

def enable_dpi_awareness() -> None:
    """Make window/monitor coordinates physical pixels on scaled displays."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def is_process_elevated() -> bool:
    """Return whether this process can inject input into elevated windows."""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def is_window_elevated(hwnd: int) -> Optional[bool]:
    """Best-effort elevation check for the process owning a window."""
    try:
        import win32api
        import win32con
        import win32process
        import win32security

        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = win32api.OpenProcess(
            win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        try:
            token = win32security.OpenProcessToken(
                process, win32con.TOKEN_QUERY
            )
            try:
                value = win32security.GetTokenInformation(
                    token, win32security.TokenElevation
                )
                return bool(getattr(value, "TokenIsElevated", value))
            finally:
                token.Close()
        finally:
            process.Close()
    except Exception:
        return None


_SHELL32 = ctypes.WinDLL("shell32", use_last_error=True)
_SHELL32.ShellExecuteW.argtypes = [
    wintypes.HWND,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    ctypes.c_int,
]
_SHELL32.ShellExecuteW.restype = ctypes.c_void_p


def build_elevation_command(
    script: str,
    argv: list[str],
    *,
    python: Optional[str] = None,
    frozen: bool = False,
) -> tuple[str, str]:
    """Build the executable and parameter string for an elevated relaunch."""
    executable = python or sys.executable
    if frozen:
        parameters = subprocess.list2cmdline([str(arg) for arg in argv])
    else:
        parameters = subprocess.list2cmdline(
            [str(Path(script).resolve()), *(str(arg) for arg in argv)]
        )
    return executable, parameters


def relaunch_elevated(
    argv: Optional[list[str]] = None,
    *,
    script: Optional[str] = None,
) -> bool:
    """Ask Windows to relaunch this script with an elevated token.

    Returns True only when the ShellExecuteW request started a new process.
    A False result includes the user cancelling the UAC prompt, so callers
    should stop instead of continuing with a medium-integrity token.
    """
    if os.name != "nt" or is_process_elevated():
        return False
    executable, parameters = build_elevation_command(
        script or sys.argv[0],
        list(sys.argv[1:] if argv is None else argv),
        frozen=bool(getattr(sys, "frozen", False)),
    )
    result = _SHELL32.ShellExecuteW(
        None,
        "runas",
        executable,
        parameters,
        os.getcwd(),
        1,  # SW_SHOWNORMAL
    )
    return int(result or 0) > 32


class _TimerResolution:
    def __init__(self) -> None:
        self._active = False
        try:
            self._winmm = ctypes.WinDLL("winmm")
            self._winmm.timeBeginPeriod(1)
            self._active = True
        except Exception:
            self._winmm = None

    def close(self) -> None:
        if self._active and self._winmm is not None:
            try:
                self._winmm.timeEndPeriod(1)
            except Exception:
                pass
            self._active = False


# ---------------------------------------------------------------- windows

def list_windows() -> list[tuple[int, str, str]]:
    result: list[tuple[int, str, str]] = []

    def callback(hwnd: int, _extra: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if title:
            result.append((hwnd, title, win32gui.GetClassName(hwnd)))
        return True

    win32gui.EnumWindows(callback, None)
    return result


def find_window(title_substring: str) -> Optional[int]:
    if not title_substring:
        hwnd = win32gui.GetForegroundWindow()
        return hwnd or None
    needle = title_substring.lower()
    matches = [(hwnd, title) for hwnd, title, _ in list_windows()
               if needle in title.lower()]
    if not matches:
        return None
    foreground = win32gui.GetForegroundWindow()
    for hwnd, _ in matches:
        if hwnd == foreground:
            return hwnd
    return matches[0][0]


def window_title(hwnd: int) -> str:
    try:
        return win32gui.GetWindowText(hwnd)
    except Exception:
        return ""


def window_alive(hwnd: int) -> bool:
    try:
        return bool(win32gui.IsWindow(hwnd))
    except Exception:
        return False


def client_region(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    """Return (left, top, width, height) of the client area, or None."""
    try:
        if not win32gui.IsWindow(hwnd) or win32gui.IsIconic(hwnd):
            return None
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            return None
        origin = win32gui.ClientToScreen(hwnd, (0, 0))
        return (int(origin[0]), int(origin[1]), int(width), int(height))
    except Exception:
        return None


def is_foreground(hwnd: int) -> bool:
    try:
        foreground = win32gui.GetForegroundWindow()
        if foreground == hwnd:
            return True
        # Some games use a child window as the foreground window.
        return win32gui.GetAncestor(foreground, 2) == hwnd  # GA_ROOT
    except Exception:
        return False


# ---------------------------------------------------------------- SendInput

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT),
    ]


_USER32 = ctypes.WinDLL("user32", use_last_error=True)
_USER32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
_USER32.SendInput.restype = ctypes.c_uint
_USER32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_USER32.GetAsyncKeyState.restype = ctypes.c_short
_USER32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
_USER32.GetCursorPos.restype = wintypes.BOOL
_USER32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
_USER32.SetCursorPos.restype = wintypes.BOOL


def _send_mouse(dx: int, dy: int, flags: int) -> None:
    inp = INPUT(type=0, mi=MOUSEINPUT(int(dx), int(dy), 0, flags, 0, None))
    sent = _USER32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        err = ctypes.get_last_error()
        raise OSError(f"SendInput failed (err={err})")


def send_relative_move(dx: int, dy: int) -> None:
    _send_mouse(dx, dy, MOUSEEVENTF_MOVE)


def send_left_down() -> None:
    _send_mouse(0, 0, MOUSEEVENTF_LEFTDOWN)


def send_left_up() -> None:
    _send_mouse(0, 0, MOUSEEVENTF_LEFTUP)


def cursor_pos() -> Optional[tuple[int, int]]:
    point = wintypes.POINT()
    if not _USER32.GetCursorPos(ctypes.byref(point)):
        return None
    return (int(point.x), int(point.y))


def set_cursor_pos(x: int, y: int) -> bool:
    return bool(_USER32.SetCursorPos(int(x), int(y)))


def ensure_cursor_in_region(region: tuple[int, int, int, int]) -> bool:
    """Keep the OS cursor inside the game client area before a click.

    Relative mouse motion accumulates, so a long aiming session can leave the
    system cursor outside a windowed game.  In that case a synthetic click may
    be routed to the desktop/another overlay window.  Recenter only when it is
    actually outside; a cursor already inside the client area is left alone.
    """
    left, top, width, height = region
    point = cursor_pos()
    if point is not None:
        x, y = point
        if left <= x < left + width and top <= y < top + height:
            return False
    return set_cursor_pos(left + width // 2, top + height // 2)


class Hotkeys:
    VK = {"f7": 0x76, "f8": 0x77, "f9": 0x78}

    def __init__(self) -> None:
        # GetAsyncKeyState's low bit is a "pressed since last read" latch.  It
        # survives a quick press+release while the main loop is busy with a
        # 510 ms swing, unlike the high "currently down" bit.
        self._pressed_latch: dict[str, bool] = {}

    def _read(self, name: str) -> int:
        state = int(_USER32.GetAsyncKeyState(self.VK[name]))
        if state & 0x0001:
            # Any GetAsyncKeyState call clears the low bit; keep our own latch
            # so a frequent down() check (e.g. F9 abort polling) cannot eat a
            # press before pressed() sees it.
            self._pressed_latch[name] = True
        return state

    def down(self, name: str) -> bool:
        return bool(self._read(name) & 0x8000)

    def pressed(self, name: str) -> bool:
        self._read(name)
        return self._pressed_latch.pop(name, False)


# ---------------------------------------------------------------- human motion

@dataclass
class HumanConfig:
    pre_move_ms: tuple[float, float] = (10.0, 35.0)
    steps: tuple[int, int] = (14, 20)
    step_delay_ms: tuple[float, float] = (3.0, 5.0)
    long_pause_prob: float = 0.03
    long_pause_ms: tuple[float, float] = (5.0, 12.0)
    bend_ratio: tuple[float, float] = (0.06, 0.16)
    jitter_counts: float = 2.8
    jitter_outlier_prob: float = 0.08
    jitter_outlier_counts: float = 4.5
    overshoot_ratio: tuple[float, float] = (0.06, 0.10)
    overshoot_max_counts: float = 30.0
    overshoot_pause_ms: tuple[float, float] = (4.0, 10.0)
    post_move_ms: tuple[float, float] = (8.0, 20.0)
    pre_click_ms: tuple[float, float] = (15.0, 35.0)
    post_click_ms: tuple[float, float] = (8.0, 20.0)
    between_builds_ms: tuple[float, float] = (20.0, 60.0)
    long_break_every: int = 8
    long_break_ms: tuple[float, float] = (200.0, 350.0)


def _sleep_ms(ms: float, abort: Optional[Callable[[], bool]] = None) -> bool:
    """Sleep in small chunks so F9 can abort quickly.  False means aborted."""
    end = time.perf_counter() + max(0.0, ms) / 1000.0
    while True:
        if abort is not None and abort():
            return False
        remaining = end - time.perf_counter()
        if remaining <= 0:
            return True
        time.sleep(min(0.008, remaining))


class MouseMover:
    """Relative mouse output with a curved, jittery, human-like trajectory."""

    def __init__(self, cfg: Optional[HumanConfig] = None, dry_run: bool = False,
                 rng: Optional[random.Random] = None):
        self.cfg = cfg or HumanConfig()
        self.dry_run = dry_run
        self.rng = rng or random.Random()
        self._timer = _TimerResolution()
        self._left_down = False
        self.last_click_timing = (0.0, 0.0, 0.0)

    def move(self, dx: int, dy: int,
             abort: Optional[Callable[[], bool]] = None) -> bool:
        dx, dy = int(round(dx)), int(round(dy))
        if dx == 0 and dy == 0:
            return True
        cfg, rng = self.cfg, self.rng
        if not _sleep_ms(rng.uniform(*cfg.pre_move_ms), abort):
            return False

        if abs(dx) + abs(dy) <= 3:
            if not self.dry_run:
                send_relative_move(dx, dy)
            if not _sleep_ms(rng.uniform(4, 12), abort):
                return False
            return True

        length = math.hypot(dx, dy)
        # Perpendicular control-point offsets make the path bend instead of
        # drawing a perfectly straight line.
        nx, ny = (-dy / length, dx / length)
        sign = rng.choice((-1.0, 1.0))
        bend1 = length * rng.uniform(*cfg.bend_ratio) * sign
        bend2 = length * rng.uniform(-0.10, 0.10)
        p1 = (dx * 0.30 + nx * bend1, dy * 0.30 + ny * bend1)
        p2 = (dx * 0.70 + nx * bend2, dy * 0.70 + ny * bend2)

        steps = rng.randint(*cfg.steps)
        sent_x = sent_y = 0

        def random_jitter() -> int:
            sigma = (cfg.jitter_outlier_counts
                     if rng.random() < cfg.jitter_outlier_prob
                     else cfg.jitter_counts)
            return int(round(rng.gauss(0.0, sigma)))

        for index in range(1, steps + 1):
            t = index / steps
            eased = 0.5 - 0.5 * math.cos(math.pi * t)  # ease-in-out
            omt = 1.0 - eased
            bx = (omt ** 3 * 0.0 + 3 * omt ** 2 * eased * p1[0]
                  + 3 * omt * eased ** 2 * p2[0] + eased ** 3 * dx)
            by = (omt ** 3 * 0.0 + 3 * omt ** 2 * eased * p1[1]
                  + 3 * omt * eased ** 2 * p2[1] + eased ** 3 * dy)
            target_x, target_y = int(round(bx)), int(round(by))
            jitter_x = random_jitter()
            jitter_y = random_jitter()
            step_x = target_x - sent_x + jitter_x
            step_y = target_y - sent_y + jitter_y
            if step_x or step_y:
                if not self.dry_run:
                    send_relative_move(step_x, step_y)
                sent_x += step_x
                sent_y += step_y
            delay = rng.uniform(*cfg.step_delay_ms)
            if rng.random() < cfg.long_pause_prob:
                delay += rng.uniform(*cfg.long_pause_ms)
            if not _sleep_ms(delay, abort):
                return False

        # Remove accumulated jitter without a visible final snap.
        rem_x, rem_y = dx - sent_x, dy - sent_y
        for _ in range(2):
            if abs(rem_x) + abs(rem_y) <= 8:
                break
            part_x = int(round(rem_x * 0.6))
            part_y = int(round(rem_y * 0.6))
            if not self.dry_run and (part_x or part_y):
                send_relative_move(part_x, part_y)
            rem_x -= part_x
            rem_y -= part_y
            if not _sleep_ms(rng.uniform(4, 10), abort):
                return False
        if not self.dry_run and (rem_x or rem_y):
            send_relative_move(rem_x, rem_y)

        # A small terminal overshoot + correction makes the flick look like a
        # practised player instead of a perfect servo move.  The net endpoint
        # remains exact.
        if length >= 24.0 and cfg.overshoot_ratio[1] > 0:
            ratio = rng.uniform(*cfg.overshoot_ratio)
            ox, oy = dx * ratio, dy * ratio
            cap = min(cfg.overshoot_max_counts, length * 0.12)
            over_len = math.hypot(ox, oy)
            if over_len > cap:
                scale = cap / over_len
                ox *= scale
                oy *= scale
            ox, oy = int(round(ox)), int(round(oy))
            if ox or oy:
                if not self.dry_run:
                    send_relative_move(ox, oy)
                if not _sleep_ms(rng.uniform(*cfg.overshoot_pause_ms), abort):
                    return False
                if not self.dry_run:
                    send_relative_move(-ox, -oy)
                if not _sleep_ms(rng.uniform(3.0, 8.0), abort):
                    return False

        if not _sleep_ms(rng.uniform(*cfg.post_move_ms), abort):
            return False
        return True

    def move_fast(self, dx: int, dy: int,
                  abort: Optional[Callable[[], bool]] = None) -> bool:
        """Short, low-latency correction used near the hammer strike.

        This keeps the human-like curve for normal aiming, but a chain swing
        often has only 80-150 ms left after the first correction.  Using
        14-20 jittered segments there overruns the strike window; a short
        flick finishes the last 20-50 px in time.
        """
        dx, dy = int(round(dx)), int(round(dy))
        if dx == 0 and dy == 0:
            return True
        if not _sleep_ms(self.rng.uniform(2.0, 6.0), abort):
            return False
        length = math.hypot(dx, dy)
        steps = max(5, min(10, int(length / 18.0) + 5))
        sent_x = sent_y = 0
        for index in range(1, steps + 1):
            t = index / steps
            eased = 1.0 - (1.0 - t) ** 3
            target_x = int(round(dx * eased))
            target_y = int(round(dy * eased))
            jitter_x = int(round(self.rng.gauss(0.0, 0.7)))
            jitter_y = int(round(self.rng.gauss(0.0, 0.7)))
            step_x = target_x - sent_x + jitter_x
            step_y = target_y - sent_y + jitter_y
            if step_x or step_y:
                if not self.dry_run:
                    send_relative_move(step_x, step_y)
                sent_x += step_x
                sent_y += step_y
            if not _sleep_ms(self.rng.uniform(1.2, 2.8), abort):
                return False
        rem_x, rem_y = dx - sent_x, dy - sent_y
        if rem_x or rem_y:
            if not self.dry_run:
                send_relative_move(rem_x, rem_y)
        if not _sleep_ms(self.rng.uniform(2.0, 6.0), abort):
            return False
        return True

    def click(self, hold_ms: float = 510.0,
              abort: Optional[Callable[[], bool]] = None) -> bool:
        self.last_click_timing = (0.0, 0.0, 0.0)
        click_start = time.perf_counter()
        if not _sleep_ms(self.rng.uniform(*self.cfg.pre_click_ms), abort):
            return False
        pre_ms = (time.perf_counter() - click_start) * 1000.0
        if self.dry_run:
            hold_start = time.perf_counter()
            ok = _sleep_ms(hold_ms, abort)
            actual_hold = (time.perf_counter() - hold_start) * 1000.0
            post_start = time.perf_counter()
            _sleep_ms(self.rng.uniform(*self.cfg.post_click_ms), abort)
            post_ms = (time.perf_counter() - post_start) * 1000.0
            self.last_click_timing = (pre_ms, actual_hold, post_ms)
            return ok
        send_left_down()
        self._left_down = True
        hold_start = time.perf_counter()
        try:
            end = time.perf_counter() + hold_ms / 1000.0
            while True:
                if abort is not None and abort():
                    return False
                remaining = end - time.perf_counter()
                if remaining <= 0:
                    break
                time.sleep(min(0.01, remaining))
        finally:
            actual_hold = (time.perf_counter() - hold_start) * 1000.0
            send_left_up()
            self._left_down = False
        post_start = time.perf_counter()
        _sleep_ms(self.rng.uniform(*self.cfg.post_click_ms), abort)
        post_ms = (time.perf_counter() - post_start) * 1000.0
        self.last_click_timing = (pre_ms, actual_hold, post_ms)
        return True

    def click_with_aim(self, hold_ms: float, aim_callback: Callable[[float], bool],
                       strike_guard_ms: float = 60.0,
                       abort: Optional[Callable[[], bool]] = None,
                       stop_check: Optional[Callable[[], bool]] = None) -> bool:
        """Hold LMB while ``aim_callback`` moves the crosshair.

        The callback receives the monotonic deadline for aiming and should
        return before the strike guard.  The button is released at hold_ms.
        """
        self.last_click_timing = (0.0, 0.0, 0.0)
        click_start = time.perf_counter()
        if not _sleep_ms(self.rng.uniform(*self.cfg.pre_click_ms), abort):
            return False
        pre_ms = (time.perf_counter() - click_start) * 1000.0

        if self.dry_run:
            hold_start = time.perf_counter()
            deadline = hold_start + max(0.0, hold_ms - strike_guard_ms) / 1000.0
            ok = aim_callback(deadline)
            hold_end = hold_start + hold_ms / 1000.0
            remaining = max(0.0, (hold_end - time.perf_counter()) * 1000.0)
            if stop_check is not None and stop_check():
                remaining = 0.0
            if not _sleep_ms(remaining, abort):
                return False
            actual_hold = (time.perf_counter() - hold_start) * 1000.0
            post_start = time.perf_counter()
            _sleep_ms(self.rng.uniform(*self.cfg.post_click_ms), abort)
            post_ms = (time.perf_counter() - post_start) * 1000.0
            self.last_click_timing = (pre_ms, actual_hold, post_ms)
            return ok

        send_left_down()
        self._left_down = True
        hold_start = time.perf_counter()
        deadline = hold_start + max(0.0, hold_ms - strike_guard_ms) / 1000.0
        try:
            aim_callback(deadline)
            hold_end = hold_start + hold_ms / 1000.0
            while True:
                if abort is not None and abort():
                    return False
                if stop_check is not None and stop_check():
                    break
                remaining = hold_end - time.perf_counter()
                if remaining <= 0:
                    break
                time.sleep(min(0.01, remaining))
        finally:
            actual_hold = (time.perf_counter() - hold_start) * 1000.0
            send_left_up()
            self._left_down = False
        post_start = time.perf_counter()
        _sleep_ms(self.rng.uniform(*self.cfg.post_click_ms), abort)
        post_ms = (time.perf_counter() - post_start) * 1000.0
        self.last_click_timing = (pre_ms, actual_hold, post_ms)
        return True

    def release_left(self) -> None:
        if self._left_down:
            try:
                send_left_up()
            except Exception:
                pass
            self._left_down = False

    def close(self) -> None:
        self.release_left()
        self._timer.close()
