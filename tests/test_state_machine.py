import sys
import time
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wardogs.auto_build import BuildBot  # noqa: E402
from wardogs.detector import Candidate, Detection  # noqa: E402
import wardogs.win_input as win_input  # noqa: E402
from wardogs.win_input import HumanConfig  # noqa: E402


ZERO_HUMAN = HumanConfig(
    pre_move_ms=(0, 0), steps=(9, 16), step_delay_ms=(0, 0),
    long_pause_prob=0.0, long_pause_ms=(0, 0), bend_ratio=(0.08, 0.22),
    jitter_counts=0.0, post_move_ms=(0, 0), pre_click_ms=(0, 0),
    post_click_ms=(0, 0), between_builds_ms=(0, 0),
    long_break_every=0, long_break_ms=(0, 0),
)


def marker(cx, cy, w=100, h=60):
    area = 2000
    return Candidate(
        bbox=(int(cx - w / 2), int(cy - h / 2), w, h),
        center=(float(cx), float(cy)),
        centroid=(float(cx), float(cy)),
        area=area,
        fill=area / (w * h),
        aspect=w / h,
        hsv_median=(29.0, 170.0, 255.0),
    )


def detection(yellow=None, locked=False):
    return Detection(
        width=2560, height=1440,
        yellow=[yellow] if yellow else [],
        white=[], best=yellow, locked=locked,
        center_v=255.0 if locked else 100.0,
        center_s=1.0 if locked else 20.0,
        ring_v=150.0,
    )


def detection_multi(yellow, locked=False):
    return Detection(
        width=2560, height=1440,
        yellow=list(yellow),
        white=[], best=yellow[0] if yellow else None, locked=locked,
        center_v=255.0 if locked else 100.0,
        center_s=1.0 if locked else 20.0,
        ring_v=150.0,
    )


class FakeMover:
    cfg = ZERO_HUMAN

    def __init__(self):
        self.moves = []
        self.clicks = 0
        self.last_click_timing = (0.0, 0.0, 0.0)

    def move(self, dx, dy, abort=None):
        self.moves.append((dx, dy))
        return True

    def move_fast(self, dx, dy, abort=None):
        self.moves.append((dx, dy))
        return True

    def click(self, hold_ms=510, abort=None):
        self.clicks += 1
        return True

    def click_with_aim(self, hold_ms, aim_callback, strike_guard_ms=60, abort=None,
                       stop_check=None):
        self.clicks += 1
        self.last_click_timing = (0.0, hold_ms, 0.0)
        return True

    def release_left(self):
        pass


class SwingAimMover(FakeMover):
    def click_with_aim(self, hold_ms, aim_callback, strike_guard_ms=60, abort=None,
                       stop_check=None):
        self.clicks += 1
        self.last_click_timing = (0.0, hold_ms, 0.0)
        aim_callback(time.perf_counter() + 1.0)
        return True


class FakeKeys:
    def down(self, name):
        return False

    def pressed(self, name):
        return False


class StateMachineTest(unittest.TestCase):
    def test_hotkey_quick_tap_survives_busy_loop(self):
        class FakeUser32:
            def __init__(self):
                self.state = 0

            def GetAsyncKeyState(self, _vk):
                value = self.state
                self.state = 0  # any read clears the low latch bit
                return value

        hotkeys = win_input.Hotkeys()
        fake = FakeUser32()
        with mock.patch.object(win_input, "_USER32", fake):
            fake.state = 0x0001  # tap completed before the loop polls
            self.assertTrue(hotkeys.pressed("f8"))
            self.assertFalse(hotkeys.pressed("f8"))

            fake.state = 0x0001
            self.assertFalse(hotkeys.down("f8"))  # not held, but latched
            self.assertTrue(hotkeys.pressed("f8"))

            fake.state = 0x8001  # currently held + fresh press
            self.assertTrue(hotkeys.pressed("f8"))
            fake.state = 0x8000  # still held, no new press
            self.assertFalse(hotkeys.pressed("f8"))

    def test_new_build_resets_aim_iterations(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {"post_release_ms": [0, 0], "build_wait_s": 0.01,
                      "poll_ms": 0},
            "miss": {"every_n_builds": 0},
            "game": {"allow_cold_white": True},
        }
        bot = BuildBot(
            config, None, None, FakeMover(), FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=True, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        bot.mode = "CLICK"
        bot._iterations = 8
        bot._close_checks = 4
        bot._record_build_success(detection(marker(1300, 800)))
        self.assertEqual(bot.mode, "AIM")
        self.assertEqual(bot._iterations, 0)
        self.assertEqual(bot._close_checks, 0)

    def test_white_off_center_can_be_recovered(self):
        config = {
            "aim": {"settle_ms": [0, 0], "white_recover_radius_px": 450,
                    "white_recover_max_attempts": 2},
            "click": {}, "miss": {}, "game": {},
        }
        mover = FakeMover()
        bot = BuildBot(
            config, None, None, mover, FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=True, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        white = marker(1500, 800)
        det = Detection(
            width=2560, height=1440, yellow=[], white=[white], best=None,
            locked=False, center_v=180.0, center_s=20.0, ring_v=100.0,
        )
        bot._last_yellow = marker(1200, 760)
        self.assertTrue(bot._recover_white(det))
        self.assertEqual(len(mover.moves), 1)
        self.assertEqual(bot._white_recover_attempts, 1)

    def test_swing_aims_while_button_is_held(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {}, "miss": {},
            "swing": {"enabled": True},
            "game": {},
        }
        mover = FakeMover()
        bot = BuildBot(
            config, None, None, mover, FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=True, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        start = detection(marker(1200, 760))
        target = detection(marker(1300, 800))
        locked = detection(locked=True)
        queue = [target, locked]
        bot._detect = lambda **_kwargs: queue.pop(0)
        self.assertTrue(bot._swing_aim(time.perf_counter() + 1.0, start))
        self.assertEqual(len(mover.moves), 1)

    def test_stale_white_lock_targets_the_next_yellow(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {}, "miss": {},
            "swing": {
                "enabled": True,
                "stale_lock_min_px": 55,
                "stale_lock_hint_px": 180,
                "stale_lock_max_age_s": 2.0,
            },
            "game": {},
        }
        bot = BuildBot(
            config, None, None, FakeMover(), FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=True, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        ghost = marker(1280, 720)
        nxt = marker(1080, 580)
        det = detection_multi([ghost, nxt], locked=True)
        bot._pending_target = nxt.center
        bot._pending_target_time = time.monotonic()
        target = bot._stale_lock_target(det)
        self.assertIsNotNone(target)
        self.assertEqual(target.center, nxt.center)

    def test_overlapping_next_yellow_is_not_filtered_near_crosshair(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {}, "miss": {},
            "swing": {
                "enabled": True,
                "stale_lock_min_px": 55,
                "stale_lock_hint_px": 180,
                "stale_lock_max_age_s": 2.0,
            },
            "game": {},
        }
        bot = BuildBot(
            config, None, None, FakeMover(), FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=True, max_builds=0, debug_dir=None,
        )
        ghost = marker(1280, 720)
        overlapping_next = marker(1252, 704)  # only 32 px from the crosshair
        det = detection_multi([ghost, overlapping_next], locked=True)
        bot._pending_target = overlapping_next.center
        bot._pending_target_time = time.monotonic()
        target = bot._stale_lock_target(det)
        self.assertIsNotNone(target)
        self.assertEqual(target.center, overlapping_next.center)

    def test_overlapping_new_x_is_verified_by_small_screen_move(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {
                "hold_ms": 510, "post_release_ms": [0, 0],
                "build_wait_s": 0.02, "poll_ms": 0,
                "ensure_cursor_in_window": False,
            },
            "miss": {"every_n_builds": 0},
            "swing": {"enabled": True, "overlap_min_move_px": 1.0},
            "game": {},
        }
        mover = SwingAimMover()
        bot = BuildBot(
            config, None, None, mover, FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=False, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        old = marker(1280, 720)
        overlapping_next = marker(1282, 719)
        bot._last_yellow = old
        bot._last_yellow_time = time.monotonic()
        queue = [detection(locked=True), detection(overlapping_next)]
        bot._detect = lambda **_kwargs: queue.pop(0)
        bot.mode = "CLICK"
        bot._step()
        self.assertEqual(bot.stats.aimed_builds, 1)
        self.assertEqual(bot.stats.failed_clicks, 0)

    def test_failed_hit_micro_adjusts_instead_of_pausing(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {
                "hold_ms": 510, "post_release_ms": [0, 0],
                "build_wait_s": 0.01, "poll_ms": 0, "max_failed_clicks": 1,
                "failed_hit_recovery_delay_ms": [0, 0],
                "ensure_cursor_in_window": False,
            },
            "miss": {"every_n_builds": 0},
            "swing": {"enabled": False},
            "game": {},
        }
        mover = FakeMover()
        bot = BuildBot(
            config, None, None, mover, FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=False, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        bot._detect = lambda **_kwargs: detection(locked=True)
        bot.mode = "CLICK"
        bot._step()
        self.assertFalse(bot.hard_pause)
        self.assertEqual(bot.mode, "CLICK")
        self.assertEqual(bot.stats.failed_clicks, 1)
        self.assertEqual(len(mover.moves), 1)

    def test_confirmed_next_target_turning_white_is_clicked(self):
        config = {
            "aim": {"settle_ms": [0, 0], "white_frames": 1},
            "click": {}, "miss": {},
            "swing": {
                "enabled": True,
                "stale_lock_hint_px": 180,
                "stale_lock_max_age_s": 2.0,
            },
            "game": {"allow_cold_white": True},
        }
        bot = BuildBot(
            config, None, None, FakeMover(), FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=True, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        bot._pending_target = (1280.0, 720.0)
        bot._pending_target_time = time.monotonic()
        bot._detect = lambda **_kwargs: detection(locked=True)
        bot.mode = "AIM"
        bot._step()
        self.assertEqual(bot.mode, "CLICK")

    def test_swing_aim_ignores_stale_white_at_crosshair(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {}, "miss": {},
            "swing": {"enabled": True, "stale_lock_min_px": 55},
            "game": {},
        }
        mover = FakeMover()
        bot = BuildBot(
            config, None, None, mover, FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=True, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        ghost = marker(1280, 720)
        nxt = marker(1080, 580)
        start = detection_multi([ghost, nxt], locked=True)
        still_stale = detection_multi([ghost, nxt], locked=True)
        real_lock = detection_multi([], locked=True)
        queue = [still_stale, real_lock]
        bot._detect = lambda **_kwargs: queue.pop(0)
        self.assertTrue(
            bot._swing_aim(time.perf_counter() + 1.0, start, nxt)
        )
        self.assertEqual(len(mover.moves), 1)

    def test_click_after_build_does_not_swing_in_place(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {
                "hold_ms": 510, "post_release_ms": [0, 0],
                "build_wait_s": 0.05, "poll_ms": 0,
                "ensure_cursor_in_window": False,
            },
            "miss": {"every_n_builds": 0},
            "swing": {
                "enabled": True,
                "stale_lock_min_px": 55,
                "stale_lock_hint_px": 180,
                "stale_lock_max_age_s": 2.0,
            },
            "game": {},
        }
        mover = SwingAimMover()
        bot = BuildBot(
            config, None, None, mover, FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=False, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        ghost = marker(1280, 720)
        nxt = marker(1080, 580)
        after = marker(1200, 800)
        bot._pending_target = nxt.center
        bot._pending_target_time = time.monotonic()
        stale = detection_multi([ghost, nxt], locked=True)
        real_lock = detection_multi([], locked=True)
        queue = [stale, stale, real_lock, detection(after), detection(after)]
        bot._detect = lambda **_kwargs: queue.pop(0)
        bot.mode = "CLICK"
        bot._step()
        self.assertEqual(bot.stats.aimed_builds, 1)
        self.assertGreaterEqual(len(mover.moves), 1)
        self.assertEqual(bot._pending_target, after.center)

    def test_unmoved_x_after_missed_swing_is_not_a_build(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {
                "hold_ms": 510, "post_release_ms": [0, 0],
                "build_wait_s": 0, "poll_ms": 0,
                "ensure_cursor_in_window": False, "retry_delay_ms": 0,
            },
            "miss": {"every_n_builds": 0},
            "swing": {
                "enabled": True, "max_aim_moves": 0,
                "stale_lock_min_px": 55, "stale_lock_hint_px": 180,
                "stale_lock_max_age_s": 2.0,
            },
            "game": {},
        }
        mover = SwingAimMover()
        bot = BuildBot(
            config, None, None, mover, FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=False, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        ghost = marker(1280, 720)
        nxt = marker(1080, 580)
        bot._pending_target = nxt.center
        bot._pending_target_time = time.monotonic()
        stale = detection_multi([ghost, nxt], locked=True)
        queue = [stale, detection(nxt), detection(nxt)]
        bot._detect = lambda **_kwargs: queue.pop(0)
        bot.mode = "CLICK"
        bot._step()
        self.assertEqual(bot.stats.aimed_builds, 0)
        self.assertEqual(bot.stats.failed_clicks, 1)

    def test_white_lock_does_not_recount_last_yellow(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {
                "hold_ms": 510, "post_release_ms": [0, 0],
                "build_wait_s": 0.02, "poll_ms": 0,
                "ensure_cursor_in_window": False, "retry_delay_ms": 0,
            },
            "miss": {"every_n_builds": 0},
            "swing": {"enabled": False},
            "game": {},
        }
        bot = BuildBot(
            config, None, None, FakeMover(), FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=False, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        last_yellow = marker(1151, 729)
        bot._last_yellow = last_yellow
        bot._last_yellow_time = time.monotonic()
        bot._last_det = detection(locked=True)
        bot._last_det_time = time.monotonic()
        bot._detect = lambda **_kwargs: detection(last_yellow)
        bot.mode = "CLICK"
        bot._step()
        self.assertEqual(bot.stats.aimed_builds, 0)
        self.assertEqual(bot.stats.failed_clicks, 1)

    def test_click_reuses_verified_frame_without_recapture(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {"post_release_ms": [0, 0]},
            "miss": {"every_n_builds": 0},
            "swing": {"enabled": True},
            "game": {"allow_cold_white": True},
        }
        bot = BuildBot(
            config, None, None, FakeMover(), FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=True, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        target = marker(1100, 700)
        det = detection(target)
        bot._last_det = det
        bot._last_det_time = time.monotonic()
        bot._click_target = target
        bot._detect = lambda **_kwargs: self.fail("unexpected recapture")
        bot.mode = "CLICK"
        bot._step()
        self.assertEqual(bot.stats.aimed_builds, 1)

    def test_chain_starts_next_swing_before_verification(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {
                "hold_ms": 510, "post_release_ms": [0, 0],
                "build_wait_s": 0.05, "poll_ms": 0,
                "ensure_cursor_in_window": False,
            },
            "miss": {"every_n_builds": 0},
            "swing": {
                "enabled": True, "chain": True,
                "stale_lock_min_px": 55, "chain_min_move_px": 40,
            },
            "game": {},
        }
        mover = SwingAimMover()
        bot = BuildBot(
            config, None, None, mover, FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=False, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        first = detection(marker(1200, 700))
        locked = detection(locked=True)
        second = detection_multi([marker(1400, 800)])
        third = detection_multi([marker(1250, 750)])
        queue = [first, locked, second, locked, locked, third, locked, locked]
        bot._detect = lambda **_kwargs: queue.pop(0)
        bot.mode = "CLICK"
        bot._step()   # first swing -> chain pending
        self.assertIsNotNone(bot._chain_pending)
        bot._step()   # immediate chain swing; finds new X and counts build #1
        self.assertEqual(bot.stats.aimed_builds, 1)
        self.assertIsNotNone(bot._chain_pending)
        bot._step()   # another chain swing; counts build #2
        self.assertEqual(bot.stats.aimed_builds, 2)
        self.assertEqual(mover.clicks, 3)

    def test_chain_accepts_overlapping_spawn_instead_of_stopping(self):
        config = {
            "aim": {"settle_ms": [0, 0]},
            "click": {"hold_ms": 510, "max_failed_clicks": 1},
            "miss": {"every_n_builds": 0},
            "swing": {
                "enabled": True, "chain": True,
                "chain_min_move_px": 40, "overlap_min_move_px": 1,
            },
            "game": {},
        }
        bot = BuildBot(
            config, None, None, FakeMover(), FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=True, max_builds=0, debug_dir=None,
        )
        old = marker(1280, 720)
        overlapping = marker(1282, 719)
        prev = {
            "expected_old": old.center,
            "aim_ok": True,
            "build_start": time.perf_counter(),
            "pre_target": old.center,
            "pre_candidate": old,
            "verify_move": 40.0,
        }
        self.assertTrue(bot._chain_judge_previous(overlapping, prev))
        self.assertEqual(bot.stats.aimed_builds, 1)
        self.assertEqual(bot.stats.failed_clicks, 0)
        self.assertFalse(bot._chain_stop_after)

    def test_max_swings_counts_only_full_hammers(self):
        config = {
            "aim": {}, "click": {"hold_ms": 510, "max_swings": 2},
            "miss": {}, "swing": {"enabled": True}, "game": {},
        }
        bot = BuildBot(
            config, None, None, FakeMover(), FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=True, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        self.assertFalse(bot._count_swing(510))
        self.assertFalse(bot._count_swing(200))   # early release, no strike
        self.assertTrue(bot._count_swing(510))
        self.assertTrue(bot.stop)
        self.assertEqual(bot.stats.swings, 2)
        self.assertEqual(bot.stats.aborted_swings, 1)

    def test_missing_yellow_keeps_searching(self):
        config = {
            "aim": {"lost_timeout_s": 0.01, "settle_ms": [0, 0]},
            "click": {}, "miss": {},
            "game": {"allow_cold_white": True},
        }
        bot = BuildBot(
            config, None, None, FakeMover(), FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=True, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        bot.mode = "SEARCH"
        bot._ever_seen = True
        bot._last_seen = time.monotonic() - 100.0
        bot._detect = lambda **_kwargs: detection(None)
        bot._step()
        self.assertFalse(bot.hard_pause)
        self.assertFalse(bot.stop)
        self.assertEqual(bot.mode, "SEARCH")

    def test_aim_white_click_and_non_aim_click(self):
        config = {
            "aim": {
                "gain_x": -0.55, "gain_y": -0.55,
                "max_iterations": 10, "settle_ms": [0, 0],
                "white_frames": 1, "stable_frames": 2,
                "calibrate_min_error_px": 150,
            },
            "click": {
                "hold_ms": 510, "post_release_ms": [0, 0],
                "build_wait_s": 0.05, "poll_ms": 0, "max_failed_clicks": 3,
            },
            "swing": {"enabled": False},
            "miss": {"every_n_builds": 1, "min_distance_px": 120,
                     "look_away_max_tries": 2, "hold_ms": 510},
            "game": {"allow_cold_white": True},
        }
        mover = FakeMover()
        bot = BuildBot(
            config, None, None, mover, FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=False, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None

        first = marker(1200, 760)      # error (80, -40), near the screen centre
        locked = detection(locked=True)
        moved = marker(1500, 800)      # far from centre: safe for non-aim click
        queue = [detection(first), detection(first), detection(first),
                 locked, locked, detection(moved), detection(moved)]
        bot._detect = lambda **_kwargs: queue.pop(0)

        for _ in range(6):
            bot._step()

        self.assertEqual(bot.stats.aimed_builds, 1)
        self.assertEqual(mover.clicks, 2)          # aimed click + non-aim click
        self.assertEqual(bot.stats.non_aim_clicks, 1)
        self.assertEqual(len(mover.moves), 1)      # the aim correction
        self.assertEqual(bot.mode, "SEARCH")

    def test_white_is_clicked_after_two_frames(self):
        config = {
            "aim": {"white_frames": 2, "settle_ms": [0, 0], "stable_frames": 2},
            "click": {"hold_ms": 510, "post_release_ms": [0, 0],
                      "build_wait_s": 0.02, "poll_ms": 0},
            "miss": {"every_n_builds": 0},
            "game": {"allow_cold_white": True},
        }
        mover = FakeMover()
        bot = BuildBot(
            config, None, None, mover, FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=False, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        locked = detection(locked=True)
        new = marker(1300, 800)
        queue = [locked, locked, locked, detection(new)]
        bot._detect = lambda **_kwargs: queue.pop(0)
        bot.mode = "SEARCH"
        bot._step()   # white hit 1
        bot._step()   # white hit 2 -> CLICK
        self.assertEqual(bot.mode, "CLICK")
        bot._step()   # fresh lock, click, then new yellow
        self.assertEqual(bot.stats.aimed_builds, 1)
        self.assertEqual(mover.clicks, 1)

    def test_far_target_calibrates_both_axes(self):
        config = {
            "aim": {
                "gain_x": -0.55, "gain_y": -0.55, "max_iterations": 10,
                "settle_ms": [0, 0], "white_frames": 2, "stable_frames": 2,
                "calibrate_min_error_px": 150, "calibrate_probe_counts": 12,
            },
            "click": {"hold_ms": 510, "post_release_ms": [0, 0],
                      "build_wait_s": 0.05, "poll_ms": 0},
            "swing": {"enabled": False},
            "miss": {"every_n_builds": 0},
            "game": {"allow_cold_white": True},
        }
        mover = FakeMover()
        bot = BuildBot(
            config, None, None, mover, FakeKeys(),
            hwnd=None, fixed_region=(0, 0, 2560, 1440),
            dry_run=False, max_builds=0, debug_dir=None,
        )
        bot._sleep = lambda ms: None
        far = marker(900, 600)
        after_x = marker(880, 600)
        after_y = marker(880, 590)
        queue = [detection(far), detection(after_x), detection(after_y),
                 detection(after_y), detection(after_y)]
        bot._detect = lambda: queue.pop(0)
        bot.mode = "AIM"
        bot._step()
        self.assertTrue(bot._calibrated)
        self.assertAlmostEqual(bot.gain_x, -20.0 / 12.0, places=2)
        self.assertAlmostEqual(bot.gain_y, -10.0 / 12.0, places=2)
        self.assertEqual(mover.moves[:2], [(12, 0), (0, 12)])


if __name__ == "__main__":
    unittest.main()
