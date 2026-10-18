import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import wardogs.auto_build as auto_build
from wardogs.win_input import build_elevation_command


class ElevationCommandTests(unittest.TestCase):
    def test_script_arguments_are_quoted_for_shell_execute(self):
        executable, parameters = build_elevation_command(
            r"C:\Program Files\wardogs\auto_build.py",
            ["--debug", "--max-builds", "3"],
            python=r"C:\Python311\python.exe",
        )

        self.assertEqual(executable, r"C:\Python311\python.exe")
        self.assertTrue(
            parameters.startswith(r'"C:\Program Files\wardogs\auto_build.py"')
        )
        self.assertIn("--debug --max-builds 3", parameters)

    def test_frozen_build_does_not_repeat_the_script_path(self):
        executable, parameters = build_elevation_command(
            r"C:\ignored\auto_build.py",
            ["--chain"],
            python=r"C:\Apps\wardogs.exe",
            frozen=True,
        )

        self.assertEqual(executable, r"C:\Apps\wardogs.exe")
        self.assertEqual(parameters, "--chain")
        self.assertNotIn("auto_build.py", parameters)


class ElevationFlowTests(unittest.TestCase):
    def test_live_run_relaunches_with_all_arguments(self):
        with mock.patch.object(
            auto_build, "is_process_elevated", return_value=False
        ), mock.patch.object(
            auto_build, "relaunch_elevated", return_value=True
        ) as relaunch:
            code = auto_build.main(["--test-click", "1", "--debug"])

        self.assertEqual(code, 0)
        elevated_args = relaunch.call_args.args[0]
        self.assertIn("--test-click", elevated_args)
        self.assertIn("--debug", elevated_args)
        self.assertIn("--elevation-attempted", elevated_args)
        self.assertTrue(
            relaunch.call_args.kwargs["script"].endswith("auto_build.py")
        )

    def test_second_attempt_without_admin_fails_fast(self):
        with mock.patch.object(
            auto_build, "is_process_elevated", return_value=False
        ), mock.patch.object(
            auto_build, "relaunch_elevated"
        ) as relaunch:
            code = auto_build.main([
                "--test-click", "1", "--elevation-attempted",
            ])

        self.assertEqual(code, 1)
        relaunch.assert_not_called()

    def test_offline_image_mode_does_not_elevate(self):
        with mock.patch.object(
            auto_build, "is_process_elevated", return_value=False
        ), mock.patch.object(
            auto_build, "relaunch_elevated"
        ) as relaunch, mock.patch.object(
            auto_build, "run_test_images", return_value=0
        ) as run_test_images:
            code = auto_build.main(["--test-images", "dummy.png"])

        self.assertEqual(code, 0)
        relaunch.assert_not_called()
        run_test_images.assert_called_once()

    def test_log_file_receives_records(self):
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "run.log"

            def fake_run_test_images(*_args, **_kwargs):
                auto_build.log.warning("log-file-test-marker")
                return 0

            try:
                with mock.patch.object(
                    auto_build, "is_process_elevated", return_value=False
                ), mock.patch.object(
                    auto_build, "relaunch_elevated"
                ) as relaunch, mock.patch.object(
                    auto_build, "run_test_images",
                    side_effect=fake_run_test_images,
                ):
                    code = auto_build.main([
                        "--test-images", "dummy.png",
                        "--log-file", str(log_path),
                    ])

                self.assertEqual(code, 0)
                relaunch.assert_not_called()
                self.assertIn(
                    "log-file-test-marker",
                    log_path.read_text(encoding="utf-8"),
                )
            finally:
                for handler in list(logging.getLogger().handlers):
                    if (isinstance(handler, auto_build._FlushFileHandler)
                            and Path(handler.baseFilename).resolve()
                            == log_path.resolve()):
                        handler.close()
                        logging.getLogger().removeHandler(handler)

    def test_bootstrap_does_not_truncate_elevated_log_file(self):
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "run.log"
            with mock.patch.object(
                auto_build, "is_process_elevated", return_value=False
            ), mock.patch.object(
                auto_build, "relaunch_elevated", return_value=True
            ) as relaunch:
                code = auto_build.main([
                    "--test-click", "1", "--log-file", str(log_path),
                ])

            self.assertEqual(code, 0)
            relaunch.assert_called_once()
            self.assertFalse(log_path.exists())


if __name__ == "__main__":
    unittest.main()
