"""Tests für computer.py (PHASE 6 – Computer steuern).

Alle Tests laufen ohne echten Bildschirm / Hardware durch Mocking.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Hilfsfunktion: Mock-pyautogui erzeugen
# ---------------------------------------------------------------------------


def _make_pag():
    pag = MagicMock()
    pag.FAILSAFE = True
    pag.position.return_value = MagicMock(x=100, y=200)
    return pag


# ---------------------------------------------------------------------------
# open_program
# ---------------------------------------------------------------------------


class TestOpenProgram(unittest.TestCase):
    def test_empty_name_raises(self):
        import computer
        with self.assertRaises(computer.ControlError):
            computer.open_program("   ")

    @patch("computer.sys.platform", "win32")
    @patch("computer.subprocess.Popen")
    def test_windows_startfile(self, mock_popen):
        import computer
        import os as _os
        had_attr = hasattr(_os, "startfile")
        mock_sf = MagicMock()
        setattr(_os, "startfile", mock_sf)
        try:
            computer.open_program("notepad.exe")
        finally:
            if not had_attr:
                delattr(_os, "startfile")
        mock_sf.assert_called_once_with("notepad.exe")

    @patch("computer.sys.platform", "linux")
    @patch("computer.subprocess.Popen")
    def test_linux_xdg_open(self, mock_popen):
        import computer
        computer.open_program("gedit")
        mock_popen.assert_called_once_with(["xdg-open", "gedit"])


# ---------------------------------------------------------------------------
# close_program
# ---------------------------------------------------------------------------


class TestCloseProgram(unittest.TestCase):
    def test_empty_name_raises(self):
        import computer
        with self.assertRaises(computer.ControlError):
            computer.close_program("")

    def test_no_matching_process_raises(self):
        import computer
        mock_psutil = MagicMock()
        mock_psutil.process_iter.return_value = []
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = Exception
        with patch("computer._psutil", return_value=mock_psutil):
            with self.assertRaises(computer.ControlError):
                computer.close_program("nonexistent_xyz")

    def test_matching_process_terminated(self):
        import computer
        proc = MagicMock()
        proc.info = {"name": "chrome.exe", "pid": 1234}
        mock_psutil = MagicMock()
        mock_psutil.process_iter.return_value = [proc]
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = Exception
        with patch("computer._psutil", return_value=mock_psutil):
            count = computer.close_program("chrome")
        self.assertEqual(count, 1)
        proc.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# open_browser
# ---------------------------------------------------------------------------


class TestOpenBrowser(unittest.TestCase):
    @patch("computer.webbrowser.open")
    def test_opens_url(self, mock_open):
        import computer
        computer.open_browser("https://example.com")
        mock_open.assert_called_once_with("https://example.com")

    @patch("computer.webbrowser.open")
    def test_adds_https(self, mock_open):
        import computer
        computer.open_browser("example.com")
        mock_open.assert_called_once_with("https://example.com")

    def test_empty_url_raises(self):
        import computer
        with self.assertRaises(computer.ControlError):
            computer.open_browser("")


# ---------------------------------------------------------------------------
# find_files
# ---------------------------------------------------------------------------


class TestFindFiles(unittest.TestCase):
    def test_empty_query_raises(self):
        import computer
        with self.assertRaises(computer.ControlError):
            computer.find_files("")

    def test_nonexistent_dir_raises(self):
        import computer
        with self.assertRaises(computer.ControlError):
            computer.find_files("test", search_dir="/nonexistent/path/xyz")

    def test_finds_files(self, tmp_path=None):
        import tempfile
        import computer
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test_file.txt"
            p.write_text("hello")
            results = computer.find_files("test_file", search_dir=tmp)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "test_file.txt")


# ---------------------------------------------------------------------------
# open_folder / open_explorer
# ---------------------------------------------------------------------------


class TestOpenFolder(unittest.TestCase):
    def test_nonexistent_raises(self):
        import computer
        with self.assertRaises(computer.ControlError):
            computer.open_folder("/this/does/not/exist/xyz")

    @patch("computer.sys.platform", "linux")
    @patch("computer.subprocess.Popen")
    def test_opens_home(self, mock_popen):
        import computer
        computer.open_folder(Path.home())
        mock_popen.assert_called_once()

    @patch("computer.sys.platform", "linux")
    @patch("computer.subprocess.Popen")
    def test_open_explorer_alias(self, mock_popen):
        import computer
        computer.open_explorer(Path.home())
        mock_popen.assert_called_once()


# ---------------------------------------------------------------------------
# take_screenshot
# ---------------------------------------------------------------------------


class TestTakeScreenshot(unittest.TestCase):
    @patch("computer._pyautogui")
    def test_returns_path(self, mock_pag_fn):
        import tempfile
        import computer
        pag = MagicMock()
        img = MagicMock()
        pag.screenshot.return_value = img
        mock_pag_fn.return_value = pag

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "shot.png"
            result = computer.take_screenshot(save_path=out)

        self.assertEqual(result, out)
        img.save.assert_called_once_with(str(out))


# ---------------------------------------------------------------------------
# Zwischenablage
# ---------------------------------------------------------------------------


class TestClipboard(unittest.TestCase):
    def test_write_and_read_via_pyperclip(self):
        import computer
        mock_pyperclip = MagicMock()
        mock_pyperclip.paste.return_value = "Hallo"
        with patch.dict("sys.modules", {"pyperclip": mock_pyperclip}):
            # clipboard_write nutzt pyperclip direkt
            computer.clipboard_write("Hallo")
            # Für clipboard_read: pyperclip mocken, _pyautogui NICHT aufrufen
            result = computer.clipboard_read()
        mock_pyperclip.copy.assert_called_once_with("Hallo")
        self.assertEqual(result, "Hallo")


# ---------------------------------------------------------------------------
# Maus
# ---------------------------------------------------------------------------


class TestMouse(unittest.TestCase):
    def test_mouse_move(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            computer.mouse_move(100, 200)
        pag.moveTo.assert_called_once_with(100, 200, duration=0.25)

    def test_mouse_click_with_coords(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            computer.mouse_click(50, 75)
        pag.click.assert_called_once_with(50, 75, button="left", clicks=1)

    def test_mouse_click_without_coords(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            computer.mouse_click()
        pag.click.assert_called_once_with(button="left", clicks=1)

    def test_mouse_scroll(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            computer.mouse_scroll(3)
        pag.scroll.assert_called_once_with(3)

    def test_mouse_position(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            pos = computer.mouse_position()
        self.assertEqual(pos, (100, 200))


# ---------------------------------------------------------------------------
# Tastatur
# ---------------------------------------------------------------------------


class TestKeyboard(unittest.TestCase):
    def test_type_text(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            computer.keyboard_type("Hallo")
        pag.typewrite.assert_called_once_with("Hallo", interval=0.0)

    def test_type_empty_raises(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            with self.assertRaises(computer.ControlError):
                computer.keyboard_type("")

    def test_hotkey(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            computer.keyboard_hotkey("ctrl", "c")
        pag.hotkey.assert_called_once_with("ctrl", "c")

    def test_hotkey_empty_raises(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            with self.assertRaises(computer.ControlError):
                computer.keyboard_hotkey()

    def test_press_key(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            computer.keyboard_press("enter")
        pag.press.assert_called_once_with("enter")

    def test_press_empty_raises(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            with self.assertRaises(computer.ControlError):
                computer.keyboard_press("")


# ---------------------------------------------------------------------------
# Lautstärke / Stumm
# ---------------------------------------------------------------------------


class TestVolume(unittest.TestCase):
    def test_invalid_level_raises(self):
        import computer
        with self.assertRaises(computer.ControlError):
            computer.set_volume(101)
        with self.assertRaises(computer.ControlError):
            computer.set_volume(-1)

    def test_mute_presses_key(self):
        import computer
        pag = _make_pag()
        with patch("computer._pyautogui", return_value=pag):
            computer.mute_volume()
        pag.press.assert_called_once_with("volumemute")


# ---------------------------------------------------------------------------
# Helligkeit
# ---------------------------------------------------------------------------


class TestBrightness(unittest.TestCase):
    def test_invalid_level_raises(self):
        import computer
        with self.assertRaises(computer.ControlError):
            computer.set_brightness(101)

    @patch("computer.sys.platform", "linux")
    def test_non_windows_raises(self):
        import computer
        with self.assertRaises(computer.ControlError):
            computer.set_brightness(50)


# ---------------------------------------------------------------------------
# System-Befehle
# ---------------------------------------------------------------------------


class TestSystemCommands(unittest.TestCase):
    @patch("computer.sys.platform", "win32")
    @patch("computer.subprocess.run")
    def test_shutdown_windows(self, mock_run):
        import computer
        computer.shutdown(0)
        mock_run.assert_called_once_with(["shutdown", "/s", "/t", "0"], check=True)

    @patch("computer.sys.platform", "win32")
    @patch("computer.subprocess.run")
    def test_restart_windows(self, mock_run):
        import computer
        computer.restart(0)
        mock_run.assert_called_once_with(["shutdown", "/r", "/t", "0"], check=True)

    @patch("computer.sys.platform", "win32")
    @patch("computer.subprocess.Popen")
    def test_open_task_manager_windows(self, mock_popen):
        import computer
        computer.open_task_manager()
        mock_popen.assert_called_once_with(["taskmgr.exe"])

    @patch("computer.sys.platform", "darwin")
    @patch("computer.subprocess.Popen")
    def test_open_task_manager_mac(self, mock_popen):
        import computer
        computer.open_task_manager()
        mock_popen.assert_called_once_with(["open", "-a", "Activity Monitor"])


# ---------------------------------------------------------------------------
# list_processes
# ---------------------------------------------------------------------------


class TestListProcesses(unittest.TestCase):
    def test_returns_all_processes(self):
        import computer
        proc1 = MagicMock()
        proc1.info = {"pid": 1, "name": "python.exe", "status": "running"}
        proc2 = MagicMock()
        proc2.info = {"pid": 2, "name": "chrome.exe", "status": "running"}
        mock_psutil = MagicMock()
        mock_psutil.process_iter.return_value = [proc1, proc2]
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = Exception
        with patch("computer._psutil", return_value=mock_psutil):
            result = computer.list_processes()
        self.assertEqual(len(result), 2)

    def test_filters_by_name(self):
        import computer
        proc1 = MagicMock()
        proc1.info = {"pid": 1, "name": "python.exe", "status": "running"}
        proc2 = MagicMock()
        proc2.info = {"pid": 2, "name": "chrome.exe", "status": "running"}
        mock_psutil = MagicMock()
        mock_psutil.process_iter.return_value = [proc1, proc2]
        mock_psutil.NoSuchProcess = Exception
        mock_psutil.AccessDenied = Exception
        with patch("computer._psutil", return_value=mock_psutil):
            result = computer.list_processes("chrome")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "chrome.exe")


if __name__ == "__main__":
    unittest.main()
