"""PC-Steuerung für JARVIS (Phase 6).

Alle Funktionen sind plattformübergreifend implementiert (Windows / macOS / Linux).
Windows-spezifische Funktionen (pywin32, Helligkeit) werden nur ausgeführt, wenn
das entsprechende Modul verfügbar ist; auf anderen Plattformen wird ein
ControlError ausgelöst.

Abhängigkeiten:
  pyautogui  – Maus, Tastatur, Screenshot, Fenster
  psutil     – Prozesse, Lautstärke (indirekt)
  pywin32    – Fenster verschieben / Lautstärke / Helligkeit (Windows only)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Fehlerklasse
# ---------------------------------------------------------------------------


class ControlError(RuntimeError):
    """Wird bei Fehlern in der PC-Steuerung ausgelöst."""


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _pyautogui():
    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        return pyautogui
    except ImportError as exc:
        raise ControlError("pyautogui ist nicht installiert.") from exc


def _psutil():
    try:
        import psutil
        return psutil
    except ImportError as exc:
        raise ControlError("psutil ist nicht installiert.") from exc


def _win32():
    try:
        import win32gui
        import win32con
        import win32api
        return win32gui, win32con, win32api
    except ImportError as exc:
        raise ControlError(
            "pywin32 ist nicht installiert oder dieses System ist kein Windows."
        ) from exc


# ---------------------------------------------------------------------------
# Programme
# ---------------------------------------------------------------------------


def open_program(name_or_path: str) -> None:
    """Öffnet ein Programm anhand seines Namens oder Pfads.

    Auf Windows wird `start` genutzt, auf macOS `open`, auf Linux `xdg-open`
    oder ein direktes subprocess-Aufruf.
    """
    name_or_path = name_or_path.strip()
    if not name_or_path:
        raise ControlError("Kein Programmname angegeben.")

    try:
        if sys.platform == "win32":
            os.startfile(name_or_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", name_or_path])
        else:
            subprocess.Popen(["xdg-open", name_or_path])
    except (OSError, FileNotFoundError) as exc:
        # Fallback: versuche direkt als Kommando
        path = shutil.which(name_or_path)
        if path:
            subprocess.Popen([path])
        else:
            raise ControlError(
                f"Programm '{name_or_path}' konnte nicht geöffnet werden: {exc}"
            ) from exc


def close_program(name: str) -> int:
    """Beendet alle Prozesse, deren Name den Suchbegriff enthält.

    Gibt die Anzahl beendeter Prozesse zurück.
    """
    psutil = _psutil()
    name_lower = name.strip().lower()
    if not name_lower:
        raise ControlError("Kein Prozessname angegeben.")

    count = 0
    for proc in psutil.process_iter(["name", "pid"]):
        try:
            if name_lower in proc.info["name"].lower():
                proc.terminate()
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if count == 0:
        raise ControlError(f"Kein laufender Prozess mit Namen '{name}' gefunden.")
    return count


# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------


def open_browser(url: str = "https://www.google.com") -> None:
    """Öffnet eine URL im Standard-Browser."""
    url = url.strip()
    if not url:
        raise ControlError("Keine URL angegeben.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)


# ---------------------------------------------------------------------------
# Dateisystem
# ---------------------------------------------------------------------------


def find_files(
    name: str,
    search_dir: str | Path = Path.home(),
    max_results: int = 20,
) -> list[Path]:
    """Sucht Dateien nach Name (glob) und gibt eine Liste von Pfaden zurück."""
    name = name.strip()
    if not name:
        raise ControlError("Kein Suchbegriff angegeben.")

    search_path = Path(search_dir)
    if not search_path.exists():
        raise ControlError(f"Verzeichnis existiert nicht: {search_path}")

    results: list[Path] = []
    try:
        for match in search_path.rglob(f"*{name}*"):
            results.append(match)
            if len(results) >= max_results:
                break
    except PermissionError:
        pass  # Überspringe gesperrte Ordner

    return results


def open_folder(path: str | Path = Path.home()) -> None:
    """Öffnet einen Ordner im Dateimanager / Explorer."""
    folder = Path(path).expanduser().resolve()
    if not folder.exists():
        raise ControlError(f"Ordner existiert nicht: {folder}")

    try:
        if sys.platform == "win32":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except OSError as exc:
        raise ControlError(f"Ordner konnte nicht geöffnet werden: {exc}") from exc


def open_explorer(path: str | Path = Path.home()) -> None:
    """Alias für open_folder – öffnet den Explorer / Finder / Dateimanager."""
    open_folder(path)


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------


def take_screenshot(save_path: str | Path | None = None) -> Path:
    """Erstellt einen Screenshot und speichert ihn.

    Falls kein Pfad angegeben wird, wird der Screenshot im Home-Verzeichnis
    unter 'jarvis_screenshot_<timestamp>.png' gespeichert.
    Gibt den Pfad zur gespeicherten Datei zurück.
    """
    import datetime

    pag = _pyautogui()

    if save_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path.home() / f"jarvis_screenshot_{ts}.png"

    save_path = Path(save_path)
    try:
        img = pag.screenshot()
        img.save(str(save_path))
    except Exception as exc:
        raise ControlError(f"Screenshot fehlgeschlagen: {exc}") from exc

    return save_path


# ---------------------------------------------------------------------------
# Zwischenablage
# ---------------------------------------------------------------------------


def clipboard_read() -> str:
    """Liest den aktuellen Inhalt der Zwischenablage."""
    try:
        import pyperclip
        return pyperclip.paste()
    except ImportError:
        pass

    # Fallback über tkinter
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.destroy()
        return text
    except Exception as exc:
        raise ControlError(f"Zwischenablage konnte nicht gelesen werden: {exc}") from exc


def clipboard_write(text: str) -> None:
    """Schreibt Text in die Zwischenablage."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return
    except ImportError:
        pass

    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
    except Exception as exc:
        raise ControlError(f"Zwischenablage konnte nicht beschrieben werden: {exc}") from exc


# ---------------------------------------------------------------------------
# Maus
# ---------------------------------------------------------------------------


def mouse_move(x: int, y: int, duration: float = 0.25) -> None:
    """Bewegt den Mauszeiger zu den angegebenen Koordinaten."""
    pag = _pyautogui()
    try:
        pag.moveTo(x, y, duration=duration)
    except Exception as exc:
        raise ControlError(f"Mausbewegung fehlgeschlagen: {exc}") from exc


def mouse_click(
    x: int | None = None,
    y: int | None = None,
    button: str = "left",
    clicks: int = 1,
) -> None:
    """Klickt an der angegebenen Position (oder der aktuellen Position)."""
    pag = _pyautogui()
    try:
        if x is not None and y is not None:
            pag.click(x, y, button=button, clicks=clicks)
        else:
            pag.click(button=button, clicks=clicks)
    except Exception as exc:
        raise ControlError(f"Mausklick fehlgeschlagen: {exc}") from exc


def mouse_scroll(amount: int, x: int | None = None, y: int | None = None) -> None:
    """Scrollt die Maus. Positive Werte = hoch, negative = runter."""
    pag = _pyautogui()
    try:
        if x is not None and y is not None:
            pag.scroll(amount, x=x, y=y)
        else:
            pag.scroll(amount)
    except Exception as exc:
        raise ControlError(f"Scrollen fehlgeschlagen: {exc}") from exc


def mouse_position() -> tuple[int, int]:
    """Gibt die aktuelle Mausposition zurück."""
    pag = _pyautogui()
    pos = pag.position()
    return (pos.x, pos.y)


# ---------------------------------------------------------------------------
# Tastatur
# ---------------------------------------------------------------------------


def keyboard_type(text: str, interval: float = 0.0) -> None:
    """Tippt den angegebenen Text."""
    pag = _pyautogui()
    if not text:
        raise ControlError("Kein Text angegeben.")
    try:
        pag.typewrite(text, interval=interval)
    except Exception as exc:
        raise ControlError(f"Texteingabe fehlgeschlagen: {exc}") from exc


def keyboard_hotkey(*keys: str) -> None:
    """Drückt eine Tastenkombination (z.B. 'ctrl', 'c')."""
    pag = _pyautogui()
    if not keys:
        raise ControlError("Keine Tasten angegeben.")
    try:
        pag.hotkey(*keys)
    except Exception as exc:
        raise ControlError(f"Tastenkombination fehlgeschlagen: {exc}") from exc


def keyboard_press(key: str) -> None:
    """Drückt eine einzelne Taste."""
    pag = _pyautogui()
    if not key:
        raise ControlError("Keine Taste angegeben.")
    try:
        pag.press(key)
    except Exception as exc:
        raise ControlError(f"Tastendruck fehlgeschlagen: {exc}") from exc


# ---------------------------------------------------------------------------
# Fenster (Windows only via pywin32)
# ---------------------------------------------------------------------------


def _get_window_handle(title_fragment: str):
    """Sucht ein Fenster anhand eines Titelbestandteils (Windows only)."""
    win32gui, win32con, win32api = _win32()

    found_hwnd = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title_fragment.lower() in title.lower():
                found_hwnd.append(hwnd)

    win32gui.EnumWindows(callback, None)

    if not found_hwnd:
        raise ControlError(
            f"Kein sichtbares Fenster mit Titel '{title_fragment}' gefunden."
        )
    return found_hwnd[0]


def move_window(title_fragment: str, x: int, y: int, width: int = 0, height: int = 0) -> None:
    """Verschiebt ein Fenster (Windows only).

    Falls width/height 0 sind, werden die aktuellen Maße beibehalten.
    """
    win32gui, win32con, win32api = _win32()
    hwnd = _get_window_handle(title_fragment)
    rect = win32gui.GetWindowRect(hwnd)
    cur_w = rect[2] - rect[0]
    cur_h = rect[3] - rect[1]
    w = width if width > 0 else cur_w
    h = height if height > 0 else cur_h
    try:
        win32gui.MoveWindow(hwnd, x, y, w, h, True)
    except Exception as exc:
        raise ControlError(f"Fenster verschieben fehlgeschlagen: {exc}") from exc


def list_windows() -> list[str]:
    """Gibt Titel aller sichtbaren Fenster zurück (Windows only)."""
    win32gui, _, _ = _win32()
    titles: list[str] = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if t:
                titles.append(t)

    win32gui.EnumWindows(callback, None)
    return titles


# ---------------------------------------------------------------------------
# Lautstärke (Windows only)
# ---------------------------------------------------------------------------


def set_volume(level: int) -> None:
    """Setzt die Systemlautstärke (0–100). Windows only."""
    if not 0 <= level <= 100:
        raise ControlError("Lautstärke muss zwischen 0 und 100 liegen.")

    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        scalar = level / 100.0
        volume.SetMasterVolumeLevelScalar(scalar, None)
        return
    except ImportError:
        pass

    # Fallback: Lautstärke über Tastenkombination (plattformübergreifend, ungenau)
    pag = _pyautogui()
    try:
        # Normalisierung: erst auf 0 setzen (40x leiser), dann auf Ziel
        for _ in range(40):
            pag.press("volumedown")
        steps = round(level / 2.5)
        for _ in range(steps):
            pag.press("volumeup")
    except Exception as exc:
        raise ControlError(f"Lautstärke konnte nicht gesetzt werden: {exc}") from exc


def mute_volume() -> None:
    """Schaltet den Ton stumm / hebt die Stummschaltung auf."""
    pag = _pyautogui()
    try:
        pag.press("volumemute")
    except Exception as exc:
        raise ControlError(f"Stummschalten fehlgeschlagen: {exc}") from exc


# ---------------------------------------------------------------------------
# Helligkeit (Windows only)
# ---------------------------------------------------------------------------


def set_brightness(level: int) -> None:
    """Setzt die Bildschirmhelligkeit (0–100). Windows only via WMI."""
    if not 0 <= level <= 100:
        raise ControlError("Helligkeit muss zwischen 0 und 100 liegen.")

    if sys.platform != "win32":
        raise ControlError("Helligkeitssteuerung ist nur auf Windows verfügbar.")

    try:
        import wmi
        c = wmi.WMI(namespace="wmi")
        methods = c.WmiMonitorBrightnessMethods()[0]
        methods.WmiSetBrightness(level, 0)
    except ImportError as exc:
        raise ControlError("wmi-Modul ist nicht installiert (pip install wmi).") from exc
    except Exception as exc:
        raise ControlError(f"Helligkeit konnte nicht gesetzt werden: {exc}") from exc


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


def shutdown(delay_seconds: int = 0) -> None:
    """Fährt den Computer herunter."""
    try:
        if sys.platform == "win32":
            subprocess.run(["shutdown", "/s", "/t", str(delay_seconds)], check=True)
        elif sys.platform == "darwin":
            subprocess.run(
                ["osascript", "-e", f'tell application "System Events" to shut down'],
                check=True,
            )
        else:
            subprocess.run(["shutdown", "-h", f"+{delay_seconds // 60}"], check=True)
    except subprocess.CalledProcessError as exc:
        raise ControlError(f"Herunterfahren fehlgeschlagen: {exc}") from exc


def restart(delay_seconds: int = 0) -> None:
    """Startet den Computer neu."""
    try:
        if sys.platform == "win32":
            subprocess.run(["shutdown", "/r", "/t", str(delay_seconds)], check=True)
        elif sys.platform == "darwin":
            subprocess.run(
                ["osascript", "-e", 'tell application "System Events" to restart'],
                check=True,
            )
        else:
            subprocess.run(["shutdown", "-r", f"+{delay_seconds // 60}"], check=True)
    except subprocess.CalledProcessError as exc:
        raise ControlError(f"Neustart fehlgeschlagen: {exc}") from exc


def open_task_manager() -> None:
    """Öffnet den Taskmanager / Systemmonitor."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["taskmgr.exe"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-a", "Activity Monitor"])
        else:
            for app in ("gnome-system-monitor", "xfce4-taskmanager", "htop", "top"):
                path = shutil.which(app)
                if path:
                    subprocess.Popen([path])
                    return
            raise ControlError("Kein Taskmanager gefunden.")
    except OSError as exc:
        raise ControlError(f"Taskmanager konnte nicht geöffnet werden: {exc}") from exc


# ---------------------------------------------------------------------------
# Prozess-Info
# ---------------------------------------------------------------------------


def list_processes(name_filter: str = "") -> list[dict]:
    """Gibt eine Liste laufender Prozesse zurück.

    Optional nach Namen filtern. Gibt Dicts mit 'pid', 'name', 'status' zurück.
    """
    psutil = _psutil()
    result = []
    name_lower = name_filter.strip().lower()
    for proc in psutil.process_iter(["pid", "name", "status"]):
        try:
            info = proc.info
            if name_lower and name_lower not in info["name"].lower():
                continue
            result.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result
