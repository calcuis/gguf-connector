
import platform
import subprocess
import shutil
import sys

def windows_open_terminal():
    command = "openclaw tui"

    # Prefer Windows Terminal (wt) if installed
    wt = shutil.which("wt")
    if wt:
        subprocess.Popen([
            "wt",
            "wsl",
            "-d", "Ubuntu",
            "--",
            "bash",
            "-ic", command
        ])
        return

    # Fallback: cmd start
    subprocess.Popen(
        f'start "" wsl -d Ubuntu -- bash -ic "{command}"',
        shell=True
    )

def openclaw_tui():
    system = platform.system().lower()

    if system == "windows":
        windows_open_terminal()

    elif system in ("linux", "darwin"):
        subprocess.Popen(["bash", "-ic", "openclaw tui"])

    else:
        print("Unsupported OS")
        sys.exit(1)

openclaw_tui()
