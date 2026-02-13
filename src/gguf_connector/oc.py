
import platform
import subprocess
import shutil

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
            "-ic",
            command
        ])
    else:
        subprocess.Popen(
            f'wsl -d Ubuntu -- bash -ic "{command}"',
            shell=True
        )

def macos_open_terminal():
    # AppleScript → open new interactive Terminal window
    cmd = 'openclaw tui'

    script = f'''
    tell application "Terminal"
        do script "{cmd}"
        activate
    end tell
    '''

    subprocess.Popen(["osascript", "-e", script])

def openclaw_tui():
    system = platform.system().lower()

    if system == "windows":
        windows_open_terminal()
    elif system == "darwin":
        macos_open_terminal()
    else:  # linux
        subprocess.Popen(["bash", "-ic", "openclaw tui"])

openclaw_tui()
