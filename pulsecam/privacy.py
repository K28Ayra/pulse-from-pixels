"""Live mode writes nothing and sends nothing. This makes Python refuse both.

An audit hook (PEP 578) sees every open(), os.* file change and socket call made
from Python, and can't be removed once added. Code inside C++ libraries (OpenCV,
MediaPipe, onnxruntime) doesn't go through it, which is why tests/test_privacy.py
also checks that no new files show up.
"""

import os
import sys

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
FILE_CHANGES = {
    "os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.truncate", "os.symlink",
    "os.link", "os.chmod", "os.chown", "os.utime", "shutil.copyfile", "shutil.rmtree",
    "shutil.move", "shutil.make_archive",
}


def _refuse(event, args):
    if event == "open":
        mode = args[1]
        flags = args[2]
        if mode is not None and any(c in mode for c in "wax+"):
            raise PermissionError("live mode: no file writes (" + str(args[0]) + ")")
        if mode is None and flags & WRITE_FLAGS:
            raise PermissionError("live mode: no file writes (" + str(args[0]) + ")")
    elif event in FILE_CHANGES:
        raise PermissionError("live mode: no file changes (" + event + ")")
    elif event.startswith("socket.") or event.startswith("urllib.") or event == "http.client.connect":
        raise PermissionError("live mode: no network (" + event + ")")


def block_writes_and_network():
    """Can't be undone for the rest of the process."""
    sys.dont_write_bytecode = True
    sys.addaudithook(_refuse)
