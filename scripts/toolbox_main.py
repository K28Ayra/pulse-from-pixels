"""Runs rPPG-Toolbox's main.py with fewer preprocessing workers.

    python toolbox_main.py <workers> --config_file <config>     (from inside tools/rPPG-Toolbox)

Each worker holds a whole video in memory. A 1024x1024 UBFC-Phys video is about 35 GB,
so the toolbox default of 8 workers ran out of memory, and the toolbox just left those
recordings out without an error.
"""

import runpy
import sys

sys.path.insert(0, ".")
from dataset.data_loader.BaseLoader import BaseLoader  # noqa: E402

BaseLoader.multi_process_manager.__defaults__ = (int(sys.argv.pop(1)),)
sys.argv[0] = "main.py"
runpy.run_path("main.py", run_name="__main__")
