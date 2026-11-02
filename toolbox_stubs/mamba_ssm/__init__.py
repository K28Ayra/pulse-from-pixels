"""Stand-in for mamba_ssm so rPPG-Toolbox's main.py can start.

The toolbox imports every trainer at startup, PhysMamba included, and the
real mamba_ssm needs a CUDA compiler to build. I don't run PhysMamba.
"""


class Mamba:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("mamba_ssm is a stub here, PhysMamba can't run")
