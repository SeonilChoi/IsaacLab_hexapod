# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Mimic imitation env (Gym registration in hexapod.mimic.hexapod_imitate)
from . import mimic  # noqa: F401

# Register UI extensions.
from .ui_extension_example import *
