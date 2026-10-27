# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task registrations for the project."""

from isaaclab_tasks.utils import import_packages

# 하위 폴더 내 task 탐색
import_packages(__name__, ["utils", ".mdp"])
