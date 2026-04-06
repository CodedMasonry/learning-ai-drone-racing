import math

import numpy as np


class Mapping:
    # [x, y, z]
    grid: dict[np.ndarray, bool]

    def is_colliding(self, pose3d: np.ndarray) -> bool:
        if pose3d in self.grid:
            return True
        return False


def dist(initial_pose3d: np.ndarray, final_pose3d: np.ndarray):
    i = initial_pose3d
    f = final_pose3d
    return math.dist(i, f)


def pathfind_to_pos(initial_pose3d: np.ndarray, final_pose3d: np.ndarray):
    pass
