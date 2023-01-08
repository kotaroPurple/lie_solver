
import numpy as np


def y_axis_rotation_matrix(theta: float) -> np.ndarray:
    # rotate
    rot_mat = np.eye(3)
    _cos = np.cos(theta)
    _sin = np.sin(theta)
    rot_mat[0, 0] = rot_mat[2, 2] = _cos
    rot_mat[2, 0] = - _sin
    rot_mat[0, 2] = _sin
    return rot_mat


def circle_data(
        radius: float, number: int, rot_mat: np.ndarray,
        start: float = 0., end: float = 2 * np.pi) -> np.ndarray:
    # circle in xy plane
    phi = np.arange(number) / (number - 1) * (end - start) + start
    xyz = np.c_[radius * np.cos(phi), radius * np.sin(phi), np.zeros(number)]
    ori1 = np.c_[-np.sin(phi), np.cos(phi), np.zeros(number)]
    ori2 = np.c_[-np.cos(phi), -np.sin(phi), np.zeros(number)]
    ori3 = np.c_[np.zeros(number), np.zeros(number), np.ones(number)]
    # rotation
    rot_xyz = xyz @ rot_mat.T
    rot_ori1 = ori1 @ rot_mat.T
    rot_ori2 = ori2 @ rot_mat.T
    rot_ori3 = ori3 @ rot_mat.T
    rot_mat = np.dstack((rot_ori1, rot_ori2, rot_ori3))
    # transformation matrix
    t_mat = np.zeros((number, 4, 4))
    t_mat[:, :3, 3] = rot_xyz
    t_mat[:, :3, :3] = rot_mat
    t_mat[:, 3, 3] = 1.
    return t_mat
