
import numpy as np
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular
from manifpy import SE2
from manifpy import SE3
from manifpy import SE2Tangent
from manifpy import SE3Tangent


def Covariance(is_se3: bool = True):
    if is_se3:
        return np.zeros((SE3.DoF, SE3.DoF))
    else:
        return np.zeros((SE2.DoF, SE2.DoF))


def Jacobian(is_se3: bool = True):
    if is_se3:
        return np.zeros((SE3.DoF, SE3.DoF))
    else:
        return np.zeros((SE2.DoF, SE2.DoF))


def uniform_random(dim: int) -> np.ndarray:
    return np.random.uniform([-1.] * dim, [1.] * dim)


def quaternion_from_rot_mat(rot_mat: np.ndarray):
    w = np.sqrt(1 + np.trace(rot_mat)) / 2
    if w == 0.:
        return np.array([1., 0., 0., 0.])
    w4 = 4 * w
    x = (rot_mat[2, 1] - rot_mat[1, 2]) / w4
    y = (rot_mat[0, 2] - rot_mat[2, 0]) / w4
    z = (rot_mat[1, 0] - rot_mat[0, 1]) / w4
    return np.array([x, y, z, w])


def rot_mat_from_quaternion(q_vec: np.ndarray):
    # prepair
    q2 = q_vec * q_vec
    q_pair = []
    for i in range(4):
        q_pair.append(q_vec[i] * q_vec[(i + 1) % 4])  # (xy, yz, zw, wx)
    q_pair.append(q_vec[0] * q_vec[2])  # (xz)
    q_pair.append(q_vec[1] * q_vec[3])  # (yw)
    # rotation matrix
    rot = np.empty((3, 3))
    rot[0, 0] = q2[3] + q2[0] - q2[1] - q2[2]
    rot[1, 1] = q2[3] - q2[0] + q2[1] - q2[2]
    rot[2, 2] = q2[3] - q2[0] - q2[1] + q2[2]
    rot[0, 1] = 2. * (q_pair[0] - q_pair[2])
    rot[0, 2] = 2. * (q_pair[4] + q_pair[5])
    rot[1, 0] = 2. * (q_pair[0] + q_pair[2])
    rot[1, 2] = 2. * (q_pair[1] - q_pair[3])
    rot[2, 0] = 2. * (q_pair[4] - q_pair[5])
    rot[2, 1] = 2. * (q_pair[1] + q_pair[3])
    return rot


def rot_mat_from_angle(angle: float):
    cos_ = np.cos(angle)
    sin_ = np.sin(angle)
    rot = np.empty((2, 2))
    rot[0, 0] = rot[1, 1] = cos_
    rot[0, 1] = -sin_
    rot[1, 0] = sin_
    return rot


def se3_from_transformation(t_mat: np.ndarray):
    t_vec = t_mat[:3, 3]
    q = quaternion_from_rot_mat(t_mat[:3, :3])
    t_q = np.r_[t_vec, q]
    return SE3(t_q)


def se2_from_transformation(t_mat: np.ndarray):
    tmp = np.r_[t_mat[:2, 2], t_mat[:2, 0]]  # translation, cos, sin
    return SE2(tmp)


def transformation_from_se3(data: SE3):
    t_mat = np.eye(4)
    t_mat[:3, 3] = data.translation()
    t_mat[:3, :3] = rot_mat_from_quaternion(data.quat())
    return t_mat


def transformation_from_se2(data: SE2):
    t_mat = np.eye(3)
    t_mat[:2, 2] = data.translation()
    t_mat[:2, :2] = rot_mat_from_angle(data.angle())
    return t_mat


def solve_J(J_mat: np.ndarray, r_vec: np.ndarray, use_cholesky: bool = True):
    if use_cholesky:
        # solve dx = -(J.T J)^(-1) * J.T * r
        #       (J.T J) * dx = -J.T * r
        #   J.T * J => L * L.T (cholesky decomposition)
        #   L. (L.T dx) = b (= -J.T * r)
        #      L.T dx = c
        #          dx = Answer
        J_T = J_mat.T
        L = cholesky(J_T @ J_mat, lower=True)
        c = solve_triangular(L, -J_T @ r_vec, lower=True)
        dx = solve_triangular(L.T, c)
        return dx
    else:
        # solve dx = -(J.T J)^(-1) * J.T * r
        # J = QR
        # => R.J = -Q.T * r
        Q, R = np.linalg.qr(J_mat)
        b = -Q.T @ r_vec
        dx = solve_triangular(R, b)
        return dx


def measure_point(point: np.ndarray, t_mat: np.ndarray):
    # inv(t_mat) = [R.T, -R.T * t]
    dim = point.size
    rot = t_mat[:dim, :dim]
    translation = t_mat[:dim, dim]
    result = rot.T @ (point - translation)
    return result


def noised_transformation_matrix(t_mat: np.ndarray, sigmas: np.ndarray):
    is_se3 = True if t_mat.shape == (4, 4) else False
    size = SE3.DoF if is_se3 else SE2.DoF
    noise = sigmas * uniform_random(size)
    if is_se3:
        X = se3_from_transformation(t_mat)
        noised_X = X + SE3Tangent(noise)
        return transformation_from_se3(noised_X)
    else:
        X = se2_from_transformation(t_mat)
        noised_X = X + SE2Tangent(noise)
        return transformation_from_se2(noised_X)
