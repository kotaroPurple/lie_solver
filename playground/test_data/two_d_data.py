
import numpy as np


def circle_data(
        radius: float, number: int,
        start: float = 0., end: float = 2 * np.pi) -> np.ndarray:
    # circle in xy plane
    phi = np.arange(number) / (number - 1) * (end - start) + start
    xy = np.c_[radius * np.cos(phi), radius * np.sin(phi)]
    ori1 = np.c_[-np.sin(phi), np.cos(phi)]
    ori2 = np.c_[-np.cos(phi), -np.sin(phi)]
    rot_mat = np.dstack((ori1, ori2))
    # transformation matrix
    t_mat = np.zeros((number, 3, 3))
    t_mat[:, :2, 2] = xy
    t_mat[:, :2, :2] = rot_mat
    t_mat[:, 2, 2] = 1.
    return t_mat


def rectangular_data(good_data: bool) -> np.ndarray:
    # straight
    length1 = 1.5
    length2 = 1.0
    n1 = 60
    n2 = 40
    delta_theta = 0. if good_data else np.deg2rad(-5)
    # turn
    angle = np.pi / 2
    ratio = 1.0 if good_data else 0.95
    n_turn = 20
    # start
    start = np.eye(3)  # origin
    # straight 1
    move = _go_straight(start, length1, n1, delta_theta)
    # turn 1
    tmp = _turn(move[-1], angle, n_turn, ratio)
    move = np.concatenate((move, tmp), axis=0)
    # straight 2
    tmp = _go_straight(move[-1], length2, n2, delta_theta)
    move = np.concatenate((move, tmp), axis=0)
    # turn 2
    tmp = _turn(move[-1], angle, n_turn, ratio)
    move = np.concatenate((move, tmp), axis=0)
    # straight 3
    tmp = _go_straight(move[-1], length1, n1, delta_theta)
    move = np.concatenate((move, tmp), axis=0)
    # turn 3
    tmp = _turn(move[-1], angle, n_turn, ratio)
    move = np.concatenate((move, tmp), axis=0)
    # straight 3 (home)
    tmp = _go_straight(move[-1], length2, n2, delta_theta)
    move = np.concatenate((move, tmp), axis=0)
    return move


def _go_straight(start: np.ndarray, length: float, number: int, delta_theta: float) -> np.ndarray:
    _cos = np.cos(delta_theta)
    _sin = np.sin(delta_theta)
    straight = np.arange(number) / (number - 1) * length
    x = straight * _cos
    y = straight * _sin
    rot_mat = np.array([[_cos, -_sin], [_sin, _cos]])
    # transformation mtarix
    t_mat = np.zeros((number, 3, 3))
    t_mat[:, 0, 2] = x
    t_mat[:, 1, 2] = y
    t_mat[:, 2, 2] = 1.
    t_mat[:, :2, :2] = rot_mat
    # from start
    t_mat = np.zeros((number, 3, 3))
    for i in range(number):
        # current t
        current = np.eye(3)
        current[0, 2] = x[i]
        current[1, 2] = y[i]
        current[:2, :2] = rot_mat
        t_mat[i] = start @ current
    return t_mat


def _turn(start: np.ndarray, angle: float, number: int, ratio: float = 1.0) -> np.ndarray:
    thetas = np.arange(number) / (number - 1) * angle * ratio
    _cos = np.cos(thetas)
    _sin = np.sin(thetas)
    # from start
    t_mat = np.zeros((number, 3, 3))
    for i in range(number):
        # current t
        current = np.eye(3)
        current[:2, :2] = np.array([[_cos[i], -_sin[i]], [_sin[i], _cos[i]]])
        t_mat[i] = start @ current
    return t_mat


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    result = rectangular_data()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(result[:, 0, 2], result[:, 1, 2])

    # # axis
    colors = ['black', 'red']
    step = 5
    for i in range(0, len(result), step):
        rot_mat = result[i, None, :2, :2]
        starts = result[i:i+1, :2, 2]
        for c in range(2):
            goals = starts + 0.05 * rot_mat[0:1, :, c]
            s_g = np.r_[starts, goals]
            ax.plot(s_g[:, 0], s_g[:, 1], c=colors[c])

    plt.axis('equal')
    plt.show()
