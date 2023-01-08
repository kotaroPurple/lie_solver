
import numpy as np
import matplotlib.pyplot as plt
from util.manif import uniform_random
from util.manif import se3_from_transformation
from util.manif import measure_point
from util.manif import noised_transformation_matrix
from util.manif import transformation_from_se3
from solver.sam import SAM

# manif
from manifpy import SE3
from manifpy import SE3Tangent

# test data
from test_data.three_d_data import y_axis_rotation_matrix
from test_data.three_d_data import circle_data


def main():
    # generate data
    radius = 1.0
    number = 60
    tilt_angle = np.deg2rad(30)
    _rot_mat = y_axis_rotation_matrix(tilt_angle)
    t_mat_sim = circle_data(radius, number, _rot_mat, 0., 2 * np.pi * (36/36))
    t_mat_input = circle_data(radius * 1., number, _rot_mat, 0., 2 * np.pi * (33/36))

    # sigmas for information matrix
    init_sigma = np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005])  # initial pose does not moved  # NOQA
    y_sigma = np.array([0.02, 0.02, 0.02])  # measurement
    u_sigma = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # movement
    loop_sigma = np.array([0.005, 0.005, 0.005])  # loop closure

    # Landmarks
    landmarks_sim = np.array([[0., 0., 0.]])  # a landmark at origin

    # measurement
    _pairs = [[0] for _ in range(number)]  # landmark index
    measurements = {}
    with_measure_noise = True
    for i in range(len(_pairs)):
        if len(_pairs[i]) > 0:
            measurements[i] = {}
            for k in _pairs[i]:
                noise = np.zeros(SE3.Dim)
                if with_measure_noise:
                    noise += y_sigma * uniform_random(SE3.Dim)
                measurements[i][k] = measure_point(landmarks_sim[k], t_mat_sim[i]) + noise

    # controls (movement between t_mat[i+1] and t_mat[i])
    controls = []
    for i in range(number - 1):
        # Xi = se3_from_transformation(t_mat_sim[i])
        # Xj = se3_from_transformation(t_mat_sim[i + 1])
        Xi = se3_from_transformation(t_mat_input[i])  # noisy
        Xj = se3_from_transformation(t_mat_input[i + 1])
        u = (Xj - Xi)  # tangent space
        u_noise = u_sigma * uniform_random(SE3.DoF)
        u_with_noise = (u.coeffs() + u_noise)
        diff_X = SE3Tangent(u_with_noise).exp()
        diff_T = transformation_from_se3(diff_X)
        controls.append(diff_T)  # transformation matrix
    controls = np.array(controls)

    # Loops
    # index (0), index (number - 1)
    loops = {0: [number - 1]}

    # add noise to input
    tmp_t_mat = []
    for _t_mat in t_mat_input:
        tmp_t_mat.append(noised_transformation_matrix(_t_mat, 15. * u_sigma))
    t_mat_input = np.array(tmp_t_mat)

    # set initial pose : t_mat_input[0] = t_mat_sim[0]
    t_mat_input[0] = t_mat_sim[0].copy()

    # SOLVE
    iterations = 20
    norm_threshold = 1e-6

    solver = SAM(is_se3=True)
    solver.input_data(t_mat_input, landmarks_sim, measurements, controls, loops)
    solver.information_matrix_from_sigma(init_sigma, y_sigma, u_sigma, loop_sigma)
    solver.solve(iterations, norm_threshold, verbose=True)

    # show result
    predicted_t_mat = solver.predicted_poses()
    predicted_lmks = solver.predicted_landmakrs()

    # show result
    lim_value = radius + radius / 10
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # # trajectories
    ax.plot(
        predicted_t_mat[:, 0, 3], predicted_t_mat[:, 1, 3], predicted_t_mat[:, 2, 3], label='SAM')
    ax.plot(t_mat_input[:, 0, 3], t_mat_input[:, 1, 3], t_mat_input[:, 2, 3], label='Input')
    ax.plot(
        t_mat_sim[:, 0, 3], t_mat_sim[:, 1, 3], t_mat_sim[:, 2, 3],
        c='black', alpha=0.5, label='GT')

    # # landmarks
    ax.scatter(predicted_lmks[:, 0], predicted_lmks[:, 1], predicted_lmks[:, 2], c='black')

    # # axis
    colors = ['black', 'red', 'blue']
    step = 2
    for i in range(0, number, step):
        rot_mat = predicted_t_mat[i, None, :3, :3]
        starts = predicted_t_mat[i:i+1, :3, 3]
        for c in range(3):
            goals = starts + 0.05 * rot_mat[0:1, :, c]
            s_g = np.r_[starts, goals]
            ax.plot(s_g[:, 0], s_g[:, 1], s_g[:, 2], c=colors[c])

    ax.set_xlim(-lim_value, lim_value)
    ax.set_ylim(-lim_value, lim_value)
    ax.set_zlim(-lim_value, lim_value)
    plt.title('SAM in SE(3)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
