
import numpy as np
import matplotlib.pyplot as plt
from util.manif import uniform_random
from util.manif import measure_point
from util.manif import noised_transformation_matrix
from util.manif import se2_from_transformation
from util.manif import transformation_from_se2
from solver.sam import SAM

# manif
from manifpy import SE2
from manifpy import SE2Tangent

# test data
from test_data.two_d_data import circle_data


def main():
    # データ作成
    radius = 1.0
    number = 60
    t_mat_sim = circle_data(radius, number, 0., 2 * np.pi * (36/36))
    t_mat_input = circle_data(radius, number, 0., 2 * np.pi * (33/36))

    # sigmas for covariance matrix
    init_sigma = np.array([0.005, 0.005, 0.005])  # initial pose does not moved  # NOQA
    y_sigma = np.array([0.02, 0.02])  # measurement
    u_sigma = np.array([0.01, 0.01, 0.01])  # movement
    loop_sigma = np.array([0.005, 0.005])  # loop closure

    # Landmarks
    landmarks_sim = np.array([[0., 0.]])  # a landmark at origin

    # measurement
    _pairs = [[0] for _ in range(number)]  # landmark index
    measurements = {}
    with_measure_noise = True
    for i in range(len(_pairs)):
        if len(_pairs[i]) > 0:
            measurements[i] = {}
            for k in _pairs[i]:
                noise = np.zeros(SE2.Dim)
                if with_measure_noise:
                    noise += y_sigma * uniform_random(SE2.Dim)
                measurements[i][k] = measure_point(landmarks_sim[k], t_mat_sim[i]) + noise

    # controls (movement between t_mat[i+1] and t_mat[i])
    controls = []
    for i in range(number - 1):
        # Xi = se2_from_transformation(t_mat_sim[i])
        # Xj = se2_from_transformation(t_mat_sim[i + 1])
        Xi = se2_from_transformation(t_mat_input[i])  # noisy
        Xj = se2_from_transformation(t_mat_input[i + 1])
        u = (Xj - Xi)  # tangent space
        u_noise = u_sigma * uniform_random(SE2.DoF)
        u_with_noise = (u.coeffs() + u_noise)
        diff_X = SE2Tangent(u_with_noise).exp()
        diff_T = transformation_from_se2(diff_X)
        controls.append(diff_T)  # transformation matrix
    controls = np.array(controls)

    # Loops
    # index (0), index (number - 1)
    loops = {0: [number - 1]}

    # add noise to input
    tmp_t_mat = []
    for _t_mat in t_mat_input:
        tmp_t_mat.append(noised_transformation_matrix(_t_mat, 10. * u_sigma))
    t_mat_input = np.array(tmp_t_mat)

    # set initial pose : t_mat_input[0] = t_mat_sim[0]
    t_mat_input[0] = t_mat_sim[0].copy()

    # SOLVE
    iterations = 20
    norm_threshold = 1e-6

    solver = SAM(is_se3=False)
    solver.input_data(t_mat_input, landmarks_sim, measurements, controls, loops)
    solver.information_matrix_from_sigma(init_sigma, y_sigma, u_sigma, loop_sigma)
    solver.solve(iterations, norm_threshold, verbose=True)

    # show result
    predicted_t_mat = solver.predicted_poses()
    predicted_lmks = solver.predicted_landmakrs()

    # show result
    lim_value = radius + 2 * radius / 10
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)

    # # trajectories
    ax.plot(predicted_t_mat[:, 0, 2], predicted_t_mat[:, 1, 2], label='SAM')
    ax.plot(t_mat_input[:, 0, 2], t_mat_input[:, 1, 2], label='Input')
    ax.plot(t_mat_sim[:, 0, 2], t_mat_sim[:, 1, 2], c='black', alpha=0.5, label='GT')

    # # landmarks
    ax.scatter(predicted_lmks[:, 0], predicted_lmks[:, 1], c='black')

    # # axis
    colors = ['black', 'red']
    step = 2
    for i in range(0, number, step):
        rot_mat = predicted_t_mat[i, None, :2, :2]
        starts = predicted_t_mat[i:i+1, :2, 2]
        for c in range(2):
            goals = starts + 0.05 * rot_mat[0:1, :, c]
            s_g = np.r_[starts, goals]
            ax.plot(s_g[:, 0], s_g[:, 1], c=colors[c])

    ax.set_xlim(-lim_value, lim_value)
    ax.set_ylim(-lim_value, lim_value)
    plt.title('SAM in SE(2)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
