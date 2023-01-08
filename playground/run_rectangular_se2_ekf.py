
import numpy as np
import matplotlib.pyplot as plt
from util.manif import uniform_random
from util.manif import noised_transformation_matrix
from util.manif import se2_from_transformation
from util.manif import transformation_from_se2
from solver.ekf import EKF

# manif
from manifpy import SE2
from manifpy import SE2Tangent

# test data
from test_data.two_d_data import rectangular_data


def main():
    # generate data
    t_mat_sim = rectangular_data(good_data=False)
    t_mat_input = rectangular_data(good_data=False)
    number = len(t_mat_input)

    # sigmas for covariance matrix
    u_sigma = np.array([0.01, 0.01, 0.01])  # model
    y_sigma = np.array([0.1, 0.1, 0.1])  # measurement

    # controls (movement between t_mat[i+1] and t_mat[i])
    controls = [np.eye(3)]
    for i in range(number - 1):
        Xi = se2_from_transformation(t_mat_sim[i])
        Xj = se2_from_transformation(t_mat_sim[i + 1])
        u = (Xj - Xi)  # tangent space
        u_noise = u_sigma * uniform_random(SE2.DoF)
        u_with_noise = (u.coeffs() + u_noise)
        diff_X = SE2Tangent(u_with_noise).exp()
        diff_T = transformation_from_se2(diff_X)
        controls.append(diff_T)  # transformation matrix
    controls = np.array(controls)

    # add noise to input
    tmp_t_mat = []
    for _t_mat in t_mat_input:
        tmp_t_mat.append(noised_transformation_matrix(_t_mat, 5 * u_sigma))
    t_mat_input = np.array(tmp_t_mat)

    # set initial pose : t_mat_input[0] = t_mat_sim[0]
    t_mat_input[0] = t_mat_sim[0].copy()

    # EKF
    solver = EKF(is_se3=False, with_movement=True)
    # solver = EKF(is_se3=False, with_movement=False)
    solver.set_sigmas(u_sigma, y_sigma)
    filtered_t = []

    for current_t, _control in zip(t_mat_input, controls):
        predcited_t = solver.predict(current_t, _control)
        # predcited_t = solver.predict(current_t)  # without movement
        filtered_t.append(predcited_t)

    # show result
    filtered_t = np.array(filtered_t)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)

    # # trajectories
    ax.plot(filtered_t[:, 0, 2], filtered_t[:, 1, 2], label='EKF')
    ax.plot(t_mat_input[:, 0, 2], t_mat_input[:, 1, 2], label='Input')
    ax.plot(t_mat_sim[:, 0, 2], t_mat_sim[:, 1, 2], c='black', alpha=0.5, label='GT')

    # # axis
    colors = ['black', 'red']
    step = 4
    for i in range(0, number, step):
        rot_mat = filtered_t[i, None, :2, :2]
        starts = filtered_t[i:i+1, :2, 2]
        for c in range(2):
            goals = starts + 0.05 * rot_mat[0:1, :, c]
            s_g = np.r_[starts, goals]
            ax.plot(s_g[:, 0], s_g[:, 1], c=colors[c])

    plt.title('EKF in SE(2)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
