
import numpy as np
import matplotlib.pyplot as plt
from util.manif import uniform_random
from util.manif import se3_from_transformation
from util.manif import noised_transformation_matrix
from util.manif import transformation_from_se3
from solver.ekf import EKF

# manif
from manifpy import SE3
from manifpy import SE3Tangent

# test data
from test_data.three_d_data import y_axis_rotation_matrix
from test_data.three_d_data import circle_data


def main():
    # generate data
    radius = 1.0
    number = 120
    tilt_angle = np.deg2rad(30)
    _rot_mat = y_axis_rotation_matrix(tilt_angle)
    t_mat_sim = circle_data(radius, number, _rot_mat, 0., 2 * np.pi)
    t_mat_input = circle_data(radius, number, _rot_mat, 0., 2 * np.pi)

    # sigmas for covariance matrix
    u_sigma = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # model
    y_sigma = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # measurement

    # controls (movement between t_mat[i+1] and t_mat[i])
    controls = [np.eye(4)]
    for i in range(number - 1):
        Xi = se3_from_transformation(t_mat_sim[i])
        Xj = se3_from_transformation(t_mat_sim[i + 1])
        u = (Xj - Xi)  # tangent space
        u_noise = u_sigma * uniform_random(SE3.DoF)
        u_with_noise = (u.coeffs() + u_noise)
        diff_X = SE3Tangent(u_with_noise).exp()
        diff_T = transformation_from_se3(diff_X)
        controls.append(diff_T)  # transformation matrix
    controls = np.array(controls)

    # add noise to input
    tmp_t_mat = []
    for _t_mat in t_mat_input:
        tmp_t_mat.append(noised_transformation_matrix(_t_mat, 10. * u_sigma))
    t_mat_input = np.array(tmp_t_mat)

    # set initial pose : t_mat_input[0] = t_mat_sim[0]
    t_mat_input[0] = t_mat_sim[0].copy()

    # EKF
    solver = EKF(is_se3=True, with_movement=True)
    # solver = EKF(is_se3=True, with_movement=False)
    solver.set_sigmas(u_sigma, y_sigma)
    filtered_t = []

    for current_t, _control in zip(t_mat_input, controls):
        predcited_t = solver.predict(current_t, _control)
        # predcited_t = solver.predict(current_t)  # without movement
        filtered_t.append(predcited_t)

    # show result
    filtered_t = np.array(filtered_t)
    lim_value = radius + radius / 10
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # # trajectories
    ax.plot(
        filtered_t[:, 0, 3], filtered_t[:, 1, 3], filtered_t[:, 2, 3], label='EKF')
    ax.plot(t_mat_input[:, 0, 3], t_mat_input[:, 1, 3], t_mat_input[:, 2, 3], label='Input')
    ax.plot(
        t_mat_sim[:, 0, 3], t_mat_sim[:, 1, 3], t_mat_sim[:, 2, 3],
        c='black', alpha=0.5, label='GT')

    # # axis
    colors = ['black', 'red', 'blue']
    step = 2
    for i in range(0, number, step):
        rot_mat = filtered_t[i, None, :3, :3]
        starts = filtered_t[i:i+1, :3, 3]
        for c in range(3):
            goals = starts + 0.05 * rot_mat[0:1, :, c]
            s_g = np.r_[starts, goals]
            ax.plot(s_g[:, 0], s_g[:, 1], s_g[:, 2], c=colors[c])

    ax.set_xlim(-lim_value, lim_value)
    ax.set_ylim(-lim_value, lim_value)
    ax.set_zlim(-lim_value, lim_value)
    plt.title('EKF in SE(3)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
