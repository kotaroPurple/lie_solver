
import copy
import numpy as np
from numpy.linalg import norm
from typing import Optional
from util.manif import Jacobian
from util.manif import se2_from_transformation
from util.manif import se3_from_transformation
from util.manif import solve_J
from util.manif import transformation_from_se2
from util.manif import transformation_from_se3

# manif
from manifpy import SE2
from manifpy import SE3
from manifpy import SE2Tangent
from manifpy import SE3Tangent


class SAM:
    """Smoothing and Mapping (SE2, SE3)

    How to use:
    1. instance class and set data
        solver = SAM(is_se3=True)  # or is_se3=False (SE2)
        solver.input_data(...)
        solver.information_matrix_from_sigma(...)
    2. solve by graph optimization
        solver.solve(iteration=20, norm_threshold=1e^-6, verbose=False)
    3. get result
        predicted_t_mat = solver.predicted_poses()
        predicted_landmarks = solver.predicted_landmakrs()
    """
    def __init__(self, is_se3: bool):
        self.is_se3 = is_se3
        # redefinition of class, function
        if is_se3:
            self.SE = SE3
            self.SETangent = SE3Tangent
            self.Jacobian = lambda: Jacobian(is_se3=True)
            self.se_from_transformation = se3_from_transformation
        else:
            self.SE = SE2
            self.SETangent = SE2Tangent
            self.Jacobian = lambda: Jacobian(is_se3=False)
            self.se_from_transformation = se2_from_transformation

    def input_data(
            self, t_mat: np.ndarray, landmarks: np.ndarray, measurements: dict,
            controls: Optional[np.ndarray], loops: Optional[dict]) -> None:
        """data setting

        Args:
            t_mat (np.ndarray): initial transformation matrices (N,4,4) or (N,3,3)
            landmarks (np.ndarray): initial landmark positions (M,3) or (M,2)
            measurements (dict): measured landmark positions at n (0~N-1)
                {n: {i: np.ndarray, j: np.ndarray, ...}, m: {...}}
            controls (Optional[np.ndarray]): relative pose between t_mat[i+1] and t_mat[i]
                if None, controls are calculated from input t_mat (you may have noisy result)
                (N-1,4,4) : each element at n has a transformation matrix
            loops (Optional[dict]): dict for loop closure {position index: list of position index}
                ({i; [j,...]}, j > i)
        """
        if self.is_se3:
            assert (t_mat.shape[1:] == (4, 4))
            assert (landmarks.shape[1] == 3)
        else:
            assert (t_mat.shape[1:] == (3, 3))
            assert (landmarks.shape[1] == 2)
        self.t_mat = t_mat
        self.init_landmarks = landmarks
        self.landmarks = copy.deepcopy(landmarks)
        self.measurements = measurements
        self.loops = loops if loops is not None else dict()
        self._prepare_pair_of_position_and_landmark()
        self._prepare_poses_and_controls(controls)

    def information_matrix_from_sigma(
            self, init_sigma: np.ndarray, measurement_sigma: np.ndarray,
            control_sigma: np.ndarray, loop_sigma: np.ndarray) -> None:
        """information matrix for solver

        Information Matrix is generated from numpy.diagflat(1. / sigmas)

        Args:
            init_sigma (np.ndarray): 1st position is anchored (6) or (3)
            measurement_sigma (np.ndarray): measurement sigma (3) or (2)
            control_sigma (np.ndarray): control sigma (6) or (3)
            loop_sigma (np.ndarray): loop closure sigma (3) or (2)
        """
        if self.is_se3:
            assert (len(init_sigma) == 6)
            assert (len(measurement_sigma) == 3)
            assert (len(control_sigma) == 6)
            assert (len(loop_sigma) == 3)
        else:
            assert (len(init_sigma) == 3)
            assert (len(measurement_sigma) == 2)
            assert (len(control_sigma) == 3)
            assert (len(loop_sigma) == 2)
        self.W_init = np.diagflat(1. / init_sigma)
        self.W_measure = np.diagflat(1. / measurement_sigma)
        self.W_move = np.diagflat(1. / control_sigma)
        self.W_loop = np.diagflat(1. / loop_sigma)

    def solve(
            self, iteration: int, norm_threshold: float = 1e-6,
            verbose: bool = False) -> np.ndarray:
        """Graph-Based Optimization

        Args:
            iteration (int): number of iteration
            norm_threshold (float, optional): threshold of norm residual. Defaults to 1e-6.
            verbose (bool, optional): if True, print logs. Defaults to False.

        Returns:
            np.ndarray: list including norms of residuals
        """
        # constants
        DoF = self.SE.DoF
        Dim = self.SE.Dim
        NUM_POSES = len(self.t_mat)
        NUM_LMKS = len(self.init_landmarks)
        NUM_FACTORS = sum([len(x) for x in self.measurements.values()])
        NUM_LOOPS = sum([len(x) for x in self.loops.values()])
        NUM_STATES = NUM_POSES * DoF + NUM_LMKS * Dim
        NUM_MEAS = NUM_POSES * DoF + NUM_FACTORS * Dim + NUM_LOOPS * Dim

        # Jacobian, residual
        J_d_xi = self.Jacobian()  # J^d_xi (d: diff)
        J_d_xj = self.Jacobian()  # J^d_xj
        J_ix_x = self.Jacobian()
        J_r_p0 = self.Jacobian()
        J_e_ix = np.zeros((Dim, DoF))
        J_e_b = np.zeros((Dim, Dim))
        J_e_dx = np.zeros((Dim, DoF))
        residual = np.zeros(NUM_MEAS)
        J = np.zeros((NUM_MEAS, NUM_STATES))

        # zero array
        _n = 3 if self.is_se3 else 2

        def _zero_array():
            return np.zeros(_n)

        # solve
        scores = []

        for it in range(iteration):
            # Clear residual vector and Jacobian matrix
            residual.fill(0)
            J.fill(0)
            row = 0
            col = 0

            # r[row:row+DoF] = W_init @ poses[0].lminus(X_init, J_r_p0).coeffs()
            residual[row:row+DoF] = self.W_init @ self.poses[0].rminus(self.X_init, J_r_p0).coeffs()
            J[row:row+DoF, col:col+DoF] = self.W_init @ J_r_p0
            row += DoF

            # loop poses
            for i in range(NUM_POSES):
                # move
                if i < NUM_POSES - 1:
                    # move from Xi to Xj
                    j = i + 1
                    Xi = self.poses[i]
                    Xj = self.poses[j]
                    u = self.SETangent(self.controls[i])

                    # expectation
                    d = Xj.rminus(Xi, J_d_xj, J_d_xi)
                    # residual
                    residual[row:row+DoF] = self.W_move @ (d - u).coeffs()
                    # Jacobian of residual
                    col = i * DoF
                    J[row:row+DoF, col:col+DoF] = self.W_move @ J_d_xi
                    col = j * DoF
                    J[row:row+DoF, col:col+DoF] = self.W_move @ J_d_xj
                    # next
                    row += DoF

                # measuremnt
                for k in self.pairs[i]:
                    # recover related states and data
                    Xi = self.poses[i]
                    b = self.landmarks[k]
                    y = self.measurements[i][k]
                    # expectation
                    e = Xi.inverse(J_ix_x).act(b, J_e_ix, J_e_b)  # inv(X).b
                    J_e_x = J_e_ix @ J_ix_x  # chain rule
                    # residual
                    residual[row:row+Dim] = self.W_measure @ (e - y)
                    # Jacobian of residual
                    col = i * DoF
                    J[row:row+Dim, col:col+DoF] = self.W_measure @ J_e_x
                    col = NUM_POSES * DoF + k * Dim
                    J[row:row+Dim, col:col+Dim] = self.W_measure @ J_e_b
                    # next
                    row += Dim

                # loop closure
                if i in self.loops:
                    for k in self.loops[i]:
                        Xi = self.poses[i]
                        Xk = self.poses[k]
                        # expectation
                        e = Xi.between(Xk, J_d_xi, J_d_xj).act(_zero_array(), J_e_dx)
                        # residual
                        residual[row:row+Dim] = self.W_loop @ e  # (e - zero)
                        # Jacobian
                        col = i * DoF
                        J[row:row+Dim, col:col+DoF] = self.W_loop @ J_e_dx @ J_d_xi
                        col = k * DoF
                        J[row:row+Dim, col:col+DoF] = self.W_loop @ J_e_dx @ J_d_xj
                        # next
                        row += Dim

            # sove
            dX = solve_J(J, residual, use_cholesky=True)  # with cholesky or QR decomposition

            # update all poses
            for i in range(NUM_POSES):
                # we go very verbose here
                row = i * DoF
                size = DoF
                dx = dX[row:row+size]
                self.poses[i] = self.poses[i] + self.SETangent(dx)

            # update all landmarks
            for k in range(NUM_LMKS):
                # we go very verbose here
                row = NUM_POSES * DoF + k * Dim
                size = Dim
                db = dX[row:row+size]
                self.landmarks[k] = self.landmarks[k] + db

            # redisual info
            norm_of_res = norm(residual)
            norm_of_dx = norm(dX)
            scores.append(norm_of_res)

            # verbose
            if verbose:
                print(f'{it}: residual norm: {norm_of_res}, step norm: {norm_of_dx}')

            # conditional exit
            if norm_of_dx < norm_threshold:
                if verbose:
                    print(f'exit: norm of dx < {norm_threshold}')
                break
        # end of loop
        else:
            if verbose:
                print('the end of iterarion: norm(dx) is larger than threshold')

        # output score (loss)
        return np.array(scores)

    def predicted_poses(self) -> np.ndarray:
        """get predicted poses as transformatioin matrix

        Returns:
            np.ndarray: (N,3,3) or (N,2,2) (N is the length of input data)
        """
        get_transformation = transformation_from_se3 if self.is_se3 else transformation_from_se2
        t_mat = []
        for X in self.poses:
            t_mat.append(get_transformation(X))
        return np.array(t_mat)

    def predicted_landmakrs(self) -> np.ndarray:
        """get predicted landmark positions

        Returns:
            np.ndarray: (M,3) or (M,2) (M is the number of landmarks)
        """
        return np.array(self.landmarks)

    def _prepare_pair_of_position_and_landmark(self):
        """pairs (list): pair of position and landmark (size: N).
            every element has a list ([i,j,...])
        """
        self.pairs = []
        for i in range(len(self.t_mat)):
            landmark_at_i = []
            if i in self.measurements:
                for k in self.measurements[i].keys():
                    landmark_at_i.append(k)
            self.pairs.append(landmark_at_i)

    def _prepare_poses_and_controls(self, controls: np.ndarray) -> None:
        """poses at X, controls (movement between Xj and Xi)
        """
        self.poses = []
        self.controls = []
        calc_control = True if controls is None else False
        # 1st
        self.X_init = self.se_from_transformation(self.t_mat[0])
        self.poses.append(self.se_from_transformation(self.t_mat[0]))
        # pose at j, movement between i and j
        for i in range(len(self.t_mat) - 1):
            Xi = self.se_from_transformation(self.t_mat[i])
            Xj = self.se_from_transformation(self.t_mat[i+1])
            self.poses.append(Xj)
            if calc_control:
                self.controls.append((Xj - Xi).coeffs())
            else:
                diff_X = self.se_from_transformation(controls[i])
                self.controls.append(diff_X.log().coeffs())  # tangent space
