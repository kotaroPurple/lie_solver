
import numpy as np
from typing import Optional
from util.manif import Covariance
from util.manif import Jacobian
from util.manif import se2_from_transformation
from util.manif import se3_from_transformation
from util.manif import transformation_from_se2
from util.manif import transformation_from_se3

# manif
from manifpy import SE2
from manifpy import SE3
from manifpy import SE2Tangent
from manifpy import SE3Tangent


class EKF:
    """Extended Kalman Filter with Lie Algebra

    The class smoothes observations of (visual) odometry.

    Model: Xj = Xi + u (X: transformation matrix, u: movement to Xj from Xi)
    Measurement: Yk = Xk (X,Y: transformation matrix)

    How to use:
    1. prepare data
        transformation matrix (current pose and movement from last pose)
        solver = EKF(is_se3=True, with_movement=True)  # or is_se3 = False (SE2)
        # solver = EKF(is_se3=True, with_movement=False)  # without movement (Model: Xj = Xi)
    2. input data and get filtered data (return input data at first time)
        predicted_t = solver.predict(t_mat, movement)
        # predicted_t = solver.predict(t_mat)  # with_movement=False
    3. iterate 2
        while True:  # data comes
            t_mat, movement = your_good_function()
            # EKF
            predicted_t = solver.predict(t_mat, movement)
    """
    def __init__(self, is_se3: bool, with_movement: bool):
        self.is_se3 = is_se3
        self.with_movement = with_movement
        # redefinition of class, function
        if is_se3:
            self.SE = SE3
            self.SETangent = SE3Tangent
            self.Covariance = lambda: Covariance(is_se3=True)
            self.Jacobian = lambda: Jacobian(is_se3=True)
            self.se_from_transformation = se3_from_transformation
            self.transformation_from_se = transformation_from_se3
            self.dim = SE3.Dim
            self.dof = SE3.DoF
        else:
            self.SE = SE2
            self.SETangent = SE2Tangent
            self.Covariance = lambda: Covariance(is_se3=False)
            self.Jacobian = lambda: Jacobian(is_se3=False)
            self.se_from_transformation = se2_from_transformation
            self.transformation_from_se = transformation_from_se2
            self.dim = SE2.Dim
            self.dof = SE2.DoF
        # prepare
        self.clear()

    def set_sigmas(
            self, model_sigma: np.ndarray, measurement_sigma: np.ndarray) -> None:
        """make covariance matrix

        Args:
            measurement_sigma (np.ndarray): measurement sigma (6) or (3)
            model_sigma (Optional[np.ndarray]): model sigma (6) or (3)
                input None when without movement mode
        """
        if self.is_se3:
            assert (len(model_sigma) == 6)
            assert (len(measurement_sigma) == 6)
        else:
            assert (len(model_sigma) == 3)
            assert (len(measurement_sigma) == 3)
        self.U = np.diagflat(np.square(model_sigma))
        self.R = np.diagflat(np.square(measurement_sigma))

    def predict(self, t_mat: np.ndarray, movement: Optional[np.ndarray] = None) -> np.ndarray:
        """predict transformation matrix by EKF

        Args:
            t_mat (np.ndarray): current pose (transformation matrix)
            movement (Optional[np.ndarray]): movement between current and last pose

        Returns:
            np.ndarray: predicted current pose
        """
        # move to next
        if (self.with_movement is False) or (movement is None):
            u_estimation = self.SETangent.Zero()
        else:
            u_estimation = self.se_from_transformation(movement).log()

        if self.is_1st:
            self.X = self.se_from_transformation(t_mat)
            self.is_1st = False
            return t_mat.copy()

        #
        # estimation
        #

        # measurement
        measurement = self.se_from_transformation(t_mat).log().coeffs()

        # move
        self.X = self.X.plus(u_estimation, self.J_x, self.J_u)  # X * exp(u)
        self.P = self.J_x @ self.P @ self.J_x.transpose() + self.J_u @ self.U @ self.J_u.transpose()
        # expectation
        _H = np.eye(self.dof)
        _E = _H @ self.P @ _H.transpose()
        # innovation
        _z = (self.SETangent(measurement).exp() - self.X).coeffs()
        _Z = _E + self.R
        # Kalman gain
        _K = self.P @ _H.transpose() @ np.linalg.inv(_Z)
        # correction step
        dx = _K @ _z
        # update
        self.X = self.X + self.SETangent(dx)
        self.P = self.P - _K @ _Z @ _K.transpose()
        # output self.X as transformation matrix
        return self.transformation_from_se(self.X)

    def clear(self) -> None:
        """clear data

        You need to call set_sigmas() before calling predict().
        """
        self.is_1st = True
        self.P = self.Covariance()
        self.J_x = self.Jacobian()
        self.J_u = self.Jacobian()
