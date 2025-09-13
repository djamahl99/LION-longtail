import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

def wrap_angle(yaw):
    return np.arctan2(np.sin(yaw), np.cos(yaw))

class PoseKalmanFilter(object):
    """
    A Kalman filter for tracking 3D object pose (position and rotation).

    The 8-dimensional state space

        x, y, z, rz, vx, vy, vz, vrz
    """

    def __init__(self, dt=0.1):
        """
        Initialize the Kalman filter.
        
        Parameters
        ----------
        dt : float
            Time step between measurements (default: 0.1)
        """
        self.ndim = 4  # [x, y, z, rz]
        self.dt = dt

        # Motion model: position + heading with velocities
        self._motion_mat = np.eye(8, 8)
        # Position integration: pos(t+1) = pos(t) + vel(t)*dt
        self._motion_mat[0, 4] = dt  # x += vx*dt
        self._motion_mat[1, 5] = dt  # y += vy*dt  
        self._motion_mat[2, 6] = dt  # z += vz*dt
        self._motion_mat[3, 7] = dt  # yaw += vyaw*dt
        
        # Observation matrix: we observe [x, y, z, yaw]
        self._update_mat = np.eye(4, 8)

        # Standard deviations for different parameter types
        self.std_pos = 1.0        # meters for x, y, z
        self.std_rot = 0.1       # radians for rx, ry, rz (~2.9 degrees)
        
        self.std_vel_pos = 4.0    # m/s for vx, vy, vz
        self.std_vel_rot = 0.1   # rad/s for vrx, vry, vrz

    def box_initiate(self, box: np.ndarray):
        measurement = PoseKalmanFilter.box_to_pose_vector(box)

        return self.initiate(measurement)

    def initiate(self, measurement):
        # measurement is [x, y, z, yaw]
        mean_pos = measurement  # [x, y, z, yaw]
        mean_vel = np.zeros(4)   # [vx, vy, vz, vyaw] 
        mean = np.concatenate([mean_pos, mean_vel])
        
        # Initial uncertainties
        std = [
            1.0,  # x position
            1.0,  # y position
            0.5,  # z position (more certain - terrain constrained)
            0.1,  # yaw 
            2.0,  # vx velocity
            2.0,  # vy velocity
            1.0,  # vz velocity (smaller - terrain constrained)
            0.2   # vyaw rate
        ]
        
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 12 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 12x12 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.

        """
        # Process noise standard deviations (typically smaller than initial uncertainty)
        std_pos = [self.std_pos * 0.3, self.std_pos * 0.3, self.std_pos * 0.3]  # x, y, z
        std_rot = [self.std_rot * 0.3]  # rz
        std_vel_pos = [self.std_vel_pos * 0.3, self.std_vel_pos * 0.3, self.std_vel_pos * 0.3]  # vx, vy, vz
        std_vel_rot = [self.std_vel_rot * 0.3]  # vrz
        
        motion_cov = np.diag(np.square(np.r_[std_pos, std_rot, std_vel_pos, std_vel_rot]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (12 dimensional array).
        covariance : ndarray
            The state's covariance matrix (12x12 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # Measurement noise standard deviations
        std = [
            self.std_pos, self.std_pos, self.std_pos,        # x, y, z
            self.std_rot,        # rz
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        assert mean.shape == (8,) and covariance.shape == (8,8)
        
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Handle yaw wraparound in innovation
        innovation = measurement - projected_mean
        # innovation[3] = wrap_angle(measurement[3] - projected_mean[3])
        
        # Cleaner Kalman gain computation
        kalman_gain = covariance @ self._update_mat.T @ np.linalg.inv(projected_cov)
        
        # Or for numerical stability, use your Cholesky approach more directly:
        # kalman_gain = scipy.linalg.cho_solve(
        #     (chol_factor, lower), 
        #     self._update_mat @ covariance
        # ).T
        
        new_mean = mean + kalman_gain @ innovation  # More standard notation
        new_mean[3] = wrap_angle(new_mean[3])
        
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        
        return new_mean, new_covariance

    @staticmethod  
    def box_to_pose_vector(box):
        return np.array([box[0], box[1], box[2], box[6]])  # [x, y, z, yaw]

    @staticmethod
    def pose_vector_to_transform(pose_vector):
        """Convert 6D pose vector to 4x4 transformation matrix.
        
        Parameters
        ----------
        pose_vector : ndarray
            4D vector [x, y, z, rz] where 0,0,rz is rotation vector
            
        Returns
        -------
        ndarray
            4x4 transformation matrix
        """
        assert len(pose_vector) == 4, f"pose_vector={np.round(pose_vector, 2)}"

        T = np.eye(4)
        T[:3, 3] = pose_vector[:3]  # translation
        
        # Convert rotation vector to rotation matrix
        rotation_vector = np.zeros((3,))
        rotation_vector[2] = pose_vector[3]
        if np.linalg.norm(rotation_vector) > 1e-8:
            T[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
            
        return T
    
    @staticmethod
    def transform_to_pose_vector(transform):
        """Convert 4x4 transformation matrix to 6D pose vector.
        
        Parameters
        ----------
        transform : ndarray
            4x4 transformation matrix
            
        Returns
        -------
        ndarray
            4D pose vector [x, y, z, rz]
        """
        assert transform.shape == (4,4), f"Got transform of shape {transform.shape}"
        pose_vector = np.zeros(4)
        pose_vector[:3] = transform[:3, 3]  # translation
        
        # Convert rotation matrix to rotation vector
        rotation_matrix = transform[:3, :3]
        pose_vector[3] = Rotation.from_matrix(rotation_matrix).as_rotvec()[2]
        
        return pose_vector