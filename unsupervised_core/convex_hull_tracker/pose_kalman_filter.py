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


class PoseKalmanFilter(object):
    """
    A Kalman filter for tracking 3D object pose (position and rotation).

    The 12-dimensional state space

        x, y, z, rx, ry, rz, vx, vy, vz, vrx, vry, vrz

    contains the 3D position (x, y, z), rotation vector (rx, ry, rz) in axis-angle
    representation, and their respective velocities.

    Object motion follows a constant velocity model. The pose parameters
    (x, y, z, rx, ry, rz) are taken as direct observation of the state space.

    """

    def __init__(self, dt=1.0):
        """
        Initialize the Kalman filter.
        
        Parameters
        ----------
        dt : float
            Time step between measurements (default: 0.1)
        """
        ndim = 6  # [x, y, z, rx, ry, rz]
        self.dt = dt

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Standard deviations for different parameter types
        self.std_pos = 1.0        # meters for x, y, z
        self.std_rot = 0.1       # radians for rx, ry, rz (~2.9 degrees)
        
        self.std_vel_pos = 4.0    # m/s for vx, vy, vz
        self.std_vel_rot = 0.1   # rad/s for vrx, vry, vrz

    def box_initiate(self, box: np.ndarray):
        measurement = PoseKalmanFilter.box_to_pose_vector(box)

        return self.initiate(measurement)

    def initiate(self, measurement):
        """Create track from unassociated measurement (absolute pose).

        Parameters
        ----------
        measurement : ndarray or 4x4 matrix
            Either 6D pose vector (x, y, z, rx, ry, rz) where rx,ry,rz is rotation
            vector, or 4x4 transformation matrix

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (12 dimensional) and covariance matrix (12x12
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        if measurement.shape == (4, 4):
            measurement = PoseKalmanFilter.transform_to_pose_vector(measurement)
        
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Standard deviations for initial uncertainty
        std = [
            # Position uncertainties (x, y, z)
            self.std_pos, self.std_pos, self.std_pos,
            # Rotation uncertainties (rx, ry, rz)
            self.std_rot, self.std_rot, self.std_rot,
            # Velocity uncertainties
            self.std_vel_pos, self.std_vel_pos, self.std_vel_pos,  # vx, vy, vz
            self.std_vel_rot, self.std_vel_rot, self.std_vel_rot,  # vrx, vry, vrz
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
        std_rot = [self.std_rot * 0.3, self.std_rot * 0.3, self.std_rot * 0.3]  # rx, ry, rz
        std_vel_pos = [self.std_vel_pos * 0.3, self.std_vel_pos * 0.3, self.std_vel_pos * 0.3]  # vx, vy, vz
        std_vel_rot = [self.std_vel_rot * 0.3, self.std_vel_rot * 0.3, self.std_vel_rot * 0.3]  # vrx, vry, vrz
        
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
            self.std_rot, self.std_rot, self.std_rot,        # rx, ry, rz
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, is_relative=False):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (12 dimensional).
        covariance : ndarray
            The state's covariance matrix (12x12 dimensional).
        measurement : ndarray or 4x4 matrix
            Either 6D pose vector (x, y, z, rx, ry, rz) or 4x4 transformation matrix.
            If is_relative=True, this represents the relative transformation since
            the last update. If False, it's an absolute pose measurement.
        is_relative : bool
            If True, measurement is a relative transformation. If False, it's absolute.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        if measurement.shape == (4, 4):
            measurement = PoseKalmanFilter.transform_to_pose_vector(measurement)
            
        if is_relative:
            # Convert relative measurement to absolute by applying it to current state
            current_pose = mean[:6]
            current_transform = PoseKalmanFilter.pose_vector_to_transform(current_pose)
            relative_transform = PoseKalmanFilter.pose_vector_to_transform(measurement)
            new_transform = current_transform @ relative_transform
            measurement = PoseKalmanFilter.transform_to_pose_vector(new_transform)
        
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, is_relative=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 6 degrees of
        freedom, otherwise 3.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (12 dimensional).
        covariance : ndarray
            Covariance of the state distribution (12x12 dimensional).
        measurements : ndarray
            An Nx6 dimensional matrix of N measurements, each in
            format (x, y, z, rx, ry, rz).
        only_position : Optional[bool]
            If True, distance computation is done with respect to the 3D
            position (x, y, z) only.
        is_relative : bool
            If True, measurements are relative transformations.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        assert ((measurements.shape[-1] == 6) and not only_position) or ((measurements.shape[-1] == 3) and only_position)
        if is_relative:
            # Convert relative measurements to absolute
            current_pose = mean[:6]
            current_transform = PoseKalmanFilter.pose_vector_to_transform(current_pose)
            absolute_measurements = []
            for measurement in measurements:
                relative_transform = PoseKalmanFilter.pose_vector_to_transform(measurement)
                new_transform = current_transform @ relative_transform
                absolute_measurements.append(PoseKalmanFilter.transform_to_pose_vector(new_transform))
            measurements = np.array(absolute_measurements)
            
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:3], covariance[:3, :3]
            measurements = measurements[:, :3]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

    @staticmethod
    def box_to_pose_vector(box: np.ndarray):
        return np.array([box[0], box[1], box[2], 0.0, box[-1], 0.0], dtype=np.float32)

    @staticmethod
    def pose_vector_to_transform(pose_vector):
        """Convert 6D pose vector to 4x4 transformation matrix.
        
        Parameters
        ----------
        pose_vector : ndarray
            6D vector [x, y, z, rx, ry, rz] where rx,ry,rz is rotation vector
            
        Returns
        -------
        ndarray
            4x4 transformation matrix
        """
        assert len(pose_vector) == 6, f"pose_vector={np.round(pose_vector, 2)}"

        T = np.eye(4)
        T[:3, 3] = pose_vector[:3]  # translation
        
        # Convert rotation vector to rotation matrix
        rotation_vector = pose_vector[3:6]
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
            6D pose vector [x, y, z, rx, ry, rz]
        """
        pose_vector = np.zeros(6)
        pose_vector[:3] = transform[:3, 3]  # translation
        
        # Convert rotation matrix to rotation vector
        rotation_matrix = transform[:3, :3]
        pose_vector[3:6] = Rotation.from_matrix(rotation_matrix).as_rotvec()
        
        return pose_vector

    def get_transform_matrix(self):
        """Get 4x4 transformation matrix from current state mean.
                    
        Returns
        -------
        ndarray
            4x4 transformation matrix representing current pose
        """
        pose_vector = self.mean[:6]
        return PoseKalmanFilter.pose_vector_to_transform(pose_vector)

    def set_noise_params(self, std_pos=None, std_rot=None,
                        std_vel_pos=None, std_vel_rot=None):
        """Update noise parameters.
        
        Parameters
        ----------
        std_pos : float, optional
            Standard deviation for position (x, y, z) in meters
        std_rot : float, optional  
            Standard deviation for rotation (rx, ry, rz) in radians
        std_vel_pos : float, optional
            Standard deviation for position velocities in m/s
        std_vel_rot : float, optional
            Standard deviation for rotation velocities in rad/s
        """
        if std_pos is not None:
            self.std_pos = std_pos
        if std_rot is not None:
            self.std_rot = std_rot
        if std_vel_pos is not None:
            self.std_vel_pos = std_vel_pos
        if std_vel_rot is not None:
            self.std_vel_rot = std_vel_rot


# Example usage
if __name__ == "__main__":
    # Create Kalman filter
    kf = PoseKalmanFilter(dt=0.1)  # 10 Hz updates
    
    # Example initial pose: [x, y, z, rx, ry, rz]
    initial_pose = np.array([10.0, 5.0, 1.5, 0.1, 0.2, 0.3])
    
    # Initialize track with absolute pose
    mean, covariance = kf.initiate(initial_pose)
    print("Initial pose:", mean[:6])
    print("Initial 4x4 transform:\n", kf.get_transform_matrix())
    
    # Simulate a few prediction/update cycles with relative measurements
    for i in range(5):
        # Predict
        mean, covariance = kf.predict(mean, covariance)
        print(f"\nPredicted pose {i+1}:", mean[:6])
        
        # Simulate relative measurement (small movement)
        relative_pose = np.array([0.1, 0.02, 0.01, 0.01, 0.005, 0.02])
        
        # Update with relative measurement
        mean, covariance = kf.update(mean, covariance, relative_pose, is_relative=True)
        print(f"Updated pose {i+1}:", mean[:6])
        print(f"Velocity {i+1}:", mean[6:])
        
        # Show as transformation matrix
        T = kf.get_transform_matrix()
        print(f"Transform matrix {i+1}:")
        print(T)