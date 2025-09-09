
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sklearn.metrics import r2_score

from lion.unsupervised_core.box_utils import compute_ppscore, icp, icp_open3d_robust
from lion.unsupervised_core.convex_hull_tracker.convex_hull_object import (
    ConvexHullObject,
)
from lion.unsupervised_core.convex_hull_tracker.convex_hull_utils import (
    relative_object_pose,
    relative_object_rotation,
    rigid_icp,
    voxel_sampling_fast,
)
from lion.unsupervised_core.convex_hull_tracker.pose_kalman_filter import (
    PoseKalmanFilter,
)
from lion.unsupervised_core.outline_utils import points_rigid_transform
from lion.unsupervised_core.trajectory_optimizer import (
    optimize_with_gtsam_timed,
    optimize_with_gtsam_timed_positions,
)


class ConvexHullTrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class ConvexHullTrack:
    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature, convex_hull: ConvexHullObject):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        assert len(mean) == 12

        self.state = ConvexHullTrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self.history = [convex_hull]
        self.timestamps = [convex_hull.timestamp]
        self.last_points = convex_hull.original_points
        self.last_mesh = convex_hull.mesh
        self.last_avg_velocity = None
        self.last_yaw = None
        self.last_used_heading = False
        self.last_iou = 0.0

        # linear box prediction
        self.c = convex_hull.box.copy()
        self.A = np.zeros((7,), float)


        self.last_predict_timestamp = self.timestamps[-1]

        # initialise: todo -> update
        self.object_points = convex_hull.object_points

        # hmmm could change -> but we are only tracking poses with the kalman filter
        self.lwh = convex_hull.box[3:6]
        self.yaw = convex_hull.box[6]
        self.yaw_method = "init"

        self.positions = []
        self.positions.append(mean[:3])

        # final after global optimization
        self.optimized_poses = None
        self.optimized_boxes = None
        self.prev_pose = np.eye(4)

        self.source = convex_hull.source

        
        self.stagger_step = 2
        self.max_stagger_gap = 5
        self.icp_max_iterations = 5
        self.heading_speed_thresh_ms = 3.6 # m/s

        self._n_init = n_init
        self._max_age = max_age

        self._constraint_cache = {}
        self._cached_initial_poses = []
        self._last_processed_frame = 0

    def extrapolate_box(self, timestamps):
        init_timestamp = min(self.timestamps)*1e-9

        A = self.A
        c = self.c

        new_timestamps = np.array([timestamp*1e-9 - init_timestamp for timestamp in timestamps], float)
        new_prediction = A[np.newaxis, :] * new_timestamps[:, np.newaxis] + c[np.newaxis, :]

        return new_prediction.reshape(-1, 7)

    def to_box(self):
        A = self.A
        c = self.c

        timestamp = self.last_predict_timestamp
        init_timestamp = min(self.timestamps)*1e-9

        new_timestamps = np.array([timestamp*1e-9 - init_timestamp], float)
        new_prediction = A[np.newaxis, :] * new_timestamps[:, np.newaxis] + c[np.newaxis, :]

        return new_prediction.reshape(7)


    def extrapolate_kalman_box(self, timestamp: int):
        dt = (timestamp - self.timestamps[-1]) * 1e-9
        
        ndim = 6
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        new_mean = np.dot(self._motion_mat, self.mean.copy())

        centre = new_mean[:3]
        yaw = new_mean[5]
        x, y, z = centre
        l, w, h = self.lwh

        box = np.array([x, y, z, l, w, h, yaw], dtype=np.float32)

        return box

    # def to_box(self):
    #     assert len(self.mean) == 12

    #     centre = self.mean[:3]
    #     yaw = self.mean[5]
    #     # yaw = self.yaw

    #     # if abs(self.mean[5] - self.yaw) > 0.4:
    #     #     print(f"{self.yaw=:.3f} {self.mean[5]=:.3} {self.hits=} {self.age=} {self.yaw_method=}") 

    #     x, y, z = centre
    #     l, w, h = self.lwh

    #     # box = np.concatenate([centre.reshape(3), lwh.reshape(3), ry.reshape(1)], axis=0)
    #     box = np.array([x, y, z, l, w, h, yaw], dtype=np.float32)

    #     return box

    def to_points(self):
        transform = self.to_pose_matrix()

        return points_rigid_transform(self.object_points, transform)

    def to_shape_dict(self) -> Dict:
        points = self.to_points()
        centre = self.mean[:3]
        # probably could do more efficient...
        mesh = trimesh.convex.convex_hull(points)

        return {
            'original_points': points,
            'centroid_3d': centre,
            'mesh': mesh
        }

    def to_pose_matrix(self):
        return PoseKalmanFilter.pose_vector_to_transform(self.mean[:6])

    def predict(self, kf: PoseKalmanFilter, timestamp: int):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        # update previous pose
        self.prev_pose = self.to_pose_matrix()

        self.last_predict_timestamp = timestamp

        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

        self.positions.append(self.mean[:3])

    def compute_poses(self):
        timestamps = self.timestamps
        n_frames = len(timestamps)

        world_points_per_timestamp = []
        world_centers = []
        
        # Collect world points and centers
        for i, timestamp_ns in enumerate(timestamps):
            obj: ConvexHullObject = self.history[i]
            world_points = obj.original_points
            world_points_per_timestamp.append(world_points.copy())
            # world_points_per_timestamp.append(obj.mesh.vertices)
            center = obj.box[:3]
            world_centers.append(center)

        world_centers = np.array(world_centers)
        times_secs = np.array(timestamps, dtype=float) * 1e-9

        # heading -> 0, 0, 0 if not moving else we calculate the average.
        heading = np.zeros((2,))
        if n_frames > 1:
            velocities = []
            for i in range(1, min(n_frames, 5)):
                velocity = (world_centers[i] - world_centers[i-1]) / (times_secs[i] - times_secs[i-1])
                velocities.append(velocity)

            velocities = np.stack(velocities, axis=0)

            avg_velocity = np.mean(velocities, axis=0)
            avg_speed = np.linalg.norm(avg_velocity[:2])

            self.last_avg_velocity = avg_velocity
            
            if avg_speed > self.heading_speed_thresh_ms:
                heading = avg_velocity[:2]

        start_frame = max(1, self._last_processed_frame)
        
        if start_frame == 1:
            # First time or full recompute - compute everything
            first_object_pose = self._create_pose_from_object_points_and_heading(
                world_points_per_timestamp[0], heading, world_centers[0]
            )
            initial_poses = [first_object_pose]
            constraints = []
            cumulative_pose = first_object_pose.copy()
            self._cached_initial_poses = [first_object_pose]
            self._constraint_cache = {}
        else:
            # Incremental - reuse cached poses and constraints
            initial_poses = self._cached_initial_poses.copy()
            constraints = [self._constraint_cache[key] for key in sorted(self._constraint_cache.keys())]
            cumulative_pose = initial_poses[-1].copy()

        for i in range(start_frame, n_frames):
            constraint_key = (i-1, i)  # Key for frame pair

            undo_pose = np.linalg.inv(cumulative_pose)

            prev_points = points_rigid_transform(
                world_points_per_timestamp[i - 1], undo_pose
            )
            cur_points = points_rigid_transform(
                world_points_per_timestamp[i], undo_pose
            )

            R, t, A_inliers, B_inliers, icp_cost = relative_object_pose(prev_points, cur_points, max_iterations=self.icp_max_iterations)

            relative_pose = np.eye(4)
            relative_pose[:3, :3] = R
            relative_pose[:3, 3] = t
            confidence = len(A_inliers) / len(cur_points)

            cumulative_pose = cumulative_pose @ relative_pose
            initial_poses.append(cumulative_pose.copy())

            constraint = {
                'frame_i': i - 1,
                'frame_j': i, 
                'relative_pose': relative_pose,
                'confidence': confidence,
            }
            self._constraint_cache[constraint_key] = constraint
            constraints.append(constraint)

        # Cache the computed initial poses and update processed frame
        self._cached_initial_poses = initial_poses.copy()
        self._last_processed_frame = n_frames

        assert len(initial_poses) == len(
            timestamps
        ), "Must have one initial pose per timestamp"

        optimized_poses, marginals, quality = optimize_with_gtsam_timed(
            initial_poses, constraints, timestamps
        )

        # optimized_poses = initial_poses

        # optimized_poses, marginals, quality = optimize_with_gtsam_timed_positions(world_centers, constraints, timestamps)


        # update points
        # Now transform points to object-centric coordinates
        object_centric_points = []
        for convex_hull_obj, obj_pose in zip(self.history, optimized_poses):
            # Get world points
            world_points = convex_hull_obj.original_points

            # Transform to object-centric: multiply by inverse of object pose
            world_to_object = np.linalg.inv(obj_pose)
            object_points = points_rigid_transform(world_points, world_to_object)
            object_centric_points.append(object_points)

        self.object_points = np.concatenate(object_centric_points, axis=0)
        self.object_points = voxel_sampling_fast(self.object_points, 0.05, 0.05, 0.05)

        self.last_points = points_rigid_transform(self.object_points, optimized_poses[-1])
        self.last_mesh = trimesh.convex.convex_hull(self.last_points)
        self.optimized_poses = optimized_poses

        undo_last_pose = np.linalg.inv(optimized_poses[-1])
        object_mesh_vertices = points_rigid_transform(self.last_mesh.vertices.copy(), undo_last_pose)

        self.positions = [x[:3, 3] for x in optimized_poses]

        self.optimized_boxes = self._compute_oriented_boxes(timestamps, self.optimized_poses, object_mesh_vertices)

        init_timestamp = min(self.timestamps)*1e-9
        measurements = np.stack([convex_hull_obj.box for convex_hull_obj in self.history], axis=0)

        times_offsets = times_secs - init_timestamp


        n_samples, n_features = measurements.shape
        
        # Create design matrix [t, 1] for each timestamp
        X = np.column_stack([times_offsets, np.ones(n_samples)])
        
        # Solve least squares: (X^T X)^-1 X^T y for each feature
        # This gives us [A_j, c_j] for each feature j
        coefficients = np.linalg.lstsq(X, measurements, rcond=None)[0]
        
        A = coefficients[0, :]  # Slopes (7,)
        c = coefficients[1, :]  # Intercepts (7,)

        self.A, self.c = A, c

        # y = A*t + c (broadcasting handles multiple timestamps)
        predictions = A[np.newaxis, :] * times_offsets[:, np.newaxis] + c[np.newaxis, :]

        eval_r2 = r2_score(measurements, predictions, multioutput='uniform_average')
        print(f"Box R2: {eval_r2:.2f}")

        dims_mins = self.object_points.min(axis=0)
        dims_maxes = self.object_points.max(axis=0)

        self.lwh = dims_maxes - dims_mins

        return optimized_poses


    def _compute_oriented_boxes(self, timestamps: List[int], optimized_poses: List[np.ndarray], merged_object_points: np.ndarray):
        """Compute oriented bounding boxes based on object motion direction"""
        boxes = []

        optimized_positions = [x[:3, 3] for x in optimized_poses]
        vectors = []
        for i in range(1, len(optimized_positions)):
            vectors.append(optimized_positions[i] - optimized_positions[i-1])

        # same vector for last
        vectors.append(vectors[-1])


        for i, cur_pose in enumerate(optimized_poses):
            # rotvec = Rotation.from_matrix(cur_pose[:3, :3]).as_rotvec()
            # yaw = rotvec[2]

            vector = vectors[i]
            yaw = np.arctan2(vector[1], vector[0])

            cur_points = points_rigid_transform(merged_object_points, cur_pose)

            centre = cur_pose[:3, 3]

            # Create oriented bounding box from 3D vertices
            box = self._points_to_oriented_box(cur_points, centre, yaw)
            boxes.append(box)

        return boxes

    def _compute_yaw(self, vertices_3d: np.ndarray, centre: np.ndarray, prev_centre: np.ndarray, dt: float):
        
        velocity = (centre - prev_centre) / dt
        speed = np.linalg.norm(velocity[:2])

        if speed > self.heading_speed_thresh_ms and dt >= 0.1:
            return np.arctan2(velocity[1], velocity[0])

        points_2d = vertices_3d[:, :2]
        centered_vertices = points_2d - centre[:2]
        cov_matrix = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        primary_direction = eigenvectors[:, 0]  # Length direction

        return np.arctan2(primary_direction[1], primary_direction[0])

    @staticmethod
    def _points_to_oriented_box(vertices_3d, centre, yaw):
        """Convert 3D vertices to oriented bounding box [x, y, z, l, w, h, yaw]"""
        if len(vertices_3d) < 3:
            return None

        centered_vertices = vertices_3d - centre.reshape(1, 3)

        # Rotate vertices to align with yaw=0
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        rotated_xy = centered_vertices[:, :2] @ rotation_matrix.T

        # Compute bounding box in rotated frame
        min_x, max_x = np.min(rotated_xy[:, 0]), np.max(rotated_xy[:, 0])
        min_y, max_y = np.min(rotated_xy[:, 1]), np.max(rotated_xy[:, 1])
        min_z, max_z = np.min(centered_vertices[:, 2]), np.max(centered_vertices[:, 2])

        x_max_abs = np.abs(rotated_xy[:, 0]).max()
        y_max_abs = np.abs(rotated_xy[:, 1]).max()

        # Box center and dimensions
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2

        # FIXED: Rotate center back to original frame (inverse rotation)
        center_rotated = np.array([center_x, center_y]) @ rotation_matrix.T

        center_rotated = center_rotated + centre[:2]

        # length = max_x - min_x  # Forward/backward extent
        # width = max_y - min_y   # Left/right extent  
        length = x_max_abs
        width = y_max_abs
        height = max_z - min_z

        return np.array(
            [center_rotated[0], center_rotated[1], center_z, length, width, height, yaw]
        )

    def sync_kalman_with_gtsam(self, kf: PoseKalmanFilter, confidence_factor: float = 0.7):
        """
        Update Kalman filter state to match GTSAM optimized trajectory.
        This "trains" the Kalman filter with the more accurate GTSAM results.
        
        Parameters
        ----------
        kf : PoseKalmanFilter
            The Kalman filter to update
        confidence_factor : float
            How much to trust GTSAM vs current uncertainty (0.0 = trust current, 1.0 = trust GTSAM completely)
        """
        if self.optimized_poses is None or len(self.optimized_poses) < 2:
            return
        
        # Extract latest optimized pose
        latest_pose = self.optimized_poses[-1]
        latest_pose_vector = kf.transform_to_pose_vector(latest_pose)

        orig_mean = self.mean.copy()
        
        # Estimate velocity from optimized trajectory
        if len(self.optimized_poses) >= 2 and len(self.timestamps) >= 2:
            # Use last two poses to estimate current velocity
            prev_pose = self.optimized_poses[0]
            curr_pose = self.optimized_poses[-1]
            
            dt = (self.timestamps[-1] - self.timestamps[0]) * 1e-9  # Convert ns to seconds
            
            # Linear velocity
            linear_vel = (curr_pose[:3, 3] - prev_pose[:3, 3]) / dt
            
            # Compute relative rotation matrix
            R1 = prev_pose[:3, :3]  
            R2 = curr_pose[:3, :3]
            R_rel = R2 @ R1.T  # Relative rotation

            # Convert to axis-angle for angular velocity
            rel_rot_vec = Rotation.from_matrix(R_rel).as_rotvec()
            angular_vel = rel_rot_vec / dt
            
            print("linear_vel", linear_vel)
            print("angular_vel", angular_vel)
        else:
            # No velocity estimate available
            linear_vel = np.zeros(3)
            angular_vel = np.zeros(3)
        
        # Create new state vector with GTSAM pose + estimated velocity
        gtsam_mean = np.concatenate([latest_pose_vector, linear_vel, angular_vel])

        prev_mean = np.copy(self.mean)
        
        # Blend with current Kalman estimate based on confidence
        if confidence_factor > 0.99:
            # Complete replacement
            self.mean = gtsam_mean
        else:
            # Weighted blend
            self.mean = confidence_factor * gtsam_mean + (1 - confidence_factor) * self.mean
        
        diff = self.mean - prev_mean

        # Reduce uncertainty in covariance since we have more accurate estimate
        self.covariance *= (1 - confidence_factor * 0.5)  # Reduce uncertainty by up to 50%
        
        # print(f"Track {self.track_id}: Synced Kalman with GTSAM - pose updated, uncertainty reduced by {confidence_factor*50:.1f}%")
        # print(f"Updated mean=", np.round(self.mean, 2), "orig_mean", np.round(orig_mean, 2))

    def _create_pose_from_heading(self, position, heading_vector):
        """Create a pose matrix with given position and heading direction."""
        pose = np.eye(4)
        pose[:3, 3] = position
        
        # Calculate yaw angle from heading vector
        yaw = np.arctan2(heading_vector[1], heading_vector[0])
        
        # Create rotation matrix around z-axis
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        pose[:3, :3] = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        
        return pose
    

    def _create_pose_from_object_points_and_heading(self, points3d: np.ndarray, heading_vector: np.ndarray, position: np.ndarray):
        """Create a pose matrix with given position and heading direction."""
        pose = np.eye(4)
        pose[:3, 3] = position

        centre = position

        heading_2d = heading_vector[:2]
        heading_norm = np.linalg.norm(heading_2d)
        use_heading = heading_norm > 0

        # heading_2d = heading_2d / (heading_norm + 1e-6)

        points_2d = points3d[:, :2]

        if not use_heading:
            # Estimate orientation using PCA
            centered_vertices = points_2d - centre[:2]
            cov_matrix = np.cov(centered_vertices.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # priority_scores = []

            # for i in range(len(eigenvalues)):
            #     eigenvector = eigenvectors[:, i]
            #     dot_prod = np.dot(eigenvector, heading_2d)

            #     priority_scores.append((np.abs(dot_prod), eigenvalues[i]))

            # idx = np.array(sorted([0, 1], key=lambda i: priority_scores[i], reverse=True))
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            primary_direction = eigenvectors[:, 0]  # Length direction
            # secondary_direction = eigenvectors[:, 1]  # Width direction
            
            # Calculate yaw from primary direction
            yaw = np.arctan2(primary_direction[1], primary_direction[0])
            
        else:
            yaw = np.arctan2(heading_2d[1], heading_2d[0])

        box = ConvexHullObject.points_to_bounding_box(points3d, position)
        box_yaw = box[-1]

        zero_yaw = np.arctan2(0, 1)
        yaw_11 = np.arctan2(0.5, 0.5)

        if use_heading:
            print(f"use heading", heading_2d)

            print(f"{box_yaw=:.3f} {yaw=:.3f} {yaw_11=:.3f} {zero_yaw=:.3f}")

            x, y, z, rx, ry, rz, vx, vy, vz, vrx, vry, vrz = self.mean
            print(f"rx={rx:.2f} ry={ry:.2f} rz={rz:.2f}")
            self.last_used_heading = True

        self.last_yaw = yaw

        # Create rotation matrix around z-axis
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        pose[:3, :3] = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        
        return pose

    def update(self, kf: PoseKalmanFilter, convex_hull: ConvexHullObject):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        conxex_hull : ConvexHullObject
            The associated conxex_hull.

        """
        prev_pose = self.prev_pose
        undo_pose = np.linalg.inv(prev_pose)

        # # Transform to reference frame
        # prev_points = points_rigid_transform(self.last_points, undo_pose)
        # cur_points = points_rigid_transform(convex_hull.original_points, undo_pose)

        # Transform to previous frame
        prev_points = points_rigid_transform(self.last_mesh.vertices, undo_pose)
        cur_points = points_rigid_transform(convex_hull.mesh.vertices, undo_pose)

        # TODO: without getting the relative pose, although slow, we will have poor performance for moving objects they will keep getting picked up again and then deleted.
        R, t, _, _, icp_cost = relative_object_pose(
            prev_points, 
            cur_points, 
            max_iterations=1, 
            debug=False
        )
        
        # Build relative pose in reference frame
        rel_pose = np.eye(4)
        rel_pose[:3, :3] = R
        rel_pose[:3, 3] = t

        cumulative_pose = prev_pose @ rel_pose
        cumulative_pose_vector = kf.transform_to_pose_vector(cumulative_pose)

        new_mesh_vertices = np.concatenate([points_rigid_transform(self.last_mesh.vertices.copy(), rel_pose), convex_hull.mesh.vertices.copy()], axis=0)
        new_mesh = trimesh.convex.convex_hull(new_mesh_vertices)

        dt = (convex_hull.timestamp - self.history[-1].timestamp) * 1e-9
        self.yaw = self._compute_yaw(new_mesh.vertices, new_mesh.centroid, self.last_mesh.centroid, dt)
        self.yaw_method = "update"

        # do not update -> handled by gtsam sync?
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, cumulative_pose_vector, is_relative=False)


        # rel_pose_vector = kf.transform_to_pose_vector(rel_pose)
        # yaw = rel_pose_vector[5]
        # self.yaw = yaw

            # heading = convex_hull.mesh.centroid - self.last_mesh.centroid
            # dt = (convex_hull.timestamp - self.history[-1].timestamp) * 1e-9
            # yaw = np.arctan2(heading[1], heading[0])
            # self.yaw = self._compute_yaw(convex_hull.mesh.vertices, convex_hull.mesh.centroid, self.last_mesh.centroid, dt)

            # rotation_vector = np.zeros((3,))
            # rotation_vector[2] = yaw

            # cumulative_pose = np.eye(4)
            # cumulative_pose[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
            # cumulative_pose[:3, 3] = convex_hull.mesh.centroid

        # use current points for the pose
        world_to_object = np.linalg.inv(cumulative_pose)
        cur_object_points = points_rigid_transform(convex_hull.original_points, world_to_object)

        self.object_points = cur_object_points

        dims_mins = cur_object_points.min(axis=0)
        dims_maxes = cur_object_points.max(axis=0)

        lwh = dims_maxes - dims_mins
        self.lwh = lwh

        self.last_points = convex_hull.original_points
        self.last_mesh = convex_hull.mesh

        self.features.append(convex_hull.feature)
        self.history.append(convex_hull)
        self.timestamps.append(convex_hull.timestamp)

        assert len(self.lwh) == 3, f"lwh={self.lwh}"

        self.mark_hit()

        # After every few measurements, sync with GTSAM
        if len(self.history) >= 3 and len(self.history) % 3 == 0:
            self.optimized_poses = self.compute_poses()

            self.sync_kalman_with_gtsam(kf, confidence_factor=0.8)

        self.prev_pose = self.to_pose_matrix()

        # # update A, c ###########################
        # init_timestamp = min(self.timestamps)*1e-9
        # n_opt = len(self.optimized_boxes) if self.optimized_boxes else 0
        # optimized_boxes = self.optimized_boxes if self.optimized_boxes else []
        # start = 0 if n_opt == 0 else n_opt
        # measurements = np.stack(optimized_boxes + [convex_hull_obj.box for convex_hull_obj in self.history[start:]], axis=0)

        # assert len(measurements) == len(self.timestamps), f"{len(measurements)=} {len(self.timestamps)=} {n_opt=} {len(self.history)=} {start=}"

        # times_secs = np.array(self.timestamps, dtype=float) * 1e-9
        # times_offsets = times_secs - init_timestamp

        # n_samples, n_features = measurements.shape
        
        # # Create design matrix [t, 1] for each timestamp
        # X = np.column_stack([times_offsets, np.ones(n_samples)])
        
        # # Solve least squares: (X^T X)^-1 X^T y for each feature
        # # This gives us [A_j, c_j] for each feature j
        # coefficients = np.linalg.lstsq(X, measurements, rcond=None)[0]
        
        # A = coefficients[0, :]  # Slopes (7,)
        # c = coefficients[1, :]  # Intercepts (7,)

        # self.A, self.c = A, c

        # # y = A*t + c (broadcasting handles multiple timestamps)
        # predictions = A[np.newaxis, :] * times_offsets[:, np.newaxis] + c[np.newaxis, :]

        # eval_r2 = r2_score(measurements, predictions, multioutput='uniform_average')
        # print(f"Box R2: {eval_r2:.2f}")
        # # update A, c ###########################

    def update_raw_lidar(self, kf: PoseKalmanFilter, points3d: np.array):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        conxex_hull : ConvexHullObject
            The associated conxex_hull.

        """
        prev_n_points = len(self.last_points)
        cur_n_points = len(points3d)
        confidences = np.array([x.confidence for x in self.history])
        iou_2ds = np.array([x.iou_2d for x in self.history])
        objectness_scores = np.array([x.objectness_score for x in self.history])

        mean_iou_2d = np.mean(iou_2ds)
        mean_objectness = np.mean(objectness_scores)
        confidence = np.mean(confidences)

        features = np.stack(self.features, axis=0)
        if confidence > 0.0:
            mean_feature = np.average(features, axis=0, weights=confidences)
        else:
            mean_feature = np.mean(features, axis=0)

        mean_feature = mean_feature / (1e-6 + np.linalg.norm(mean_feature))

        convex_hull = ConvexHullObject(
            original_points=points3d.copy(),
            confidence=confidence,
            iou_2d=mean_iou_2d,
            objectness_score=mean_objectness,
            feature=mean_feature,
            timestamp=self.last_predict_timestamp,
            source="track_project_and_query_ball_point"
        )

        if convex_hull.original_points is not None:
            self.update(kf, convex_hull)

    def mark_hit(self):
        self.hits += 1
        self.time_since_update = 0
        if self.state == ConvexHullTrackState.Tentative and self.hits >= self._n_init:
            self.state = ConvexHullTrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        # if self.state == ConvexHullTrackState.Tentative:
        #     self.state = ConvexHullTrackState.Deleted
        # elif self.time_since_update > self._max_age:
        #     self.state = ConvexHullTrackState.Deleted

        if self.time_since_update > self._max_age:
            self.state = ConvexHullTrackState.Deleted


    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == ConvexHullTrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == ConvexHullTrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == ConvexHullTrackState.Deleted