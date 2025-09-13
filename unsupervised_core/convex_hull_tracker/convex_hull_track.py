
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.interpolate import UnivariateSpline, splev, splprep
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_squared_error, r2_score

from lion.unsupervised_core.box_utils import compute_ppscore, icp, icp_open3d_robust
from lion.unsupervised_core.convex_hull_tracker.convex_hull_object import (
    ConvexHullObject,
)
from lion.unsupervised_core.convex_hull_tracker.convex_hull_utils import (
    analytical_z_rotation_centered,
    circular_weighted_mean,
    compute_confidence_from_icp,
    hungarian_matching,
    icp_hungarian,
    predict_pose_with_motion_model,
    relative_object_pose,
    relative_object_pose_multiresolution,
    relative_object_rotation,
    rigid_icp,
    voxel_sampling_fast,
    yaw_circular_mean,
)
from lion.unsupervised_core.convex_hull_tracker.pose_kalman_filter import (
    PoseKalmanFilter,
    wrap_angle,
)
from lion.unsupervised_core.outline_utils import points_rigid_transform
from lion.unsupervised_core.trajectory_optimizer import (
    optimize_with_gtsam_timed,
    optimize_with_gtsam_timed_positions,
    refine_with_gtsam,
    smooth_trajectory_with_vehicle_model,
    smooth_with_vehicle_model,
)

NANOSEC_TO_SEC = 1e-9

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

        assert len(mean) == 8, f"{mean.shape=} {covariance.shape=}"

        self.state = ConvexHullTrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self.trees = [cKDTree(convex_hull.original_points)]
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
        self.initial_poses = None
        self.optimized_poses = None
        self.optimized_boxes = None
        self.spline_boxes = None
        self.prev_pose = PoseKalmanFilter.pose_vector_to_transform(mean[:4])

        self.source = convex_hull.source

        
        self.icp_max_iterations = 5
        self.icp_max_cost = 0.3
        self.heading_speed_thresh_ms = 3.6 # m/s

        self._n_init = n_init
        self._max_age = max_age

        self._constraint_cache = {}
        self._cached_initial_poses = []
        self._last_processed_frame = 0

        self.icp_cache = {}

    def extrapolate_box(self, timestamps):
        init_timestamp_sec = min(self.timestamps) * NANOSEC_TO_SEC

        A = self.A
        c = self.c

        time_offsets = np.array(timestamps, dtype=np.float64) * NANOSEC_TO_SEC - init_timestamp_sec
        new_prediction = A[np.newaxis, :] * time_offsets[:, np.newaxis] + c[np.newaxis, :]

        return new_prediction.reshape(-1, 7)

    # def to_box(self):
    #     A = self.A
    #     c = self.c

    #     timestamp = self.last_predict_timestamp
    #     init_timestamp = min(self.timestamps)*1e-9

    #     new_timestamps = np.array([timestamp*1e-9 - init_timestamp], float)
    #     new_prediction = A[np.newaxis, :] * new_timestamps[:, np.newaxis] + c[np.newaxis, :]

    #     return new_prediction.reshape(7)

    def to_box(self):
        # return self.optimized_boxes[-1] # last seen box...
        return self.extrapolate_box([self.last_predict_timestamp])[0]

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


    def extrapolate_kalman_box(self, timestamp: int):
        dt = (timestamp - self.timestamps[-1]) * NANOSEC_TO_SEC
        
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

    def to_points(self):
        # return self.last_points
        transform = self.to_pose_matrix()
        return points_rigid_transform(self.object_points, transform)

    def to_shape_dict(self) -> Dict:
        points = self.to_points()
        box = self.to_box()
        centre = box[:3]
        # probably could do more efficient...
        mesh = trimesh.convex.convex_hull(points)

        return {
            'original_points': points,
            'centroid_3d': centre,
            'mesh': mesh
        }

    def to_pose_matrix(self):
        if self.optimized_poses is not None:
            return self.optimized_poses[-1]
        else:
            pose_vec = PoseKalmanFilter.box_to_pose_vector(self.to_box())
            return PoseKalmanFilter.pose_vector_to_transform(pose_vec)

        # pose_vec = PoseKalmanFilter.box_to_pose_vector(self.to_box())
        # return PoseKalmanFilter.pose_vector_to_transform(pose_vec)

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

        # self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

        pos = self.extrapolate_box([timestamp])[0][:3]

        self.positions.append(pos)

    def get_or_compute_relative_pose(self, points_i, points_j, center, cache_key):
        if cache_key in self.icp_cache:
            R_cached, t_cached, cost, center_old = self.icp_cache[cache_key]
            # Transform translation for new center
            delta = center_old - center
            t_new = t_cached + (np.eye(3) - R_cached) @ delta
            return R_cached, t_new, cost
        
        # Compute and cache
        centered_i = points_i - center
        centered_j = points_j - center
        R, t, inliers, _, cost = relative_object_pose(centered_i, centered_j)
        # R, t, inliers, _, cost = relative_object_pose_multiresolution(centered_i, centered_j)
        self.icp_cache[cache_key] = (R, t, cost, center.copy())
        return R, t, cost

    def compute_poses(self):
        timestamps = self.timestamps
        n_frames = len(timestamps)

        points_list = []
        boxes = self.extrapolate_box(timestamps)
        initial_poses = []

        # use initial pose as reference
        ref_pose_vector = PoseKalmanFilter.box_to_pose_vector(boxes[0])
        ref_pose_matrix = PoseKalmanFilter.pose_vector_to_transform(ref_pose_vector)
        world_to_object = np.linalg.inv(ref_pose_matrix)

        last_position = ref_pose_vector[:3].copy()
        initial_position = ref_pose_vector[:3].copy()
        last_yaw = ref_pose_vector[3].copy()
        initial_yaw = ref_pose_vector[3].copy()
        last_velocity = np.zeros(3)
        last_angular_velocity = 0

        # Collect world points and centers
        for i, timestamp_ns in enumerate(timestamps):
            obj: ConvexHullObject = self.history[i]
            world_points = obj.original_points
            # world_points = obj.mesh.vertices

            # add our pose here
            pose_vector = PoseKalmanFilter.box_to_pose_vector(boxes[i])
            pose_matrix = PoseKalmanFilter.pose_vector_to_transform(pose_vector)
            initial_poses.append(pose_matrix)

            # object_points = points_rigid_transform(world_points, world_to_object)

            points_list.append(world_points)

        optimized_poses = [ref_pose_matrix]
        simple_poses = [ref_pose_matrix]

        last_optimized_yaw = np.copy(last_yaw)
        last_optimized_position = np.copy(last_position)

        constraints = []
        for i in range(1, n_frames):
            expected_pose = initial_poses[i]
            expected_yaw = Rotation.from_matrix(expected_pose[:3, :3]).as_rotvec()[2]
            expected_position = expected_pose[:3, 3]

            # rel_R, rel_pos, A_inliers, B_inliers, icp_cost_rel = relative_object_pose(centered_ref, centered_curr1)
            R_rel, t_rel, icp_cost_rel = self.get_or_compute_relative_pose(
                points_list[i-1], points_list[i], 
                last_optimized_position,
                (i-1, i, 'relative')
            )
    
            rel_yaw = Rotation.from_matrix(R_rel).as_rotvec()[2]
            rel_yaw_guess = wrap_angle(last_optimized_yaw + rel_yaw)

            if i == 1:
                R_init, t_init, icp_cost_initial = R_rel, t_rel, icp_cost_rel
            else:
                R_init, t_init, icp_cost_initial = self.get_or_compute_relative_pose(
                    points_list[0], points_list[i], 
                    initial_position,
                    (0, i, 'initial')
                )

            initial_yaw_guess = wrap_angle(initial_yaw + Rotation.from_matrix(R_init).as_rotvec()[2])

            # pose guesses
            rel_pos_guess = last_position + t_rel
            initial_pos_guess = initial_position + t_init

            # Compute current velocities
            dt = (timestamps[i] - timestamps[i-1]) * 1e-9
            velocity_rel = (rel_pos_guess - last_optimized_position) / dt
            velocity_init = (initial_pos_guess - last_optimized_position) / dt
            angular_vel_rel = (rel_yaw_guess - last_optimized_yaw) / dt
            angular_vel_init = (initial_yaw_guess - last_optimized_yaw) / dt

            # Smoothness penalties
            lambda_smooth = 0.1  # Tunable parameter
            smooth_cost_rel = lambda_smooth * np.linalg.norm(velocity_rel - last_velocity)**2 #+ lambda_smooth * np.abs(angular_vel_rel - last_angular_velocity)
            smooth_cost_init = lambda_smooth * np.linalg.norm(velocity_init - last_velocity)**2 #+ lambda_smooth * np.abs(angular_vel_init - last_angular_velocity)

            # weight between the two
            total_cost_rel = icp_cost_rel + smooth_cost_rel
            total_cost_init = icp_cost_initial + smooth_cost_init
            weight_rel = 1.0 / (total_cost_rel + 1e-6)
            weight_initial = 1.0 / (total_cost_init + 1e-6)

            predicted_pos, predicted_yaw = predict_pose_with_motion_model(
                last_optimized_position, last_optimized_yaw, 
                last_velocity, last_angular_velocity, dt
            )

            # TODO: predicted weight just the mean of the others?
            weight_predicted = (weight_rel + weight_initial) / 2.0

            total_weight = weight_rel + weight_initial #+ weight_predicted
            weight_rel /= total_weight
            weight_initial /= total_weight
            # weight_predicted /= total_weight

            # print("weight_initial", weight_initial, weight_rel)

            # assert weight_rel + weight_initial == 1.0, f"weight_rel + weight_initial = {weight_rel + weight_initial}"

            optimized_position = rel_pos_guess * weight_rel + initial_pos_guess * weight_initial
            optimized_yaw = circular_weighted_mean(
                np.array([rel_yaw_guess, initial_yaw_guess]), 
                np.array([weight_rel, weight_initial])
            )

            # print("optimized_position", optimized_position)
            # print("optimized_yaw", optimized_yaw)

            # optimized_position = rel_pos_guess * weight_rel + initial_pos_guess * weight_initial + predicted_pos * weight_predicted
            # optimized_yaw = circular_weighted_mean(
            #     np.array([rel_yaw_guess, initial_yaw_guess, predicted_yaw]), 
            #     np.array([weight_rel, weight_initial, weight_predicted])
            # )

            # print(f"{expected_yaw=:.2f} {initial_yaw_guess=:.2f} {rel_yaw_guess=:.2f}")
            # print(f"{expected_position=} {rel_pos_guess=} {initial_pos_guess=} {predicted_pos=}")
            # print(f"{velocity_rel=} {velocity_init=}")
            # print(f"{angular_vel_rel=} {angular_vel_init=}")
            # print(f"{smooth_cost_rel=} {smooth_cost_init=}")

            last_yaw = expected_yaw
            last_position = expected_position

            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_rotvec(np.array([0.0, 0.0, optimized_yaw], float)).as_matrix()
            pose[:3, 3] = optimized_position
            optimized_poses.append(pose)

            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_rotvec(np.array([0.0, 0.0, initial_yaw_guess], float)).as_matrix()
            pose[:3, 3] = initial_pos_guess
            simple_poses.append(pose)

            # print("pose", pose)

            # Update for next iteration
            last_velocity = (optimized_position - last_optimized_position) / dt
            last_angular_velocity = (optimized_yaw - last_optimized_yaw) / dt

            last_optimized_position = np.copy(optimized_position)
            last_optimized_yaw = np.copy(optimized_yaw)

        # smoothing to fix icp accumulated errors
        optimized_poses = smooth_trajectory_with_vehicle_model(self.timestamps, optimized_poses, heading_speed_thresh=self.heading_speed_thresh_ms)

        optimized_pose_vectors = np.stack([PoseKalmanFilter.transform_to_pose_vector(x) for x in optimized_poses], axis=0)
        initial_pose_vectors = np.stack([PoseKalmanFilter.transform_to_pose_vector(x) for x in initial_poses], axis=0)

        dists = np.linalg.norm((optimized_pose_vectors[:, :3] - initial_pose_vectors[:, :3]), axis=1)
        mean_dist = np.mean(dists)

        assert mean_dist < 5.0

        l, w, h = self.c[3:6]
        self.optimized_poses = optimized_poses

        optimized_boxes = []
        for i in range(n_frames):
            yaw = Rotation.from_matrix(optimized_poses[i][:3, :3]).as_rotvec()[2]

            x, y, z = optimized_poses[i][:3, 3]

            box = np.array([x, y, z, l, w, h, yaw])
            optimized_boxes.append(box)

        self.optimized_boxes = optimized_boxes

        # update points
        # Now transform points to object-centric coordinates
        object_centric_points = []
        l, w, h = self.c[3:6]

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

        return optimized_poses

    def _compute_oriented_boxes(self, timestamps: List[int], optimized_poses: List[np.ndarray], merged_object_points: np.ndarray): #object_mesh: trimesh.Trimesh):
        """Compute oriented bounding boxes based on object motion direction"""
        boxes = []

        for i, cur_pose in enumerate(optimized_poses):
            rotvec = Rotation.from_matrix(cur_pose[:3, :3]).as_rotvec()
            yaw = rotvec[2]

            cur_points = points_rigid_transform(merged_object_points, cur_pose)
            # cur_mesh = object_mesh.copy().apply_transform(cur_pose)

            # centre = cur_mesh.centroid
            centre = cur_pose[:3, 3]

            # Create oriented bounding box from 3D vertices
            # box = self._points_to_oriented_box(cur_mesh.vertices, centre, yaw)
            box = self._points_to_oriented_box(cur_points, centre, yaw)
            boxes.append(box)

        return boxes

    def _compute_yaw(self, vertices_3d: np.ndarray, centre: np.ndarray, prev_centre: np.ndarray, dt: float):
        
        velocity = (centre - prev_centre) / dt
        speed = np.linalg.norm(velocity[:2])

        if speed > self.heading_speed_thresh_ms and dt >= 0.1:
            return np.arctan2(velocity[1], velocity[0])

        points_2d = vertices_3d[:, :2]
        centred_vertices = points_2d - centre[:2]
        cov_matrix = np.cov(centred_vertices.T)
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

        centred_vertices = vertices_3d - centre.reshape(1, 3)

        # Rotate vertices to align with yaw=0
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        rotated_xy = centred_vertices[:, :2] @ rotation_matrix.T

        # Compute bounding box in rotated frame
        min_x, max_x = np.min(rotated_xy[:, 0]), np.max(rotated_xy[:, 0])
        min_y, max_y = np.min(rotated_xy[:, 1]), np.max(rotated_xy[:, 1])
        min_z, max_z = np.min(centred_vertices[:, 2]), np.max(centred_vertices[:, 2])

        x_max_abs = np.abs(rotated_xy[:, 0]).max()
        y_max_abs = np.abs(rotated_xy[:, 1]).max()

        # Box centre and dimensions
        centre_x = (min_x + max_x) / 2
        centre_y = (min_y + max_y) / 2
        centre_z = (min_z + max_z) / 2

        # FIXED: Rotate centre back to original frame (inverse rotation)
        # centre_rotated = np.array([centre_x, centre_y]) @ rotation_matrix.T
        # centre_rotated = centre_rotated + centre[:2]

        centre_rotated = centre[:2]

        length = max_x - min_x  # Forward/backward extent
        width = max_y - min_y   # Left/right extent  
        # length = x_max_abs*2
        # width = y_max_abs*2
        height = max_z - min_z

        return np.array(
            [centre_rotated[0], centre_rotated[1], centre_z, length, width, height, yaw]
        )

    def sync_kalman_with_gtsam(self, kf, confidence_factor=0.7):
        if len(self.optimized_poses) < 2:
            return
            
        # Estimate 4D velocity from pose trajectory
        latest_pose = self.optimized_poses[-1] 
        prev_pose = self.optimized_poses[-2]
        dt = (self.timestamps[-1] - self.timestamps[-2]) * NANOSEC_TO_SEC
        
        # Linear velocities
        vx, vy, vz = (latest_pose[:3, 3] - prev_pose[:3, 3]) / dt
        
        # Angular velocity (handle wraparound)
        yaw_curr = np.arctan2(latest_pose[1, 0], latest_pose[0, 0])
        yaw_prev = np.arctan2(prev_pose[1, 0], prev_pose[0, 0]) 
        vyaw = wrap_angle(yaw_curr - yaw_prev) / dt

        # Update Kalman state
        pose_vector = [latest_pose[0, 3], latest_pose[1, 3], latest_pose[2, 3], yaw_curr]
        velocity_vector = [vx, vy, vz, vyaw]
        
        gtsam_mean = np.concatenate([pose_vector, velocity_vector])
        self.mean = confidence_factor * gtsam_mean + (1 - confidence_factor) * self.mean
        self.mean[3] = wrap_angle(self.mean[3])  # Ensure yaw is wrapped

        print("sync_kalman_with_gtsam updated mean", self.mean[:4])
        box = self.extrapolate_box([self.last_predict_timestamp])[0]
        print("latest prediction from", box)

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
            centred_vertices = points_2d - centre[:2]
            cov_matrix = np.cov(centred_vertices.T)
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
            # prev_pose = self.prev_pose
            # undo_pose = np.linalg.inv(prev_pose)

            # # Transform to reference frame
            # prev_points = points_rigid_transform(self.last_points, undo_pose)
            # cur_points = points_rigid_transform(convex_hull.original_points, undo_pose)

            # # # Transform to previous frame
            # # prev_points = points_rigid_transform(self.last_mesh.vertices, undo_pose)
            # # cur_points = points_rigid_transform(convex_hull.mesh.vertices, undo_pose)

            # # TODO: without getting the relative pose, although slow, we will have poor performance for moving objects they will keep getting picked up again and then deleted.
            # R, t, prev_indices, cur_indices, icp_cost = relative_object_pose(
            #     prev_points, 
            #     cur_points, 
            #     max_iterations=self.icp_max_iterations, 
            #     debug=False
            # )

            # prev_inlier_points = self.last_points[prev_indices]
            # cur_inlier_points = convex_hull.original_points[cur_indices]

            # # if icp_cost > self.icp_max_cost:
            # #     return

            # print(f"{icp_cost=:.2f}")
            
            # # Build relative pose in reference frame
            # rel_pose = np.eye(4)
            # rel_pose[:3, :3] = R
            # rel_pose[:3, 3] = t

            # cumulative_pose = prev_pose @ rel_pose
            # cumulative_pose_vector = kf.transform_to_pose_vector(cumulative_pose)

            # box_pose_vector = PoseKalmanFilter.box_to_pose_vector(convex_hull.box)
            # print("box_pose_vector", box_pose_vector)
            # print("cumulative_pose_vector", cumulative_pose_vector)

        # new_mesh_vertices = np.concatenate([points_rigid_transform(self.last_mesh.vertices.copy(), rel_pose), convex_hull.mesh.vertices.copy()], axis=0)
        # new_mesh = trimesh.convex.convex_hull(new_mesh_vertices)

        # merged_points = np.concatenate([
        #     points_rigid_transform(prev_inlier_points, rel_pose),
        #     cur_inlier_points
        # ], axis=0)

        # merged_mesh = trimesh.convex.convex_hull(merged_points)


        # dt = (convex_hull.timestamp - self.history[-1].timestamp) * 1e-9
        # self.yaw = self._compute_yaw(new_mesh.vertices, new_mesh.centroid, self.last_mesh.centroid, dt)
        # self.yaw_method = "update"

        # # do not update -> handled by gtsam sync?
        # self.mean, self.covariance = kf.update(
        #     self.mean, self.covariance, cumulative_pose_vector)


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

        # # use current points for the pose
        # world_to_object = np.linalg.inv(cumulative_pose)
        # cur_object_points = points_rigid_transform(cur_inlier_points, world_to_object)

        # self.object_points = cur_object_points

        # dims_mins = cur_object_points.min(axis=0)
        # dims_maxes = cur_object_points.max(axis=0)

        # lwh = dims_maxes - dims_mins
        # self.lwh = lwh

        # self.last_points = cur_inlier_points
        # # self.last_mesh = convex_hull.mesh

        self.features.append(convex_hull.feature)
        self.history.append(convex_hull)
        self.trees.append(cKDTree(convex_hull.original_points))
        self.timestamps.append(convex_hull.timestamp)

        assert len(self.lwh) == 3, f"lwh={self.lwh}"

        self.mark_hit()

        # # After every few measurements, sync with GTSAM
        # if len(self.history) >= 3 and len(self.history) % 3 == 0:
        #     self.optimized_poses = self.compute_poses()

            # self.sync_kalman_with_gtsam(kf, confidence_factor=0.5)
        self.optimized_poses = self.compute_poses()

        self.prev_pose = self.to_pose_matrix()

        self.update_box_parameters()
        # self.update_box_parameters_weighted_lstsq()
        # self.update_box_parameters_dual_fit()

    def update_box_parametersx(self) -> None:
        """Concise version with key improvements."""        
        # Prepare data
        init_timestamp_sec = min(self.timestamps) * NANOSEC_TO_SEC
        time_offsets = np.array(self.timestamps, dtype=np.float64) * NANOSEC_TO_SEC - init_timestamp_sec
        
        # Stack measurements
        # n_optimized = len(self.optimized_boxes) if self.optimized_boxes else 0
        optimized_boxes = self.optimized_boxes
        measurements = np.stack(optimized_boxes, axis=0)

        print("measurements", measurements)
        exit()
        
        lwhs = measurements[:, 3:6]
        print("lwhs", lwhs.shape, lwhs.min(axis=0), lwhs.max(axis=0))
        
        measurements[:, 3:6] = np.mean(lwhs, axis=0).reshape(1, 3).repeat(len(measurements), 0)

        # Validate
        assert len(measurements) == len(self.timestamps), (
            f"Measurement/timestamp mismatch: {len(measurements)} vs {len(self.timestamps)}")
        
        # Fit model
        design_matrix = np.column_stack([time_offsets, np.ones(len(time_offsets))])
        # coefficients, *_ = np.linalg.lstsq(design_matrix, measurements, rcond=None)
        coefficients, residuals, *_ = np.linalg.lstsq(design_matrix, measurements, rcond=None)
        self.A, self.c = coefficients[0, :], coefficients[1, :]
        
        # Evaluate
        predictions = self.A * time_offsets[:, np.newaxis] + self.c
        errs = (measurements - predictions)
        r2 = r2_score(measurements, predictions, multioutput='uniform_average')
        rmse = mean_squared_error(measurements, predictions, multioutput='uniform_average')
        print(f"Box Parameter Update - R²: {r2:.3f} - rmse {rmse:.3f}")
        # print("errs", errs)

        box = self.to_box()
        self.mean[:3] = box[:3]
        self.mean[3] = box[6]

    def update_box_parameters_weighted_lstsq(self) -> None:
        """
        Single lstsq with observation weights
        """
        init_timestamp_sec = min(self.timestamps) * NANOSEC_TO_SEC
        time_offsets = np.array(self.timestamps, dtype=np.float64) * NANOSEC_TO_SEC - init_timestamp_sec
        
        history_boxes = np.array([obj.box for obj in self.history])
        optimized_boxes = np.array(self.optimized_boxes) if self.optimized_boxes else history_boxes.copy()
        
        n_frames = len(self.timestamps)
        
        # Stack both sets of measurements
        # all_measurements = np.vstack([history_boxes, optimized_boxes])
        # all_time_offsets = np.tile(time_offsets, 2)
        
        # # Create weights: early frames favor history, later favor optimized
        # time_weights = np.linspace(0.0, 1.0, n_frames)
        # time_weights = 1 / (1 + np.exp(-4 * (time_weights - 0.5)))  # Sigmoid
        
        # hist_weights = 1 - time_weights
        # opt_weights = time_weights
        print("history positions", history_boxes[:, :3])
        print("optimized positions", optimized_boxes[:, :3])


        def get_box_weights(boxes):
            weights = np.zeros(len(boxes))
            for i, box in enumerate(boxes):
                points = self.history[i].original_points.copy()
                centre = box[:3]
                yaw = box[6]

                l, w, h = box[3:6]

                centred_vertices = points - centre.reshape(1, 3)

                # Rotate vertices to align with yaw=0
                cos_yaw = np.cos(-yaw)
                sin_yaw = np.sin(-yaw)
                rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

                rotated_xy = centred_vertices[:, :2] @ rotation_matrix.T
                x_mask = (rotated_xy[:, 0] >= -l/2) & (rotated_xy[:, 0] <= l/2)
                y_mask = (rotated_xy[:, 1] >= -w/2) & (rotated_xy[:, 1] <= w/2)
                z_mask = (centred_vertices[:, 2] >= -h/2) & (centred_vertices[:, 2] <= h/2)

                inliers = x_mask & y_mask & z_mask

                weights[i] = inliers.sum() / len(inliers)

            return weights

        hist_weights = get_box_weights(history_boxes)
        opt_weights = get_box_weights(optimized_boxes)

        print("hist_weights (inliers)", hist_weights)
        print("opt_weights (inliers)", opt_weights)

        total_weights = hist_weights + opt_weights
        hist_weights /= total_weights
        opt_weights /= total_weights

        print("hist_weights + opt_weights", hist_weights + opt_weights)
        
        # Process measurements
        lwhs_hist = history_boxes[:, 3:6]
        lwhs_opt = optimized_boxes[:, 3:6]
        avg_lwh = np.mean(np.vstack([lwhs_hist, lwhs_opt]), axis=0)
        
        print("avg_lwh", avg_lwh)

        measurements = np.zeros((n_frames, 7))
        measurements[:, :3] = hist_weights[:, np.newaxis] * history_boxes[:, :3] + opt_weights[:, np.newaxis] * optimized_boxes[:, :3]
        measurements[:, 3:6] = avg_lwh

        for i in range(n_frames):
            measurements[i, 6] = yaw_circular_mean(history_boxes[i, 6], optimized_boxes[i, 6], hist_weights[i], opt_weights[i])

        # positions = measurements[:, :3]
        
        # dt = np.diff(time_offsets)
        # velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
        # speeds = np.linalg.norm(velocities[:, :2], axis=1)
        # velocity_yaws = np.arctan2(velocities[:, 1], velocities[:, 0])
        # avg_speed = np.mean(speeds)
        
        # if avg_speed > self.heading_speed_thresh_ms:
        #     yaws = np.concatenate([velocity_yaws, velocity_yaws[[-1]]], axis=0)
        # else:
        #     yaws = np.median(measurements[:, 6]).reshape(1).repeat(len(measurements))
            
        
        # measurements[:, 6] = yaws
        # Fit model
        design_matrix = np.column_stack([time_offsets, np.ones(len(time_offsets))])
        # coefficients, *_ = np.linalg.lstsq(design_matrix, measurements, rcond=None)
        coefficients, residuals, *_ = np.linalg.lstsq(design_matrix, measurements, rcond=None)
        self.A, self.c = coefficients[0, :], coefficients[1, :]
        
        # Evaluate on original time points
        predictions = self.A * time_offsets[:, np.newaxis] + self.c
        
        
        r2 = r2_score(measurements, predictions, multioutput='uniform_average')
        rmse = mean_squared_error(measurements, predictions, multioutput='uniform_average')
        
        print(f"Weighted LSTSQ Box Update - R²: {r2:.3f}, RMSE: {rmse:.3f}")

    def update_box_parameters_dual_fit(self) -> None:
        """
        Fit both separately then combine the A and c parameters
        """
        init_timestamp_sec = min(self.timestamps) * NANOSEC_TO_SEC
        time_offsets = np.array(self.timestamps, dtype=np.float64) * NANOSEC_TO_SEC - init_timestamp_sec
        
        history_boxes = np.array([obj.box for obj in self.history])
        optimized_boxes = np.array(self.optimized_boxes) if self.optimized_boxes else history_boxes.copy()
        
        design_matrix = np.column_stack([time_offsets, np.ones(len(time_offsets))])
        
        def fit_boxes(boxes):
            # Process boxes (LWH consistency, yaw from velocity)
            measurements = boxes.copy()
            lwhs = measurements[:, 3:6]
            measurements[:, 3:6] = np.mean(lwhs, axis=0).reshape(1, 3).repeat(len(measurements), 0)
            
            positions = measurements[:, :3]
            dt = np.diff(time_offsets)
            velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
            speeds = np.linalg.norm(velocities[:, :2], axis=1)
            velocity_yaws = np.arctan2(velocities[:, 1], velocities[:, 0])
            avg_speed = np.mean(speeds)
            
            if avg_speed > self.heading_speed_thresh_ms:
                yaws = np.concatenate([velocity_yaws, velocity_yaws[[-1]]], axis=0)
            else:
                yaws = np.median(measurements[:, 6]).reshape(1).repeat(len(measurements))
            
            measurements[:, 6] = yaws[:len(measurements)]
            
            # Fit
            coefficients, residuals, *_ = np.linalg.lstsq(design_matrix, measurements, rcond=None)
            A, c = coefficients[0, :], coefficients[1, :]
            
            # Quality metric
            predictions = A * time_offsets[:, np.newaxis] + c
            r2 = r2_score(measurements, predictions, multioutput='uniform_average')
            
            return A, c, r2, measurements
        
        # Fit both
        A_hist, c_hist, r2_hist, measurements_hist = fit_boxes(history_boxes)
        A_opt, c_opt, r2_opt, measurements_opt = fit_boxes(optimized_boxes)
        
        # Combine based on fit quality and time
        # Weight by R² quality 
        quality_weight_opt = r2_opt / (r2_hist + r2_opt + 1e-6)
        quality_weight_hist = 1 - quality_weight_opt
        
        # Time-based weight (favor optimized later)
        time_factor = 0.7  # How much to favor later frames
        
        # Combine parameters
        self.A = quality_weight_hist * A_hist + quality_weight_opt * A_opt
        self.c = quality_weight_hist * c_hist + quality_weight_opt * c_opt
        
        # Evaluate combined model
        predictions = self.A * time_offsets[:, np.newaxis] + self.c
        combined_measurements = quality_weight_hist * measurements_hist + quality_weight_opt * measurements_opt
        
        r2_combined = r2_score(combined_measurements, predictions, multioutput='uniform_average')
        rmse_combined = mean_squared_error(combined_measurements, predictions, multioutput='uniform_average')
        
        print(f"Dual-fit Box Update:")
        print(f"  History R²: {r2_hist:.3f}, Optimized R²: {r2_opt:.3f}")
        print(f"  Combined R²: {r2_combined:.3f}, RMSE: {rmse_combined:.3f}")
        print(f"  Parameter weights - history: {quality_weight_hist:.2f}, optimized: {quality_weight_opt:.2f}")



    def update_box_parameters(self) -> None:
        """Concise version with key improvements."""        
        # Prepare data
        init_timestamp_sec = min(self.timestamps) * NANOSEC_TO_SEC
        time_offsets = np.array(self.timestamps, dtype=np.float64) * NANOSEC_TO_SEC - init_timestamp_sec
        
        # Stack measurements
        # n_optimized = len(self.optimized_boxes) if self.optimized_boxes else 0
        # optimized_boxes = self.optimized_boxes or []
        # history_start = n_optimized if n_optimized > 0 else 0
        history_start = 0
        optimized_boxes = []
        history_boxes = [obj.box for obj in self.history[history_start:]]
        measurements = np.stack(optimized_boxes + history_boxes, axis=0)
        
        lwhs = measurements[:, 3:6]
        measurements[:, 3:6] = np.mean(lwhs, axis=0).reshape(1, 3).repeat(len(measurements), 0)
        positions = measurements[:, :3]
        
        if len(time_offsets) >= 4:
            # Extract positions and yaws
            positions = measurements[:, :3]
            original_yaws = measurements[:, 6]
            
            n_points = len(positions)
            
            # Base smoothing on path length and noise level
            path_length = np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1))
            smoothing_factor = max(n_points * 0.1, path_length * 0.01)
            
            
            # Parametric spline for XY path
            tck_xy, u_xy = splprep([positions[:, 0], positions[:, 1]], 
                                u=time_offsets, s=smoothing_factor, k=min(3, n_points-1))
            
            # Smooth Z separately
            z_spline = UnivariateSpline(time_offsets, positions[:, 2], 
                                    s=smoothing_factor * 0.5, k=min(3, n_points-1))
            
            # Evaluate smoothed path
            smoothed_xy = np.array(splev(time_offsets, tck_xy)).T
            smoothed_z = z_spline(time_offsets)
            smoothed_positions = np.column_stack([smoothed_xy, smoothed_z])
            measurements[:, :3] = smoothed_positions
            
            # Step 2: Compute velocities and speeds from smoothed path
            xy_derivatives = np.array(splev(time_offsets, tck_xy, der=1)).T
            z_derivative = z_spline.derivative()(time_offsets)
            
            velocities_3d = np.column_stack([xy_derivatives, z_derivative])
            speeds_2d = np.linalg.norm(xy_derivatives, axis=1)
            
            # Step 3: Derive yaw from path tangents (when moving fast enough)
            path_yaws = np.arctan2(xy_derivatives[:, 1], xy_derivatives[:, 0])
            
            # Determine which frames to use path-based yaw vs original yaw
            use_path_yaw = speeds_2d > self.heading_speed_thresh_ms
            n_fast_frames = np.sum(use_path_yaw)
            
            # Initialize output yaws
            smoothed_yaws = original_yaws.copy()
            
            if n_fast_frames > 0:
                # For fast frames, use path tangent yaw
                # Handle angle wrapping carefully
                fast_indices = np.where(use_path_yaw)[0]

                fps = path_yaws[fast_indices]
                xps = time_offsets[fast_indices]
                smoothed_yaws = np.interp(time_offsets, xps, fps)
            else:
                median_yaw = np.median(original_yaws)
                smoothed_yaws[:] = median_yaw
            
            measurements[:, 6] = smoothed_yaws
        else:
            # Calculate velocities and speeds
            dt = np.diff(time_offsets)
            velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
            speeds = np.linalg.norm(velocities[:, :2], axis=1)
            
            # Calculate yaws from velocity
            velocity_yaws = np.arctan2(velocities[:, 1], velocities[:, 0])
            
            avg_speed = np.mean(speeds)

            if avg_speed > self.heading_speed_thresh_ms:
                yaws = velocity_yaws
                yaws = np.concatenate([yaws, yaws[[-1]]], axis=0)
            else:
                yaws = np.median(measurements[:, 6]).reshape(1).repeat(len(measurements))

            measurements[:, 6] = yaws[:len(measurements)]


        # Validate
        assert len(measurements) == len(self.timestamps), (
            f"Measurement/timestamp mismatch: {len(measurements)} vs {len(self.timestamps)}")
        
        # Fit model
        design_matrix = np.column_stack([time_offsets, np.ones(len(time_offsets))])
        # coefficients, *_ = np.linalg.lstsq(design_matrix, measurements, rcond=None)
        coefficients, residuals, *_ = np.linalg.lstsq(design_matrix, measurements, rcond=None)
        self.A, self.c = coefficients[0, :], coefficients[1, :]
        
        # Evaluate
        predictions = self.A * time_offsets[:, np.newaxis] + self.c
        # errs = (measurements - predictions)
        r2 = r2_score(measurements, predictions, multioutput='uniform_average')
        rmse = mean_squared_error(measurements, predictions, multioutput='uniform_average')
        print(f"Box Parameter Update - R²: {r2:.3f} - rmse {rmse:.3f}")


        box = self.to_box()
        self.mean[:3] = box[:3]
        self.mean[3] = box[6]

    def _blend_angles(self, angle1: np.ndarray, angle2: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Properly blend angles accounting for circular nature."""
        # Convert to complex representation for proper averaging
        c1 = np.exp(1j * angle1)
        c2 = np.exp(1j * angle2)
        
        # Weighted average in complex plane
        blended_complex = (1 - weight) * c1 + weight * c2
        
        # Convert back to angle
        return np.angle(blended_complex)

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