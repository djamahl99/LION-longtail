
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_squared_error, r2_score

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
    wrap_angle,
)
from lion.unsupervised_core.outline_utils import points_rigid_transform
from lion.unsupervised_core.trajectory_optimizer import (
    optimize_with_gtsam_timed,
    optimize_with_gtsam_timed_positions,
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


    def compute_poses(self):
        timestamps = self.timestamps
        n_frames = len(timestamps)

        world_points_per_timestamp = []
        
        # Collect world points and centres
        for i, timestamp_ns in enumerate(timestamps):
            obj: ConvexHullObject = self.history[i]
            world_points = obj.original_points
            world_points_per_timestamp.append(world_points.copy())

        pred_boxes = self.extrapolate_box(timestamps)
        world_centres = pred_boxes[:, :3]

        initial_poses = []
        for box in pred_boxes:
            pose_vector = PoseKalmanFilter.box_to_pose_vector(box)
            pose = PoseKalmanFilter.pose_vector_to_transform(pose_vector)
            initial_poses.append(pose)

        start_frame = max(1, self._last_processed_frame)
        # start_frame = 1
        constraints = [self._constraint_cache[key] for key in sorted(self._constraint_cache.keys())]
        
        self._constraint_cache = {}

        for i in range(start_frame, n_frames):
            constraint_key = (i-1, i)  # Key for frame pair

            prev_pose = initial_poses[i-1]
            cur_pose = initial_poses[i]
            undo_pose = np.linalg.inv(prev_pose)

            prev_points = points_rigid_transform(
                world_points_per_timestamp[i - 1], undo_pose
            )
            cur_points = points_rigid_transform(
                world_points_per_timestamp[i], undo_pose
            )

            prev_yaw = Rotation.from_matrix(prev_pose[:3, :3]).as_rotvec()[2]
            prev_pose_pos = prev_pose[:3, 3]

            cur_yaw = Rotation.from_matrix(cur_pose[:3, :3]).as_rotvec()[2]
            cur_pose_pos = cur_pose[:3, 3]

            print("prev_yaw, prev_pose_pos", prev_yaw, prev_pose_pos)
            print("cur_yaw, cur_pose_pos", cur_yaw, cur_pose_pos)

            rel_yaw = cur_yaw - prev_yaw
            rel_position = cur_pose_pos - prev_pose_pos

            print("rel_yaw, rel_position", rel_yaw, rel_position)

            R, t, A_inliers, B_inliers, icp_cost = relative_object_pose(prev_points, cur_points, max_iterations=self.icp_max_iterations)

            rel_yaw = Rotation.from_matrix(R).as_rotvec()[2]
            print("relative_object_pose rel_yaw", rel_yaw)
            print("t", t)

            relative_pose = np.eye(4)
            relative_pose[:3, :3] = R
            relative_pose[:3, 3] = t
            confidence = len(A_inliers) / len(cur_points)

            constraint = {
                'frame_i': i - 1,
                'frame_j': i, 
                'relative_pose': relative_pose,
                'confidence': confidence,
            }
            self._constraint_cache[constraint_key] = constraint
            self._last_processed_frame = i
            constraints.append(constraint)

        optimized_poses, marginals, quality = optimize_with_gtsam_timed(
            initial_poses, constraints, timestamps
        )
            
        # update points
        # Now transform points to object-centric coordinates
        object_centric_points = []
        for i, (convex_hull_obj, obj_pose) in enumerate(zip(self.history, optimized_poses)):
            # Get world points
            world_points = convex_hull_obj.original_points

            # Transform to object-centric: multiply by inverse of object pose
            world_to_object = np.linalg.inv(obj_pose)
            object_points = points_rigid_transform(world_points, world_to_object)
            object_centric_points.append(object_points)

        self.object_points = np.concatenate(object_centric_points, axis=0)
        self.object_points = voxel_sampling_fast(self.object_points, 0.05, 0.05, 0.05)

        self.last_points = points_rigid_transform(self.object_points, optimized_poses[-1])
        self.last_mesh: trimesh.Trimesh = trimesh.convex.convex_hull(self.last_points)
        self.optimized_poses = optimized_poses

        undo_last_pose = np.linalg.inv(optimized_poses[-1])
        object_mesh = self.last_mesh.copy().apply_transform(undo_last_pose)
        self.optimized_boxes = self._compute_oriented_boxes(timestamps, self.optimized_poses, object_mesh)

        self.positions = [x[:3, 3] for x in optimized_poses]

        return optimized_poses

    def compute_poses_(self):
        timestamps = self.timestamps
        n_frames = len(timestamps)

        world_points_per_timestamp = []
        # world_centres = []
        
        # Collect world points and centres
        for i, timestamp_ns in enumerate(timestamps):
            obj: ConvexHullObject = self.history[i]
            world_points = obj.original_points
            world_points_per_timestamp.append(world_points.copy())
            # world_points_per_timestamp.append(obj.mesh.vertices)
            # centre = obj.box[:3]
            # world_centres.append(centre)

        pred_boxes = self.extrapolate_box(timestamps)
        world_centres = pred_boxes[:, :3]

        world_centres = np.array(world_centres)
        times_secs = np.array(timestamps, dtype=float) * NANOSEC_TO_SEC

        # heading -> 0, 0, 0 if not moving else we calculate the average.
        # heading = np.zeros((2,))
        # if n_frames > 1:
        #     velocities = []
        #     for i in range(1, min(n_frames, 5)):
        #         velocity = (world_centres[i] - world_centres[i-1]) / (times_secs[i] - times_secs[i-1])
        #         velocities.append(velocity)

        #     velocities = np.stack(velocities, axis=0)

        #     avg_velocity = np.mean(velocities, axis=0)
        #     avg_speed = np.linalg.norm(avg_velocity[:2])

        #     self.last_avg_velocity = avg_velocity
            
        #     if avg_speed > self.heading_speed_thresh_ms:
        #         heading = avg_velocity[:2]

        cur_box = self.to_box()
        yaw_c = cur_box[6]
        # yaw = np.arctan2(heading[1], heading[0])

        heading = np.array([np.cos(yaw_c), np.sin(yaw_c)])
        # start_frame = max(1, self._last_processed_frame)
        start_frame = 1
        
        if start_frame == 1:
            # First time or full recompute - compute everything
            first_object_pose = self._create_pose_from_object_points_and_heading(
                world_points_per_timestamp[0], heading, world_centres[0]
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

            # Extract current state
            current_yaw = Rotation.from_matrix(cumulative_pose[:3, :3]).as_rotvec()[2]
            current_pos = cumulative_pose[:3, 3]

            # Extract relative motion
            relative_yaw = Rotation.from_matrix(R).as_rotvec()[2]

            # Compose: add yaws (since Z-rotations commute)
            new_yaw = wrap_angle(current_yaw + relative_yaw)

            # Compose: rotate relative translation by current orientation, then add
            current_rot = cumulative_pose[:3, :3]
            new_pos = current_pos + current_rot @ t

            # Rebuild cumulative pose
            cumulative_pose = np.eye(4)
            cumulative_pose[:3, :3] = Rotation.from_rotvec([0, 0, new_yaw]).as_matrix()
            cumulative_pose[:3, 3] = new_pos
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

        # for pose in initial_poses:
        #     rot_vec = Rotation.from_matrix(pose[:3, :3]).as_rotvec()

        #     print("initial rot_vec", rot_vec)

        # optimized_poses, marginals, quality = optimize_with_gtsam_timed(
        #     initial_poses, constraints, timestamps
        # )

        # for initial_pose, optimized_pose in zip(initial_poses, optimized_poses):
        #     rot_vec = Rotation.from_matrix(initial_pose[:3, :3]).as_rotvec()
        #     print('initial_pose rot_vec', rot_vec)
        #     rot_vec = Rotation.from_matrix(optimized_pose[:3, :3]).as_rotvec()
        #     print('optimized_pose rot_vec', rot_vec)

        optimized_poses = initial_poses

        # update points
        # Now transform points to object-centric coordinates
        extrapolated_object_points = []
        object_centric_points = []
        extrapolated_boxes = self.extrapolate_box([x.timestamp for x in self.history])
        for i, (convex_hull_obj, obj_pose) in enumerate(zip(self.history, optimized_poses)):
            # Get world points
            world_points = convex_hull_obj.original_points

            obj_pose_vector_box = PoseKalmanFilter.box_to_pose_vector(extrapolated_boxes[i])
            obj_pose_box = PoseKalmanFilter.pose_vector_to_transform(obj_pose_vector_box)

            optimized_obj_pose_vector = PoseKalmanFilter.transform_to_pose_vector(obj_pose)

            print("obj_pose_vector_box", obj_pose_vector_box)
            print("optimized_obj_pose_vector", optimized_obj_pose_vector)


            # Transform to object-centric: multiply by inverse of object pose
            world_to_object = np.linalg.inv(obj_pose)
            object_points = points_rigid_transform(world_points, world_to_object)
            object_centric_points.append(object_points)

            # world_to_object = np.linalg.inv(obj_pose_box)
            # object_points = points_rigid_transform(world_points, world_to_object)
            # extrapolated_object_points.append(object_points)

        self.object_points = np.concatenate(object_centric_points, axis=0)
        self.object_points = voxel_sampling_fast(self.object_points, 0.05, 0.05, 0.05)

        self.last_points = points_rigid_transform(self.object_points, optimized_poses[-1])
        self.last_mesh: trimesh.Trimesh = trimesh.convex.convex_hull(self.last_points)
        self.optimized_poses = optimized_poses

        undo_last_pose = np.linalg.inv(optimized_poses[-1])
        # object_mesh_vertices = points_rigid_transform(self.last_mesh.vertices.copy(), undo_last_pose)

        object_mesh = self.last_mesh.copy().apply_transform(undo_last_pose)

        self.positions = [x[:3, 3] for x in optimized_poses]

        self.optimized_boxes = self._compute_oriented_boxes(timestamps, self.optimized_poses, object_mesh)

        # init_timestamp = min(self.timestamps)*1e-9
        # measurements = np.stack([convex_hull_obj.box for convex_hull_obj in self.history], axis=0)

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
        # print(f"Compute poses box R2: {eval_r2:.2f}")

        dims_mins = self.object_points.min(axis=0)
        dims_maxes = self.object_points.max(axis=0)

        self.lwh = dims_maxes - dims_mins

        return optimized_poses


    def _compute_oriented_boxes(self, timestamps: List[int], optimized_poses: List[np.ndarray], object_mesh: trimesh.Trimesh):
        """Compute oriented bounding boxes based on object motion direction"""
        boxes = []

        for i, cur_pose in enumerate(optimized_poses):
            rotvec = Rotation.from_matrix(cur_pose[:3, :3]).as_rotvec()
            yaw = rotvec[2]

            # cur_points = points_rigid_transform(merged_object_points, cur_pose)
            cur_mesh = object_mesh.copy().apply_transform(cur_pose)

            centre = cur_mesh.centroid

            # Create oriented bounding box from 3D vertices
            box = self._points_to_oriented_box(cur_mesh.vertices, centre, yaw)
            boxes.append(box)

        return boxes

    def _compute_yaw(self, vertices_3d: np.ndarray, centre: np.ndarray, prev_centre: np.ndarray, dt: float):
        
        velocity = (centre - prev_centre) / dt
        speed = np.linalg.norm(velocity[:2])

        if speed > self.heading_speed_thresh_ms and dt >= 0.1:
            return np.arctan2(velocity[1], velocity[0])

        points_2d = vertices_3d[:, :2]
        centreed_vertices = points_2d - centre[:2]
        cov_matrix = np.cov(centreed_vertices.T)
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

        centreed_vertices = vertices_3d - centre.reshape(1, 3)

        # Rotate vertices to align with yaw=0
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        rotated_xy = centreed_vertices[:, :2] @ rotation_matrix.T

        # Compute bounding box in rotated frame
        min_x, max_x = np.min(rotated_xy[:, 0]), np.max(rotated_xy[:, 0])
        min_y, max_y = np.min(rotated_xy[:, 1]), np.max(rotated_xy[:, 1])
        min_z, max_z = np.min(centreed_vertices[:, 2]), np.max(centreed_vertices[:, 2])

        x_max_abs = np.abs(rotated_xy[:, 0]).max()
        y_max_abs = np.abs(rotated_xy[:, 1]).max()

        # Box centre and dimensions
        centre_x = (min_x + max_x) / 2
        centre_y = (min_y + max_y) / 2
        centre_z = (min_z + max_z) / 2

        # FIXED: Rotate centre back to original frame (inverse rotation)
        centre_rotated = np.array([centre_x, centre_y]) @ rotation_matrix.T

        centre_rotated = centre_rotated + centre[:2]

        # length = max_x - min_x  # Forward/backward extent
        # width = max_y - min_y   # Left/right extent  
        length = x_max_abs*2
        width = y_max_abs*2
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
            centreed_vertices = points_2d - centre[:2]
            cov_matrix = np.cov(centreed_vertices.T)
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
        
        print('measurements', measurements.shape)
        lwhs = measurements[:, 3:6]
        print("lwhs", lwhs.shape, lwhs.min(axis=0), lwhs.max(axis=0))
        print("lwhs mean", np.mean(lwhs, axis=0))

        measurements[:, 3:6] = np.mean(lwhs, axis=0).reshape(1, 3).repeat(len(measurements), 0)

        # yaws = measurements[:, 6]
        # yaws_diffs = np.diff(yaws, 1, axis=0)
        # print("yaws", yaws.shape, yaws.min(axis=0), yaws.max(axis=0))
        # print("yaws median", np.median(yaws))
        # print("yaws_diffs", yaws_diffs.shape, yaws_diffs.min(axis=0), yaws_diffs.max(axis=0))

        # # chosen_yaw = np.median([x[6] for x in optimized_boxes]) if optimized_boxes else np.median(yaws)
        # # print("chosen_yaw", chosen_yaw)
        # chosen_yaw = np.median(yaws)

        # heading = np.array([np.cos(chosen_yaw), np.sin(chosen_yaw)])
        # n_frames = len(measurements)
        # world_centres = measurements[:, :3]
        # if n_frames > 1:
        #     velocities = []
        #     for i in range(1, min(n_frames, 5)):
        #         velocity = (world_centres[i] - world_centres[i-1]) / (time_offsets[i] - time_offsets[i-1])
        #         velocities.append(velocity)

        #     velocities = np.stack(velocities, axis=0)

        #     avg_velocity = np.mean(velocities, axis=0)
        #     avg_speed = np.linalg.norm(avg_velocity[:2])
            
        #     if avg_speed > self.heading_speed_thresh_ms:
        #         heading = avg_velocity[:2]

        # chosen_yaw = np.arctan2(heading[1], heading[0])

        # measurements[:, 6] = chosen_yaw.reshape(1).repeat(len(measurements))

        positions = measurements[:, :3]
        
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
        errs = (measurements - predictions)
        r2 = r2_score(measurements, predictions, multioutput='uniform_average')
        rmse = mean_squared_error(measurements, predictions, multioutput='uniform_average')
        print(f"Box Parameter Update - R²: {r2:.3f} - rmse {rmse:.3f}")
        # print("errs", errs)

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