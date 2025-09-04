
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from lion.unsupervised_core.box_utils import compute_ppscore, icp, icp_open3d_robust
from lion.unsupervised_core.convex_hull_tracker.convex_hull_object import (
    ConvexHullObject,
)
from lion.unsupervised_core.convex_hull_tracker.convex_hull_utils import (
    relative_object_pose,
    relative_object_rotation,
    rigid_icp,
)
from lion.unsupervised_core.convex_hull_tracker.pose_kalman_filter import (
    PoseKalmanFilter,
)
from lion.unsupervised_core.outline_utils import points_rigid_transform, voxel_sampling
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

        self.state = ConvexHullTrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self.history = [convex_hull]
        self.timestamps = [convex_hull.timestamp]
        self.last_points = convex_hull.original_points
        # self.last_box = convex_hull.box

        self.last_predict_timestamp = self.timestamps[-1]

        # initialise: todo -> update
        self.object_points = convex_hull.object_points

        # hmmm could change -> but we are only tracking poses with the kalman filter
        self.lwh = convex_hull.box[3:6]

        self.positions = []
        self.positions.append(mean[:3])

        # final after global optimization
        self.merged_mesh = None
        self.optimized_poses = None
        self.optimized_boxes = None
        self.prev_pose = np.eye(4)

        self.source = convex_hull.source

        
        self.stagger_step = 2
        self.max_stagger_gap = 5
        self.icp_max_iterations = 5
        self.heading_speed_thresh_ms = 1.2 # m/s (walking pace) -> about 1.2 * 3.6 km/h

        self._n_init = n_init
        self._max_age = max_age

    def to_box(self):
        centre = self.mean[:3]
        yaw = self.mean[5]

        # x, y, z, rx, ry, rz, vx, vy, vz, vrx, vry, vrz = self.mean
        # print(f"rx={rx:.2f} ry={ry:.2f} rz={rz:.2f}")

        x, y, z = centre
        l, w, h = self.lwh

        # box = np.concatenate([centre.reshape(3), lwh.reshape(3), ry.reshape(1)], axis=0)
        box = np.array([x, y, z, l, w, h, yaw], dtype=np.float32)

        return box

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
            # world_points_per_timestamp.append(obj.mesh.vertices.copy())

            center = obj.box[:3]
            world_centers.append(center)


        world_centers = np.array(world_centers)

        times_secs = np.array(timestamps, dtype=float) * 1e-9

        # heading -> 0, 0, 0 if not moving else we calculate the average.
        heading = np.zeros((2,))
        if n_frames > 1:
            velocities = []
            for i in range(1, min(n_frames-1, 10)):
                velocity = (world_centers[i] - world_centers[i-1]) / (times_secs[i] - times_secs[i-1])
                velocities.append(velocity)

            velocities = np.stack(velocities, axis=0)

            avg_velocity = np.mean(velocities, axis=0)
            avg_speed = np.linalg.norm(avg_velocity)
            
            if avg_speed > self.heading_speed_thresh_ms:
                heading = avg_velocity[:2]

        first_object_pose = self._create_pose_from_object_points_and_heading(world_points_per_timestamp[0], heading, world_centers[0])
        initial_poses = [first_object_pose]

        constraints = []
        cumulative_pose = first_object_pose.copy()
        for i in range(1, len(timestamps)):
            undo_pose = np.linalg.inv(cumulative_pose)

            prev_points = points_rigid_transform(
                world_points_per_timestamp[i - 1], undo_pose
            )
            cur_points = points_rigid_transform(
                world_points_per_timestamp[i], undo_pose
            )

            ###############################
            R, t, A_inliers, B_inliers, icp_cost = relative_object_pose(prev_points, cur_points, max_iterations=self.icp_max_iterations)
            # Run ICP
            # R, t, A_inliers, B_inliers = icp(
            #     prev_points, cur_points,
            #     max_iterations=self.icp_max_iterations,
            #     return_inliers=True, 
            #     ret_err=False
            # )

            relative_pose = np.eye(4)
            relative_pose[:3, :3] = R
            relative_pose[:3, 3] = t
            confidence = len(A_inliers) / len(cur_points)
            ######################

            # relative_pose, _ = icp_open3d_robust(prev_points, cur_points)
            # confidence = 0.8


            cumulative_pose = cumulative_pose @ relative_pose
            initial_poses.append(cumulative_pose.copy())

            constraint = {
                'frame_i': i - 1,
                'frame_j': i, 
                'relative_pose': relative_pose,
                'confidence': confidence,
            }
            constraints.append(constraint)


        # # Try different stagger gaps: 2, 3, 4, etc.
        # for gap in range(self.stagger_step, min(self.max_stagger_gap + 1, n_frames)):
        #     for i in range(n_frames - gap):
        #         j = i + gap
                        
        #         # Use the same coordinate frame approach as consecutive poses
        #         # Transform both point sets using the reference pose at frame i
        #         undo_pose = np.linalg.inv(initial_poses[i])
                
        #         cur_points_normalized = points_rigid_transform(
        #             world_points_per_timestamp[i], undo_pose
        #         )
        #         target_points_normalized = points_rigid_transform(
        #             world_points_per_timestamp[j], undo_pose
        #         )

        #         #######################

        #         R, t, A_inliers, B_inliers, icp_cost = relative_object_pose(cur_points_normalized, target_points_normalized, max_iterations=self.icp_max_iterations)

        #         # # Run ICP in the normalized coordinate frame
        #         # R, t, A_inliers, B_inliers = icp(
        #         #     cur_points_normalized, target_points_normalized,
        #         #     max_iterations=self.icp_max_iterations,
        #         #     return_inliers=True, 
        #         #     ret_err=False
        #         # )
                
        #         # Create relative pose directly from ICP result
        #         relative_pose = np.eye(4)
        #         relative_pose[:3, :3] = R
        #         relative_pose[:3, 3] = t
        #         confidence = len(A_inliers) / len(cur_points_normalized)
        #         #######################

        #         # relative_pose, _ = icp_open3d_robust(cur_points_normalized, target_points_normalized)
        #         # confidence = 0.8

                
        #         constraint = {
        #             'frame_i': i,
        #             'frame_j': j, 
        #             'relative_pose': relative_pose,
        #             'confidence': confidence,
        #         }
        #         constraints.append(constraint)
                        


        assert len(initial_poses) == len(
            timestamps
        ), "Must have one initial pose per timestamp"
        # assert (
        #     len(constraints) > len(timestamps) 
        # ), "Should have more constraints than timestamps"

        optimized_poses, marginals, quality = optimize_with_gtsam_timed(
            initial_poses, constraints, timestamps
        )


##################################################################
        # optimized_poses_positions, marginals, quality = optimize_with_gtsam_timed_positions(
        #     world_centers, constraints, timestamps
        # )

        # positions0 = np.array([x[:3, 3] for x in optimized_poses])
        # positions1 = np.array([x[:3, 3] for x in optimized_poses_positions])

        # differences = (positions1 - positions0)

        # print(f'optimized position diffs', differences)

##################################################################

        # # get initial object points ##################################################################
        # object_centric_points = []
        # for convex_hull_obj, obj_pose in zip(self.history, optimized_poses):
        #     # Get world points
        #     world_points = convex_hull_obj.original_points

        #     # Transform to object-centric: multiply by inverse of object pose
        #     world_to_object = np.linalg.inv(obj_pose)
        #     object_points = points_rigid_transform(world_points, world_to_object)

        #     lwh = convex_hull_obj.box[3:6]

        #     mask = np.all(np.abs(object_points) <= lwh, axis=1)
        #     mask_prop = mask.sum() / max(1, len(mask))
        #     print(f"object points in box {mask_prop*100:.3f}")

        #     object_centric_points.append(object_points)

        # object_points = np.concatenate(object_centric_points, axis=0)
        
        # dims_mins = object_points.min(axis=0)
        # dims_maxes = object_points.max(axis=0)

        # object_centre_offset = (dims_mins + dims_maxes) / 2
        # print('object_centre_offset', object_centre_offset)

        # prev_positions = np.array([x[:3, 3] for x in optimized_poses])

        # offset_T = np.eye(4)
        # offset_T[:3, 3] = -object_centre_offset  # shift origin

        # corrected_poses = [pose @ offset_T for pose in optimized_poses]

        # new_positions = np.array([x[:3, 3] for x in corrected_poses])

        # world_offset = (new_positions - prev_positions).mean(axis=0)
        # print(f"world_offset", world_offset)

        # optimized_poses = corrected_poses

        # # get initial object points ##################################################################


        # centre_offsets = [self.positions[i] - world_centers[i] for i in range(n_frames)]
        # print('centre_offsets', centre_offsets)

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

        # # plot the points ##################################################################
        # colors = plt.cm.tab20(np.linspace(0, 1, len(object_centric_points)))

        # fig, ax = plt.subplots(figsize=(5, 5))

        # t0 = timestamps[0] * 1e-9
        # for idx, points in enumerate(object_centric_points):
        #     t = timestamps[idx] * 1e-9 - t0
        #     ax.scatter(
        #         points[:, 0],
        #         points[:, 1],
        #         s=4,
        #         color=colors[idx],
        #         label=f"Frame {t}",
        #         alpha=0.5,
        #     )
        # ax.set_aspect("equal")
        # ax.grid(True, alpha=0.3)
        # ax.set_xlabel("X (meters)", fontsize=12)
        # ax.set_ylabel("Y (meters)", fontsize=12)

        # save_path = Path("./convex_hull_track")
        # save_path.mkdir(exist_ok=True)
        # plt.savefig(save_path / f"{self.track_id}.png")
        # plt.close()
        # # plot the points ##################################################################

        self.object_points = np.concatenate(object_centric_points, axis=0)
        # ppscore = compute_ppscore(self.object_points, object_centric_points)
        # ppscore_mask = ppscore > min(np.median(ppscore), 0.7)
        # ppscore_prop = ppscore_mask.sum() / max(1, len(ppscore_mask))
        # print(f"ppscore_prop {ppscore_prop*100:.3f}")
        # self.object_points = self.object_points[ppscore_mask]
        self.object_points = voxel_sampling(self.object_points, 0.05, 0.05, 0.05)

        self.last_points = points_rigid_transform(self.object_points, optimized_poses[-1])
        self.optimized_poses = optimized_poses

        self.positions = [x[:3, 3] for x in optimized_poses]

        self.optimized_boxes = self._compute_oriented_boxes(timestamps, self.optimized_poses, self.object_points)

        dims_mins = self.object_points.min(axis=0)
        dims_maxes = self.object_points.max(axis=0)

        self.lwh = dims_maxes - dims_mins

        return optimized_poses

    def _compute_oriented_boxes(self, timestamps: List[int], optimized_poses: List[np.ndarray], merged_object_points: np.ndarray):
        """Compute oriented bounding boxes based on object motion direction"""
        boxes = []
        for i, cur_pose in enumerate(optimized_poses):
            rotvec = Rotation.from_matrix(cur_pose[:3, :3]).as_rotvec()
            yaw = rotvec[2]

            cur_points = points_rigid_transform(merged_object_points, cur_pose)

            # Create oriented bounding box from 3D vertices
            box = self._vertices_to_oriented_box(cur_points, yaw)
            boxes.append(box)

        return boxes

    @staticmethod
    def _vertices_to_oriented_box(vertices_3d, yaw):
        """Convert 3D vertices to oriented bounding box [x, y, z, l, w, h, yaw]"""
        if len(vertices_3d) < 3:
            return None

        # Rotate vertices to align with yaw=0
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        rotated_xy = vertices_3d[:, :2] @ rotation_matrix.T

        # Compute bounding box in rotated frame
        min_x, max_x = np.min(rotated_xy[:, 0]), np.max(rotated_xy[:, 0])
        min_y, max_y = np.min(rotated_xy[:, 1]), np.max(rotated_xy[:, 1])
        min_z, max_z = np.min(vertices_3d[:, 2]), np.max(vertices_3d[:, 2])

        # Box center and dimensions
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2

        # FIXED: Rotate center back to original frame (inverse rotation)
        center_rotated = np.array([center_x, center_y]) @ rotation_matrix.T

        length = max_x - min_x  # Forward/backward extent
        width = max_y - min_y   # Left/right extent  
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
            prev_pose = self.optimized_poses[-2]
            curr_pose = self.optimized_poses[-1]
            
            dt = (self.timestamps[-1] - self.timestamps[-2]) * 1e-9  # Convert ns to seconds
            
            # Linear velocity
            linear_vel = (curr_pose[:3, 3] - prev_pose[:3, 3]) / dt
            
            # Angular velocity (approximate using rotation vectors)
            prev_rot_vec = kf.transform_to_pose_vector(prev_pose)[3:6]
            curr_rot_vec = kf.transform_to_pose_vector(curr_pose)[3:6]
            angular_vel = (curr_rot_vec - prev_rot_vec) / dt
            
            # If we have more history, use smoother velocity estimate
            if len(self.optimized_poses) >= 3:
                linear_vels = []
                angular_vels = []
                
                for i in range(len(self.optimized_poses) - 2, len(self.optimized_poses)):
                    if i <= 0:
                        continue
                    p1 = self.optimized_poses[i-1]
                    p2 = self.optimized_poses[i]
                    dt_i = (self.timestamps[i] - self.timestamps[i-1]) * 1e-9
                    
                    if dt_i > 0:
                        linear_vels.append((p2[:3, 3] - p1[:3, 3]) / dt_i)
                        
                        r1 = kf.transform_to_pose_vector(p1)[3:6]
                        r2 = kf.transform_to_pose_vector(p2)[3:6]
                        angular_vels.append((r2 - r1) / dt_i)
                
                if linear_vels:
                    # Weighted average (more weight to recent velocities)
                    weights = np.exp(np.linspace(-1, 0, len(linear_vels)))
                    weights /= weights.sum()
                    
                    linear_vel = np.average(linear_vels, axis=0, weights=weights)
                    angular_vel = np.average(angular_vels, axis=0, weights=weights)
        else:
            # No velocity estimate available
            linear_vel = np.zeros(3)
            angular_vel = np.zeros(3)
        
        # Create new state vector with GTSAM pose + estimated velocity
        gtsam_mean = np.concatenate([latest_pose_vector, linear_vel, angular_vel])
        
        # Blend with current Kalman estimate based on confidence
        if confidence_factor > 0.99:
            # Complete replacement
            self.mean = gtsam_mean
        else:
            # Weighted blend
            self.mean = confidence_factor * gtsam_mean + (1 - confidence_factor) * self.mean
        
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

        points_2d = points3d[:, :2]

        if not use_heading:
            # Estimate orientation using PCA
            centered_vertices = points_2d - centre[:2]
            cov_matrix = np.cov(centered_vertices.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            priority_scores = []

            for i in range(len(eigenvalues)):
                eigenvector = eigenvectors[:, i]
                dot_prod = np.dot(eigenvector, heading_2d)

                priority_scores.append((np.abs(dot_prod), eigenvalues[i]))

            idx = np.array(sorted([0, 1], key=lambda i: priority_scores[i], reverse=True))
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            primary_direction = eigenvectors[:, 0]  # Length direction
            # secondary_direction = eigenvectors[:, 1]  # Width direction

            primary_dot = np.dot(primary_direction, heading_2d)
            sign = 1 if primary_dot > 0 else -1

            primary_direction *= sign
            
            # Calculate yaw from primary direction
            yaw = np.arctan2(primary_direction[1], primary_direction[0])
            
        else:
            yaw = np.arctan2(heading_2d[1], heading_2d[0])

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

        # print(f"update {self.last_points=} {convex_hull.original_points=}")
        # print(f"update {self.last_points.shape=} {convex_hull.original_points.shape=}")

        # Transform to reference frame
        prev_points = points_rigid_transform(self.last_points, undo_pose)
        cur_points = points_rigid_transform(convex_hull.original_points, undo_pose)

        #######################################################
        # centre1 = np.mean(prev_points, axis=0)
        # centre2 = np.mean(cur_points, axis=0)

        # R, t_centered_2, _, _, icp_cost = relative_object_pose(
        #     prev_points - centre1, 
        #     cur_points - centre2, 
        #     max_iterations=5, 
        #     debug=False
        # )

        # t = (centre2 - centre1) + t_centered_2

        # rel_pose = np.eye(4)
        # rel_pose[:3, :3] = R
        # rel_pose[:3, 3] = t
        #######################################################


        # R, t, _ = icp(prev_points, cur_points, max_iterations=5)

        #######################################################
        R, t, _, _, icp_cost = relative_object_pose(
            prev_points, 
            cur_points, 
            max_iterations=5, 
            debug=False
        )
        
        # Build relative pose in reference frame
        rel_pose = np.eye(4)
        rel_pose[:3, :3] = R
        rel_pose[:3, 3] = t

        #######################################################
        # R, t, _ = icp(prev_points, cur_points, max_iterations=20)

        # # Build relative pose in reference frame
        # rel_pose = np.eye(4)
        # rel_pose[:3, :3] = R
        # rel_pose[:3, 3] = t
        #######################################################


        #######################################################

                            # prev_pose = self.prev_pose
                            # convex_hull_pose = kf.pose_vector_to_transform(kf.box_to_pose_vector(convex_hull.box))

                            # print("prev_pose" , kf.transform_to_pose_vector(prev_pose))
                            # print("convex_hull_pose", kf.transform_to_pose_vector(convex_hull_pose))

                            # rel_pose = np.linalg.inv(convex_hull_pose) @ prev_pose
        #######################################################

        cumulative_pose = prev_pose @ rel_pose
        cumulative_pose_vector = kf.transform_to_pose_vector(cumulative_pose)

        # create new last points
        #######################################################
        # last_points = self.last_points.copy()
        # last_points = points_rigid_transform(last_points, rel_pose)

        # self.last_points = np.concatenate([last_points, convex_hull.original_points], axis=0)
        #######################################################

        self.last_points = convex_hull.original_points

        # do not update -> handled by gtsam sync?
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, cumulative_pose_vector, is_relative=False)

        self.features.append(convex_hull.feature)
        self.history.append(convex_hull)
        self.timestamps.append(convex_hull.timestamp)

        # use current points for the pose
        world_to_object = np.linalg.inv(cumulative_pose)
        cur_object_points = points_rigid_transform(convex_hull.original_points, world_to_object)

        self.object_points = cur_object_points

        dims_mins = cur_object_points.min(axis=0)
        dims_maxes = cur_object_points.max(axis=0)

        lwh = dims_maxes - dims_mins
        self.lwh = lwh

        # self.lwh = 0.5 * self.lwh + convex_hull.box[3:6] * 0.5

        assert len(self.lwh) == 3, f"lwh={self.lwh}"

        self.mark_hit()

        # After every few measurements, sync with GTSAM
        if len(self.history) >= 3 and len(self.history) % 3 == 0:
            self.optimized_poses = self.compute_poses()

            self.sync_kalman_with_gtsam(kf, confidence_factor=1.0)

        self.prev_pose = self.to_pose_matrix()


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
        confidence = np.mean(confidences)

        features = np.stack(self.features, axis=0)
        mean_feature = np.average(features, axis=0, weights=confidences)
        mean_feature = mean_feature / (1e-6 + np.linalg.norm(mean_feature))

        convex_hull = ConvexHullObject(
            original_points=points3d.copy(),
            confidence=confidence,
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