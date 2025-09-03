
from pathlib import Path
from typing import Dict, List
import numpy as np
import trimesh
from lion.unsupervised_core.box_utils import icp
from lion.unsupervised_core.convex_hull_tracker.convex_hull_object import ConvexHullObject
from lion.unsupervised_core.convex_hull_tracker.convex_hull_utils import relative_object_pose, relative_object_rotation, rigid_icp
from lion.unsupervised_core.convex_hull_tracker.pose_kalman_filter import PoseKalmanFilter
from lion.unsupervised_core.outline_utils import points_rigid_transform, voxel_sampling
from scipy.spatial import cKDTree
from lion.unsupervised_core.trajectory_optimizer import optimize_with_gtsam_timed
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

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
        self.last_box = convex_hull.box

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

        self.source = convex_hull.source

        
        self.stagger_step = 2
        self.max_stagger_gap = 5
        self.max_icp_iterations = 5

        self._n_init = n_init
        self._max_age = max_age

    def to_box(self):
        centre = self.mean[:3]
        ry = self.mean[4]

        lwh = self.lwh

        x, y, z = centre
        l, w, h = self.lwh

        # box = np.concatenate([centre.reshape(3), lwh.reshape(3), ry.reshape(1)], axis=0)
        box = np.array([x, y, z, l, w, h, ry], dtype=np.float32)

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
        linear_vel = self.mean[6:9]   # vx, vy, vz
        angular_vel = self.mean[9:12] # vrx, vry, vrz

        debug = np.linalg.norm(linear_vel) > 0.1  # Use correct linear_vel

        if debug:
            print(f"convex_hull_track debug, linear_vel", linear_vel)
            print('angular_vel', angular_vel)
            print(f"kf._motion_mat", kf._motion_mat)

            last_timestamp = self.last_predict_timestamp
            dt = (float(timestamp) - float(last_timestamp)) * 1e-9
            print(f"dt = {dt:.3f} seconds (should be 0.1?)")

            updated_mean = np.dot(kf._motion_mat, self.mean)

            print("last mean", self.mean)
            print("updated_mean", updated_mean)



        self.last_predict_timestamp = timestamp

        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

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

            center = obj.box[:3]
            world_centers.append(center)

        world_centers = np.array(world_centers)

        # Compute initial heading from velocity
        if len(world_centers) > 1:
            initial_velocity = world_centers[1] - world_centers[0]
            speed = np.linalg.norm(initial_velocity) / (
                timestamps[1] - timestamps[0]
            )
            if speed > 0.1:  # Has significant motion
                heading = initial_velocity / speed
            else:
                heading = np.array([1, 0, 0])  # Default heading
        else:
            heading = np.array([1, 0, 0])

        # Build first object pose with aligned heading
        first_object_pose = self._create_pose_from_heading(
            world_centers[0], heading
        )
        initial_poses = [first_object_pose]

        relative_poses = []
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

            # R, t, _, _, icp_cost = relative_object_pose(prev_points, cur_points, max_iterations=5, debug=False)
            # Run ICP
            R, t, A_inliers, B_inliers = icp(
                prev_points, target_points,
                max_iterations=self.max_icp_iterations,
                return_inliers=True, 
                ret_err=False
            )

            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = t


            cumulative_pose = cumulative_pose @ transform
            initial_poses.append(cumulative_pose)

            constraint = {
                'frame_i': i - 1,
                'frame_j': i, 
                'relative_pose': relative_pose,
                'confidence': len(A_inliers) / len(cur_points),
            }
            constraints.append(constraint)


        # Try different stagger gaps: 2, 3, 4, etc.
        for gap in range(self.stagger_step, min(self.max_stagger_gap + 1, n_frames)):
            for i in range(n_frames - gap):
                j = i + gap
                
                cur_points = world_points_per_timestamp[i] 
                target_points = world_points_per_timestamp[j]
                
                # Initialize with accumulated pose from pairwise stage
                initial_relative = np.linalg.inv(initial_poses[i]) @ initial_poses[j]
                
                # Apply initial transform to get better ICP starting point
                transformed_cur = points_rigid_transform(cur_points, initial_relative)
                
                # Run ICP
                R, t, A_inliers, B_inliers = icp(
                    transformed_cur, target_points,
                    max_iterations=self.max_icp_iterations,
                    return_inliers=True, 
                    ret_err=False
                )
                
                # Create relative pose
                icp_pose = np.eye(4)
                icp_pose[:3, :3] = R
                icp_pose[:3, 3] = t
                
                # Combine with initial estimate
                relative_pose = initial_relative @ icp_pose
                
                constraint = {
                    'frame_i': i,
                    'frame_j': j, 
                    'relative_pose': relative_pose,
                    'confidence': len(A_inliers) / len(cur_points),
                }
                constraints.append(constraint)
                


        assert len(initial_poses) == len(
            timestamps
        ), "Must have one initial pose per timestamp"
        assert (
            len(constraints) == len(timestamps) - 1
        ), "Should have N-1 constraints for N poses"

        optimized_poses, marginals, quality = optimize_with_gtsam_timed(
            initial_poses, constraints, timestamps
        )

        self.positions = [x[:3, 3] for x in optimized_poses]
        self.last_points = world_points_per_timestamp[-1]
        self.optimized_poses = optimized_poses

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
        self.object_points = voxel_sampling(self.object_points, 0.05, 0.05, 0.05)

        object_boxes = self._compute_oriented_boxes(timestamps, self.optimized_poses, self.object_points)



        self.last_box = object_boxes[-1]
        self.lwh = self.last_box[3:6]

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

        # Rotate center back to original frame
        center_rotated = np.array([center_x, center_y]) @ rotation_matrix

        length = max_x - min_x
        width = max_y - min_y
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
        
        print(f"Track {self.track_id}: Synced Kalman with GTSAM - pose updated, uncertainty reduced by {confidence_factor*50:.1f}%")
        print(f"Updated mean=", np.round(self.mean, 2), "orig_mean", np.round(orig_mean, 2))

    def _create_pose_from_heading(self, position, heading_vector):
        """Create a pose matrix with given position and heading direction."""
        pose = np.eye(4)
        pose[:3, 3] = position

        # Build rotation matrix where x-axis aligns with heading
        forward = heading_vector / np.linalg.norm(heading_vector)

        # Choose up vector (z-axis up in world)
        up_world = np.array([0, 0, 1])

        # Compute right vector
        right = np.cross(forward, up_world)
        if np.linalg.norm(right) > 0.01:
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)

            # Rotation matrix [forward, right, up]
            pose[:3, :3] = np.column_stack([forward, right, up])
        else:
            # Handle case where heading is vertical
            if abs(forward[0]) < 0.9:
                right = np.array([1, 0, 0])
            else:
                right = np.array([0, 1, 0])
            right = right - np.dot(right, forward) * forward
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            pose[:3, :3] = np.column_stack([forward, right, up])

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
        prev_pose = self.optimized_poses[-1] if self.optimized_poses is not None else self.to_pose_matrix()
        undo_pose = np.linalg.inv(prev_pose)

        # Transform to reference frame
        prev_points = points_rigid_transform(self.last_points, undo_pose)
        cur_points = points_rigid_transform(convex_hull.original_points, undo_pose)

        centre1 = prev_points.mean(axis=0)
        centre2 = cur_points.mean(axis=0)

        dist = np.linalg.norm(centre1 - centre2)
        vec = centre2 - centre1
        vec_norm = vec / (dist + 1e-6)

        # R = relative_object_rotation(
        # R, t_centered_2, _, _, icp_cost = relative_object_pose(
        #     prev_points - centre1, 
        #     cur_points - centre2, 
        #     max_iterations=5, 
        #     debug=True
        # )

        # t_wo_centre = (centre2 - centre1)

        # t = (centre2 - centre1) + t_centered_2

        R, t, _ = icp(prev_points, cur_points, max_iterations=5)

        # Build relative pose in reference frame
        rel_pose = np.eye(4)
        rel_pose[:3, :3] = R
        rel_pose[:3, 3] = t

        cumulative_pose = prev_pose @ rel_pose
        cumulative_pose_vector = PoseKalmanFilter.transform_to_pose_vector(cumulative_pose)


        # create new last points
        last_points = self.last_points.copy()
        last_points = points_rigid_transform(last_points, rel_pose)

        self.last_points = np.concatenate([last_points, convex_hull.original_points], axis=0)

        prev_semantics = []
        for x in self.features:
            prev_semantics.append(np.dot(x, convex_hull.feature))

        # do not update -> handled by gtsam sync
        # self.mean, self.covariance = kf.update(
            # self.mean, self.covariance, cumulative_pose_vector, is_relative=False)
        self.features.append(convex_hull.feature)


        self.history.append(convex_hull)
        self.timestamps.append(convex_hull.timestamp)

        # use current points for the pose
        # world_to_object = np.linalg.inv(cumulative_pose)
        # cur_object_points = points_rigid_transform(self.last_points.copy(), world_to_object)

        # self.object_points = cur_object_points

        # dims_mins = self.object_points.min(axis=0)
        # dims_maxes = self.object_points.max(axis=0)

        # self.lwh = dims_maxes - dims_mins

        assert len(self.lwh) == 3, f"lwh={self.lwh}"

        self.last_box = self.to_box()

        self.hits += 1
        self.time_since_update = 0
        if self.state == ConvexHullTrackState.Tentative and self.hits >= self._n_init:
            self.state = ConvexHullTrackState.Confirmed

        self.optimized_poses = self.compute_poses()

        # After every few measurements, sync with GTSAM
        if len(self.history) >= 3 and len(self.history) % 3 == 0:
            self.sync_kalman_with_gtsam(kf, confidence_factor=1.0)

    def update_(self, kf: PoseKalmanFilter, convex_hull: ConvexHullObject):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        conxex_hull : ConvexHullObject
            The associated conxex_hull.

        """
        # if dist > 5.0:
        #     print(f"{dist=}")
        #     self.mark_missed()
        #     return

        # R, t, icp_cost = icp(self.last_points, convex_hull.original_points, max_iterations=5, ret_err=True)

        cur_pose = self.to_pose_matrix()
        undo_pose = np.linalg.inv(cur_pose)

        # Transform to reference frame
        prev_points = points_rigid_transform(self.last_points, undo_pose)
        cur_points = points_rigid_transform(convex_hull.original_points, undo_pose)

        centre1 = prev_points.mean(axis=0)
        centre2 = cur_points.mean(axis=0)

        dist = np.linalg.norm(centre1 - centre2)
        vec = centre2 - centre1
        vec_norm = vec / (dist + 1e-6)

            # # Center both point clouds by centre1 for ICP
            # prev_points_centered = prev_points - centre1
            # cur_points_centered = cur_points - centre2

            # # ICP in centered space
            # R, t_centered, _, _, icp_cost = relative_object_pose(
            #     prev_points_centered, 
            #     cur_points_centered, 
            #     max_iterations=5, 
            #     debug=True
            # )

            # # Convert transformation back to reference frame
            # # In centered space: cur_centered = R @ prev_centered + t_centered
            # # In ref frame: cur - centre1 = R @ (prev - centre1) + t_centered
            # # Therefore: cur = R @ prev + (centre1 - R @ centre1 + t_centered)
            # t = t_centered + centre1 - R @ centre1  # <-- THIS IS THE KEY LINE!

            #     print('t', np.round(t, 2))
            #     print('t_centered', np.round(t_centered, 2))

            #     t = t_centered

        # R = relative_object_rotation(
        R, t_centered_2, _, _, icp_cost = relative_object_pose(
            prev_points - centre1, 
            cur_points - centre2, 
            max_iterations=5, 
            debug=True
        )

        t_wo_centre = (centre2 - centre1)
        # t = centre2 - R @ centre1

        t = (centre2 - centre1) + t_centered_2

        # Build relative pose in reference frame
        rel_pose = np.eye(4)
        rel_pose[:3, :3] = R
        rel_pose[:3, 3] = t

        # Update cumulative pose
        # cumulative_pose = rel_pose @ cur_pose  
        cumulative_pose = cur_pose @ rel_pose

        cur_pose_vector = PoseKalmanFilter.transform_to_pose_vector(cur_pose)
        rel_pose_vector = PoseKalmanFilter.transform_to_pose_vector(rel_pose)
        cumulative_pose_vector = PoseKalmanFilter.transform_to_pose_vector(cumulative_pose)
        self.positions.append(cumulative_pose_vector[:3])

        # Debug output
        t_norm = t / (np.linalg.norm(t) + 1e-6)
        t_dot = np.dot(vec_norm, t_norm)

        print("t_dot=", np.round(t_dot, 2))
        print('vec_norm', np.round(vec_norm, 2))
        print('t_norm', np.round(t_norm, 2))

        print("t_w_centres", np.round(t_wo_centre, 2))
        print("t", np.round(t, 2))
        print('vec_norm', np.round(vec_norm, 2))
        print('t_norm', np.round(t_norm, 2))
        print(f"cur_pose_vector", np.round(cur_pose_vector, 2))
        print(f"cumulative_pose_vector", np.round(cumulative_pose_vector, 2))
        print(f"rel_pose_vector", np.round(rel_pose_vector, 2))
        print(f"R", np.round(R, 2))
        # assert t_dot > 0.5, f"{t_dot=} should be > 0.5!"

        # create new last points
        last_points = self.last_points.copy()
        last_points = points_rigid_transform(last_points, rel_pose)

        self.last_points = np.concatenate([last_points, convex_hull.original_points], axis=0)

        # tree = cKDTree(convex_hull.original_points)
        # distances, n_indices = tree.query(last_points)

        # print("last_points distances", distances.min(), distances.mean(), distances.max())
        # inliers_mask = (distances < 0.25)

        # # keep inliers of each


        prev_semantics = []
        for x in self.features:
            prev_semantics.append(np.dot(x, convex_hull.feature))

        # print(f"{prev_semantics=}")

        # print("prev mean", np.round(self.mean[:6], 2))
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, cumulative_pose_vector, is_relative=False)
        self.features.append(convex_hull.feature)

        # print("updated mean", np.round(self.mean[:6],2))



        self.history.append(convex_hull)
        self.timestamps.append(convex_hull.timestamp)


        # use current points for the pose
        cur_pose = self.to_pose_matrix()
        world_to_object = np.linalg.inv(cur_pose)
        cur_object_points = points_rigid_transform(convex_hull.original_points.copy(), world_to_object)

        # fig, ax = plt.subplots(figsize=(5, 5))

        # ax.scatter(
        #     self.object_points[:, 0],
        #     self.object_points[:, 1],
        #     s=3,
        #     c="blue",
        #     label=f"self.object_points",
        #     alpha=1.0,
        # )
        # ax.scatter(
        #     cur_object_points[:, 0],
        #     cur_object_points[:, 1],
        #     s=3,
        #     c="green",
        #     label=f"cur_object_points",
        #     alpha=1.0,
        # )
        # ax.set_aspect("equal")
        # ax.grid(True, alpha=0.3)
        # ax.set_xlabel("X (meters)", fontsize=12)
        # ax.set_ylabel("Y (meters)", fontsize=12)

        # ax.legend()

        # save_path = Path("./convex_hull_track")
        # save_path.mkdir(exist_ok=True)
        # plt.savefig(save_path / f"track_{self.track_id}_{self.hits}.png")
        # plt.close()

        # self.object_points = np.concatenate([self.object_points, cur_object_points], axis=0)
        self.object_points = cur_object_points


        dims_mins = self.object_points.min(axis=0)
        dims_maxes = self.object_points.max(axis=0)

        self.lwh = dims_maxes - dims_mins

        assert len(self.lwh) == 3, f"lwh={self.lwh}"

        self.last_box = self.to_box()

        # print("new last_box", np.round(self.last_box, 2))

        # probably should do otherwise...
        # self.object_points = np.concatenate([self.object_points, convex_hull.object_points], axis=0)

        # print("hits before", self.hits)
        self.hits += 1
        self.time_since_update = 0
        if self.state == ConvexHullTrackState.Tentative and self.hits >= self._n_init:
            self.state = ConvexHullTrackState.Confirmed

        # print(f"Updated {self.track_id=} {self.hits=} {self.state}")

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