
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.interpolate import UnivariateSpline, splev, splprep
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from shapely.geometry import MultiPoint
from sklearn.covariance import EmpiricalCovariance
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
    compute_spline_poses,
    create_spline_fitness_function,
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

        self.n_rop_better = 0
        self.n_o3d_better = 0

        self.trees = [cKDTree(convex_hull.original_points)]
        self.history = [convex_hull]
        self.ellipses = [self.fit_ellipse_to_points(convex_hull.original_points, convex_hull.mesh.centroid)]
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
        self.path_box = None
        self.spline_boxes = None
        self.ellipse_size_splines = None
        self.ellipse_pos_model = None
        self.ellipse_theta_model = None
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

    # def extrapolate_box(self, timestamps):

    #     boxes = self.predict_ellipse_motion_model(timestamps)

    #     boxes[:, 3:5] *= 2

    #     assert boxes[:, 3:5].shape[1] == 2

    #     return boxes

    def optimized_motion_box(self, history=False):
        times_secs = np.array(self.timestamps, dtype=np.float64) * NANOSEC_TO_SEC
        time_offsets = times_secs - times_secs[0]  # Make relative to first timestamp
        
        if len(times_secs) < 3:
            return None

        try:
            A = np.column_stack([
                np.ones(len(time_offsets)),
                time_offsets,
                0.5 * time_offsets**2
            ])

            if not history:
                measurements = np.stack(self.optimized_boxes, axis=0)
            else:
                measurements = np.stack([obj.box for obj in self.history], axis=0)

            coeffs = np.linalg.lstsq(A, measurements, rcond=None)[0]  # [p0, v0, a] for each dimension

            t = self.last_predict_timestamp * NANOSEC_TO_SEC - times_secs[0]
            prediction = (coeffs[0] + coeffs[1] * t + 0.5 * coeffs[2] * t**2)

            return prediction
        except Exception as e:
            print(f"optimized_motion_box: {e}")
            return None


    def to_box(self):
        A = self.A
        c = self.c

        timestamp = self.last_predict_timestamp
        init_timestamp = min(self.timestamps)*NANOSEC_TO_SEC

        new_timestamps = np.array([timestamp*NANOSEC_TO_SEC - init_timestamp], float)
        new_prediction = A[np.newaxis, :] * new_timestamps[:, np.newaxis] + c[np.newaxis, :]

        return new_prediction.reshape(7)

    def to_box(self):
        optimized_box = self.optimized_motion_box()

        if optimized_box is None:
            A = self.A
            c = self.c

            timestamp = self.last_predict_timestamp
            init_timestamp = min(self.timestamps)*NANOSEC_TO_SEC

            new_timestamps = np.array([timestamp*NANOSEC_TO_SEC - init_timestamp], float)
            new_prediction = A[np.newaxis, :] * new_timestamps[:, np.newaxis] + c[np.newaxis, :]

            return new_prediction.reshape(7)
        else:
            return optimized_box

    # def to_box(self):
    #     # return self.optimized_boxes[-1] # last seen box...
    #     # return self.extrapolate_box([self.last_predict_timestamp])[0]

    #     cx, cy, cz, a, b, h, theta = self.predict_ellipse_motion_model([self.last_predict_timestamp])[0]

    #     # length is diameter whilst a,b  are radius / semi-axis
    #     l = 2*a
    #     w = 2*b

    #     box = np.array([cx, cy, cz, l, w, h, theta], np.float32)

    #     return box

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

    def extrapolate_ellipse_box(self, timestamps: List[int]):
        boxes = self.predict_ellipse_motion_model([self.last_predict_timestamp])

        boxes[:, 3:5] *= 2

        return boxes

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

    def to_bev_hull(self):
        pose_vec = PoseKalmanFilter.box_to_pose_vector(self.to_box())
        transform = PoseKalmanFilter.pose_vector_to_transform(pose_vec)

        points = points_rigid_transform(self.object_points, transform)

        z_min, z_max = points[:, 2].min(), points[:, 2].max()

        return {
            "hull": MultiPoint(points).convex_hull,
            "z_min": z_min,
            "z_max": z_max
        }


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
            R_cached, t_cached, cost, iou, center_old = self.icp_cache[cache_key]
            # Transform translation for new center
            delta = center_old - center
            t_new = t_cached + (np.eye(3) - R_cached) @ delta
            return R_cached, t_new, cost, iou
        
        # Compute and cache
        centered_i = points_i - center
        centered_j = points_j - center
        R, t, inliers, _, cost = relative_object_pose(centered_i, centered_j)

        shape1_points = (R @ centered_i.copy().T).T + t
        shape2_points = centered_j

        # Project to BEV (XY plane only)
        shape1_bev = shape1_points[:, :2]  # Take only X,Y coordinates
        shape2_bev = shape2_points[:, :2]

        hull1 = MultiPoint(shape1_bev).convex_hull
        hull2 = MultiPoint(shape2_bev).convex_hull

        # Calculate intersection
        z1_min, z1_max = shape1_points[:, 2].min(), shape1_points[:, 2].max()
        z2_min, z2_max = shape2_points[:, 2].min(), shape2_points[:, 2].max()
        intersection_min = max(z1_min, z2_min)
        intersection_max = min(z1_max, z2_max)
        intersection_height = max(0, intersection_max - intersection_min)

        # Calculate union
        union_min = min(z1_min, z2_min)
        union_max = max(z1_max, z2_max)
        union_height = union_max - union_min

        z_iou = 0.0

        if union_height == 0:
            z_iou = 1.0  # Both have same z
        
        else:
            z_iou = intersection_height / union_height

        bev_iou = 0.0
        # Handle degenerate cases (points, lines)
        if hull1.area == 0 or hull2.area == 0:
            bev_iou = 0.0
        else:

            # Compute intersection and union
            intersection = hull1.intersection(hull2).area
            union = hull1.union(hull2).area

            # Return IoU
            bev_iou = intersection / union if union > 0 else 0.0

        iou = bev_iou * z_iou

        self.icp_cache[cache_key] = (R, t, cost, iou, center.copy())
        return R, t, cost, iou

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

            # add our pose here
            pose_vector = PoseKalmanFilter.box_to_pose_vector(boxes[i])
            pose_matrix = PoseKalmanFilter.pose_vector_to_transform(pose_vector)
            initial_poses.append(pose_matrix)
            points_list.append(world_points)

        optimized_poses = [ref_pose_matrix]

        last_optimized_yaw = np.copy(last_yaw)
        last_optimized_position = np.copy(last_position)

        constraints = []
        for i in range(1, n_frames):
            # rel_R, rel_pos, A_inliers, B_inliers, icp_cost_rel = relative_object_pose(centered_ref, centered_curr1)
            R_rel, t_rel, icp_cost_rel, iou_rel = self.get_or_compute_relative_pose(
                points_list[i-1], points_list[i], 
                last_optimized_position,
                (i-1, i, 'relative')
            )
    
            rel_yaw = Rotation.from_matrix(R_rel).as_rotvec()[2]
            rel_yaw_guess = wrap_angle(last_optimized_yaw + rel_yaw)

            if i == 1:
                R_init, t_init, icp_cost_initial, iou_initial = R_rel, t_rel, icp_cost_rel, iou_rel
            else:
                R_init, t_init, icp_cost_initial, iou_initial = self.get_or_compute_relative_pose(
                    points_list[0], points_list[i], 
                    initial_position,
                    (0, i, 'initial')
                )

            initial_yaw_guess = wrap_angle(initial_yaw + Rotation.from_matrix(R_init).as_rotvec()[2])

            # pose guesses
            rel_pos_guess = last_optimized_position + t_rel
            initial_pos_guess = initial_position + t_init

            # Compute current velocities
            dt = (timestamps[i] - timestamps[i-1]) * 1e-9
            velocity_rel = (rel_pos_guess - last_optimized_position) / dt
            velocity_init = (initial_pos_guess - last_optimized_position) / dt

            # Smoothness penalties
            lambda_smooth = 0.1  # Tunable parameter
            smooth_cost_rel = lambda_smooth * np.linalg.norm(velocity_rel - last_velocity)**2 
            smooth_cost_init = lambda_smooth * np.linalg.norm(velocity_init - last_velocity)**2

            iou_cost_rel = (1.0 - iou_rel)
            iou_cost_initial = (1.0 - iou_initial)

            # weight between the two
            total_cost_rel = icp_cost_rel + smooth_cost_rel + iou_cost_rel
            total_cost_init = icp_cost_initial + smooth_cost_init + iou_cost_initial
            weight_rel = 1.0 / (total_cost_rel + 1e-6)
            weight_initial = 1.0 / (total_cost_init + 1e-6)

            total_weight = weight_rel + weight_initial
            weight_rel /= total_weight
            weight_initial /= total_weight

            print(f"{weight_rel=} {weight_initial=}")

            optimized_position = rel_pos_guess * weight_rel + initial_pos_guess * weight_initial
            optimized_yaw = circular_weighted_mean(
                np.array([rel_yaw_guess, initial_yaw_guess]), 
                np.array([weight_rel, weight_initial])
            )

            velocity = (optimized_position - last_optimized_position) / dt
            speed = np.linalg.norm(velocity[:2])

            if speed < self.heading_speed_thresh_ms:
                optimized_yaw = last_optimized_yaw


            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_rotvec(np.array([0.0, 0.0, optimized_yaw], float)).as_matrix()
            pose[:3, 3] = optimized_position
            optimized_poses.append(pose)

            # Update for next iteration
            last_velocity = (optimized_position - last_optimized_position) / dt
            last_angular_velocity = (optimized_yaw - last_optimized_yaw) / dt

            last_optimized_position = np.copy(optimized_position)
            last_optimized_yaw = np.copy(optimized_yaw)

        # smoothing to fix icp accumulated errors
        optimized_poses = smooth_trajectory_with_vehicle_model(self.timestamps, optimized_poses, heading_speed_thresh=self.heading_speed_thresh_ms)

        # Usage with CMA-ES
        total_time_secs = (max(self.timestamps) - min(self.timestamps)) * NANOSEC_TO_SEC
        n_knots = max(2, int(total_time_secs))  # every second?
        print("n_knots", n_knots)
        n_params = 6 * n_knots  # 6 splines × n_knots each

        fitness_func = create_spline_fitness_function(timestamps, optimized_poses, boxes, self.lwh, n_knots)

        # Initialize with small random coefficients (smooth trajectories)
        initial_guess = np.random.normal(0, 0.05, n_params)
        sigma = 0.2

        import cma

        es = cma.CMAEvolutionStrategy(initial_guess, sigma)
        es.optimize(fitness_func, iterations=50, maxfun=50)

        print("result", es.result)
        xbest = es.result.xbest

        self.spline_poses = compute_spline_poses(self.timestamps, optimized_poses, xbest, n_knots)

        self.optimized_poses = optimized_poses

        optimized_boxes = []
        spline_boxes = []
        for i in range(n_frames):
            world_points = self.history[i].mesh.vertices

            # Transform to object-centric: multiply by inverse of object pose
            world_to_object = np.linalg.inv(optimized_poses[i])
            object_points = points_rigid_transform(world_points, world_to_object)

            dims_mins, dims_maxes = object_points.min(axis=0), object_points.max(axis=0)
            l, w, h = dims_maxes - dims_mins

            yaw = Rotation.from_matrix(optimized_poses[i][:3, :3]).as_rotvec()[2]
            x, y, z = optimized_poses[i][:3, 3]
            box = np.array([x, y, z, l, w, h, yaw])
            optimized_boxes.append(box)

            # Transform to object-centric: multiply by inverse of object pose
            world_to_object = np.linalg.inv(self.spline_poses[i])
            object_points = points_rigid_transform(world_points, world_to_object)

            dims_mins, dims_maxes = object_points.min(axis=0), object_points.max(axis=0)
            l, w, h = dims_maxes - dims_mins

            yaw = Rotation.from_matrix(self.spline_poses[i][:3, :3]).as_rotvec()[2]
            x, y, z = self.spline_poses[i][:3, 3]
            box = np.array([x, y, z, l, w, h, yaw])
            spline_boxes.append(box)

        self.spline_boxes = spline_boxes
        self.optimized_boxes = optimized_boxes

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

        dims_mins, dims_maxes = self.object_points.min(axis=0), self.object_points.max(axis=0)
        self.lwh = dims_maxes - dims_mins

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
            # centred_vertices = points_2d - centre[:2]
            # cov_matrix = np.cov(centred_vertices.T)
            # eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # # priority_scores = []

            # # for i in range(len(eigenvalues)):
            # #     eigenvector = eigenvectors[:, i]
            # #     dot_prod = np.dot(eigenvector, heading_2d)

            # #     priority_scores.append((np.abs(dot_prod), eigenvalues[i]))

            # # idx = np.array(sorted([0, 1], key=lambda i: priority_scores[i], reverse=True))
            # idx = np.argsort(eigenvalues)[::-1]
            # eigenvalues = eigenvalues[idx]
            # eigenvectors = eigenvectors[:, idx]

            # primary_direction = eigenvectors[:, 0]  # Length direction
            # # secondary_direction = eigenvectors[:, 1]  # Width direction

            # # Calculate yaw from primary direction
            # yaw = np.arctan2(primary_direction[1], primary_direction[0])
                        
            box = ConvexHullObject.points_to_bounding_box(points3d, centre)
            yaw = box[6]
            pose[:3, 3] = box[:3]

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

    def fit_ellipse_to_points(self, points_3d, center):
        """Fit oriented ellipse to 2D points, returns [cx, cy, a, b, theta]"""

        points_2d = points_3d[:, :2]  
        # Use provided center consistently
        points_centered = points_2d - center[:2]

        # Height could be more robust
        h = np.percentile(points_3d[:, 2], 95) - np.percentile(points_3d[:, 2], 5)
        
        cov = EmpiricalCovariance().fit(points_centered)
        # center = np.mean(points_2d, axis=0)
        
        # Eigendecomposition for orientation and axes
        eigenvals, eigenvecs = np.linalg.eigh(cov.covariance_)
        
        # 95% confidence ellipse (adjust multiplier as needed)
        confidence = 2.447  # ~95% for 2D
        a, b = confidence * np.sqrt(eigenvals[::-1])  # Semi-axes
        theta = np.arctan2(eigenvecs[1, -1], eigenvecs[0, -1])  # Orientation

        return np.array([center[0], center[1], center[2], a, b, h, theta])

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
        ellipse = self.fit_ellipse_to_points(convex_hull.original_points, convex_hull.mesh.centroid)
        self.ellipses.append(ellipse)

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
        self.update_ellipse_with_motion_constraints()

    def update_ellipse_with_motion_constraints(self):
        """Use vehicle motion model constraints"""
        time_offsets = np.array(self.timestamps, dtype=np.float64) * NANOSEC_TO_SEC
        time_offsets = time_offsets - time_offsets[0]  # Make relative to first timestamp
        measurements = np.array(self.ellipses, np.float32)
        
        # Assume ellipse size (a,b) changes slowly, position follows motion model
        # Split into: position (cx,cy,theta) with dynamics, size (a,b) with smooth evolution
        
        positions = measurements[:, :3]  # cx, cy, cz
        orientations = measurements[:, 6]  # theta
        sizes = measurements[:, 3:6]  # a, b, h
        
        n_points = len(measurements)
        # Smooth sizes with splines (they should change gradually)
        size_splines = []
        for i in range(3):
            spline = UnivariateSpline(time_offsets, sizes[:, i], s=0.05, k=min(3, n_points-1))
            size_splines.append(spline)


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
                directions = np.stack([np.cos(original_yaws), np.sin(original_yaws)], axis=1)
                primary_direction = np.mean(directions, axis=0)

                primary_yaw = np.arctan2(primary_direction[1], primary_direction[0])

                smoothed_yaws[:] = primary_yaw
            
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
                # yaws = np.mean(measurements[:, 6]).reshape(1).repeat(len(measurements))

                directions = np.stack([np.cos(measurements[:, 6]), np.sin(measurements[:, 6])], axis=1)
                # cov_matrix = np.cov(directions.T)
                # eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                # # Primary direction (largest eigenvalue)
                # primary_direction = eigenvectors[:, np.argmax(eigenvalues)]
                primary_direction = np.mean(directions, axis=0)

                primary_yaw = np.arctan2(primary_direction[1], primary_direction[0])
                median_yaw = np.median(measurements[:, 6])
                mean_yaw = np.mean(measurements[:, 6])
                print(f"{primary_yaw=:.3f} {median_yaw=:.3f} {mean_yaw=:.3f}")
                yaws = primary_yaw.reshape(1).repeat(len(measurements))

            measurements[:, 6] = yaws[:len(measurements)]

        # after smoothing
        positions = measurements[:, :3]  # cx, cy, cz
        orientations = measurements[:, 6]  # theta
        sizes = measurements[:, 3:6]  # a, b
        
        # Fit constant acceleration model to position
        if len(time_offsets) >= 3:
            # Fit: pos = p0 + v0*t + 0.5*a*t^2
            A = np.column_stack([
                np.ones(len(time_offsets)),
                time_offsets,
                0.5 * time_offsets**2
            ])
            
            pos_coeffs = np.linalg.lstsq(A, positions, rcond=None)[0]
            self.ellipse_pos_model = pos_coeffs  # [p0, v0, a] for each dimension
            
            # Similar for orientation (but be careful with wraparound)
            # unwrapped_theta = np.unwrap(orientations)
            unwrapped_theta = orientations
            theta_coeffs = np.linalg.lstsq(A, unwrapped_theta, rcond=None)[0]
            self.ellipse_theta_model = theta_coeffs

        else:
            self.ellipse_pos_model = np.zeros((3,3), np.float32)
            self.ellipse_pos_model[0] = positions[0]

            self.ellipse_theta_model = np.zeros((3,), np.float32)
            directions = np.stack([np.cos(measurements[:, 6]), np.sin(measurements[:, 6])], axis=1)
            primary_direction = np.mean(directions, axis=0)

            primary_yaw = np.arctan2(primary_direction[1], primary_direction[0])
            self.ellipse_theta_model[0] = primary_yaw
        
        self.ellipse_size_splines = size_splines
        self.ellipse_time_ref = time_offsets[0]        

        predictions = self.predict_ellipse_motion_model(self.timestamps)
        
        r2 = r2_score(measurements, predictions, multioutput='uniform_average')
        rmse = mean_squared_error(measurements, predictions, multioutput='uniform_average')

        print(f"Ellipse Update - R²: {r2:.3f}, RMSE: {rmse:.3f}")

    def predict_ellipse_motion_model(self, future_timestamps):
        """Predict using motion model"""
        init_timestamp_sec = min(self.timestamps) * NANOSEC_TO_SEC
        future_times = np.array(future_timestamps, dtype=np.float64) * NANOSEC_TO_SEC - init_timestamp_sec
        time_offsets = np.array(self.timestamps, dtype=np.float64) * NANOSEC_TO_SEC - init_timestamp_sec

        if self.ellipse_size_splines is None:
            measurements = np.array(self.ellipses, np.float32)
            design_matrix = np.column_stack([time_offsets, np.ones(len(time_offsets))])
            coefficients, residuals, *_ = np.linalg.lstsq(design_matrix, measurements, rcond=None)
            A, c = coefficients[0, :], coefficients[1, :]

            predictions = A * future_times[:, np.newaxis] + c

            return predictions

        predictions = np.zeros((len(future_times), 7))
        
        for i, t in enumerate(future_times):
            # Position: p0 + v0*t + 0.5*a*t^2
            predictions[i, :3] = (self.ellipse_pos_model[0] + 
                                self.ellipse_pos_model[1] * t + 
                                0.5 * self.ellipse_pos_model[2] * t**2)
            
            # Sizes from splines (or last value if extrapolating)
            last_time = max(self.timestamps) * NANOSEC_TO_SEC - init_timestamp_sec
            if t <= last_time:
                predictions[i, 3] = self.ellipse_size_splines[0](t)  # a
                predictions[i, 4] = self.ellipse_size_splines[1](t)  # b
                predictions[i, 5] = self.ellipse_size_splines[2](t)  # h
            else:
                predictions[i, 3] = self.ellipse_size_splines[0](last_time)  # a
                predictions[i, 4] = self.ellipse_size_splines[1](last_time)  # b
                predictions[i, 5] = self.ellipse_size_splines[2](last_time)  # h
            
            # Orientation: p0 + v0*t + 0.5*a*t^2 then wrap
            theta_pred = (self.ellipse_theta_model[0] + 
                        self.ellipse_theta_model[1] * t + 
                        0.5 * self.ellipse_theta_model[2] * t**2)
            predictions[i, 6] = np.arctan2(np.sin(theta_pred), np.cos(theta_pred))
        
        return predictions

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

        self.lwh = avg_lwh
        
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
                # yaws = np.median(measurements[:, 6]).reshape(1).repeat(len(measurements))
                directions = np.stack([np.cos(measurements[:, 6]), np.sin(measurements[:, 6])], axis=1)

                primary_direction = np.mean(directions, axis=0)

                primary_yaw = np.arctan2(primary_direction[1], primary_direction[0])
                median_yaw = np.median(measurements[:, 6])
                mean_yaw = np.mean(measurements[:, 6])
                print(f"{primary_yaw=:.3f} {median_yaw=:.3f} {mean_yaw=:.3f}")
                yaws = primary_yaw.reshape(1).repeat(len(measurements))
            
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
                # yaws = np.median(measurements[:, 6]).reshape(1).repeat(len(measurements))

                directions = np.stack([np.cos(measurements[:, 6]), np.sin(measurements[:, 6])], axis=1)
                # cov_matrix = np.cov(directions.T)
                # eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                # # Primary direction (largest eigenvalue)
                # primary_direction = eigenvectors[:, np.argmax(eigenvalues)]
                primary_direction = np.mean(directions, axis=0)

                primary_yaw = np.arctan2(primary_direction[1], primary_direction[0])
                median_yaw = np.median(measurements[:, 6])
                mean_yaw = np.mean(measurements[:, 6])
                print(f"{primary_yaw=:.3f} {median_yaw=:.3f} {mean_yaw=:.3f}")
                yaws = primary_yaw.reshape(1).repeat(len(measurements))


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