import cProfile
import io
import pstats
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lion.unsupervised_core.box_utils import *
from lion.unsupervised_core.convex_hull_tracker import linear_assignment, nn_matching
from lion.unsupervised_core.convex_hull_tracker.convex_hull_track import (
    ConvexHullTrack,
    ConvexHullTrackState,
)
from lion.unsupervised_core.convex_hull_tracker.convex_hull_utils import (
    analytical_z_rotation_centered,
    box_iou_3d,
    circular_weighted_mean,
    relative_object_pose,
    voxel_sampling_fast,
    yaw_circular_mean,
)
from lion.unsupervised_core.convex_hull_tracker.pose_kalman_filter import (
    PoseKalmanFilter,
    wrap_angle,
)
from lion.unsupervised_core.outline_utils import points_rigid_transform
from lion.unsupervised_core.tracker.box_op import register_bbs

from .convex_hull_object import ConvexHullObject
from .convex_hull_track import BoxPredictor

np.set_printoptions(suppress=True, precision=2)

NANOSEC_TO_SEC = 1e-9


class GlobalTrack:
    history: List[ConvexHullObject]
    track_id: int
    object_points: np.ndarray
    positions: np.ndarray
    optimized_positions: np.ndarray
    refined_positions: np.ndarray
    cumulative_positions: np.ndarray


    def __init__(self, track_id: int, history: List[ConvexHullObject]) -> None:
        self.track_id = track_id
        self.history = history
        self.object_points = np.zeros((0, 3))
        self.optimized_positions = np.zeros((0, 3))
        self.refined_positions = np.zeros((0, 3))
        self.cumulative_positions = np.zeros((0, 3))
        
        self.state = ConvexHullTrackState.Confirmed

        self.heading_speed_thresh_ms = 3.6 # m/s

        self.box_predictor = BoxPredictor(self.heading_speed_thresh_ms)

        self.positions = np.stack([x.box[:3] for x in self.history], axis=0)

        self.timestamps = [x.timestamp for x in history]
        self.icp_cache = {}
        self.icp_cache_shape = {}
        self.flow_cache = {}
        self.optimized_boxes = []
        

    def get_or_compute_relative_pose(self, points_i, points_j, center, cache_key):
        if cache_key in self.icp_cache:
            R_cached, t_cached, cost, inliersi, inliersj, center_old = self.icp_cache[cache_key]
            # Transform translation for new center
            delta = center_old - center
            t_new = t_cached + (np.eye(3) - R_cached) @ delta
            return R_cached, t_new, cost, inliersi, inliersj
        
        points_i_shape = points_i.shape
        points_j_shape = points_j.shape

        # Compute and cache
        centered_i = points_i - center
        centered_j = points_j - center
        R, t, inliersi, inliersj, cost = relative_object_pose(centered_i, centered_j)

        self.icp_cache[cache_key] = (R, t, cost, inliersi, inliersj, center.copy())
        self.icp_cache_shape[cache_key] = (len(centered_i), len(centered_j), points_i_shape, points_j_shape, centered_i.shape, centered_j.shape)
        return R, t, cost, inliersi, inliersj


    def get_or_compute_relative_pose_flow(self, points_i, flow_i, center, cache_key):
        if cache_key in self.flow_cache:
            R_cached, t_cached, center_old = self.flow_cache[cache_key]
            # Transform translation for new center
            delta = center_old - center
            t_new = t_cached + (np.eye(3) - R_cached) @ delta
            return R_cached, t_new
        
        centered_i = points_i.copy() - center
        centered_j = centered_i + flow_i
        R_flow, t_flow = analytical_z_rotation_centered(centered_i, centered_j)

        self.flow_cache[cache_key] = (R_flow, t_flow, center.copy())
        return R_flow, t_flow


    def compute_poses_(self):
        timestamps = [x.timestamp for x in self.history]
        n_frames = len(timestamps)

        points_list = []
        flow_list = []
        boxes = np.stack([x.box for x in self.history], axis=0)


        # use initial pose as reference
        ref_pose_vector = PoseKalmanFilter.box_to_pose_vector(boxes[0])
        ref_pose_matrix = PoseKalmanFilter.box_to_transform(boxes[0])
        world_to_object = np.linalg.inv(ref_pose_matrix)

        boxes_poses = [PoseKalmanFilter.box_to_transform(box) for box in boxes]


        last_position = ref_pose_vector[:3].copy()
        initial_position = ref_pose_vector[:3].copy()
        last_yaw = ref_pose_vector[3].copy()
        initial_yaw = ref_pose_vector[3].copy()
        last_velocity = np.zeros(3)

        # Collect world points and centers
        for i, timestamp_ns in enumerate(timestamps):
            obj: ConvexHullObject = self.history[i]
            world_points = obj.original_points

            points_list.append(world_points)
            flow_list.append(obj.flow)

        optimized_poses = [ref_pose_matrix]

        last_optimized_yaw = np.copy(last_yaw)
        last_optimized_position = np.copy(last_position)

        flow_positions = [last_optimized_position]

        for i in range(1, n_frames):
            R_rel, t_rel, icp_cost_rel, _, _ = self.get_or_compute_relative_pose(
                points_list[i-1], points_list[i], 
                last_optimized_position,
                (i-1, i)
            )
    
            rel_yaw = Rotation.from_matrix(R_rel).as_rotvec()[2]
            rel_yaw_guess = wrap_angle(last_optimized_yaw + rel_yaw)

            if i == 1:
                R_init, t_init, icp_cost_initial = R_rel, t_rel, icp_cost_rel
            else:
                R_init, t_init, icp_cost_initial, _, _ = self.get_or_compute_relative_pose(
                    points_list[0], points_list[i], 
                    initial_position,
                    (0, i)
                )

            initial_yaw_guess = wrap_angle(initial_yaw + Rotation.from_matrix(R_init).as_rotvec()[2])

            R_flow, t_flow = self.get_or_compute_relative_pose_flow(points_list[i-1], flow_list[i-1], last_optimized_position, (i-1, i))
            flow_positions.append(last_optimized_position + t_flow)
            flow_yaw_guess = wrap_angle(last_optimized_yaw + Rotation.from_matrix(R_flow).as_rotvec()[2])


            mean_distance_rel = 0
            mean_distance_init = 0
            mean_distance_flow = 0
            mean_distance_box = 0

            # pose guesses
            rel_pos_guess = last_optimized_position + t_rel
            initial_pos_guess = initial_position + t_init
            flow_pos_guess = last_optimized_position + t_flow

            # compute boxes for each
            box_rel = np.concatenate([rel_pos_guess.reshape(3), boxes[i, 3:6].reshape(3), rel_yaw_guess.reshape(1)], axis=0)
            box_init = np.concatenate([initial_pos_guess.reshape(3), boxes[i, 3:6].reshape(3), initial_yaw_guess.reshape(1)], axis=0)
            box_flow = np.concatenate([flow_pos_guess.reshape(3), boxes[i, 3:6].reshape(3), flow_yaw_guess.reshape(1)], axis=0)
            box_orig = boxes[i]

            box_mask_rel = ConvexHullObject.points_in_box(box_rel, points_list[i], ret_mask=True)
            box_mask_init = ConvexHullObject.points_in_box(box_init, points_list[i], ret_mask=True)
            box_mask_orig = ConvexHullObject.points_in_box(box_orig, points_list[i], ret_mask=True)
            box_mask_flow = ConvexHullObject.points_in_box(box_flow, points_list[i], ret_mask=True)

            box_iou_rel = box_mask_rel.sum() / box_mask_rel.size
            box_iou_init = box_mask_init.sum() / box_mask_rel.size
            box_iou_orig = box_mask_orig.sum() / box_mask_rel.size
            box_iou_flow = box_mask_flow.sum() / box_mask_flow.size

            # Compute current velocities
            dt = (timestamps[i] - timestamps[i-1]) * 1e-9
            velocity_rel = (rel_pos_guess - last_optimized_position) / dt
            velocity_init = (initial_pos_guess - last_optimized_position) / dt
            velocity_box = (box_orig[:3] - last_optimized_position) / dt
            velocity_flow = (flow_pos_guess - last_optimized_position) / dt

            # Smoothness penalties
            lambda_smooth = 0.1  # Tunable parameter
            smooth_cost_rel = lambda_smooth * np.linalg.norm(velocity_rel - last_velocity)**2 
            smooth_cost_init = lambda_smooth * np.linalg.norm(velocity_init - last_velocity)**2
            smooth_cost_orig = lambda_smooth * np.linalg.norm(velocity_box - last_velocity)**2
            smooth_cost_flow = lambda_smooth * np.linalg.norm(velocity_flow - last_velocity)**2

            # iou_cost_rel = (1.0 - iou_rel)
            # iou_cost_initial = (1.0 - iou_initial)


            # weight between the two
            # total_cost_rel = icp_cost_rel + smooth_cost_rel + iou_cost_rel
            # total_cost_init = icp_cost_initial + smooth_cost_init + iou_cost_initial
            total_cost_rel = (1.0 - box_iou_rel) + smooth_cost_rel + mean_distance_rel
            total_cost_init = (1.0 - box_iou_init) + smooth_cost_init + mean_distance_init
            total_cost_orig = (1.0 - box_iou_orig) + smooth_cost_orig + mean_distance_box
            total_cost_flow = (1.0 - box_iou_flow) + smooth_cost_flow + mean_distance_flow

            weight_rel = 1.0 / (total_cost_rel + 1e-6)
            weight_initial = 1.0 / (total_cost_init + 1e-6)
            weight_orig = 1.0 / (total_cost_orig + 1e-6)
            weight_flow = 1.0 / (total_cost_flow + 1e-6)

            total_weight = weight_rel + weight_initial + weight_orig + weight_flow
            weight_rel /= total_weight
            weight_initial /= total_weight
            weight_orig /= total_weight
            weight_flow /= total_weight

            print(f"{float(weight_rel)=} {float(weight_initial)=} {float(weight_orig)=} {float(weight_flow)=}")

            optimized_position = rel_pos_guess * weight_rel + initial_pos_guess * weight_initial + box_orig[:3] * weight_orig + weight_flow * flow_pos_guess
            optimized_yaw = circular_weighted_mean(
                np.array([rel_yaw_guess, initial_yaw_guess, box_orig[6], flow_yaw_guess]), 
                np.array([weight_rel, weight_initial, weight_orig, weight_flow])
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

        self.optimized_poses = optimized_poses

        self.optimized_positions = np.array([x[:3, 3].copy() for x in optimized_poses])
        self.flow_positions = flow_positions

        # update points
        # Now transform points to object-centric coordinates
        object_centric_points = []
        optimized_boxes = []


        # for convex_hull_obj, obj_pose, inlier_indices in zip(self.history, optimized_poses, inlier_indices_list):
        for convex_hull_obj, obj_pose in zip(self.history, optimized_poses):
            # Get world points
            # world_points = convex_hull_obj.original_points[inlier_indices]
            world_points = convex_hull_obj.original_points

            # Transform to object-centric: multiply by inverse of object pose
            world_to_object = np.linalg.inv(obj_pose)
            object_points = points_rigid_transform(world_points, world_to_object)

            object_centric_points.append(object_points)

            dims_mins, dims_maxes = object_points.min(axis=0), object_points.max(axis=0)
            l, w, h = dims_maxes - dims_mins

            yaw = Rotation.from_matrix(optimized_poses[i][:3, :3]).as_rotvec()[2]
            x, y, z = optimized_poses[i][:3, 3]
            box = np.array([x, y, z, l, w, h, yaw])
            optimized_boxes.append(box)

        self.optimized_boxes = optimized_boxes

        self.box_predictor.update(timestamps, self.optimized_boxes)

        self.object_points = np.concatenate(object_centric_points, axis=0)
        self.object_points = voxel_sampling_fast(self.object_points, 0.05, 0.05, 0.05)

        self.object_mesh = trimesh.convex.convex_hull(self.object_points)

        dims_mins, dims_maxes = self.object_points.min(axis=0), self.object_points.max(axis=0)
        self.lwh = dims_maxes - dims_mins

        return optimized_poses

    def compute_poses(self):
        timestamps = [x.timestamp for x in self.history]
        n_frames = len(timestamps)

        points_list = []
        flow_list = []

        # Collect world points and centers
        for obj in self.history:
            world_points = obj.original_points

            points_list.append(world_points)
            flow_list.append(obj.flow)

        last_optimized_position = points_list[0].mean(axis=0)
        initial_position = last_optimized_position.copy()

        for i in range(1, n_frames):
            R_rel, t_rel, icp_cost_rel, _, _ = self.get_or_compute_relative_pose(
                points_list[i-1], points_list[i], 
                last_optimized_position,
                (i-1, i)
            )

            # R_flow, t_flow = self.get_or_compute_relative_pose_flow(points_list[i-1], flow_list[i-1], last_optimized_position, (i-1, i))

            last_optimized_position = points_list[i].mean()

        inlier_tallies = [np.zeros((len(x.original_points),), int) for x in self.history]
        print("inlier_tallies", ",".join([f"{x.shape}" for x in inlier_tallies]))
        print("x.original_points.shape", [x.original_points.shape for x in self.history])
        print("points_list", [x.shape for x in points_list])
        inlier_tallies = [np.zeros((len(x),), int) for x in points_list]

        for i in range(1, n_frames):
            # R_cached, t_cached, cost, inliersi, inliersj, center_old
            R_cached, t_cached, cost, inliersi, inliersj, center_old = self.icp_cache[(i-1, i)]
            icp_shape = self.icp_cache_shape[(i-1, i)]

            print(f'{icp_shape=} {inlier_tallies[i-1].shape=} {inlier_tallies[i].shape=}')
            print(f'{points_list[i-1].shape=} {points_list[i].shape=}')

            inlier_tallies[i-1][inliersi] += 1
            inlier_tallies[i][inliersj] += 1

        inlier_indices_list = []
        for i in range(n_frames):
            cur_tallies = inlier_tallies[i]
            inlier_indices = np.where(cur_tallies > 0)[0]
            # print(f"Frame {i} inliers {cur_tallies.min()} {cur_tallies.mean()} {cur_tallies.max()} {cur_tallies.shape=}")
            inlier_indices_list.append(inlier_indices)

        optimized_boxes = []


        cumulative_pose = np.eye(4)
        cumulative_pose[:3, :3] = Rotation.from_rotvec(np.array([0.0, 0.0, self.history[0].box[6]], float)).as_matrix()
        cumulative_pose[:3, 3] = initial_position
        cumulative_poses = [cumulative_pose]
        for i in range(1, n_frames):
            cum_R = cumulative_pose[:3, :3]
            cum_t = cumulative_pose[:3, 3]

            cum_yaw = Rotation.from_matrix(cum_R).as_rotvec()[2]
            R_rel, t_rel, icp_cost_rel, _, _ = self.get_or_compute_relative_pose(
                points_list[i-1], points_list[i], 
                cum_t,
                (i-1, i)
            )
            cur_yaw = wrap_angle(cum_yaw + Rotation.from_matrix(R_rel).as_rotvec()[2])
            cur_t = cum_t + t_rel

            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_rotvec(np.array([0.0, 0.0, cur_yaw], float)).as_matrix()
            pose[:3, 3] = cur_t

            cumulative_poses.append(pose)

        self.optimized_poses = cumulative_poses
        self.optimized_positions = np.stack([x[:3, 3] for x in cumulative_poses], axis=0)

        
        cur_points = points_list[0].copy()
        cur_points = cur_points[inlier_indices_list[0]]

        world_inlier_points = [cur_points.copy()]

        obj_pose = cumulative_poses[0]
        world_to_object = np.linalg.inv(obj_pose)

        object_points = points_rigid_transform(cur_points, world_to_object)

        object_centric_points = [object_points]

        for i in range(1, n_frames):
            cur_points = points_list[i].copy()
            cur_points = cur_points[inlier_indices_list[i]]

            world_inlier_points.append(cur_points.copy())
            
            obj_pose = cumulative_poses[i]
            world_to_object = np.linalg.inv(obj_pose)

            object_points = points_rigid_transform(cur_points, world_to_object)

            object_centric_points.append(object_points)

        # fig, ax = plt.subplots(figsize=(8, 8))
        # colors = plt.cm.tab20(np.linspace(0, 1, len(object_centric_points)))
        # for idx, points in enumerate(object_centric_points):
        #     ax.scatter(
        #         points[:, 0],
        #         points[:, 1],
        #         s=3,
        #         c=colors[idx],
        #         label="Lidar Points",
        #         alpha=0.5,
        #     )


        # ax.set_aspect("equal")
        # ax.grid(True, alpha=0.3)
        # ax.set_xlabel("X (meters)", fontsize=12)
        # ax.set_ylabel("Y (meters)", fontsize=12)

        # plt.tight_layout()

        # save_folder = Path("./global_tracker_object_points/")
        # save_folder.mkdir(exist_ok=True)
        # save_path = save_folder / f"track_{self.track_id}.png"
        # plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # plt.close()

        object_points = np.concatenate(object_centric_points, axis=0)
        mesh: trimesh.Trimesh = trimesh.convex.convex_hull(object_points)

        refined_poses1 = []

        # refine poses
        for i in range(n_frames):
            initial_pose = cumulative_poses[i]

            cur_points = world_inlier_points[i]
            cur_mesh: trimesh.Trimesh = trimesh.convex.convex_hull(cur_points)

            transformed_mesh = mesh.copy().apply_transform(initial_pose)
            
            # R, t, cost = efficient_relative_pose_trimesh(transformed_mesh, cur_mesh, samples=1000)
            R, t, _, _, _ = relative_object_pose(transformed_mesh.vertices, cur_mesh.vertices)

            rel_pose = np.eye(4)
            rel_pose[:3, :3] = R
            rel_pose[:3, 3] = t

            cur_pose = rel_pose @ initial_pose

            init_R = initial_pose[:3, :3]
            init_t = initial_pose[:3, 3]

            init_yaw = Rotation.from_matrix(init_R).as_rotvec()[2]

            cur_R = cur_pose[:3, :3]
            cur_t = cur_pose[:3, 3]

            cur_yaw = Rotation.from_matrix(cur_R).as_rotvec()[2]

            print('cur_t', cur_t, 'init_t', init_t)
            print('cur_yaw', cur_yaw, 'init_yaw', init_yaw)

            refined_poses1.append(cur_pose)

        # n_frames = len(cumulative_poses)
        
        # # 1. Build robust object mesh from ALL frames (not just convex hull)
        # merged_points = np.zeros((0, 3))
        # for i in range(n_frames):
        #     cur_points = points_list[i][inlier_indices_list[i]].copy()

        #     merged_points = np.concatenate([merged_points, cur_points], axis=0)

        #     if (i, i+1) in self.icp_cache:
        #         R_rel, t_rel, _, _, _, center_old = self.icp_cache[(i, i+1)]

        #         print("merged_points", merged_points.shape, center_old.shape)
        #         print("R_rel", R_rel.shape, t_rel.shape)
        #         merged_points = merged_points - center_old.reshape(1, 3)
        #         merged_points = (R_rel @ merged_points.T).T + t_rel.reshape(1, 3) + center_old.reshape(1, 3)
        
        # # merged_points = np.concatenate(merged_points, axis=0)
        # print('merged_points', merged_points.shape)
        
        # # Option A: Use alpha shapes instead of convex hull for better shape representation
        # reference_mesh: trimesh.Trimesh = trimesh.convex.convex_hull(merged_points)

        # reference_position = reference_mesh.centroid
        # reference_yaw = self.history[-1].box[6]

        # reference_pose = np.eye(4)
        # reference_pose[:3, :3] = Rotation.from_rotvec(np.array([0.0, 0.0, reference_yaw], float)).as_matrix()
        # reference_pose[:3, 3] = reference_position

        # print("reference_position", reference_position)
        
        # refined_poses2 = []
        
        # for i in range(n_frames):
        #     initial_pose = cumulative_poses[i].copy()
        #     cur_points = points_list[i][inlier_indices_list[i]]

        #     initial_rel_pose = reference_pose @ np.linalg.inv(initial_pose)

        #     print(f"{reference_pose[:3, 3]=} {initial_rel_pose[:3, 3]=} {initial_pose[:3, 3]=}")

        #     # transformed_mesh = mesh.copy().apply_transform(initial_pose)
            
        #     # Instead of mesh-to-mesh, do point-to-mesh registration
        #     # This is more robust for sparse LiDAR data
        #     refined_pose = self.refine_pose_point_to_mesh(
        #         cur_points, reference_mesh, initial_rel_pose
        #     )

        #     print("initial_rel_pose[:3, 3]", initial_rel_pose[:3, 3])
        #     print("refined_pose[:3, 3]", refined_pose[:3, 3])
            
        #     refined_poses2.append(refined_pose)

        refined_poses2 = [np.eye(4)] * len(cumulative_poses)

        for ref_pose1, ref_pose2, cumulative_pose in zip(refined_poses1, refined_poses2, cumulative_poses):
            out = []
            for cur_name, cur_pose in [('ref_pose1', ref_pose1), ('ref_pose2', ref_pose2), ('cumulative_pose', cumulative_pose)]:
                cur_R = cur_pose[:3, :3]
                cur_t = cur_pose[:3, 3]

                cur_yaw = Rotation.from_matrix(cur_R).as_rotvec()[2]

                out.append(f"{cur_name}: position: {cur_t[0]:.2f} {cur_t[1]:.2f} {cur_t[2]:.2f} yaw: {cur_yaw:.2f}")
    
            print(",".join(out))

        self.refined_positions = np.stack([x[:3, 3] for x in refined_poses1], axis=0)
        self.cumulative_positions = np.stack([x[:3, 3] for x in cumulative_poses], axis=0)

        optimized_boxes = []
        for convex_hull_obj, obj_pose in zip(self.history, refined_poses1):
            # Transform to object-centric: multiply by inverse of object pose
            object_points = points_rigid_transform(mesh.vertices, obj_pose)

            dims_mins, dims_maxes = object_points.min(axis=0), object_points.max(axis=0)
            l, w, h = dims_maxes - dims_mins

            yaw = Rotation.from_matrix(obj_pose[:3, :3]).as_rotvec()[2]
            x, y, z = obj_pose[:3, 3]
            box = np.array([x, y, z, l, w, h, yaw])
            optimized_boxes.append(box)

        self.optimized_boxes = optimized_boxes

        self.box_predictor.update(timestamps, [x.box for x in self.history])

        # self.object_points = np.concatenate(object_centric_points, axis=0)
        self.object_points = mesh.vertices

        self.object_mesh = mesh

        dims_mins, dims_maxes = self.object_points.min(axis=0), self.object_points.max(axis=0)
        self.lwh = dims_maxes - dims_mins

    def refine_pose_point_to_mesh(self, points, mesh, initial_rel_pose, max_iterations=20):
        """
        Refine pose using point-to-mesh ICP instead of mesh-to-mesh
        """
        current_rel_pose = initial_rel_pose.copy()
        
        for iteration in range(max_iterations):
            # Transform points to current pose
            transformed_points = points_rigid_transform(points, current_rel_pose)
            
            # Find closest points on mesh surface
            closest_points, distances, triangle_ids = trimesh.proximity.closest_point(
                mesh, transformed_points
            )
            
            # Filter outliers
            distance_threshold = np.percentile(distances, 80)
            inliers = distances < distance_threshold
            
            if np.sum(inliers) < 10:  # Need minimum points
                break
                
            # Solve for relative transformation
            source_pts = transformed_points[inliers]
            target_pts = closest_points[inliers]
            
            # Use your existing ICP solver or a robust one
            # R_rel, t_rel = self.solve_point_to_point_icp(source_pts, target_pts)
            # R_rel, t_rel, inliersi, inliersj, cost = relative_object_pose(source_pts, target_pts)
            R_rel, t_rel = analytical_z_rotation_centered(source_pts, target_pts)

            
            # Update pose
            rel_transform = np.eye(4)
            rel_transform[:3, :3] = R_rel
            rel_transform[:3, 3] = t_rel
            
            current_rel_pose = rel_transform @ current_rel_pose

            # Check convergence
            mean_error = np.mean(distances[inliers])
            if mean_error < 0.02:  # 2cm threshold
                break
        
        return current_rel_pose


    def extrapolate_box(self, timestamps):
        new_prediction = self.box_predictor.predict_boxes(timestamps)

        return new_prediction.reshape(-1, 7)

class GlobalTracker:
    objects: List[ConvexHullObject] = []
    
    def __init__(self, config=None, debug: bool = False) -> None:
        self.config = config
        self.debug = debug
        self.tracks: List[GlobalTrack] = []  # id -> track

        nn_budget = None
        max_cosine_distance = 0.2
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )

        self._next_id = 0
        self.ppscore_thresh = 0.7
        self.track_query_eps = 3.0  # metres
        self.max_iou_distance = 0.9

        self.min_semantic_threshold = 0.0 # allow no features
        self.min_iou_threshold = 0.1
        self.min_box_iou = 0.3

        self.icp_max_dist = 1.0
        self.icp_max_iterations = 5

        self.nms_iou_threshold = 0.3
        self.nms_stg2_iou_threshold = 0.7
        self.nms_semantic_threshold: float = 0.95
        self.nms_query_distance: float = 10.0
        self.n_points_err_thresh = 0.3

        self.min_component_points = 10

        self.n_init = 3
        self.max_age = 10  # 5 frames -> 0.5 seconds
        self.max_age_secs = self.max_age * 0.1

        self.updates_per_track = {}
        self.timestamps = []

        self.objects: List[ConvexHullObject] = []
        self.object_timestamps = []

    def add_objects(self, objects: List[ConvexHullObject], timestamp: int):
        timestamps = [timestamp] * len(objects)

        self.objects.extend(objects)
        self.object_timestamps.extend(timestamps)
        self.timestamps.append(timestamp)

    def calculate(self):
        N = len(self.objects)
        object_timestamps_secs = np.array([x * NANOSEC_TO_SEC for x in self.object_timestamps], float)

        best_semantic_ious = np.zeros((N,), float)
        best_ious = np.zeros((N,), float)

        rows = []
        cols = []
        costs = []
        ious = []
        ious_dflt = []
        ious_tmd_centre = []
        ious_tmd_left = []
        ious_tmd_right = []
        semantic_ious = []
        dists = []
        dists_all = []
        time_difference = []

        truck_pos = np.array([3524.0, 1938, 0.0])

        object_positions = [x.centroid_3d for x in self.objects]
        distances = np.linalg.norm(object_positions - truck_pos, axis=1)
        close_indices = np.argsort(distances)[:10]
        # close_indices = np.where(distances < 15.0)[0]
        close_indices = set([int(x) for x in close_indices])

        object_timestamps = np.array(self.object_timestamps, int)
        unique_times = np.unique(object_timestamps)
        unique_times_secs = unique_times.copy() * NANOSEC_TO_SEC
        n_times = len(unique_times)

        for time_i in range(0, n_times):
            start_time = unique_times[time_i]
            time_i_secs = unique_times_secs[time_i]

            for i in np.where(object_timestamps == start_time)[0]:
                object_i: ConvexHullObject = self.objects[i]

                for time_j in range(time_i + 1, n_times):
                    end_time = unique_times[time_j]
                    time_j_secs = unique_times_secs[time_j]

                    if abs(time_j_secs - time_i_secs) > self.max_age_secs:# or abs(time_j_secs - time_i_secs) < 0.1: # too long/short
                        break

                    for j in np.where(object_timestamps == end_time)[0]:



                        assert time_j_secs > time_i_secs

                        object_j: ConvexHullObject = self.objects[j]

                        semantic_iou = np.dot(
                            object_i.feature, object_j.feature
                        )

                        best_semantic_ious[i] = max(best_semantic_ious[i], semantic_iou)
                        best_semantic_ious[j] = max(best_semantic_ious[j], semantic_iou)

                        if semantic_iou < self.min_semantic_threshold:
                            continue


                        # between
                        new_timestamp = (object_i.timestamp + object_j.timestamp) // 2
                        object_i_box_extrapolated_centre = object_i.to_timestamped_box(new_timestamp)
                        object_j_box_extrapolated_centre = object_j.to_timestamped_box(new_timestamp)

                        # # object i timestamp
                        # object_i_box_extrapolated_left = object_i.to_timestamped_box(object_i.timestamp)
                        # object_j_box_extrapolated_left = object_j.to_timestamped_box(object_i.timestamp)

                        # # object j timestamp
                        # object_i_box_extrapolated_right = object_i.to_timestamped_box(object_j.timestamp)
                        # object_j_box_extrapolated_right = object_j.to_timestamped_box(object_j.timestamp)


                        iou_dflt = box_iou_3d(object_i.box, object_j.box)
                        iou_tmd_centre = box_iou_3d(object_i_box_extrapolated_centre, object_j_box_extrapolated_centre)
                        # iou_tmd_left = box_iou_3d(object_i_box_extrapolated_left, object_j_box_extrapolated_left)
                        # iou_tmd_right = box_iou_3d(object_i_box_extrapolated_right, object_j_box_extrapolated_right)

                        iou = max(iou_dflt, iou_tmd_centre)


                        if iou < self.min_box_iou:
                            continue

                        # R, t, inliersi, inliersj, pose_cost = relative_object_pose(object_i.original_points, object_j.original_points)

                        # relative_pose = np.eye(4)
                        # relative_pose[:3, :3] = R
                        # relative_pose[:3, 3] = t

                        # box_transformed = register_bbs(object_i.box.copy().reshape(1, 7), relative_pose)[0]

                        # iou_transformed = box_iou_3d(box_transformed, object_j.box)

                        dt1 = (new_timestamp - start_time) * NANOSEC_TO_SEC
                        dt2 = (new_timestamp - end_time) * NANOSEC_TO_SEC
                        dlt_dt = 0.1

                        dt_delta1 = dt1 / dlt_dt
                        dt_delta2 = dt2 / dlt_dt

                        object_i_points_ = object_i.original_points.copy() + object_i.flow.copy() * dt_delta1
                        object_j_points = object_j.original_points.copy() + object_j.flow.copy() * dt_delta2

                        points_dists = np.linalg.norm(object_i_points_[:, np.newaxis, :] - object_j_points[np.newaxis, :, :], axis=2)
                        points_dists = points_dists.min(axis=1)


                        assert points_dists.shape[0] == object_i_points_.shape[0], f"{points_dists.shape=} {object_i_points_.shape=}"

                        dist = points_dists.mean()

                        # if (i in close_indices or j in close_indices) and dist < 5.0:

                        #     fig, ax = plt.subplots(figsize=(4,4))

                        #     ax.scatter(
                        #         object_i.original_points[:, 0],
                        #         object_i.original_points[:, 1],
                        #         s=3,
                        #         c="red",
                        #         label="Lidar Points",
                        #         alpha=0.5,
                        #     )
                        #     ax.scatter(
                        #         object_i_points_[:, 0],
                        #         object_i_points_[:, 1],
                        #         s=3,
                        #         c="blue",
                        #         label="Lidar Points",
                        #         alpha=0.7,
                        #     )
                        #     ax.scatter(
                        #         object_j_points[:, 0],
                        #         object_j_points[:, 1],
                        #         s=3,
                        #         c="green",
                        #         label="Lidar Points",
                        #         alpha=0.7,
                        #     )


                        #     ax.set_aspect("equal")
                        #     ax.grid(True, alpha=0.3)
                        #     ax.set_xlabel("X (meters)", fontsize=12)
                        #     ax.set_ylabel("Y (meters)", fontsize=12)
                        #     # ax.title(f"dist:{dist:.2f} {semantic_iou:.2f} {iou:.2f}")
                        #     ax.set_title(f"dist:{dist:.2f} semantic_iou:{semantic_iou:.2f} iou:{iou:.2f} time: {(time_j_secs - time_i_secs):.2f}", fontsize=14, fontweight="bold")
                        #     xy1 = np.minimum(object_i_points_.min(axis=0), object_j_points.min(axis=0))
                        #     xy2 = np.minimum(object_i_points_.max(axis=0), object_j_points.max(axis=0))

                        #     x1, x2 = np.min(xy1), np.max(xy2)

                        #     # # square
                        #     # plt.xlim(x1, x2)
                        #     # plt.ylim(x1, x2)

                        #     plt.tight_layout()

                        #     save_folder = Path("./global_tracker/")
                        #     save_folder.mkdir(exist_ok=True)
                        #     save_path = save_folder / f"{i}_{j}.png"
                        #     plt.savefig(save_path, dpi=150, bbox_inches="tight")
                        #     plt.close()


                        dists_all.append(dist)


                        best_ious[i] = max(best_ious[i], iou)
                        best_ious[j] = max(best_ious[j], iou)

                        # if iou < self.min_box_iou:
                        #     continue

                        if dist > 1.0: # flow distance too far...
                            continue

                        # dist1 = np.linalg.norm(object_i.box[:3] - object_j.box[:3])
                        # dist2 = np.linalg.norm(object_i_box_extrapolated[:3] - object_j_box_extrapolated[:3])
                        # dist = (dist1 + dist2) / 2.0

                        # dist = np.linalg.norm(object_i_box_extrapolated[:3] - object_j_box_extrapolated[:3])

                        cost = (1.0 - iou) + dist + (1.0 - semantic_iou) + (time_j_secs - time_i_secs)*10

                        rows.append(i)
                        cols.append(j)
                        costs.append(cost)

                        ious.append(iou)
                        ious_dflt.append(iou_dflt)
                        ious_tmd_centre.append(iou_tmd_centre)
                        # ious_tmd_left.append(iou_tmd_left)
                        # ious_tmd_right.append(iou_tmd_right)
                        semantic_ious.append(semantic_iou)
                        dists.append(dist)
                        time_difference.append(time_j_secs - time_i_secs)

        rows = np.array(rows, int)
        cols = np.array(cols, int)
        costs = np.array(costs, float)

        print('costs', costs.shape, costs.min(), costs.max())
        graph = csr_array((costs, (rows, cols)), shape=(N, N))

        print("ious", np.min(ious), np.mean(ious), np.max(ious))
        print("ious_dflt", np.min(ious_dflt), np.mean(ious_dflt), np.max(ious_dflt))
        print("ious_tmd_centre", np.min(ious_tmd_centre), np.mean(ious_tmd_centre), np.max(ious_tmd_centre))
        # print("ious_tmd_left", np.min(ious_tmd_left), np.mean(ious_tmd_left), np.max(ious_tmd_left))
        # print("ious_tmd_right", np.min(ious_tmd_right), np.mean(ious_tmd_right), np.max(ious_tmd_right))
        print("semantic_ious", np.min(semantic_ious), np.mean(semantic_ious), np.max(semantic_ious))
        print("dists", np.min(dists), np.mean(dists), np.max(dists))
        print("dists_all", np.min(dists_all), np.mean(dists_all), np.max(dists_all))
        print("time_difference", np.min(time_difference), np.mean(time_difference), np.max(time_difference))
        print("best_ious", np.min(best_ious), np.mean(best_ious), np.max(best_ious))

        paths = find_tracks_parallel_dijkstra(graph, self.object_timestamps)

        track_lengths = []
        track_lengths_ns = []
        truck_paths = []
        pr = cProfile.Profile()
        pr.enable()
        for track_id, path in enumerate(paths):
            history = [self.objects[i] for i in path]

            track_lengths.append(len(path))

            times = [self.object_timestamps[idx] for idx in path]

            min_time = min(times)
            max_time = max(times)

            track_lengths_ns.append(max_time - min_time)

            track = GlobalTrack(track_id, history)

            track.compute_poses()

            self.tracks.append(track)

            cur_nodes = set(path)
            if len(cur_nodes.intersection(close_indices)) > 0:
                print("close to truck has path", path)
                truck_paths.append(path)

        pr.disable()

        s = io.StringIO()
        sortby = "cumtime"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open("_global_tracker.calculate.txt", "w") as f:
            f.write(s.getvalue())

        fig, ax = plt.subplots(figsize=(8, 8))

        colors = plt.cm.tab20(np.linspace(0, 1, len(truck_paths)))
        for truck_id, path in enumerate(truck_paths):

            for node in path:
                box = self.objects[node].box
                points = self.objects[node].original_points
                # Get rotated box corners
                corners = get_box_bev_corners(box)
                obj_polygon = patches.Polygon(
                    corners,
                    linewidth=1,
                    edgecolor=colors[truck_id],
                    facecolor="none",
                    alpha=0.7,
                    linestyle="-",
                )
                ax.add_patch(obj_polygon)

                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=3,
                    c=colors[truck_id],
                    label="Lidar Points",
                    alpha=0.5,
                )


        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)
        # ax.set_title(f"dist:{dist:.2f} semantic_iou:{semantic_iou:.2f} iou:{iou:.2f}", fontsize=14, fontweight="bold")

        plt.tight_layout()



        # save_folder = Path("./global_tracker/")
        # save_folder.mkdir(exist_ok=True)
        # save_path = save_folder / f"{i}_{j}.png"
        save_path = f"truck_tracklets.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        
        counts = Counter(track_lengths)
        print("find_tracks_parallel_dijkstra track length counts", counts.items())

        secs_strings = [f"{time_diff_ns*NANOSEC_TO_SEC:.2f}" for time_diff_ns in track_lengths_ns]
        counts = Counter(secs_strings)
        print(f"Seconds counts", ",".join([f"{time_diff_secs} - {count}" for time_diff_secs, count in counts.items()]))

        # n_components, labels = connected_components(csgraph=graph, directed=True, return_labels=True)


        # track_id = 0

        # track_lengths = []

        # # Process each component separately
        # for comp_id in range(n_components):
        #     # Get nodes in this component
        #     comp_nodes = np.where(labels == comp_id)[0]
            
        #     if len(comp_nodes) == 1:
        #         # Single node component
        #         # dist_matrix[comp_nodes[0], comp_nodes[0]] = 0
        #         cmp_lbl = comp_nodes[0]
                
        #         # print(f"Single cc - Best semantic: {best_semantic_ious[cmp_lbl]:.2f} best IoU: {best_ious[cmp_lbl]:.2f}")
        #         continue
            
        #     times = object_timestamps_secs[comp_nodes]
        #     min_time = times.min()
        #     times = times - min_time
        #     print("cc times", np.round(times, 3))
        #     print("comp_nodes", comp_nodes)
        #     timestamps = np.array([self.objects[i].timestamp for i in comp_nodes])
        #     print("timestamps", timestamps)

        #     unique_times = np.unique(timestamps)
        #     print("unique_times", unique_times)

        #     # Extract subgraph for this component
        #     comp_graph = graph[np.ix_(comp_nodes, comp_nodes)]

        #     comp_dist, comp_pred = dijkstra(csgraph=comp_graph, directed=True, return_predecessors=True)

        #     best_path, best_cost, time_span = find_longest_valid_path(comp_dist, comp_pred, timestamps, unique_times)

        #     # Convert back to original indices
        #     best_path_original = [comp_nodes[idx] for idx in best_path]
        #     best_path_timestamps = [timestamps[idx] for idx in best_path]

        #     print("best_path (component indices):", best_path)
        #     print("best_path (original indices):", best_path_original)
        #     print("best_path_timestamps:", best_path_timestamps)
        #     print("path length:", len(best_path))

        #     if len(best_path) == 0:
        #         print("=== DEBUGGING ===")
        #         print("comp_nodes:", comp_nodes)
        #         print("timestamps:", timestamps)

        #         # Check the subgraph structure
        #         print("comp_graph shape:", comp_graph.shape)
        #         print("comp_graph nnz:", comp_graph.nnz)
        #         print("comp_graph data:", comp_graph.data[:10] if comp_graph.nnz > 0 else "No edges!")

        #         print("comp_dist shape:", comp_dist.shape)
        #         print("comp_dist min/max:", comp_dist.min(), comp_dist.max())
        #         print("Number of inf values:", np.isinf(comp_dist).sum())
        #         print("Number of finite values:", np.isfinite(comp_dist).sum())

        #     history = [self.objects[i] for i in best_path_original]

        #     track_lengths.append(len(history))

        #     track = GlobalTrack(track_id, history)
        #     track.compute_poses()
        #     track_id += 1

        #     self.tracks.append(track)

        
        counts = Counter(track_lengths)
        print("track length counts", counts.items())

def find_longest_valid_path(comp_dist, comp_pred, timestamps, unique_times):
    """Find the longest time span with a valid path, then pick the best path for that span"""
    
    # Generate all time spans, sorted by length descending
    time_spans = []
    for t0 in range(len(unique_times)):
        for t1 in range(t0 + 1, len(unique_times)):
            time_span = (unique_times[t1] - unique_times[t0]) * NANOSEC_TO_SEC
            time_spans.append((time_span, t0, t1))
    
    time_spans.sort(key=lambda x: x[0], reverse=True)  # Longest first
    
    for time_span, t0, t1 in time_spans:
        min_timestamp = unique_times[t0]
        start_indices = np.where(timestamps == min_timestamp)[0]
        
        max_timestamp = unique_times[t1]
        last_indices = np.where(timestamps == max_timestamp)[0]
        
        # Find all valid paths for this time span
        valid_paths = []
        
        for start_idx in start_indices:
            for last_idx in last_indices:
                if start_idx != last_idx:
                    cost = comp_dist[start_idx, last_idx]
                    
                    if np.isfinite(cost):
                        # Reconstruct path
                        path = reconstruct_path(comp_pred, start_idx, last_idx)
                        valid_paths.append((cost, path, start_idx, last_idx))
        
        if valid_paths:
            # Found valid paths for this time span - pick the best one
            best_cost, best_path, start_idx, end_idx = min(valid_paths, key=lambda x: x[0])
            
            print(f"Found {len(valid_paths)} valid paths for time span {time_span:.3f}s")
            print(f"Best path: cost={best_cost:.3f}, length={len(best_path)}")
            
            return best_path, best_cost, time_span
    
    return [], np.inf, 0

def reconstruct_path(comp_pred, start_idx, end_idx):
    cur_idx = end_idx
    path = [cur_idx]
    
    while comp_pred[start_idx, cur_idx] != -9999:
        cur_idx = comp_pred[start_idx, cur_idx]
        path.append(cur_idx)
    
    path.reverse()
    return path

def find_tracks_parallel_dijkstra(graph, timestamps):
    """
    Process all timestamp groups simultaneously, then resolve conflicts
    """
    n = len(timestamps)
    available_nodes = set(range(n))
    all_tracks = []
    
    unique_times = np.unique(timestamps)
    tidx = 0

    with tqdm(total=n, desc="Processing nodes find_tracks_parallel_dijkstra") as pbar:
        while available_nodes and tidx < len(unique_times):
            nodes_before = len(available_nodes)
            
            # Check if current timestamp has any available nodes
            current_time = unique_times[tidx]
            source_nodes = [i for i in available_nodes if timestamps[i] == current_time]
            
            if not source_nodes:
                tidx += 1
                continue
            
            # Run Dijkstra from all sources simultaneously
            available_list = list(available_nodes)
            subgraph = graph[np.ix_(available_list, available_list)]
            source_indices = [available_list.index(node) for node in source_nodes]
            
            dist_matrix, pred_matrix = dijkstra(
                csgraph=subgraph,
                directed=True,
                indices=source_indices,
                return_predecessors=True
            )
            
            # Extract non-conflicting tracks
            tracks, used_nodes = resolve_track_conflicts(
                source_indices, source_nodes, dist_matrix, pred_matrix,
                available_list, timestamps
            )
            
            all_tracks.extend(tracks)
            available_nodes -= used_nodes
            
            # Update progress
            nodes_processed = nodes_before - len(available_nodes)
            if nodes_processed > 0:
                pbar.update(nodes_processed)
            
            # Always move to next timestamp
            tidx += 1
            
            # Update progress bar display
            pbar.set_postfix({
                'remaining': len(available_nodes),
                'tracks': len(all_tracks),
                'timestamp_idx': f"{tidx}/{len(unique_times)}"
            })
        
        # Handle any remaining nodes
        if available_nodes:
            pbar.update(len(available_nodes))
    
    return all_tracks

def resolve_track_conflicts(source_indices, source_nodes, dist_matrix, pred_matrix, 
                          available_list, timestamps):
    unique_times = np.unique(timestamps)
    local_to_position = {local_idx: pos for pos, local_idx in enumerate(source_indices)}
    
    time_spans = []
    for start_idx, start_node in zip(source_indices, source_nodes):
        source_time = timestamps[start_node]
        
        for t1 in unique_times:
            if t1 <= source_time:
                continue
            time_span = (t1 - source_time) * NANOSEC_TO_SEC
            time_spans.append((time_span, start_idx, start_node, t1))
    
    time_spans.sort(key=lambda x: x[0], reverse=True)
    
    valid_paths = []
    for time_span, start_idx, start_node, max_timestamp in time_spans:
        # Find global indices at max_timestamp
        last_global_indices = np.where(timestamps == max_timestamp)[0]

        src_position = local_to_position[start_idx]
        
        # Convert to local indices and filter available
        last_local_indices = [available_list.index(global_idx) 
                             for global_idx in last_global_indices 
                             if global_idx in available_list]
        
        for last_local_idx in last_local_indices:
            if start_idx != last_local_idx:
                cost = dist_matrix[src_position, last_local_idx]
                
                if np.isfinite(cost):
                    path_local = reconstruct_path(pred_matrix, src_position, last_local_idx)
                    valid_paths.append((time_span, source_time, max_timestamp, 
                                     cost, path_local, start_idx, last_local_idx))
    
    if valid_paths:
        valid_paths = sorted(valid_paths, key=lambda x: (x[0], x[3]))
        
        best_paths = []
        used_nodes = set()

        path_sets: List[set] = []
        
        for time_span, source_time, max_timestamp, cost, path_local, start_idx, last_idx in valid_paths:
            path_global = [available_list[i] for i in path_local]
            cur_nodes = set(path_global)

            # best_iou = 0.0
            # for path_set in path_sets:
            #     inter = len(path_set.intersection(cur_nodes))
            #     union = len(path_set.union(cur_nodes))

            #     if union == 0.0:
            #         iou = 0.0
            #     else:
            #         iou = inter / union

            #     best_iou = max(best_iou, iou)

            # print('best_iou', best_iou)

            # allow partial overlap, reject complete duplicates
            if len(cur_nodes.symmetric_difference(used_nodes)) == 0:
                print("no difference")
                continue  # All nodes already used - this is a duplicate path
            
            used_nodes.update(path_global)
            best_paths.append(path_global)
            path_sets.append(cur_nodes)
        
        return best_paths, used_nodes
    
    return [], set()



def efficient_relative_pose_trimesh(mesh1, mesh2, samples=1000):
    """
    Using trimesh's built-in registration capabilities
    """
    # Use trimesh's registration
    matrix, cost = trimesh.registration.mesh_other(
        mesh1, mesh2, samples=samples
    )
    
    # Extract R and t
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    
    return R, t, cost