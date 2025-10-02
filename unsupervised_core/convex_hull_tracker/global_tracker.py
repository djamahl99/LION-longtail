import cProfile
import io
import pstats
import time
from collections import Counter, defaultdict
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
from shapely.geometry import MultiPoint, Polygon, box
from tqdm import tqdm

from lion.unsupervised_core.box_utils import *
from lion.unsupervised_core.convex_hull_tracker import linear_assignment, nn_matching
from lion.unsupervised_core.convex_hull_tracker.convex_hull_track import (
    ConvexHullTrack,
    ConvexHullTrackState,
)
from lion.unsupervised_core.convex_hull_tracker.convex_hull_utils import (
    analytical_z_rotation_centered,
    bidirectional_matching,
    box_iou_3d,
    circular_weighted_mean,
    relative_object_pose,
    relative_object_pose_multiresolution,
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
        self.source = "global_track"

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


    def compute_poses(self):
        print(f"compute_poses {self.track_id=}")
        timestamps = [x.timestamp for x in self.history]
        n_frames = len(timestamps)

        times_secs = np.array([x * NANOSEC_TO_SEC for x in timestamps], float)

        points_list = []
        flow_list = []

        # Collect world points and centers
        for obj in self.history:
            world_points = obj.original_points

            points_list.append(world_points)
            flow_list.append(obj.flow)

        last_optimized_position = points_list[0].mean(axis=0)
        initial_position = last_optimized_position.copy()
        last_flow_position = self.history[0].box[:3].copy()
        last_flow_yaw = self.history[0].box[6]

        last_flow_pose = np.eye(4)
        last_flow_pose[:3, 3] = last_flow_position
        last_flow_pose[:3, :3] = Rotation.from_rotvec(np.array([0.0, 0.0, last_flow_yaw])).as_matrix()

        flow_poses = [last_flow_pose]

        for i in range(1, n_frames):
            R_rel, t_rel, icp_cost_rel, _, _ = self.get_or_compute_relative_pose(
                points_list[i-1], points_list[i], 
                last_optimized_position,
                (i-1, i)
            )

            R_flow, t_flow = self.get_or_compute_relative_pose_flow(points_list[i-1], flow_list[i-1], last_optimized_position, (i-1, i))

            flow_position = last_flow_position + t_flow
            flow_yaw =  wrap_angle(last_flow_yaw + Rotation.from_matrix(R_flow).as_rotvec()[2])
            

            flow_pose = np.eye(4)
            flow_pose[:3, 3] = flow_position
            flow_pose[:3, :3] = Rotation.from_rotvec(np.array([0.0, 0.0, flow_yaw])).as_matrix()

            flow_poses.append(flow_pose)


            last_optimized_position = points_list[i].mean()
            last_flow_position = flow_position
            last_flow_yaw = flow_yaw

        # inlier_tallies = [np.zeros((len(x.original_points),), int) for x in self.history]
        # print("inlier_tallies", ",".join([f"{x.shape}" for x in inlier_tallies]))
        # print("x.original_points.shape", [x.original_points.shape for x in self.history])
        # print("points_list", [x.shape for x in points_list])
        # inlier_tallies = [np.zeros((len(x),), int) for x in points_list]

        # for i in range(1, n_frames):
        #     # R_cached, t_cached, cost, inliersi, inliersj, center_old
        #     R_cached, t_cached, cost, inliersi, inliersj, center_old = self.icp_cache[(i-1, i)]
        #     icp_shape = self.icp_cache_shape[(i-1, i)]

        #     print(f'{icp_shape=} {inlier_tallies[i-1].shape=} {inlier_tallies[i].shape=}')
        #     print(f'{points_list[i-1].shape=} {points_list[i].shape=}')

        #     inlier_tallies[i-1][inliersi] += 1
        #     inlier_tallies[i][inliersj] += 1

        # inlier_indices_list = []
        # for i in range(n_frames):
        #     cur_tallies = inlier_tallies[i]
        #     inlier_indices = np.where(cur_tallies > 0)[0]
        #     # print(f"Frame {i} inliers {cur_tallies.min()} {cur_tallies.mean()} {cur_tallies.max()} {cur_tallies.shape=}")
        #     inlier_indices_list.append(inlier_indices)

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

        box_densities = []
        for convex_hull_obj in self.history:
            vol = np.prod(convex_hull_obj.box[3:6])
            n_points = convex_hull_obj.orig_n_points

            density = n_points / vol # points per metre^3

            box_densities.append(density)

        print("box_densities: " + ",".join([f"{x:.3f}" for x in box_densities]))

        reference_idx = np.argmax(box_densities)
        reference_points = points_list[reference_idx]
        reference_mesh1: trimesh.Trimesh = trimesh.convex.convex_hull(reference_points)

        reference_yaw = self.history[reference_idx].box[6]
        reference_position = self.history[reference_idx].box[:3]

        reference_pose = np.eye(4)
        reference_pose[:3, :3] = Rotation.from_rotvec(np.array([0.0, 0.0, reference_yaw], float)).as_matrix()
        reference_pose[:3, 3] = reference_position

        reference_to_object = np.linalg.inv(reference_pose)
        reference_object_points = points_rigid_transform(reference_points, reference_to_object)

        refined_poses21 = []
        
        for i in range(n_frames):
            initial_pose = cumulative_poses[i].copy()
            cur_points = points_list[i]

            initial_pose_inv = np.linalg.inv(initial_pose)
            pointsi = points_rigid_transform(cur_points, initial_pose_inv)
            pointsj = points_rigid_transform(reference_points, initial_pose_inv)

            R, t, inliersi, inliersj, cost = relative_object_pose_multiresolution(pointsi, pointsj)

            dt = times_secs[reference_idx] - times_secs[i]
            velocity = t / dt
            speed = np.linalg.norm(velocity[:2])

            if np.abs(t[2]) > np.max(np.abs(t)):
                print(f'z is max t = ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}) speed = {speed:.2f}')

            if speed < self.heading_speed_thresh_ms:
                print(f'speed: {speed:.2f} velocity: {velocity[0]:.2f} {velocity[1]:.2f}')
                print(f"R should be identity", R)
                R = np.eye(3)
                t = np.zeros((3,))

            relative_transform = np.eye(4)
            relative_transform[:3, :3] = R
            relative_transform[:3, 3] = t


            # # Apply inverse to get current frame pose in world coordinates
            refined_pose21 = reference_pose @ np.linalg.inv(relative_transform)
            refined_poses21.append(refined_pose21)

        refined_poses3 = []
        refined_poses3_p2m = []
        refined_poses3_trimesh = []
        ref_object_points = points_rigid_transform(reference_points, np.linalg.inv(reference_pose))
        reference_object_mesh = trimesh.convex.convex_hull(ref_object_points)

        inlier_tallies = [np.zeros((len(x.original_points),), int) for x in self.history]
        inlier_tallies = [np.zeros((len(x),), int) for x in points_list]
        
        for i in range(len(points_list)):
            cur_points = points_list[i]

            # Use the pose from the previous refinement step as the starting point
            cur_world_to_object = np.linalg.inv(refined_poses21[i])
            cur_points_object = points_rigid_transform(cur_points, cur_world_to_object)

            # Find the final correction needed to align the local points to the canonical model
            R, t, _, _, _ = relative_object_pose_multiresolution(
                cur_points_object, ref_object_points, debug=False
            )

            initial_rel_pose = np.eye(4)
            initial_rel_pose[:3, :3] = R
            initial_rel_pose[:3, 3] = t

            object_relative_pose_p2m = self.refine_pose_point_to_mesh(
                cur_points_object, reference_object_mesh, initial_rel_pose
            )

            object_relative_pose_trimesh, _, _ = trimesh.registration.icp(cur_points_object, ref_object_points, initial=initial_rel_pose, threshold=1e-05, max_iterations=20)

            transformed_cur = points_rigid_transform(cur_points_object, object_relative_pose_trimesh)
            row_indices, col_indices, distances = bidirectional_matching(transformed_cur, ref_object_points)

            threshold = max(0.3, np.quantile(distances, 0.75))    
            mask = distances <= threshold
            inliersi = row_indices[mask]
            inliersj = col_indices[mask]


            inlier_tallies[i][inliersi] += 1
            inlier_tallies[reference_idx][inliersj] += 1

            # Check if rotation is reasonable (not too large)
            angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
            if angle > np.pi/6:  # More than 30 degrees seems suspicious
                print(f"Large rotation detected: {np.degrees(angle):.1f}°")
              
            # Build the correction transform
            object_relative_pose = np.eye(4)
            object_relative_pose[:3, :3] = R
            object_relative_pose[:3, 3] = t

            # Combine the correction with the previous inverse pose to get a new, better inverse pose
            new_world_to_object_pose = object_relative_pose @ cur_world_to_object

            # Invert the final inverse pose to get the correct forward pose for rendering in the world
            correct_final_pose = np.linalg.inv(new_world_to_object_pose)

            new_world_to_object_pose_p2m = object_relative_pose_p2m @ cur_world_to_object
            correct_final_pose_p2m = np.linalg.inv(new_world_to_object_pose_p2m)

            new_world_to_object_pose_trimesh = object_relative_pose_trimesh @ cur_world_to_object
            correct_final_pose_trimesh = np.linalg.inv(new_world_to_object_pose_trimesh)

            refined_poses3_trimesh.append(correct_final_pose_trimesh)

            
            refined_poses3.append(correct_final_pose)
            refined_poses3_p2m.append(correct_final_pose_p2m)

        object_centric_points_pose21 = []
        object_centric_points_pose3 = []
        object_centric_points_cumpose = []
        object_centric_points_flowpose = []
        object_centric_points_pose3p2m = []
        object_centric_points_pose3_trimesh = []




        for i in range(n_frames):
            cur_points = points_list[i].copy()

            cur_tallies = inlier_tallies[i]
            print(f"Frame {i} tallies {np.min(cur_tallies)} {np.median(cur_tallies)} {np.max(cur_tallies)}")
            if i != reference_idx:
                inlier_indices = np.where(cur_tallies > 0)[0]
            else:
                inlier_indices = np.where(cur_tallies >= np.median(cur_tallies))[0]

            cur_points = cur_points[inlier_indices]

            object_centric_points_pose21.append(points_rigid_transform(cur_points, np.linalg.inv(refined_poses21[i])))
            object_centric_points_pose3.append(points_rigid_transform(cur_points, np.linalg.inv(refined_poses3[i])))
            object_centric_points_cumpose.append(points_rigid_transform(cur_points, np.linalg.inv(cumulative_poses[i])))
            object_centric_points_flowpose.append(points_rigid_transform(cur_points, np.linalg.inv(flow_poses[i])))
            object_centric_points_pose3p2m.append(points_rigid_transform(cur_points, np.linalg.inv(refined_poses3_p2m[i])))
            object_centric_points_pose3_trimesh.append(points_rigid_transform(cur_points, np.linalg.inv(refined_poses3_trimesh[i])))

        object_centric_points_pose21 = np.concatenate(object_centric_points_pose21, axis=0)
        object_points_pose3 = np.concatenate(object_centric_points_pose3, axis=0)
        object_centric_points_cumpose = np.concatenate(object_centric_points_cumpose, axis=0)
        object_centric_points_flowpose = np.concatenate(object_centric_points_flowpose, axis=0)
        object_centric_points_pose3p2m = np.concatenate(object_centric_points_pose3p2m, axis=0)
        object_centric_points_pose3_trimesh = np.concatenate(object_centric_points_pose3_trimesh, axis=0)

        iou_refined_poses21 = compute_bev_iou(reference_object_points, object_centric_points_pose21)
        iou_refined_poses3 = compute_bev_iou(reference_object_points, object_points_pose3)
        iou_cumulative_pose = compute_bev_iou(reference_object_points, object_centric_points_cumpose)
        iou_flowpose = compute_bev_iou(reference_object_points, object_centric_points_flowpose)
        iou_refined_pose3p2m = compute_bev_iou(reference_object_points, object_centric_points_pose3p2m)
        iou_refined_pose3_trimesh = compute_bev_iou(reference_object_points, object_centric_points_pose3_trimesh)

        self.canonical_ious = dict(iou_refined_poses21=iou_refined_poses21,
            iou_refined_poses3=iou_refined_poses3, iou_cumulative_pose=iou_cumulative_pose, 
            iou_flowpose=iou_flowpose, iou_refined_pose3p2m=iou_refined_pose3p2m, iou_refined_pose3_trimesh=iou_refined_pose3_trimesh)

        pose_data = {
            'poses21': {
                'iou': iou_refined_poses21,
                'points': object_centric_points_pose21,
                'poses': refined_poses21,
            },
            'poses3': {
                'iou': iou_refined_poses3,
                'points': object_points_pose3,
                'poses': refined_poses3,
            },
            'cumpose': {
                'iou': iou_cumulative_pose,
                'points': object_centric_points_cumpose,
                'poses': cumulative_poses,
            },
            'flowpose': {
                'iou': iou_flowpose,
                'points': object_centric_points_flowpose,
                'poses': flow_poses,
            },
            'pose3p2m': {
                'iou': iou_refined_pose3p2m,
                'points': object_centric_points_pose3p2m,
                'poses': refined_poses3_p2m,
            },
            'pose3_trimesh': {
                'iou': iou_refined_pose3_trimesh,
                'points': object_centric_points_pose3_trimesh,
                'poses': refined_poses3_trimesh,
            }
        }

        best_method_name, best_method_data = max(pose_data.items(), key=lambda x: x[1]['iou'])

        # decide which one to use
        object_points = best_method_data['points']
        optimized_poses = best_method_data['poses']

        mesh: trimesh.Trimesh = trimesh.convex.convex_hull(object_points)

        orig_positions = self.positions

        self.flow_positions = np.stack([x[:3, 3] for x in flow_poses], axis=0)

        self.refined_positions = np.stack([x[:3, 3] for x in refined_poses3_trimesh], axis=0)
        self.refined_positions2 = np.stack([x[:3, 3] for x in refined_poses21], axis=0)
        self.refined_positions3_trimesh = np.stack([x[:3, 3] for x in refined_poses3_trimesh], axis=0)
        self.cumulative_positions = np.stack([x[:3, 3] for x in cumulative_poses], axis=0)

        refined_mse = np.linalg.norm(self.refined_positions - orig_positions, axis=1).mean()
        refined2_mse = np.linalg.norm(self.refined_positions2 - orig_positions, axis=1).mean()
        refined3_trimesh_mse = np.linalg.norm(self.refined_positions3_trimesh - orig_positions, axis=1).mean()
        cumulative_mse = np.linalg.norm(self.cumulative_positions - orig_positions, axis=1).mean()

        print(f"{refined_mse=:.2f} {refined2_mse=:.2f} {refined3_trimesh_mse=:.2f} {cumulative_mse}")

        dims_mins, dims_maxes = object_points.min(axis=0), object_points.max(axis=0)
        self.lwh = dims_maxes - dims_mins
        l, w, h = self.lwh

        optimized_boxes = []
        for i, obj_pose in enumerate(optimized_poses):
            yaw = Rotation.from_matrix(obj_pose[:3, :3]).as_rotvec()[2]
            x, y, z = obj_pose[:3, 3]
            box = np.array([x, y, z, l, w, h, yaw])

            optimized_boxes.append(box)

        self.optimized_boxes = optimized_boxes
        self.box_predictor.update(timestamps, optimized_boxes)
        self.optimized_poses = optimized_poses
        self.object_points = object_points
        self.object_centric_points_pose3_trimesh = object_centric_points_pose3_trimesh
        self.object_mesh = mesh

        # plot all the meshes and the merged meshes etc
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 4))

        ### ############################################################

        


        points_dict = {
            "pose21": object_centric_points_pose21,
            "pose3": object_points_pose3,
            # "cumpose": object_centric_points_cumpose,
            # "flowpose": object_centric_points_flowpose,
            "pose3p2m": object_centric_points_pose3p2m,
            "pose3_trimesh": object_centric_points_pose3_trimesh,
        }

        object_points_color = plt.cm.tab20c(np.linspace(0, 1, len(points_dict)))

        for idx, (points_name, object_points) in enumerate(points_dict.items()):
            ax2.scatter(
                object_points[:, 0],
                object_points[:, 1],
                s=3,
                c=object_points_color[idx],
                label=points_name,
                alpha=0.5,
            )

            hull = ConvexHull(object_points[:, :2])
            vertices_2d = object_points[hull.vertices, :2]

            polygon = patches.Polygon(
                vertices_2d,
                linewidth=2,
                edgecolor=object_points_color[idx],
                facecolor="none",
                # label=points_name,
                alpha=0.5,
            )
            ax2.add_patch(polygon)   

        ax2.legend()
            

        ### ############################################################
        hull = ConvexHull(reference_mesh1.vertices[:, :2])
        vertices_2d = reference_mesh1.vertices[hull.vertices, :2]

        polygon = patches.Polygon(
            vertices_2d,
            linewidth=2,
            edgecolor="orange",
            facecolor="none",
            alpha=0.7,
        )
        ax1.add_patch(polygon)    


        colors = plt.cm.tab20(np.linspace(0, 1, n_frames))
        for frame in range(n_frames):
            points = points_list[frame]

            ax1.scatter(
                points[:, 0],
                points[:, 1],
                s=3,
                c=colors[frame],
                label="Lidar Points",
                alpha=0.5,
            )
            
            hull = ConvexHull(points[:, :2])
            vertices_2d = points[hull.vertices, :2]

            polygon = patches.Polygon(
                vertices_2d,
                linewidth=2,
                edgecolor=colors[frame],
                facecolor="none",
                alpha=0.7,
            )
            ax1.add_patch(polygon)    

        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("X (meters)", fontsize=12)
        ax1.set_ylabel("Y (meters)", fontsize=12)

        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("X (meters)", fontsize=12)
        ax2.set_ylabel("Y (meters)", fontsize=12)
        plt.tight_layout()


        save_folder = Path("./compute_poses/")
        save_folder.mkdir(exist_ok=True)
        save_path = save_folder / f"track_{self.track_id}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


    def to_mesh(self, timestamp: int) -> Optional[trimesh.Trimesh]:
        if timestamp in self.timestamps:
            index = self.timestamps.index(timestamp)

            pose = self.optimized_poses[index]

            mesh = self.object_mesh.copy().apply_transform(pose)

            return mesh

        return None

    def refine_pose_point_to_mesh(self, points, mesh, initial_rel_pose, max_iterations=10):
        """
        Refine pose using point-to-mesh ICP instead of mesh-to-mesh
        """
        current_rel_pose = initial_rel_pose.copy()

        last_mean_error = np.inf
        

        vertices_3d = mesh.vertices
        vertices_2d = vertices_3d[:, :2]

        hull = ConvexHull(vertices_2d)
        vertices_2d = vertices_2d[hull.vertices]

        with tqdm(total=max_iterations, desc="refine_pose_point_to_mesh") as pbar:

            for iteration in range(max_iterations):
                # Transform points to current pose
                transformed_points = points_rigid_transform(points, current_rel_pose)
                
                # Find closest points on mesh surface
                # t0 = time.time()
                closest_points, distances, triangle_ids = trimesh.proximity.closest_point(
                    mesh, transformed_points
                )
                # t1 = time.time()
                # print(f"trimesh.proximity.closest_point took {(t1-t0):.2f} seconds for {len(transformed_points)} points")
                
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
                source_centre = source_pts.mean(axis=0)
                target_centre = target_pts.mean(axis=0)
                print("source_centre", source_centre)
                print("target_centre", target_centre)

                # fig, ax = plt.subplots(figsize=(4, 4))

                # ax.scatter(
                #     source_pts[:, 0],
                #     source_pts[:, 1],
                #     s=3,
                #     c="blue",
                #     label="Lidar Points",
                #     alpha=0.5,
                # )

                
                # ax.scatter(
                #     target_pts[:, 0],
                #     target_pts[:, 1],
                #     s=3,
                #     c="green",
                #     label="Lidar Points",
                #     alpha=0.5,
                # )


                # # Create polygon patch for alpha shape
                # polygon = patches.Polygon(
                #     vertices_2d,
                #     linewidth=2,
                #     edgecolor="purple",
                #     facecolor="none",
                #     alpha=0.7,
                # )
                # ax.add_patch(polygon)    

                # ax.set_aspect("equal")
                # plt.tight_layout()

                # save_folder = Path("./refine_pose_point_to_mesh/")
                # save_folder.mkdir(exist_ok=True)
                # save_path = save_folder / f"track_{self.track_id}_{iteration}.png"
                # plt.savefig(save_path, dpi=100, bbox_inches="tight")
                # plt.close()


                R_rel, t_rel = analytical_z_rotation_centered(source_pts, target_pts)

                
                # Update pose
                rel_transform = np.eye(4)
                rel_transform[:3, :3] = R_rel
                rel_transform[:3, 3] = t_rel
                
                current_rel_pose = rel_transform @ current_rel_pose

                # Check convergence
                mean_error = np.mean(distances[inliers])
                if abs(mean_error - last_mean_error) < 0.1:  # 10cm threshold
                    break

                last_mean_error = mean_error
            
                pbar.update(1)
                pbar.set_postfix({
                    "mean_error": f"{mean_error:.2f}",
                    "distance_threshold": f"{distance_threshold:.2f}",
                    "inliers": f"{np.sum(inliers)}/{len(inliers)}",
                })

            pbar.update(max_iterations)

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
        self.min_box_iou = 0.5

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

    def calculate(self, k=5):
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

                edge_candidates = []

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

                        # if semantic_iou < self.min_semantic_threshold:
                        #     continue


                        # between
                        new_timestamp = (object_i.timestamp + object_j.timestamp) // 2
                        object_i_box_extrapolated_centre = object_i.to_timestamped_box(new_timestamp)
                        object_j_box_extrapolated_centre = object_j.to_timestamped_box(new_timestamp)

                        iou = box_iou_3d(object_i_box_extrapolated_centre, object_j_box_extrapolated_centre)

                        if iou < self.min_box_iou:
                            continue

                        dt1 = (new_timestamp - start_time) * NANOSEC_TO_SEC
                        dt2 = (new_timestamp - end_time) * NANOSEC_TO_SEC
                        dlt_dt = 0.1

                        dt_delta1 = dt1 / dlt_dt
                        dt_delta2 = dt2 / dlt_dt

                        # object_i_points_ = object_i.original_points.copy() + object_i.flow.copy() * dt_delta1
                        # object_j_points = object_j.original_points.copy() + object_j.flow.copy() * dt_delta2

                        # points_dists = np.linalg.norm(object_i_points_[:, np.newaxis, :] - object_j_points[np.newaxis, :, :], axis=2)
                        # points_dists = points_dists.min(axis=1)

                        # assert points_dists.shape[0] == object_i_points_.shape[0], f"{points_dists.shape=} {object_i_points_.shape=}"

                        # dist = points_dists.mean()

                        dist = np.linalg.norm(object_i.box[:3] - object_j.box[3])

                        dists_all.append(dist)

                        best_ious[i] = max(best_ious[i], iou)
                        best_ious[j] = max(best_ious[j], iou)

                        if dist > 1.0: # flow distance too far...
                            continue

                        cost = (1.0 - iou) + dist + (1.0 - semantic_iou) + (time_j_secs - time_i_secs)


                        edge_candidates.append((j, cost))

                        ious.append(iou)
                        # ious_dflt.append(iou_dflt)
                        # ious_tmd_centre.append(iou_tmd_centre)
                        semantic_ious.append(semantic_iou)
                        dists.append(dist)
                        time_difference.append(time_j_secs - time_i_secs)

                # find the best 5 candidates
                edge_candidates.sort(key=lambda x: x[1])

                for j, cost in edge_candidates[:k]:
                    rows.append(i)
                    cols.append(j)
                    costs.append(cost)


        rows = np.array(rows, int)
        cols = np.array(cols, int)
        costs = np.array(costs, float)

        print('costs', costs.shape, costs.min(), costs.max())
        graph = csr_array((costs, (rows, cols)), shape=(N, N))

        print("ious", np.min(ious), np.mean(ious), np.max(ious))
        # print("ious_dflt", np.min(ious_dflt), np.mean(ious_dflt), np.max(ious_dflt))
        # print("ious_tmd_centre", np.min(ious_tmd_centre), np.mean(ious_tmd_centre), np.max(ious_tmd_centre))
        # print("ious_tmd_left", np.min(ious_tmd_left), np.mean(ious_tmd_left), np.max(ious_tmd_left))
        # print("ious_tmd_right", np.min(ious_tmd_right), np.mean(ious_tmd_right), np.max(ious_tmd_right))
        print("semantic_ious", np.min(semantic_ious), np.mean(semantic_ious), np.max(semantic_ious))
        print("dists", np.min(dists), np.mean(dists), np.max(dists))
        print("dists_all", np.min(dists_all), np.mean(dists_all), np.max(dists_all))
        print("time_difference", np.min(time_difference), np.mean(time_difference), np.max(time_difference))
        print("best_ious", np.min(best_ious), np.mean(best_ious), np.max(best_ious))

        all_boxes = [obj.box for obj in self.objects]

        paths = find_tracks_parallel_dijkstra(graph, all_boxes, self.object_timestamps)

        track_lengths = []
        track_lengths_ns = []
        truck_paths = []
        truck_tracks: List[GlobalTrack] = []
        pr = cProfile.Profile()
        pr.enable()
        canonical_ious = defaultdict(list)
        best_canonical_ious = []
        for track_id, path in enumerate(paths):
            history = [self.objects[i] for i in path]

            track_lengths.append(len(path))

            times = [self.object_timestamps[idx] for idx in path]

            min_time = min(times)
            max_time = max(times)

            track_lengths_ns.append(max_time - min_time)

            track = GlobalTrack(track_id, history)

            track.compute_poses()

            best_iou_name = ""
            best_iou = 0.0

            for iou_name, cur_iou in track.canonical_ious.items():
                print(f"{iou_name}: {cur_iou:.2f}")

                canonical_ious[iou_name].append(cur_iou)

                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_iou_name = iou_name

            best_canonical_ious.append(best_iou_name)

            for iou_name, ious in canonical_ious.items():
                print(f"{iou_name}: min: {np.min(ious):.2f} mean: {np.mean(ious):.2f} max: {np.max(ious):.2f}")

            iou_counter = Counter(best_canonical_ious)
            print("best_canonical_ious", iou_counter.most_common())

            self.tracks.append(track)


            cur_nodes = set(path)
            if len(cur_nodes.intersection(close_indices)) > 0:
                print("close to truck has path", path)
                truck_paths.append(path)
                truck_tracks.append(track)

        pr.disable()

        s = io.StringIO()
        sortby = "cumtime"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open("_global_tracker.calculate.txt", "w") as f:
            f.write(s.getvalue())

        fig, ax = plt.subplots(figsize=(8, 8))

        colors = plt.cm.tab20(np.linspace(0, 1, len(truck_paths)))
        for truck_id, (path, truck_track) in enumerate(zip(truck_paths, truck_tracks)):

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
            
            for box in truck_track.optimized_boxes:
                corners = get_box_bev_corners(box)
                obj_polygon = patches.Polygon(
                    corners,
                    linewidth=1,
                    edgecolor=colors[truck_id],
                    facecolor="none",
                    alpha=0.7,
                    linestyle="dotted",
                )
                ax.add_patch(obj_polygon)          


            ax.plot(truck_track.positions[:, 0], truck_track.positions[:, 1], alpha=0.7, linewidth=1, color="brown", linestyle='-', label='positions')
            # ax.plot(truck_track.flow_positions[:, 0], truck_track.flow_positions[:, 1], alpha=0.7, linewidth=1, color="red", linestyle='-')
            ax.plot(truck_track.refined_positions2[:, 0], truck_track.refined_positions2[:, 1], alpha=0.7, linewidth=1, color="green", linestyle='-', label='refined_positions2')
            ax.plot(truck_track.cumulative_positions[:, 0], truck_track.cumulative_positions[:, 1], alpha=0.7, linewidth=1, color="orange", linestyle='-', label='cumulative_positions')      
            ax.plot(truck_track.refined_positions3_trimesh[:, 0], truck_track.refined_positions3_trimesh[:, 1], alpha=0.7, linewidth=1, color="red", linestyle='-', label='refined_positions3_trimesh')      


        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)
        # ax.set_title(f"dist:{dist:.2f} semantic_iou:{semantic_iou:.2f} iou:{iou:.2f}", fontsize=14, fontweight="bold")

        plt.tight_layout()

        plt.xlim(truck_pos[0] + -20, truck_pos[0] + 20)
        plt.ylim(truck_pos[1] + -20, truck_pos[1] + 20)


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

def find_tracks_parallel_dijkstra(graph, all_boxes, timestamps):
    """
    Process all timestamp groups simultaneously, then resolve conflicts
    """
    n = len(timestamps)
    available_nodes = set(range(n))
    all_tracks = []
    
    unique_times = np.unique(timestamps)
    tidx = 0

    tracks_removed = 0

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
            tracks, used_nodes, n_removed = resolve_track_conflicts(
                source_indices, source_nodes, dist_matrix, pred_matrix,
                available_list, timestamps, all_boxes
            )

            tracks_removed += n_removed
            
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
                'timestamp_idx': f"{tidx}/{len(unique_times)}",
                "tracks_removed": tracks_removed
            })
        
        # Handle any remaining nodes
        if available_nodes:
            pbar.update(len(available_nodes))
    
    return all_tracks

def resolve_track_conflicts(source_indices, source_nodes, dist_matrix, pred_matrix, 
                          available_list, timestamps, all_boxes, iou_threshold: float = 0.5, dist_thresh: float = 0.5):
    unique_times = np.unique(timestamps)
    local_to_position = {local_idx: pos for pos, local_idx in enumerate(source_indices)}

    assert len(timestamps) == len(all_boxes)
    
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
    
    position_trees: Dict[int, cKDTree] = {}
    for t0 in unique_times:
        indices = np.where(timestamps == t0)[0]
        box_positions = np.stack([all_boxes[i][:3] for i in indices], axis=0)
        position_trees[t0] = cKDTree(box_positions)

    if valid_paths:
        # sort longest to shortest time span then min to max cost. (hence negative x[3])
        valid_paths = sorted(valid_paths, key=lambda x: (x[3] / x[0]), reverse=False)
        # avg_costs = [x[-]]
        
        best_paths = []
        used_nodes = set()

        path_sets: List[set] = []
        n_removed = 0
        
        for time_span, source_time, max_timestamp, cost, path_local, start_idx, last_idx in valid_paths:
            path_global = [available_list[i] for i in path_local]
            cur_nodes = set(path_global)

            best_iou = 0.0
            for path_set in path_sets:
                inter = len(path_set.intersection(cur_nodes))
                union = len(path_set.union(cur_nodes))

                if union == 0.0:
                    iou = 0.0
                else:
                    iou = inter / union

                best_iou = max(best_iou, iou)

            overlap = 0
            for node in cur_nodes:
                t0 = timestamps[node]

                indices = position_trees[t0].query_ball_point(all_boxes[node][:3], dist_thresh, p=2.0)

                for nbr in indices:
                    if nbr in used_nodes:
                        overlap += 1
                        break

            overlap = overlap / len(cur_nodes)

            if best_iou >= iou_threshold or overlap >= iou_threshold:
                used_nodes.update(path_global)
                path_sets.append(cur_nodes)
                n_removed += 1
                continue  # nodes already used - this is a duplicate path
            
            used_nodes.update(path_global)
            path_sets.append(cur_nodes)

            best_paths.append(path_global)

        return best_paths, used_nodes, n_removed
    
    return [], set(), 0



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

def compute_bev_iou(points1, points2) -> float:
    # Project to BEV (XY plane only)
    shape1_bev = points1[:, :2]  # Take only X,Y coordinates
    shape2_bev = points2[:, :2]

    # Create convex hulls in BEV as proxy for alpha shapes
    # Note: This is a simplification - ideally we'd use actual alpha shapes in 2D
    hull1 = MultiPoint(shape1_bev).convex_hull
    hull2 = MultiPoint(shape2_bev).convex_hull

    # Handle degenerate cases (points, lines)
    if hull1.area == 0 or hull2.area == 0:
        return 0.0

    # Compute intersection and union
    intersection = hull1.intersection(hull2).area
    union = hull1.union(hull2).area

    # Return IoU
    return intersection / union if union > 0 else 0.0