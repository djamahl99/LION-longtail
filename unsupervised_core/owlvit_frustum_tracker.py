import os
from pathlib import Path
import pickle as pkl
import numpy as np
from tqdm import tqdm, trange
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
from collections import defaultdict, deque
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.optimize import linear_sum_assignment

import pandas as pd

from pprint import pprint

from av2.map.map_api import ArgoverseStaticMap

from av2.map.lane_segment import LaneSegment
import av2.rendering.vector as vector_plotting_utils
import av2.geometry.polyline_utils as polyline_utils
from av2.datasets.sensor.constants import RingCameras, StereoCameras
from av2.datasets.sensor.sensor_dataloader import (
    SensorDataloader,
    SynchronizedSensorData,
)
from av2.rendering.color import ColorFormats, create_range_map
from av2.rendering.rasterize import draw_points_xy_in_img
from av2.structures.sweep import Sweep
from av2.utils.io import read_city_SE3_ego
from av2.utils.io import read_ego_SE3_sensor, read_feather
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.timestamped_image import TimestampedImage
from av2.geometry.camera.pinhole_camera import PinholeCamera


from .outline_utils import (
    OutlineFitter,
    TrackSmooth,
    voxel_sampling,
    points_rigid_transform,
)

from .box_utils import apply_pose_to_box, quat_to_yaw, get_rotated_box, points_in_frustum, find_connected_components_lidar

from sklearn.cluster import DBSCAN

def adjust_color(color, factor=0.8):
    """Lighten or darken a given matplotlib color by a factor."""
    c = mcolors.to_rgb(color)
    return tuple(min(1, max(0, channel * factor)) for channel in c)

def plot_frustum(ax, corners_ego, color):
    # Use only x,y
    corners = corners_ego[:, :2]

    # Near (0-3), Far (4-7)
    near = corners[:4]
    far = corners[4:]

    # Order for rectangle plotting (close the loop)
    rect_order = [0, 1, 3, 2, 0]

    near_color = adjust_color(color, 1.2)   # lighter
    far_color  = adjust_color(color, 0.6)   # darker
    conn_color = adjust_color(color, 0.9)   # slightly different tone

    # Plot near rectangle
    ax.plot(near[rect_order, 0], near[rect_order, 1], '-', color=near_color)
    # Plot far rectangle
    ax.plot(far[rect_order, 0], far[rect_order, 1], '-', color=far_color)

    # Connect near ↔ far corners
    for i in range(4):
        ax.plot([near[i, 0], far[i, 0]],
                [near[i, 1], far[i, 1]],
                '--', color=conn_color)

    ax.scatter(corners_ego[:, 0], corners_ego[:, 1], color=color, s=10)

def filter_frustum_points_clustering(
    points_3d: np.ndarray,
    eps: float = 1.5,
    min_samples: int = 5,
    return_largest_cluster: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter points using DBSCAN clustering to find main object cluster.

    Args:
        points_3d: Nx3 array of 3D points in frustum
        eps: DBSCAN neighborhood radius
        min_samples: Minimum points per cluster
        return_largest_cluster: If True, return largest cluster; if False, return all non-noise

    Returns:
        indices
    """
    if len(points_3d) < min_samples:
        return None
    # Perform clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points_3d)

    if return_largest_cluster:
        # Find largest cluster (excluding noise labeled as -1)
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique_labels) == 0:
            return None

        largest_cluster_label = unique_labels[np.argmax(counts)]
        mask = labels == largest_cluster_label
    else:
        # Return all non-noise points
        mask = labels >= 0

    return np.where(mask)[0]


def cluster_by_depth_gaps(points_cam, gap_threshold=1.5, min_cluster_size=310):
    """
    Cluster points based on depth discontinuities - much faster than full DBSCAN.

    The key insight: objects at different depths are usually different physical entities.
    By sorting points by depth and finding large gaps, we can efficiently separate
    distinct objects without expensive spatial clustering.
    """
    if len(points_cam) < min_cluster_size:
        return []

    # Extract depth (assuming camera coordinate system where Z is depth)
    depths = points_cam[:, [2]]  # Z coordinate represents depth from camera

    depth_min = depths.min()
    depth_max = depths.max()

    num_bins = int((depth_max - depth_min) / gap_threshold)
    num_bins = max(num_bins, 1)
    # print(f"{num_bins=}")

    _, bin_edges_equal_width = np.histogram(depths, bins=num_bins)

    all_clusters = []

    # print("bin_edges_equal_width", bin_edges_equal_width)
    for bin_idx in range(len(bin_edges_equal_width) - 1):
        bin_left = bin_edges_equal_width[bin_idx]
        bin_right = bin_edges_equal_width[bin_idx + 1]

        indices = np.where(((depths >= bin_left) & (depths <= bin_right)))[0]

        all_clusters.append(indices)

    return all_clusters

    cluster_method = DBSCAN(gap_threshold, min_samples=min_cluster_size)

    cluster_method.fit(depths)

    num_instance = len(set(cluster_method.labels_)) - (
        1 if -1 in cluster_method.labels_ else 0
    )

    all_valid_clusters = []

    for i in range(num_instance):

        indices = np.where(cluster_method.labels_ == i)[0]

        if len(indices) >= min_cluster_size:
            all_valid_clusters.append(indices)

    if len(all_valid_clusters) == 0:
        print(
            f"depths did not create any clusters! {depths.min()} {np.quantile(depths, 0.1)} {np.quantile(depths, 0.5)} {np.quantile(depths, 0.75)} {depths.max()}"
        )
        all_valid_clusters.append(np.arange(len(depths)))

    return all_valid_clusters

    # # Sort points by depth to identify discontinuities
    # depth_sorted_indices = np.argsort(depths)
    # sorted_depths = depths[depth_sorted_indices]

    # # Find gaps larger than threshold between consecutive depth-sorted points
    # depth_gaps = np.diff(sorted_depths)
    # gap_locations = np.where(depth_gaps > gap_threshold)[0]

    # # Create cluster boundaries based on gap locations
    # cluster_boundaries = [0] + (gap_locations + 1).tolist() + [len(points_cam)]

    # clusters = []
    # for i in range(len(cluster_boundaries) - 1):
    #     start_idx = cluster_boundaries[i]
    #     end_idx = cluster_boundaries[i + 1]

    #     # Get the original indices for this depth range
    #     cluster_indices = depth_sorted_indices[start_idx:end_idx]

    #     if len(cluster_indices) >= min_cluster_size:
    #         clusters.append(cluster_indices)

    # return clusters


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B.
    Inputs:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      R: mxm rotation matrix
      t: mx1 translation vector
    """
    assert A.shape == B.shape

    # Get number of dimensions
    m = A.shape[1]

    # Translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Rotation matrix
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = centroid_B.T - R @ centroid_A.T

    return R, t


def icp(
    A,
    B,
    max_iterations=100,
    tolerance=1e-6,
    ret_err=False,
    fixed_indices=False,
    return_inliers=False,
):
    """
    Iterative Closest Point (ICP) algorithm: aligns point cloud A to point cloud B.
    Returns the final transformation from the original A to B.

    Args:
        A: Source point cloud (Nx3)
        B: Target point cloud (Mx3)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        ret_err: Whether to return the final error
        fixed_indices: Whether to use fixed correspondences
        return_inliers: Whether to return inlier indices

    Returns:
        Various combinations based on flags:
        - (R_final, t_final) if neither ret_err nor return_inliers
        - (R_final, t_final, A_transformed) if not ret_err and not return_inliers
        - (R_final, t_final, error) if ret_err and not return_inliers
        - (R_final, t_final, error, A_inliers, B_inliers) if ret_err and return_inliers
    """
    A = np.copy(A)
    B = np.copy(B)
    A_original = np.copy(A)  # Keep original for inlier calculation

    prev_error = float("inf")

    # Initialize final transformation
    R_final = np.eye(A.shape[1])
    t_final = np.zeros(A.shape[1])

    tree = cKDTree(B) if not fixed_indices else None
    final_indices = None
    final_distances = None

    for i in range(max_iterations):
        # Find the nearest neighbors in B for each point in A
        if not fixed_indices:
            distances, indices = tree.query(A)
        else:
            indices = np.arange(len(B))
            distances = np.linalg.norm(A - B, axis=1)

        # Store final correspondences
        final_indices = indices
        final_distances = distances

        # Compute the transformation
        R, t = best_fit_transform(A, B[indices])

        # Update the final transformation
        R_final = R @ R_final
        t_final = R @ t_final + t

        # Apply the transformation
        A = (R @ A.T).T + t

        # Check for convergence
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    # Handle return values based on flags
    if return_inliers and final_indices is not None and final_distances is not None:
        # Define inliers based on distance threshold
        # Use a reasonable threshold - you can adjust this based on your needs
        distance_threshold = np.percentile(
            final_distances, 75
        )  # Top 75% closest points
        # Alternatively, use a fixed threshold: distance_threshold = 0.1  # 10cm threshold

        inlier_mask = final_distances <= distance_threshold
        A_inlier_indices = np.where(inlier_mask)[0]
        B_inlier_indices = final_indices[inlier_mask]

        if ret_err:
            return R_final, t_final, prev_error, A_inlier_indices, B_inlier_indices
        else:
            return R_final, t_final, A_inlier_indices, B_inlier_indices

    if ret_err:
        return R_final, t_final, prev_error

    return R_final, t_final, A


def count_neighbors(ptc, trees, max_neighbor_dist=0.3):
    neighbor_count = {}
    for seq in trees.keys():
        neighbor_count[seq] = trees[seq].query_ball_point(
            ptc[:, :3], r=max_neighbor_dist, return_length=True
        )
    return np.stack(list(neighbor_count.values())).T


def compute_ephe_score(count):
    N = count.shape[1]
    P = count / (np.expand_dims(count.sum(axis=1), -1) + 1e-8)
    H = (-P * np.log(P + 1e-8)).sum(axis=1) / np.log(N)

    return H


def compute_ppscore(cur_frame, neighbor_traversals=None, max_neighbor_dist=0.3):

    trees = {}

    for seq_id, points in enumerate(neighbor_traversals):
        trees[seq_id] = cKDTree(points)

    count = count_neighbors(cur_frame, trees, max_neighbor_dist)

    H = compute_ephe_score(count)

    return H


class AlphafrustumUtils:
    """Utility functions for alpha frustum operations."""

    @staticmethod
    def compute_frustum_uv(points_uv: np.ndarray) -> Dict[str, Any]:
        """
        Compute 2D alpha frustum (convex hull) from 3D points.

        Args:
            points_3d: Nx3 array of 3D points

        Returns:
            Dictionary containing alpha frustum information
        """
        if len(points_uv) < 3:
            return None

        try:
            # Compute convex hull in 2D
            hull = ConvexHull(points_uv)

            # Get hull vertices in order
            hull_vertices = points_uv[hull.vertices]
            hull_vertices_mins = hull_vertices.min(axis=0)
            hull_vertices_maxes = hull_vertices.max(axis=0)

            hull_dims = hull_vertices_maxes - hull_vertices_mins

            frustum = {
                "vertices_2d": hull_vertices,
                "hull_indices": hull.vertices,
                "centroid_2d": np.mean(hull_vertices, axis=0),
                "original_points": points_uv,
            }

            return frustum

        except Exception as e:
            print(f"Error computing alpha frustum: {e}")
            return None

    @staticmethod
    def compute_voxel_set(points_3d: np.ndarray, voxel_size: float = 0.1) -> set:
        voxels = np.round(points_3d / voxel_size).astype(int)

        # Use sets for fast intersection/union
        voxel_set = set(map(tuple, voxels))

        return voxel_set

    @staticmethod
    def compute_frustum_2d(points_3d: np.ndarray) -> Dict[str, Any]:
        """
        Compute 2D alpha frustum (convex hull) from 3D points.

        Args:
            points_3d: Nx3 array of 3D points

        Returns:
            Dictionary containing alpha frustum information
        """
        if len(points_3d) < 3:
            return None

        # Extract 2D points for BEV
        points_2d = points_3d[:, :2]
        z_values = points_3d[:, 2]

        try:
            # Compute convex hull in 2D
            hull = ConvexHull(points_2d)

            # Get hull vertices in order
            hull_vertices = points_2d[hull.vertices]
            hull_vertices_mins = hull_vertices.min(axis=0)
            hull_vertices_maxes = hull_vertices.max(axis=0)

            hull_dims = hull_vertices_maxes - hull_vertices_mins

            frustum = {
                "vertices_2d": hull_vertices,
                "z_min": np.min(z_values),
                "z_max": np.max(z_values),
                "hull_indices": hull.vertices,
                "centroid_2d": np.mean(hull_vertices, axis=0),
                "area": hull.volume,  # In 2D, volume is area
                "original_points": points_3d,
                "voxel_set": AlphafrustumUtils.compute_voxel_set(points_3d),
                "volume": hull.volume * (np.max(z_values) - np.min(z_values)),
            }

            return frustum

        except Exception as e:
            print(f"Error computing alpha frustum: {e}")
            return None

    @staticmethod
    def polygon_iou_2d(frustum1: Dict, frustum2: Dict) -> float:
        """Calculate 2D IoU between two alpha frustums using polygon intersection."""
        try:
            poly1 = Polygon(frustum1["vertices_2d"])
            poly2 = Polygon(frustum2["vertices_2d"])

            if not poly1.is_valid:
                poly1 = poly1.buffer(0)
            if not poly2.is_valid:
                poly2 = poly2.buffer(0)

            intersection = poly1.intersection(poly2)
            union = poly1.union(poly2)

            if union.area == 0:
                return 0.0

            return intersection.area / union.area

        except Exception as e:
            print(f"Error calculating polygon IoU: {e}")
            return 0.0

    @staticmethod
    def z_overlap_ratio(frustum1: Dict, frustum2: Dict) -> float:
        """Calculate z-direction overlap ratio."""
        z1_min, z1_max = frustum1["z_min"], frustum1["z_max"]
        z2_min, z2_max = frustum2["z_min"], frustum2["z_max"]

        # Calculate intersection
        intersection_min = max(z1_min, z2_min)
        intersection_max = min(z1_max, z2_max)
        intersection_height = max(0, intersection_max - intersection_min)

        # Calculate union
        union_min = min(z1_min, z2_min)
        union_max = max(z1_max, z2_max)
        union_height = union_max - union_min

        if union_height == 0:
            return 1.0  # Both have same z

        return intersection_height / union_height

    @staticmethod
    def frustum_3d_iou(frustum1: Dict, frustum2: Dict) -> float:
        """Calculate 3D IoU combining 2D polygon IoU and z-overlap."""
        polygon_iou = AlphafrustumUtils.polygon_iou_2d(frustum1, frustum2)
        z_iou = AlphafrustumUtils.z_overlap_ratio(frustum1, frustum2)

        # Combine both IOUs (multiplicative approach)
        return polygon_iou * z_iou

    @staticmethod
    def voxel_iou(points1: np.ndarray, points2: np.ndarray, voxel_size=0.1):
        """Faster than mesh, more accurate than 2D+Z"""

        # Voxelize both point clouds
        voxels1 = np.round(points1 / voxel_size).astype(int)
        voxels2 = np.round(points2 / voxel_size).astype(int)

        # Use sets for fast intersection/union
        set1 = set(map(tuple, voxels1))
        set2 = set(map(tuple, voxels2))

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0
    
    @staticmethod
    def voxel_iou_from_sets(set1: set, set2: set) -> float:
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return float(intersection) / float(union) if union > 0 else 0.0

    @staticmethod
    def merge_frustums_with_icp(
        frustums_with_transforms: List[Tuple[Dict, np.ndarray]],
        ppscore_thresh: float = 0.7,
    ) -> Dict:
        """
        Merge multiple alpha frustums using ICP alignment.

        Args:
            frustums_with_transforms: List of (frustum, transform_matrix) tuples
        """
        if not frustums_with_transforms:
            return None
        if len(frustums_with_transforms) == 1:
            return frustums_with_transforms[0][0]

        # Collect all transformed points
        all_aligned_points = []

        last_points = None

        for frustum, transform_matrix in frustums_with_transforms:
            points_3d = frustum["original_points"].copy()

            # if 'original_points' in frustum and frustum['original_points'] is not None:
            #     points_3d = frustum['original_points'].copy()
            # else:
            #     # Reconstruct 3D points from 2D + z bounds
            #     vertices_2d = frustum['vertices_2d']
            #     z_center = (frustum['z_min'] + frustum['z_max']) / 2
            #     points_3d = np.column_stack([vertices_2d, np.full(len(vertices_2d), z_center)])

            last_points = points_3d

            # Apply stored transformation
            if transform_matrix is not None:
                # Convert to homogeneous coordinates
                points_homo = np.column_stack([points_3d, np.ones(len(points_3d))])
                transformed_points = (transform_matrix @ points_homo.T).T[:, :3]
                all_aligned_points.append(transformed_points)
            else:
                all_aligned_points.append(points_3d)

        # Combine all aligned points
        if all_aligned_points and last_points:
            # combined_points = np.vstack(all_aligned_points)
            ppscore = compute_ppscore(last_points, all_aligned_points)
            print(
                f"merge_frustums_with_icp ppscore={ppscore.shape} {ppscore.min()} {ppscore.mean()} {ppscore.max()}"
            )
            ppscore_mask = ppscore >= ppscore_thresh
            print("ppscore_mask", ppscore_mask.shape, ppscore_mask.sum())
            return AlphafrustumUtils.compute_frustum_2d(last_points[ppscore_mask])

        return None

    @staticmethod
    def compute_object_icp_transform(
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        ego_transform: np.ndarray,
        debug: bool = False,
        return_inlier_indices: bool = False,
    ) -> Union[
        Tuple[Optional[np.ndarray], float],
        Tuple[Optional[np.ndarray], float, Optional[np.ndarray], Optional[np.ndarray]],
    ]:
        """
        Compute object-specific ICP transformation between frames.

        Args:
            prev_points: Previous frame's original points (Nx3)
            curr_points: Current frame's original points (Nx3)
            ego_transform: 4x4 ego vehicle transformation matrix
            debug: Whether to print debug info
            return_inlier_indices: Whether to return indices of inlier points

        Returns:
            If return_inlier_indices=False:
                (combined_transform, icp_error) where combined_transform includes both ego and object motion
            If return_inlier_indices=True:
                (combined_transform, icp_error, prev_inlier_indices, curr_inlier_indices)
                where prev_inlier_indices and curr_inlier_indices are arrays of indices into the original
                prev_points and curr_points arrays respectively
        """
        if (
            prev_points is None
            or curr_points is None
            or len(prev_points) < 5
            or len(curr_points) < 5
        ):
            if debug:
                print(
                    f"    ICP: Insufficient points ({len(prev_points) if prev_points is not None else 0} -> {len(curr_points) if curr_points is not None else 0})"
                )
            if return_inlier_indices:
                return None, float("inf"), None, None
            return None, float("inf")

        try:
            # First transform previous points using ego motion
            prev_homo = np.column_stack([prev_points, np.ones(len(prev_points))])
            ego_transformed_prev = (ego_transform @ prev_homo.T).T[:, :3]

            # Keep track of original indices for subsampling
            prev_original_indices = np.arange(len(ego_transformed_prev))
            curr_original_indices = np.arange(len(curr_points))

            # Subsample if too many points (for efficiency)
            max_points = 200
            if len(ego_transformed_prev) > max_points:
                idx = np.random.choice(
                    len(ego_transformed_prev), max_points, replace=False
                )
                ego_transformed_prev = ego_transformed_prev[idx]
                prev_original_indices = prev_original_indices[idx]

            if len(curr_points) > max_points:
                idx = np.random.choice(len(curr_points), max_points, replace=False)
                curr_subset = curr_points[idx]
                curr_original_indices = curr_original_indices[idx]
            else:
                curr_subset = curr_points

            # Compute ICP between ego-transformed previous points and current points
            # Note: You'll need to modify your icp function to return inlier indices
            if return_inlier_indices:
                R, t, icp_error, prev_inlier_subset_idx, curr_inlier_subset_idx = icp(
                    ego_transformed_prev,
                    curr_subset,
                    max_iterations=50,
                    tolerance=1e-4,
                    ret_err=True,
                    return_inliers=True,
                )
            else:
                R, t, icp_error = icp(
                    ego_transformed_prev,
                    curr_subset,
                    max_iterations=50,
                    tolerance=1e-4,
                    ret_err=True,
                )

            if R is None or t is None:
                if debug:
                    print(f"    ICP: Failed to converge")
                if return_inlier_indices:
                    return None, float("inf"), None, None
                return None, float("inf")

            # Create 4x4 object transformation matrix
            object_transform = np.eye(4)
            object_transform[:3, :3] = R
            object_transform[:3, 3] = t

            # Combined transformation: first ego motion, then object-specific motion
            combined_transform = object_transform @ ego_transform

            if debug:
                print(
                    f"    ICP: Success - error: {icp_error:.3f}m, points: {len(ego_transformed_prev)}->{len(curr_subset)}"
                )
                if return_inlier_indices and prev_inlier_subset_idx is not None:
                    print(
                        f"    ICP: Inliers: {len(prev_inlier_subset_idx)} prev, {len(curr_inlier_subset_idx)} curr"
                    )

            if return_inlier_indices:
                # Map subset indices back to original indices
                if (
                    prev_inlier_subset_idx is not None
                    and curr_inlier_subset_idx is not None
                ):
                    prev_inlier_original_idx = prev_original_indices[
                        prev_inlier_subset_idx
                    ]
                    curr_inlier_original_idx = curr_original_indices[
                        curr_inlier_subset_idx
                    ]
                else:
                    prev_inlier_original_idx = None
                    curr_inlier_original_idx = None

                return (
                    combined_transform,
                    icp_error,
                    prev_inlier_original_idx,
                    curr_inlier_original_idx,
                )

            return combined_transform, icp_error

        except Exception as e:
            if debug:
                print(f"    ICP: Exception - {e}")
            if return_inlier_indices:
                return None, float("inf"), None, None
            return None, float("inf")

    @staticmethod
    def transform_frustum(frustum: Dict, transform_matrix: np.ndarray) -> Dict:
        """Transform alpha frustum using a 4x4 transformation matrix."""
        if frustum is None:
            return None

        # Transform 2D vertices to 3D, apply transform, project back to 2D
        vertices_2d = frustum["vertices_2d"]
        z_center = (frustum["z_min"] + frustum["z_max"]) / 2

        # Convert to homogeneous 3D coordinates
        vertices_3d = np.column_stack(
            [
                vertices_2d,
                np.full(len(vertices_2d), z_center),
                np.ones(len(vertices_2d)),
            ]
        )

        # Apply transformation
        transformed_vertices = (transform_matrix @ vertices_3d.T).T

        # Project back to 2D
        transformed_2d = transformed_vertices[:, :2]

        # Transform z bounds
        z_points = np.array(
            [[0, 0, frustum["z_min"], 1], [0, 0, frustum["z_max"], 1]]
        )
        transformed_z = (transform_matrix @ z_points.T).T
        new_z_min, new_z_max = transformed_z[0, 2], transformed_z[1, 2]

        # Create new alpha frustum
        transformed_frustum = frustum.copy()
        transformed_frustum["vertices_2d"] = transformed_2d
        transformed_frustum["z_min"] = min(new_z_min, new_z_max)
        transformed_frustum["z_max"] = max(new_z_min, new_z_max)
        transformed_frustum["centroid_2d"] = np.mean(transformed_2d, axis=0)

        # Recompute area if needed
        try:
            hull = ConvexHull(transformed_2d)
            transformed_frustum["area"] = hull.volume
        except:
            transformed_frustum["area"] = 0.0

        return transformed_frustum


class ModifiedSensorDataloader(SensorDataloader):
    """Extended SensorDataloader with direct access methods."""

    def get_closest_sweep_timestamp(self, log_id: str, split: str, timestamp_ns: int):
        # Construct paths
        log_dir = self.dataset_dir / split / log_id
        sensor_dir = log_dir / "sensors"
        lidar_folder = sensor_dir / "lidar"
        timestamps_all = np.array(
            [int(x.stem) for x in lidar_folder.rglob("*.feather")], dtype=int
        )

        if len(timestamps_all) == 0:
            raise Exception(f"Found no lidar data at {lidar_folder=}")

        timestamp_diffs = np.abs(timestamps_all - timestamp_ns)

        return timestamps_all[np.argmin(timestamp_diffs)]

    def get_sensor_data(
        self,
        log_id: str,
        split: str,
        timestamp_ns: int,
        cam_names: Optional[List[str]] = None,
    ) -> SynchronizedSensorData:
        """
        Load sensor data directly by log_id, split, and timestamp.

        Args:
            log_id: Log identifier
            split: Dataset split (train/val/test)
            timestamp_ns: Timestamp in nanoseconds
            cam_names: Optional list of camera names to load. If None, uses self.cam_names

        Returns:
            SynchronizedSensorData object containing the requested sensor data

        Raises:
            FileNotFoundError: If the specified lidar data doesn't exist
            ValueError: If the timestamp is not found in the log
        """

        # Use provided cam_names or fall back to instance cam_names
        if cam_names is None:
            cam_names = self.cam_names

        # Construct paths
        log_dir = self.dataset_dir / split / log_id
        sensor_dir = log_dir / "sensors"
        lidar_feather_path = sensor_dir / "lidar" / f"{timestamp_ns}.feather"

        # find the closest timestamp?

        # Verify lidar data exists
        if not lidar_feather_path.exists():
            lidar_folder = lidar_feather_path = sensor_dir / "lidar"
            timestamps_all = np.array(
                [int(x.stem) for x in lidar_folder.rglob("*.feather")], dtype=int
            )

            if len(timestamps_all) == 0:
                print("lidar_folder", lidar_folder)

            timestamp_diffs = np.abs(timestamps_all - timestamp_ns)
            print("timestamp_diffs.min()", timestamp_diffs.min())

            raise FileNotFoundError(f"Lidar data not found: {lidar_feather_path}")

        # Load sweep data
        sweep = Sweep.from_feather(lidar_feather_path=lidar_feather_path)

        # Load city SE3 ego transformations
        timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_dir)

        # Load map data
        avm = ArgoverseStaticMap.from_map_dir(log_dir / "map", build_raster=True)

        # Get sweep information for this log
        try:
            # Get all lidar records for this log to determine sweep number
            log_lidar_records = self.sensor_cache.xs((split, log_id, "lidar")).index
            num_frames = len(log_lidar_records)

            # Find the index of this timestamp
            matching_indices = np.where(log_lidar_records == timestamp_ns)[0]
            if len(matching_indices) == 0:
                raise ValueError(f"Timestamp {timestamp_ns} not found in log {log_id}")

            sweep_idx = matching_indices[0]

        except KeyError:
            # If log not in cache, we can't determine sweep number - use 0 as fallback
            print(
                f"Warning: Log {log_id} not found in sensor cache. Using default sweep number."
            )
            sweep_idx = 0
            num_frames = 1

        # Construct output datum
        datum = SynchronizedSensorData(
            sweep=sweep,
            log_id=log_id,
            timestamp_city_SE3_ego_dict=timestamp_city_SE3_ego_dict,
            sweep_number=sweep_idx,
            num_sweeps_in_log=num_frames,
            avm=avm,
            timestamp_ns=timestamp_ns,
        )

        # Load annotations if enabled
        if self.with_annotations:
            if split != "test":
                datum.annotations = self._load_annotations(split, log_id, timestamp_ns)

        # Load camera imagery if requested
        if cam_names:
            datum.synchronized_imagery = self._load_synchronized_cams(
                split, sensor_dir, log_id, timestamp_ns
            )

        return datum

class OWLViTFrustumTracker:
    def __init__(self, log_id: str, 
                 config, debug: bool = False):
        self.log_id = log_id
        self.config = config
        self.debug = debug
        self.tracked_objects = {}  # id -> tracked object
        self.next_id = 0
        self.all_pose = None
        self.frame_frustums = {}
        self.icp_fail_max = 3
        self.ppscore_thresh = 0.7
        self.track_query_eps = 5.0  # metres
        self.time_delta_ns = 1e+9 # 1 second in nanoseconds 
        
        gt_annotations_feather = Path(
            f"/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val/{log_id}/annotations.feather"
        )
        assert gt_annotations_feather.exists()
        gts = pd.read_feather(gt_annotations_feather)
        gts = gts.assign(
            log_id=pd.Series([log_id for _ in range(len(gts))], dtype="string").values
        )
        gts = gts.set_index(["log_id", "timestamp_ns"], drop=False).sort_values("category")

        self.gts = gts

    def track_frustums(self, all_frustums: List[Dict]):
        frame_assignments = []
        # Get timestamps
        timestamps = np.array([x['sweep_timestamp_ns'] for x in all_frustums], dtype=int)

        times_set = set(x['sweep_timestamp_ns'] for x in all_frustums)
        print('timestamps', len(timestamps), 'times_set', len(times_set))

        times_ordered = sorted(list(times_set))

        for time_idx, timestamp_ns in enumerate(tqdm(times_ordered[:10], desc="iterating over frustum times")): # TODO: remove :10
            frustum_indices = np.where(timestamps == timestamp_ns)[0]
            
            self._process_frame(time_idx, timestamp_ns, [all_frustums[idx] for idx in frustum_indices])
            
        # plot tracked frustums if we are debugging
        if self.debug:
            for track_id, track in self.tracked_objects.items():
                trajectory = track["trajectory"]
                timestamps = sorted(trajectory.keys())
                
                timestamp_poses = {timestamp_ns: x['pose'] for timestamp_ns, x in track['trajectory'].items()}

                # create one plot, then zoom in on each to save each
                if len(timestamps) > 1:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    
                    colors = plt.cm.tab20(np.linspace(0, 1, len(timestamps)))
                    timestamp_colors = {timestamp_ns: colors[i] for i, timestamp_ns in enumerate(timestamps)}
                    
                    for timestamp_ns in timestamps:
                        gt_frame = self.gts.loc[[(self.log_id, timestamp_ns)]]

                        # Plot GT boxes
                        for idx, (_, gt_row) in enumerate(gt_frame.iterrows()):
                            try:
                                pose = timestamp_poses[timestamp_ns]
                                center_xy, yaw, length, width = apply_pose_to_box(gt_row, pose)

                                # Get category for coloring
                                category = gt_row.get("category", "UNKNOWN")

                                # Get rotated box corners
                                corners = get_rotated_box(center_xy, length, width, yaw)

                                # Create polygon patch for GT (different style)
                                gt_polygon = patches.Polygon(
                                    corners,
                                    linewidth=3,
                                    edgecolor="red",
                                    facecolor=timestamp_colors[timestamp_ns],
                                    alpha=0.8,
                                    linestyle="-",
                                    label="Ground Truth" if _ == 0 else "",
                                )
                                ax.add_patch(gt_polygon)

                            except Exception as e:
                                print(f"Error plotting GT box: {e}")
                                continue

                    # Need multiple timestamps for consensus
                    for timestamp_ns, traj in trajectory.items():
                        frustum = traj['frustum']
                        corners_city = frustum['corners_city']
                        plot_frustum(ax, corners_city, timestamp_colors[timestamp_ns])
                        
                    # Extract the center from pose translation
                    pose = frustum['pose'] # last pose
                    center = pose[:2, 3]  # Get x, y translation from pose matrix

                    # Now zoom in centered on the pose
                    plt.xlim(center[0] - 50, center[0] + 50)
                    plt.ylim(center[1] - 50, center[1] + 50)
                        
                    ax.set_aspect("equal")
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel("X (meters)", fontsize=12)
                    ax.set_ylabel("Y (meters)", fontsize=12)

                    save_path = Path("./frustum_tracklets")
                    save_path.mkdir(exist_ok=True)
                    plt.savefig(save_path / f"tracklet_{track_id}.png")
                    plt.close()
                    
                    # icp + transform + ppscore
                    city_points_per_timestamp = []
                    log_dir = Path("/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val/") / self.log_id
                    lidar_folder = log_dir / "sensors" / "lidar"
                    
                    fig, ax = plt.subplots(figsize=(5, 5))

                    # Find the lidar timestamps
                    for sweep_timestamp_ns in timestamps:
                        lidar_feather_path = lidar_folder / f"{sweep_timestamp_ns}.feather"
                        lidar = read_feather(lidar_feather_path)

                        lidar_xyz = lidar.loc[:, ["x", "y", "z"]].to_numpy().astype(float)
                        
                        city_lidar = points_rigid_transform(lidar_xyz, timestamp_poses[sweep_timestamp_ns])
                        
                                            # for timestamp_ns, traj in track['trajectory'].items():
                        traj = trajectory[sweep_timestamp_ns]
                        frustum = traj['frustum']
                        corners_city = frustum['corners_city']
                        
                        lidar_mask = points_in_frustum(city_lidar, corners_city)
                        city_lidar_in_frustum = city_lidar[lidar_mask]
                        
                        city_points_per_timestamp.append(city_lidar_in_frustum)
                        

                    # Need multiple timestamps for consensus
                    for timestamp_ns, traj in trajectory.items():
                        frustum = traj['frustum']
                        corners_city = frustum['corners_city']
                        plot_frustum(ax, corners_city, timestamp_colors[timestamp_ns])

                    lidar_points = city_points_per_timestamp[-1]
                                       
                    ppscore = compute_ppscore(lidar_points, city_points_per_timestamp[:-1])

                    ax.scatter(
                        lidar_points[:, 0],
                        lidar_points[:, 1],
                        s=1,
                        c=ppscore,
                        cmap="jet",
                        label="Lidar Points",
                        alpha=0.5,
                    )
                        
                        
                    # Extract the center from pose translation
                    pose = frustum['pose'] # last pose
                    center = pose[:2, 3]  # Get x, y translation from pose matrix

                    # Now zoom in centered on the pose
                    plt.xlim(center[0] - 50, center[0] + 50)
                    plt.ylim(center[1] - 50, center[1] + 50)
                        
                    ax.set_aspect("equal")
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel("X (meters)", fontsize=12)
                    ax.set_ylabel("Y (meters)", fontsize=12)

                    save_path = Path("./frustum_tracklets")
                    save_path.mkdir(exist_ok=True)
                    plt.savefig(save_path / f"tracklet_{track_id}_ppscore.png")
                    plt.close()

                    city_points_inliers = []

                    fig_bev, ax_bev = plt.subplots(figsize=(8, 8))

                    # compute alpha shapes
                    # for time_i in range(len(timestamps) - 1):
                    for time_i in [0]:
                        time_j = time_i + 1
                        timestamp_ns = timestamps[time_i]
                            
                        cur_points = city_points_per_timestamp[time_i] if time_i >= len(city_points_inliers) else city_points_inliers[time_i]
                        other_points = city_points_per_timestamp[:time_idx] + city_points_per_timestamp[time_idx+1:]
                        
                        if len(cur_points) < 20:
                            continue

                        ppscore = compute_ppscore(cur_points, other_points)
                        ppscore_mask = ppscore >= 0.7

                        print(f"ppscore_mask={ppscore_mask.shape} sum={ppscore_mask.sum()} ppscore {ppscore.min()} {np.median(ppscore.reshape(-1))} {ppscore.mean()} {ppscore.max()}")
                        print(f"ppscore reduced")


                        cur_points = cur_points[ppscore_mask]

                        if len(cur_points) < 20:
                            continue

                        # find largest comnnected component
                        connected_labels, n_components = find_connected_components_lidar(cur_points)

                        num_per_component = np.array([(connected_labels==i).sum() for i in range(n_components)])
                        print("num_per_component", num_per_component)

                        best_component = np.argmax(num_per_component)
                        largest_component_mask = (connected_labels == best_component)

                        print("largest_component_mask", largest_component_mask.shape[0], largest_component_mask.sum())

                        cur_points = cur_points[largest_component_mask]


                        if len(cur_points) < 20:
                            continue 

                        # next points in sequence
                        next_points = city_points_per_timestamp[time_j]

                        R, t, A_inlier_indices, B_inlier_indices = icp(
                            cur_points, 
                            next_points, 
                            max_iterations=50,
                            return_inliers=True,
                            ret_err=False
                        )
                        
                        # Get ICP inlier points
                        icp_inliers = cur_points[A_inlier_indices]
                        city_points_inliers.append(icp_inliers)

                        # Plot ICP inliers in blue
                        ax_bev.scatter(icp_inliers[:, 0], icp_inliers[:, 1], 
                                    c='blue', s=2, alpha=0.7, label='ICP Inliers')
                        
                        # # Plot ConvexHull boundary
                        # hull = ConvexHull(icp_inliers[:, :2])  # Use only XY for 2D hull
                        # # hull_points = icp_inliers[hull.vertices]
                        # try:
                        #     for simplex in hull.simplices:
                        #         ax_bev.plot(icp_inliers[:, 0], icp_inliers[simplex, 1], color=timestamp_colors[timestamp_ns], linewidth=2)
                        # except:
                        #     pass
                        # Compute convex hull in 2D
                        hull = ConvexHull(icp_inliers[:, :2])

                        # TODO: do e.g. BFS by depth to find connected components.

                        # Get hull vertices in order
                        vertices_2d = icp_inliers[hull.vertices, :2]
                        polygon = patches.Polygon(
                            vertices_2d,
                            linewidth=2,
                            edgecolor="black",
                            facecolor='blue',
                            alpha=0.7,
                        )
                        ax_bev.add_patch(polygon)

                        if time_j == len(timestamps) - 1:
                            city_points_inliers.append(next_points[B_inlier_indices])

                        gt_frame = self.gts.loc[[(self.log_id, timestamp_ns)]]

                        # Plot GT boxes
                        for idx, (_, gt_row) in enumerate(gt_frame.iterrows()):
                            try:
                                pose = timestamp_poses[timestamp_ns]
                                center_xy, yaw, length, width = apply_pose_to_box(gt_row, pose)

                                # Get rotated box corners
                                corners = get_rotated_box(center_xy, length, width, yaw)

                                # Create polygon patch for GT (different style)
                                gt_polygon = patches.Polygon(
                                    corners,
                                    linewidth=3,
                                    edgecolor="red",
                                    facecolor=timestamp_colors[timestamp_ns],
                                    alpha=0.8,
                                    linestyle="-",
                                    label="Ground Truth" if _ == 0 else "",
                                )
                                ax.add_patch(gt_polygon)

                            except Exception as e:
                                print(f"Error plotting GT box: {e}")
                                continue

                    
                    # # Plot all points in gray
                    # ax_bev.scatter(cur_points[:, 0], cur_points[:, 1], 
                    #             c='lightgray', s=1, alpha=0.3, label='All Points')
                    

                    
                    # Plot frustum outline
                    # frustum = trajectory[timestamp_ns]['frustum']
                    # corners_city = frustum['corners_city']
                    # plot_frustum(ax_bev, corners_city, 'green')
                    
                    ax_bev.set_aspect('equal')
                    ax_bev.grid(True, alpha=0.3)
                    ax_bev.legend()
                    ax_bev.set_title(f'BEV - Track {track_id} - First Timestamp')
                    ax_bev.set_xlabel('X (meters)')
                    ax_bev.set_ylabel('Y (meters)')
                    
                    save_path = Path("./frustum_tracklets")
                    save_path.mkdir(exist_ok=True)
                    plt.savefig(save_path / f"tracklet_{track_id}_inliers.png", dpi=300)
                    plt.close()



    def _process_frame(self, time_idx: int, timestamp_ns: int, frustums: List[Dict]):
        """Process a single frame for tracking."""
        frame_assignments = []

        if time_idx == 0:
            # Initialize tracking for first frame
            for frustum in frustums:
                obj_id = self._create_new_track(timestamp_ns, frustum)
                frame_assignments.append(obj_id)

        else:
            # Match with existing tracks
            frame_assignments = self._match_frame(timestamp_ns, frustums)

        self.frame_frustums[timestamp_ns] = frame_assignments


    def  _match_frame(
        self, timestamp_ns: int, frustums: List[Dict]
    ) -> List[Tuple]:
        """Match frustums in current frame with existing tracks."""
        if not frustums:
            return []

        # Get active tracks from previous frame
        active_tracks = self._get_active_tracks(timestamp_ns)

        if self.debug:
            print(
                f"  Matching: {len(frustums)} detections vs {len(active_tracks)} active tracks"
            )

        if not active_tracks:
            # No active tracks, create new ones
            assignments = []
            for frustum in frustums:
                obj_id = self._create_new_track(timestamp_ns, frustum)
                # assignments.append((obj_id, frustum, {"score": 1.0}))
                assignments.append(obj_id)

            return assignments

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(active_tracks), len(frustums)))
        semantic_matrix = np.zeros((len(active_tracks), len(frustums)))

        if self.debug:
            print(
                f"  Computing {len(active_tracks)} x {len(frustums)} IoU matrix..."
            )

        pose = frustums[0]['pose']

        active_track_positions = []
        # collect positions of active alpha frustums
        for track_id in active_tracks:
            track = self.tracked_objects[track_id]

            last_timestamp = track['last_timestamp']

            last_data = track["trajectory"][last_timestamp]
            last_frustum = last_data["frustum"]
            last_pose = last_data["pose"]

            # Transform from last frame to current frame
            transform = np.linalg.inv(pose) @ last_pose

            last_corners_city = last_frustum['corners_city']
            # last_centre = (np.min(last_corners_city, axis=0) + np.max(last_corners_city, axis=0)) / 2.0

            # predicted_centre = points_rigid_transform(
            #     last_centre.reshape(1, 3), transform
            # )[0]

            # print("last_centre", last_centre)
            # print("predicted_centre", predicted_centre)
            # active_track_positions.append(predicted_centre)

            predicted_corners_city = points_rigid_transform(
                last_corners_city, transform
            )
        
            active_track_positions.append(predicted_corners_city.reshape(1, -1))

        active_track_positions = np.concatenate(active_track_positions, axis=0)
        print('active_track_positions', active_track_positions.shape)

        tracks_tree = cKDTree(active_track_positions)

        for j, current_frustum in enumerate(frustums):
            current_semantic_features = current_frustum.get("semantic_features", None)

            corners_city = current_frustum['corners_city']
            # centre = (np.min(corners_city, axis=0) + np.max(corners_city, axis=0)) / 2.0
            

            distances, indices = tracks_tree.query(
                corners_city.reshape(-1), k=10
            )
            
            print("distances", distances.shape, distances.min(), distances.mean(), distances.max())

            # indices = np.arange(len(active_tracks))
            # distances = np.zeros((len(active_tracks)))

            current_hull = MultiPoint(current_frustum["corners_city"]).convex_hull

            for i, distance in zip(indices, distances):
                track_id = active_tracks[i]
                track = self.tracked_objects[track_id]

                last_timestamp = track['last_timestamp']

                track_semantic_features = track["semantic_features"]
                last_data = track["trajectory"][last_timestamp]
                last_frustum = last_data["frustum"]

                last_pose = last_data["pose"]
                transform = np.linalg.inv(pose) @ last_pose

                if (
                    track_semantic_features is not None
                    and current_semantic_features is not None
                ):
                    semantic_overlap = np.dot(
                        track_semantic_features, current_semantic_features
                    )
                    semantic_matrix[i, j] = semantic_overlap

                    if self.debug:
                        print(
                            f"    Track {track_id} <-> Detection {j}: Cosine similarity = {semantic_overlap:.3f}"
                        )
                else:
                    print(f"{track_semantic_features=} {current_semantic_features=}")

                corners_city = last_frustum["corners_city"]
                track_points = points_rigid_transform(corners_city, transform)

                track_hull = MultiPoint(track_points).convex_hull

                # Handle degenerate cases (points, lines)
                if track_hull.area == 0 or current_hull.area == 0:
                    return 0.0

                # Compute intersection and union
                intersection = track_hull.intersection(current_hull).area
                union = track_hull.union(current_hull).area

                # Return IoU
                iou = intersection / union if union > 0 else 0.0
                iou_matrix[i, j] = iou

                if self.debug and iou > 0.3:  # Only log significant IoUs
                    print(
                        f"    Track {track_id} <-> Detection {j}: IoU = {iou:.3f}"
                    )

        print("semantic_matrix", semantic_matrix.min(), semantic_matrix.max())
        semantic_matrix = (semantic_matrix - semantic_matrix.min()) / max(
            1, semantic_matrix.max() - semantic_matrix.min()
        )
        print(
            "semantic_matrix after scaling",
            semantic_matrix.min(),
            semantic_matrix.max(),
        )
        print("iou_matrix", iou_matrix.min(), iou_matrix.max())

        # Perform assignment using IoU threshold
        # iou_threshold = getattr(self.config, 'frustum_iou_threshold', 0.3)
        iou_threshold = 0.1  # TODO: change?
        assignments = self._hungarian_assignment(
            iou_matrix, semantic_matrix, iou_threshold
        )

        if self.debug:
            print(f"  Assignment threshold: {iou_threshold}")
            print(f"  Successful assignments: {len(assignments)}")

            # Log IoU statistics for this frame
            valid_ious = iou_matrix[iou_matrix > 0]
            if len(valid_ious) > 0:
                print(
                    f"  IoU stats - Max: {np.max(valid_ious):.3f}, Mean: {np.mean(valid_ious):.3f}, Min: {np.min(valid_ious):.3f}"
                )

        frame_assignments = []
        used_frustums = set()
        dropped_tracks = set()

        # Process matched assignments
        for track_idx, frustum_idx in assignments:
            if track_idx < len(active_tracks) and frustum_idx < len(frustums):
                track_id = active_tracks[track_idx]
                frustum = frustums[frustum_idx]
                iou_score = iou_matrix[track_idx, frustum_idx]

                # Attempt to update track (may fail due to ICP)
                update_success = self._update_track(
                    track_id, timestamp_ns, frustum
                )

                if update_success:
                    # frame_assignments.append(
                    #     (track_id, frustum, {"score": iou_score})
                    # )
                    frame_assignments.append(track_id)
                    used_frustums.add(frustum_idx)

                    if self.debug:
                        print(
                            f"    MATCHED: Track {track_id} <- Detection {frustum_idx} (IoU: {iou_score:.3f})"
                        )
                else:
                    # Track was dropped due to ICP failure
                    dropped_tracks.add(track_id)
                    # frustum becomes available for new track creation

                    if self.debug:
                        print(
                            f"    DROPPED: Track {track_id} (ICP failed for Detection {frustum_idx})"
                        )

        # Remove dropped tracks from tracking
        for track_id in dropped_tracks:
            if track_id in self.tracked_objects:
                del self.tracked_objects[track_id]

        # Create new tracks for unmatched frustums
        unmatched_count = 0
        for j, frustum in enumerate(frustums):
            if j not in used_frustums:
                obj_id = self._create_new_track(timestamp_ns, frustum)
                # frame_assignments.append((obj_id, frustum, {"score": 1.0}))
                frame_assignments.append(obj_id)
                unmatched_count += 1

                if self.debug:
                    print(f"    NEW TRACK: {obj_id} <- Detection {j} (unmatched)")

        if self.debug:
            print(f"  Created {unmatched_count} new tracks for unmatched detections")

        return frame_assignments

    def _hungarian_assignment(
        self, iou_matrix: np.ndarray, semantic_matrix: np.ndarray, threshold: float
    ) -> List[Tuple[int, int]]:
        """Hungarian assignment based on IoU threshold."""
        if iou_matrix.size == 0:
            return []

        # Convert IoU to a cost matrix (since Hungarian solves a minimization problem)
        cost_matrix = 1.0 - iou_matrix - semantic_matrix

        print("cost_matrix", cost_matrix.shape)

        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Collect assignments that pass the threshold
        assignments = [
            (r, c) for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= threshold
        ]

        return assignments

    def _create_new_track(
        self, timestamp_ns: int, frustum: Dict
    ) -> int:
        """Create a new tentative track."""
        track_id = self.next_id
        self.next_id += 1

        pose = frustum['pose']

        self.tracked_objects[track_id] = {
            "id": track_id,
            "hits": 1,  # number of successful matches
            "missed": 0,  # consecutive misses
            "first_timestamp": timestamp_ns,
            "last_timestamp": timestamp_ns,
            "semantic_features": frustum.get("semantic_features", None),
            "trajectory": {
                timestamp_ns: {
                    "frustum": frustum,
                    "pose": pose
                }
            },
        }

        if self.debug:
            print(f"    NEW TRACK {track_id} (tentative)")

        return track_id

    def _update_track(
        self,
        track_id: int,
        timestamp_ns: int,
        frustum: Dict
    ) -> bool:
        """
        Update an existing track.
        Returns False if track should be dropped.
        """
        track = self.tracked_objects[track_id]

        # Get previous frame data
        prev_timestamp = track["last_timestamp"]
        if prev_timestamp not in track["trajectory"]:
            return False

        # prev_data = track["trajectory"][prev_timestamp]

        # Reset failure counters on success
        track["missed"] = 0
        track["hits"] = track.get("hits", 0) + 1
        track["last_timestamp"] = timestamp_ns

        # Update trajectory
        track["trajectory"][timestamp_ns] = dict(frustum=frustum, pose=frustum['pose'])

        return True

    def _get_active_tracks(self, timestamp_ns: int, timestamp_thresh: int = 1e+9) -> List[int]:
        """Get tracks that were active in the given frame."""
        active = []
        for track_id, track in self.tracked_objects.items():
            if abs(track['last_timestamp'] - timestamp_ns) < timestamp_thresh:
                active.append(track_id)
        return active
