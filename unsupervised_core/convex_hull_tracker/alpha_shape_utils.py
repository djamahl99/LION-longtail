import cProfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import pickle as pkl
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, trange
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
from collections import defaultdict
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.optimize import linear_sum_assignment
from kornia.geometry.linalg import transform_points
from pprint import pprint
import trimesh
import PIL.Image as Image
import io
import pstats

from lion.unsupervised_core.box_utils import compute_ppscore



class AlphaShapeUtils:
    """Utility functions for alpha shape operations."""

    @staticmethod
    def compute_alpha_shape_uv(points_uv: np.ndarray) -> Dict[str, Any]:
        """
        Compute 2D alpha shape (convex hull) from 3D points.

        Args:
            points_3d: Nx3 array of 3D points

        Returns:
            Dictionary containing alpha shape information
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

            alpha_shape = {
                "vertices_2d": hull_vertices,
                "hull_indices": hull.vertices,
                "centroid_2d": np.mean(hull_vertices, axis=0),
                "original_points": points_uv,
            }

            return alpha_shape

        except Exception as e:
            print(f"Error computing alpha shape: {e}")
            return None

    @staticmethod
    def compute_voxel_set(points_3d: np.ndarray, voxel_size: float = 0.1) -> set:
        voxels = np.round(points_3d / voxel_size).astype(int)

        # Use sets for fast intersection/union
        voxel_set = set(map(tuple, voxels))

        return voxel_set

    @staticmethod
    def compute_alpha_shape_2d(points_3d: np.ndarray) -> Dict[str, Any]:
        """
        Compute 2D alpha shape (convex hull) from 3D points.

        Args:
            points_3d: Nx3 array of 3D points

        Returns:
            Dictionary containing alpha shape information
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

            alpha_shape = {
                "vertices_2d": hull_vertices,
                "z_min": np.min(z_values),
                "z_max": np.max(z_values),
                "hull_indices": hull.vertices,
                "centroid_2d": np.mean(hull_vertices, axis=0),
                "area": hull.volume,  # In 2D, volume is area
                "original_points": points_3d,
                "voxel_set": AlphaShapeUtils.compute_voxel_set(points_3d),
                "volume": hull.volume * (np.max(z_values) - np.min(z_values)),
            }

            return alpha_shape

        except Exception as e:
            print(f"Error computing alpha shape: {e}")
            return None

    @staticmethod
    def compute_alpha_shape(points_3d: np.ndarray) -> Dict[str, Any]:
        """
        Compute alpha shape (convex hull) from 3D points.

        Args:
            points_3d: Nx3 array of 3D points

        Returns:
            Dictionary containing alpha shape information
        """
        if len(points_3d) < 3:
            return None

        try:
            mesh = trimesh.convex.convex_hull(points_3d)

            # Get hull vertices in order
            hull_vertices = mesh.vertices

            alpha_shape = {
                "vertices_3d": hull_vertices,
                "mesh": mesh,
                # "hull_indices": hull.vertices,
                "centroid_3d": np.mean(hull_vertices, axis=0),
                # "area": hull.area,
                # "volume": hull.volume,
                "original_points": points_3d,
                # "voxel_set": AlphaShapeUtils.compute_voxel_set(points_3d),
            }

            return alpha_shape

        except Exception as e:
            print(f"Error computing alpha shape: {e}")
            return None

    @staticmethod
    def convex_hull_iou_trimesh(shape1: Dict, shape2: Dict) -> float:
        """
        Compute IoU using cached geometric objects - much faster than original.

        Args:
            shape1, shape2: Alpha shape dictionaries with cached geometry

        Returns:
            IoU value between 0 and 1
        """
        mesh1 = shape1["mesh"]
        mesh2 = shape2["mesh"]

        try:
            # Using the 'manifold' engine for boolean operations
            intersection_mesh = mesh1.intersection(mesh2, engine='manifold')
            # The union volume can be calculated from the individual volumes and the intersection volume
            # union_volume = mesh1.volume + mesh2.volume - intersection_mesh.volume
            # Or by performing the union operation directly, which might be more robust
            union_mesh = mesh1.union(mesh2, engine='manifold')
            
            intersection_volume = intersection_mesh.volume
            union_volume = union_mesh.volume

        except ValueError:
            # This can happen if the boolean operation fails
            return 0.0

        # try:
        #     intersection_volume = mesh1.intersection(mesh2).volume
        #     union_volume = mesh1.union(mesh2).volume
        # except ValueError:
        #     return 0.0

        return intersection_volume / union_volume if union_volume > 0 else 0.0

    @staticmethod
    def convex_hull_iou_voxelized(shape1: Dict, shape2: Dict, pitch: float = 0.1) -> float:
        """
        Approximate IoU using voxelization.

        Args:
            shape1, shape2: Alpha shape dictionaries with cached geometry
            pitch: The size of a single voxel.

        Returns:
            Approximate IoU value between 0 and 1
        """
        mesh1 = shape1["mesh"]
        mesh2 = shape2["mesh"]

        # Voxelize both meshes
        voxel1 = mesh1.voxelized(pitch)
        voxel2 = mesh2.voxelized(pitch)

        # Get the filled voxels for each mesh
        filled1 = voxel1.sparse_indices
        filled2 = voxel2.sparse_indices
        
        # Convert to sets for efficient intersection and union
        set1 = {tuple(i) for i in filled1}
        set2 = {tuple(i) for i in filled2}

        intersection_count = len(set1.intersection(set2))
        union_count = len(set1.union(set2))

        return intersection_count / union_count if union_count > 0 else 0.0

    @staticmethod
    def polygon_iou_2d(shape1: Dict, shape2: Dict) -> float:
        """Calculate 2D IoU between two alpha shapes using polygon intersection."""
        try:
            poly1 = Polygon(shape1["vertices_2d"])
            poly2 = Polygon(shape2["vertices_2d"])

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
    def z_overlap_ratio(shape1: Dict, shape2: Dict) -> float:
        """Calculate z-direction overlap ratio."""
        z1_min, z1_max = shape1["z_min"], shape1["z_max"]
        z2_min, z2_max = shape2["z_min"], shape2["z_max"]

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
    def alpha_shape_3d_iou(shape1: Dict, shape2: Dict) -> float:
        """Calculate 3D IoU combining 2D polygon IoU and z-overlap."""
        polygon_iou = AlphaShapeUtils.polygon_iou_2d(shape1, shape2)
        z_iou = AlphaShapeUtils.z_overlap_ratio(shape1, shape2)

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
    def merge_alpha_shapes_with_icp(
        shapes_with_transforms: List[Tuple[Dict, np.ndarray]],
        ppscore_thresh: float = 0.7,
    ) -> Dict:
        """
        Merge multiple alpha shapes using ICP alignment.

        Args:
            shapes_with_transforms: List of (alpha_shape, transform_matrix) tuples
        """
        if not shapes_with_transforms:
            return None
        if len(shapes_with_transforms) == 1:
            return shapes_with_transforms[0][0]

        # Collect all transformed points
        all_aligned_points = []

        last_points = None

        for alpha_shape, transform_matrix in shapes_with_transforms:
            points_3d = alpha_shape["original_points"].copy()

            # if 'original_points' in alpha_shape and alpha_shape['original_points'] is not None:
            #     points_3d = alpha_shape['original_points'].copy()
            # else:
            #     # Reconstruct 3D points from 2D + z bounds
            #     vertices_2d = alpha_shape['vertices_2d']
            #     z_center = (alpha_shape['z_min'] + alpha_shape['z_max']) / 2
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
                f"merge_alpha_shapes_with_icp ppscore={ppscore.shape} {ppscore.min()} {ppscore.mean()} {ppscore.max()}"
            )
            ppscore_mask = ppscore >= ppscore_thresh
            print("ppscore_mask", ppscore_mask.shape, ppscore_mask.sum())
            return AlphaShapeUtils.compute_alpha_shape(last_points[ppscore_mask])

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
    def transform_alpha_shape(alpha_shape: Dict, transform_matrix: np.ndarray) -> Dict:
        """Transform alpha shape using a 4x4 transformation matrix."""
        if alpha_shape is None:
            return None

        # Transform 2D vertices to 3D, apply transform, project back to 2D
        vertices_2d = alpha_shape["vertices_2d"]
        z_center = (alpha_shape["z_min"] + alpha_shape["z_max"]) / 2

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
            [[0, 0, alpha_shape["z_min"], 1], [0, 0, alpha_shape["z_max"], 1]]
        )
        transformed_z = (transform_matrix @ z_points.T).T
        new_z_min, new_z_max = transformed_z[0, 2], transformed_z[1, 2]

        # Create new alpha shape
        transformed_shape = alpha_shape.copy()
        transformed_shape["vertices_2d"] = transformed_2d
        transformed_shape["z_min"] = min(new_z_min, new_z_max)
        transformed_shape["z_max"] = max(new_z_min, new_z_max)
        transformed_shape["centroid_2d"] = np.mean(transformed_2d, axis=0)

        # Recompute area if needed
        try:
            hull = ConvexHull(transformed_2d)
            transformed_shape["area"] = hull.volume
        except:
            transformed_shape["area"] = 0.0

        return transformed_shape