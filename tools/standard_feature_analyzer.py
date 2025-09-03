"""
Random Forest Feature Importance Analysis for Long-tail Detection - AV2 Integration
Analyzes which features best predict detection success/failure modes using interpretable ML
"""

import gc
import json
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, replace
from collections import defaultdict, Counter

import torch
from tqdm import tqdm
import multiprocessing as mp

from pprint import pprint

# ML imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.optimize import linear_sum_assignment

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.inspection import permutation_importance
import umap

import cProfile
import io
import os
import pstats
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Import AV2 utilities
from av2.evaluation.detection.utils import (
    accumulate,
    assign,
    compute_affinity_matrix,
    distance,
    DetectionCfg,
    AffinityType,
    DistanceType,
)
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt, NDArrayObject
from av2.datasets.sensor.constants import RingCameras, StereoCameras
from av2.datasets.sensor.sensor_dataloader import (
    SensorDataloader,
    SynchronizedSensorData,
)
from av2.rendering.color import ColorFormats, create_range_map
from av2.rendering.rasterize import draw_points_xy_in_img
from av2.structures.sweep import Sweep
from av2.utils.io import read_city_SE3_ego
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.timestamped_image import TimestampedImage

from standalone_analyze_longtail import EvaluationConfig, StandaloneLongTailEvaluator

from av2.evaluation.detection.constants import (
    MAX_NORMALIZED_ASE,
    MAX_SCALE_ERROR,
    MAX_YAW_RAD_ERROR,
    MIN_AP,
    MIN_CDS,
    NUM_DECIMALS,
    AffinityType,
    DistanceType,
    FilterMetricType,
    InterpType,
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from lion.unsupervised_core.c_proto_refine import C_PROTO, CSS
from lion.unsupervised_core.mfcf import MFCF
from lion.unsupervised_core.outline_utils import OutlineFitter, points_rigid_transform, correct_heading,\
    hierarchical_occupancy_score, smooth_points, angle_from_vector, get_registration_angle, box_rigid_transform,\
    correct_orientation, density_guided_drift,\
    KL_entropy_score
from lion.unsupervised_core.rotate_iou_cpu_eval import rotate_iou_cpu_eval, rotate_iou_cpu_one


class ModifiedSensorDataloader(SensorDataloader):
    """Extended SensorDataloader with direct access methods."""

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
        from av2.structures.sweep import Sweep
        from av2.utils.io import read_city_SE3_ego
        from av2.map.map_api import ArgoverseStaticMap

        # Use provided cam_names or fall back to instance cam_names
        if cam_names is None:
            cam_names = self.cam_names

        # Construct paths
        log_dir = self.dataset_dir / split / log_id
        sensor_dir = log_dir / "sensors"
        lidar_feather_path = sensor_dir / "lidar" / f"{timestamp_ns}.feather"

        # Verify lidar data exists
        if not lidar_feather_path.exists():
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


@dataclass
class FeatureAnalysisConfig:
    """Configuration for feature importance analysis."""

    n_estimators: int = 100
    max_depth: int = 10
    random_seed: int = 42
    test_size: float = 0.3
    min_samples_per_class: int = 20

    # Clustering parameters
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 10
    n_gmm_components: int = 5

    # Visualization parameters
    tsne_perplexity: int = 30
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1


def create_bbox_from_dts_row(row: pd.Series) -> np.ndarray:
    """
    Create 3D bounding box corners from DTS DataFrame row.

    Args:
        row: DataFrame row with cuboid parameters

    Returns:
        (8, 3) array of bounding box corners
    """
    # Extract center and dimensions
    center = np.array([row["tx_m"], row["ty_m"], row["tz_m"]])
    dims = np.array([row["length_m"], row["width_m"], row["height_m"]])

    # Extract quaternion and convert to rotation matrix
    quat = np.array(
        [row["qx"], row["qy"], row["qz"], row["qw"]]
    )  # scipy expects x,y,z,w
    rotation = Rotation.from_quat(quat).as_matrix()

    # Create local bounding box corners (centered at origin)
    l, w, h = dims / 2
    corners_local = np.array(
        [
            [-l, -w, -h],
            [l, -w, -h],
            [l, w, -h],
            [-l, w, -h],  # bottom face
            [-l, -w, h],
            [l, -w, h],
            [l, w, h],
            [-l, w, h],  # top face
        ]
    )

    # Transform to world coordinates
    corners_world = (rotation @ corners_local.T).T + center

    return corners_world


def extract_camera_positions(target_datum) -> Dict[str, np.ndarray]:
    """
    Extract camera positions from synchronized imagery data.

    Args:
        target_datum: Sensor data from ModifiedSensorDataloader

    Returns:
        Dictionary mapping camera names to positions in ego frame
    """
    camera_positions = {}

    if target_datum.synchronized_imagery:
        for cam_name, timestamped_img in target_datum.synchronized_imagery.items():
            # Extract camera position from ego_SE3_cam transformation
            camera_position = timestamped_img.camera_model.ego_SE3_cam.translation
            camera_positions[cam_name] = camera_position

    return camera_positions


def create_voxel_occupancy_map(points: np.ndarray, voxel_size: float = 0.1) -> Dict:
    """
    Create a voxel occupancy map from LiDAR points for efficient raycasting.

    Args:
        points: LiDAR points (N, 3)
        voxel_size: Size of each voxel in meters

    Returns:
        Dictionary containing voxel grid and occupancy info
    """
    if len(points) == 0:
        return {"occupied_voxels": set(), "bounds": None, "voxel_size": voxel_size}

    # Calculate voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Create set of occupied voxels for fast lookup
    occupied_voxels = set(map(tuple, voxel_indices))

    # Calculate bounds
    min_bounds = np.min(voxel_indices, axis=0) * voxel_size
    max_bounds = np.max(voxel_indices, axis=0) * voxel_size

    return {
        "occupied_voxels": occupied_voxels,
        "bounds": (min_bounds, max_bounds),
        "voxel_size": voxel_size,
    }


def raycast_through_voxels(
    start: np.ndarray, end: np.ndarray, voxel_map: Dict
) -> Tuple[bool, int]:
    """
    Cast a ray through voxel grid and count intersections.

    Args:
        start: Ray start point (3,)
        end: Ray end point (3,)
        voxel_map: Voxel occupancy map

    Returns:
        (is_occluded, num_intersections)
    """
    if not voxel_map["occupied_voxels"]:
        return False, 0

    voxel_size = voxel_map["voxel_size"]
    occupied_voxels = voxel_map["occupied_voxels"]

    # Ray direction and length
    direction = end - start
    ray_length = np.linalg.norm(direction)

    if ray_length < 1e-6:
        return False, 0

    direction_normalized = direction / ray_length

    # Sample points along the ray
    num_samples = max(int(ray_length / (voxel_size * 0.5)), 10)
    t_values = np.linspace(0, ray_length, num_samples)

    intersections = 0
    for t in t_values[1:-1]:  # Skip start and end points
        point = start + t * direction_normalized
        voxel_idx = tuple(np.floor(point / voxel_size).astype(int))

        if voxel_idx in occupied_voxels:
            intersections += 1

    # Consider occluded if multiple intersections found
    is_occluded = intersections > 2

    return is_occluded, intersections


def check_bbox_ray_intersection(
    ray_start: np.ndarray,
    ray_end: np.ndarray,
    bbox_corners: np.ndarray,
    margin: float = 0.1,
) -> bool:
    """
    Check if ray intersects with a 3D bounding box.

    Args:
        ray_start: Ray start point (3,)
        ray_end: Ray end point (3,)
        bbox_corners: Bounding box corners (8, 3)
        margin: Additional margin around bbox

    Returns:
        True if ray intersects bbox
    """
    # Get bbox bounds
    min_bounds = np.min(bbox_corners, axis=0) - margin
    max_bounds = np.max(bbox_corners, axis=0) + margin

    # Ray parameters
    ray_dir = ray_end - ray_start
    ray_length = np.linalg.norm(ray_dir)

    if ray_length < 1e-6:
        return False

    ray_dir_norm = ray_dir / ray_length

    # AABB intersection test
    t_min = 0.0
    t_max = ray_length

    for i in range(3):
        if abs(ray_dir_norm[i]) < 1e-6:
            # Ray is parallel to slab
            if ray_start[i] < min_bounds[i] or ray_start[i] > max_bounds[i]:
                return False
        else:
            # Calculate intersection distances
            t1 = (min_bounds[i] - ray_start[i]) / ray_dir_norm[i]
            t2 = (max_bounds[i] - ray_start[i]) / ray_dir_norm[i]

            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return False

    return bool((t_min <= ray_length) and (t_max >= 0))


def analyze_lidar_shadow(
    detection, lidar_points, ego_pos, debug=False, debug_path="shadow_debug.png"
):
    """Estimate LiDAR points within 3D shadow frustum behind object"""
    det_center = (
        detection[["tx_m", "ty_m", "tz_m"]].values.astype(np.float64).reshape(3)
    )
    ego_pos = np.asarray(ego_pos, dtype=np.float64).reshape(3)
    lidar_points = np.asarray(lidar_points, dtype=np.float64)
    lidar_xyz = lidar_points[:, :3]
    lidar_xy = lidar_xyz[:, :2]

    # Object dimensions and orientation
    length = float(detection["length_m"])
    width = float(detection["width_m"])
    height = float(detection["height_m"]) if "height_m" in detection else 2.0
    yaw = R.from_quat(
        [detection["qx"], detection["qy"], detection["qz"], detection["qw"]]
    ).as_euler("xyz")[2]

    box_corners = get_rotated_box(det_center[:2], length, width, yaw)  # (4, 2)

    box_diag_length = np.linalg.norm(
        np.array([length, width, height], dtype=np.float32)
    )
    shadow_length = box_diag_length

    # Find which corners form the "back" of the object (furthest from ego)
    ego_to_center = det_center[:2] - ego_pos[:2]
    ego_to_center_norm = ego_to_center / np.linalg.norm(ego_to_center)

    # Project each corner onto the ego->center direction to find the back corners
    corner_projections = []
    for corner in box_corners:
        ego_to_corner = corner - ego_pos[:2]
        projection = np.dot(ego_to_corner, ego_to_center_norm)
        corner_projections.append(projection)

    corner_projections = np.array(corner_projections)

    # # Find the two corners that are furthest along the ego->center direction
    # # These form the back edge of the object
    # sorted_indices = np.argsort(corner_projections)
    # back_corner_indices = sorted_indices[-2:]  # Two furthest corners

    # corner_left = box_corners[back_corner_indices[0]]
    # corner_right = box_corners[back_corner_indices[1]]

    # Perpendicular direction (90 degrees rotation)
    perp_dir = np.array([-ego_to_center_norm[1], ego_to_center_norm[0]])

    # Project each corner onto the perpendicular direction
    # This tells us how far left/right each corner is from the ego's view
    corner_offsets = [
        (np.dot(corner - det_center[:2], perp_dir), i)
        for i, corner in enumerate(box_corners)
    ]
    corner_offsets.sort()

    # Leftmost and rightmost (from ego's view)
    left_idx = corner_offsets[0][1]
    right_idx = corner_offsets[-1][1]

    corner_left = box_corners[left_idx]
    corner_right = box_corners[right_idx]

    # Make sure left/right are correct by checking cross product
    v1 = corner_left - det_center[:2]
    v2 = corner_right - det_center[:2]
    if np.cross(v1, v2) < 0:  # Swap if needed
        corner_left, corner_right = corner_right, corner_left

    # Now create the shadow frustum extending FROM these corners AWAY from ego
    # The shadow edges are parallel to the ego-corner rays
    shadow_dir_left = corner_left - ego_pos[:2]
    shadow_dir_left /= np.linalg.norm(shadow_dir_left)

    shadow_dir_right = corner_right - ego_pos[:2]
    shadow_dir_right /= np.linalg.norm(shadow_dir_right)

    # Extend the shadow frustum far beyond the object
    far_left = corner_left + shadow_dir_left * shadow_length
    far_right = corner_right + shadow_dir_right * shadow_length

    # Vectorized version for efficiency
    in_shadow_mask = np.zeros(len(lidar_xy), dtype=bool)

    # Check left boundary: points should be to the right of the left shadow edge
    v_left = far_left - corner_left
    n_left = np.array([-v_left[1], v_left[0]])  # Normal pointing right
    rel_to_left = lidar_xy - corner_left
    right_of_left = (rel_to_left @ n_left) >= 0

    # Check right boundary: points should be to the left of the right shadow edge
    v_right = far_right - corner_right
    n_right = np.array([v_right[1], -v_right[0]])  # Normal pointing left
    rel_to_right = lidar_xy - corner_right
    left_of_right = (rel_to_right @ n_right) >= 0

    # Check back boundary: points should be beyond the back edge
    back_edge = corner_right - corner_left
    back_normal = np.array([-back_edge[1], back_edge[0]])  # Normal pointing backward
    # Normalize to ensure it points away from ego
    if np.dot(back_normal, ego_to_center) < 0:
        back_normal = -back_normal
    rel_to_back = lidar_xy - corner_left

    # Project distances onto the back_normal
    back_dists = rel_to_back @ back_normal  # Scalar projection along back_normal

    # Keep points within a distance behind the object (e.g. 1–2 diagonal lengths)
    within_back_distance = back_dists <= box_diag_length
    beyond_back = (back_dists >= 0) & within_back_distance

    # Check front boundary (far edge) - optional, might not be needed
    # Most points won't reach this far anyway

    # check that the shadow is not too far away...

    dists = np.linalg.norm(lidar_xyz - det_center, axis=1)
    in_box_diag_radius = dists <= shadow_length

    # Rest of your bbox calculation code...
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    rot_mat = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    lidar_xy_local = (rot_mat @ (lidar_xy - det_center[:2]).T).T
    lidar_z_local = lidar_xyz[:, 2] - det_center[2]

    half_length = length / 2
    half_width = width / 2
    half_height = height / 2

    in_bbox_mask = (
        (np.abs(lidar_xy_local[:, 0]) <= half_length)
        & (np.abs(lidar_xy_local[:, 1]) <= half_width)
        & (np.abs(lidar_z_local) <= half_height)
    )

    in_shadow_mask = (
        right_of_left
        & left_of_right
        & beyond_back
        & in_box_diag_radius
        & (~in_bbox_mask)
    )

    shadow_points = np.count_nonzero(in_shadow_mask)

    bbox_point_count = np.count_nonzero(in_bbox_mask)
    shadow_ratio = (
        shadow_points / bbox_point_count if bbox_point_count > 0 else float("inf")
    )

    if debug:
        # Your debug plotting code, but update to show the correct shadow region
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.set_title(f"LiDAR Shadow Frustum {detection['category']} (BEV)")

        # print("bbox_point_count", bbox_point_count)
        # print("shadow_points", shadow_points)
        # print("shadow_ratio", shadow_ratio)

        # Plot points
        radius = shadow_length + 5
        dists = np.linalg.norm(lidar_xy - det_center[:2], axis=1)
        in_radius = dists <= radius + np.linalg.norm(det_center[:2] - ego_pos[:2])

        ax.scatter(
            lidar_xy[in_radius & ~in_shadow_mask, 0],
            lidar_xy[in_radius & ~in_shadow_mask, 1],
            s=1,
            c="blue",
            label="Other Points",
            alpha=0.5,
        )
        ax.scatter(
            lidar_xy[in_shadow_mask, 0],
            lidar_xy[in_shadow_mask, 1],
            s=2,
            c="red",
            label=f"Shadow Points ({shadow_points} points)",
            alpha=0.8,
        )

        # Object box
        object_box = np.vstack([box_corners, box_corners[0]])
        ax.plot(
            object_box[:, 0],
            object_box[:, 1],
            "k-",
            label=f"Object BBox ({bbox_point_count} points)",
        )

        # Shadow frustum edges
        ax.plot(
            [corner_left[0], far_left[0]],
            [corner_left[1], far_left[1]],
            "orange",
            linestyle="--",
        )
        ax.plot(
            [corner_right[0], far_right[0]],
            [corner_right[1], far_right[1]],
            "orange",
            linestyle="--",
        )
        ax.plot(
            [corner_left[0], corner_right[0]],
            [corner_left[1], corner_right[1]],
            "orange",
            linewidth=2,
        )
        ax.plot(
            [far_left[0], far_right[0]],
            [far_left[1], far_right[1]],
            "orange",
            linestyle="--",
        )

        # Shadow polygon
        shadow_poly = np.array(
            [corner_left, corner_right, far_right, far_left, corner_left]
        )
        ax.plot(shadow_poly[:, 0], shadow_poly[:, 1], "orange", label="Shadow Region")

        ax.plot(ego_pos[0], ego_pos[1], "go", markersize=10, label="Ego")
        ax.plot(det_center[0], det_center[1], "ro", markersize=8, label="Object Center")

        ax.legend()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.savefig(debug_path, bbox_inches="tight", dpi=150)
        plt.close()

    return shadow_ratio




def get_rotated_box(center_xy, length, width, yaw):
    """Return 4 corners of rotated rectangle (BEV)"""
    dx = length / 2
    dy = width / 2
    corners = np.array(
        [
            [dx, dy],
            [dx, -dy],
            [-dx, -dy],
            [-dx, dy],
        ]
    )
    rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rotated = (rot_mat @ corners.T).T
    return rotated + center_xy


def estimate_occlusion_raycasting_batch(
    dts: pd.DataFrame,
    ego_pos: np.ndarray,
    lidar_points: np.ndarray,
    camera_positions: Optional[Dict[str, np.ndarray]] = None,
    num_rays: int = 16,
    voxel_size: float = 0.30,
    min_distance_threshold: float = 2.0,
) -> Dict[int, Dict[str, float]]:
    """
    Enhanced batch occlusion estimation using raycasting for multiple objects.

    Args:
        dts: DataFrame with DTS_COLUMNS (cuboid parameters + score)
        ego_pos: Ego vehicle position (3,)
        lidar_points: LiDAR point cloud (N, 3)
        camera_positions: Optional dict of camera positions for multi-view analysis
        num_rays: Number of rays to cast around each object center
        voxel_size: Voxel size for occupancy grid
        min_distance_threshold: Minimum distance for objects to be considered occluders

    Returns:
        Dictionary mapping object indices to occlusion metrics
    """
    if len(dts) == 0:
        return {}

    # Create voxel occupancy map from LiDAR
    voxel_map = create_voxel_occupancy_map(lidar_points, voxel_size)

    # Extract object centers and bounding boxes
    object_centers = dts[["tx_m", "ty_m", "tz_m"]].values
    object_bboxes = {}

    for idx, row in dts.iterrows():
        object_bboxes[idx] = create_bbox_from_dts_row(row)

    results = {}

    debug_row = np.random.randint(0, len(dts))
    # Process each target object
    for target_idx, target_row in dts.iterrows():
        target_center = np.array(
            [target_row["tx_m"], target_row["ty_m"], target_row["tz_m"]]
        )

        # Find potential occluding objects (closer to ego and within reasonable distance)
        target_distance = np.linalg.norm(target_center - ego_pos)

        # Filter for objects that could occlude this target
        occluder_candidates = []
        for other_idx, other_row in dts.iterrows():
            if other_idx == target_idx:
                continue

            other_center = np.array(
                [other_row["tx_m"], other_row["ty_m"], other_row["tz_m"]]
            )
            other_distance = np.linalg.norm(other_center - ego_pos)
            center_distance = np.linalg.norm(other_center - target_center)

            # Only consider objects that are closer to ego and within reasonable distance of target
            if (
                other_distance < target_distance
                and center_distance
                < min_distance_threshold
                * max(target_row["length_m"], target_row["width_m"])
            ):
                occluder_candidates.append(object_bboxes[other_idx])

        # Generate ray endpoints around target object
        ray_endpoints = []

        # Primary ray to object center
        ray_endpoints.append(target_center)

        # Additional rays around object center (small sphere sampling)
        target_size = max(
            target_row["length_m"], target_row["width_m"], target_row["height_m"]
        )
        ray_radius = target_size * 0.3  # Sample within 30% of object size

        for i in range(num_rays - 1):
            theta = 2 * np.pi * i / (num_rays - 1)
            phi = np.pi * (i % 4) / 8  # Vary elevation

            offset = ray_radius * np.array(
                [np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)]
            )
            ray_endpoints.append(target_center + offset)

        # Cast rays and analyze occlusion
        total_rays = len(ray_endpoints)
        occluded_rays = 0
        bbox_occlusions = 0
        lidar_occlusions = 0
        total_intersections = 0

        for endpoint in ray_endpoints:
            ray_occluded = False
            intersections = 0

            # Check LiDAR-based occlusion
            lidar_occluded, lidar_intersections = raycast_through_voxels(
                ego_pos, endpoint, voxel_map
            )

            if lidar_occluded:
                ray_occluded = True
                lidar_occlusions += 1
                intersections += lidar_intersections

            # Check bounding box intersections with occluder candidates
            bbox_hit = False
            for bbox_corners in occluder_candidates:
                if check_bbox_ray_intersection(ego_pos, endpoint, bbox_corners):
                    bbox_hit = True
                    intersections += 1

            if bbox_hit:
                ray_occluded = True
                bbox_occlusions += 1

            if ray_occluded:
                occluded_rays += 1

            total_intersections += intersections

        # Calculate occlusion metrics
        occlusion_ratio = occluded_rays / total_rays
        lidar_occlusion_ratio = lidar_occlusions / total_rays
        bbox_occlusion_ratio = bbox_occlusions / total_rays

        # Multi-camera occlusion analysis (if camera positions provided)
        # camera_occlusion = {}
        # if camera_positions:
        #     for cam_name, cam_pos in camera_positions.items():
        #         cam_occluded = 0
        #         for endpoint in ray_endpoints:
        #             lidar_occ, _ = raycast_through_voxels(cam_pos, endpoint, voxel_map)
        #             bbox_occ = any(
        #                 check_bbox_ray_intersection(cam_pos, endpoint, bbox)
        #                 for bbox in occluder_candidates
        #             )
        #             if lidar_occ or bbox_occ:
        #                 cam_occluded += 1
        #         camera_occlusion[cam_name] = cam_occluded / total_rays

        ground_points = lidar_points[
            lidar_points[:, 2] < target_row["tz_m"] - target_row["height_m"] / 2
        ]
        nearby_ground = ground_points[
            np.linalg.norm(ground_points[:, :2] - target_center[None, :2], axis=1) < 1.0
        ]

        # Direct line-of-sight quality
        detection_center = target_center.copy()
        ego_to_det = detection_center - ego_pos

        # Check for obstructions along direct path
        path_points = lidar_points[
            np.abs(np.cross(lidar_points - ego_pos, ego_to_det))
            / np.linalg.norm(ego_to_det)
            < 0.5
        ]
        try:
            path_points_before_det = path_points[
                np.dot(
                    (path_points.reshape(-1, 3) - ego_pos.reshape(-1, 3)),
                    ego_to_det.reshape(-1, 3),
                ).flatten()
                < np.linalg.norm(ego_to_det) ** 2
            ]
        except:
            path_points_before_det = np.zeros((0, 3), dtype=np.float32)

        results[target_idx] = {
            "overall_occlusion": occlusion_ratio,
            "lidar_occlusion": lidar_occlusion_ratio,
            "bbox_occlusion": bbox_occlusion_ratio,
            # "camera_occlusion": camera_occlusion,
            "avg_intersections_per_ray": total_intersections / total_rays,
            "num_occluder_candidates": len(occluder_candidates),
            "distance_to_ego": target_distance,
            "confidence": min(1.0, len(lidar_points) / 1000.0),
            "shadow_point_ratio": analyze_lidar_shadow(
                target_row, lidar_points, ego_pos, debug=(target_idx == debug_row)
            ),
            "points_near_ground": len(nearby_ground),
            "height_above_nearby_ground": (
                target_row["tz_m"]
                - target_row["height_m"] / 2
                - np.max(nearby_ground[:, 2])
                if len(nearby_ground) > 0
                else 999
            ),
            "path_obstruction_ratio": len(path_points_before_det)
            / max(1, len(path_points)),
            "viewing_angle": np.arccos(
                np.dot(ego_to_det, np.array([1, 0, 0])) / np.linalg.norm(ego_to_det)
            ),
        }

    return results


def estimate_occlusion_simple_batch(
    dts: pd.DataFrame, target_datum, num_rays: int = 12, voxel_size: float = 0.25
) -> Dict[int, float]:
    """
    Simplified batch wrapper function that returns occlusion values (0-1) for all objects.
    Can replace your existing _estimate_occlusion method for batch processing.

    Args:
        dts: DataFrame with DTS_COLUMNS (cuboid parameters + score)
        target_datum: Sensor data from ModifiedSensorDataloader
        num_rays: Number of rays to cast
        voxel_size: Voxel size for occupancy grid

    Returns:
        Dictionary mapping object indices to occlusion values (0-1)
    """
    if target_datum.sweep is None or len(dts) == 0:
        return {}

    lidar_points = target_datum.sweep.xyz
    ego_pos = np.array([0.0, 0.0, 0.0])
    # camera_positions = extract_camera_positions(target_datum)
    camera_positions = None

    analysis_results = estimate_occlusion_raycasting_batch(
        dts=dts,
        ego_pos=ego_pos,
        lidar_points=lidar_points,
        camera_positions=camera_positions,
        num_rays=num_rays,
        voxel_size=voxel_size,
    )

    return analysis_results

    # # Extract just the overall occlusion values
    # return {
    #     idx: result["overall_occlusion"] for idx, result in analysis_results.items()
    # }


def _load_sweep_data_and_occlusions(
    args: Tuple[str, int, pd.DataFrame, Path],
) -> Tuple[str, int, Dict[int, float]]:
    """
    Load sweep data and compute occlusions for a single sweep.
    This runs in a separate process to parallelize I/O and computation.
    """
    log_id, timestamp_ns, sweep_dts, dataset_dir = args

    # Recreate dataloader in this process
    dataloader = ModifiedSensorDataloader(dataset_dir=dataset_dir.parent)
    split = dataset_dir.name

    # Load sensor data and compute occlusions
    target_datum = dataloader.get_sensor_data(log_id, split, timestamp_ns, cam_names=[])
    occlusions = estimate_occlusion_simple_batch(sweep_dts, target_datum)

    # occlusions = {}
    # for target_idx, target_row in sweep_dts.iterrows():
    #     occlusions[target_idx] = 0

    return log_id, timestamp_ns, occlusions


def argo2_box_to_lidar(boxes):
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    elif not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    
    cnt_xyz = boxes[:, :3]  # x, y, z centers
    lwh = boxes[:, 3:6]     # length, width, height
    quat = boxes[:, 6:]    
    
    # Convert yaw to quaternion
    yaw = quat_to_yaw(quat[:, [0]], quat[:, [1]], quat[:, [2]], quat[:, [3]])
    

    # Combine: [x, y, z, length, width, height, qw, qx, qy, qz]
    lidar_boxes = torch.cat([cnt_xyz, lwh, yaw], dim=1)
    return lidar_boxes

class RandomForestFeatureAnalyzer:
    """
    Random Forest-based feature importance analysis for AV2 detection results.

    Analyzes which object and scene features are most predictive of:
    - True Positives vs False Positives
    - Detection vs Miss
    - Error magnitude (ATE, ASE, AOE)
    - Class-specific failure modes
    """

    ignore_columns = [
        "dt_log_id",
        "dt_timestamp_ns",
        "dt_is_evaluated",
        "gt_log_id",
        "gt_timestamp_ns",
        "gt_is_evaluated",
        "ATE",
        "ASE",
        "AOE",
        "gt_ATE",
        "gt_ASE",
        "gt_AOE",
        "gt_0.5",
        "gt_1.0",
        "gt_2.0",
        "gt_4.0",
        "gt_track_uuid",
        "gt_index",
        "tx_diff",
        "ty_diff",
        "tz_diff",
        "relative_height_error",
        "dt_ATE",
        "dt_ASE",
        "dt_AOE",
        "dt_0.5",
        "dt_1.0",
        "dt_2.0",
        "dt_4.0",
        "category_match",
        "length_diff",
        "width_diff",
        "height_diff",
        "relative_length_error",
        "relative_width_error",
    ]

    def __init__(self, config: FeatureAnalysisConfig = None):
        self.config = config or FeatureAnalysisConfig()
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def extract_av2_features(
        self,
        eval_dts: pd.DataFrame,
        eval_gts: pd.DataFrame,
        cfg: DetectionCfg,
        ego_poses: Optional[pd.DataFrame] = None,
        num_processes: Optional[int] = 8,
        enable_profiling: bool = True,
        benchmark_every: int = 100,
    ) -> pd.DataFrame:
        """
        Optimized and benchmarked AV2 feature extraction with performance monitoring.

        Args:
            enable_profiling: If True, runs cProfile on feature extraction
            benchmark_every: Print timing stats every N iterations
        """
        print(
            "Extracting comprehensive features from AV2 results (optimized + benchmarked)..."
        )

        primary_threshold_idx = len(cfg.affinity_thresholds_m) // 2
        primary_threshold = cfg.affinity_thresholds_m[primary_threshold_idx]
        threshold_col = str(primary_threshold)

        assert cfg.dataset_dir is not None
        split = cfg.dataset_dir.name
        assert split in ["train", "val", "test"], f"{split=} is not valid!"

        print("eval_dts", eval_dts.shape)
        print("eval_gts", eval_gts.shape)

        # Optimized grouping using MultiIndex
        print("Creating optimized indexes...")
        eval_dts_indexed = eval_dts.set_index(["log_id", "timestamp_ns"])
        eval_gts_indexed = eval_gts.set_index(["log_id", "timestamp_ns"])
        unique_keys = eval_dts_indexed.index.unique()

        # lets remove some aha #################################################
        index_pairs_list = list(set(eval_gts_indexed.index.tolist()))
        print(f"index_pairs_list[:10]={index_pairs_list[:10]} {len(index_pairs_list)}")
        indices = np.random.choice(len(index_pairs_list), size=150)
        valid_uuids_gts = [index_pairs_list[i] for i in indices]
        # valid_uuids_gts = list(set(eval_gts_indexed.index.tolist()))[:150]

        eval_gts_indexed = eval_gts_indexed.loc[list(valid_uuids_gts)].sort_index()
        eval_dts_indexed = eval_dts_indexed.loc[list(valid_uuids_gts)].sort_index()

        unique_keys = eval_dts_indexed.index.unique()
        ########################################################################

        # Prepare arguments
        sweep_args = []
        sweep_lookup = {}

        print("Building sweep arguments with indexed lookup...")
        for key in tqdm(unique_keys, desc="Extracting sweep args"):
            sweep_dts = (
                eval_dts_indexed.loc[[key]]
                if key in eval_dts_indexed.index
                else pd.DataFrame()
            )
            sweep_gts = (
                eval_gts_indexed.loc[[key]]
                if key in eval_gts_indexed.index
                else pd.DataFrame()
            )

            log_id, timestamp_ns = key
            sweep_args.append(
                (log_id, timestamp_ns, sweep_dts.reset_index(), cfg.dataset_dir)
            )
            sweep_lookup[key] = (sweep_dts.reset_index(), sweep_gts.reset_index())

        print(
            f"Loading sensor data and computing occlusions for {len(sweep_args)} sweeps using {num_processes or mp.cpu_count()} processes..."
        )

        if num_processes is None:
            num_processes = mp.cpu_count()

        with mp.Pool(processes=num_processes) as pool:
            occlusion_results = pool.map(_load_sweep_data_and_occlusions, sweep_args)

        occlusions_by_sweep = {
            (log_id, timestamp_ns): occlusions
            for log_id, timestamp_ns, occlusions in occlusion_results
        }

        print("Extracting features from loaded data...")

        # Benchmarking setup
        features_list = []

        for i, (key, (sweep_dts, sweep_gts)) in enumerate(
            tqdm(sweep_lookup.items(), desc="Extracting features")
        ):
            log_id, timestamp_ns = key

            # Time occlusion lookup
            occlusions = occlusions_by_sweep[(log_id, timestamp_ns)]

            dts_array = sweep_dts[
                [
                    "tx_m",
                    "ty_m",
                    "tz_m",
                    "length_m",
                    "width_m",
                    "height_m",
                    "qw",
                    "qx",
                    "qy",
                    "qz",
                ]
            ].values
            gts_array = sweep_gts[
                [
                    "tx_m",
                    "ty_m",
                    "tz_m",
                    "length_m",
                    "width_m",
                    "height_m",
                    "qw",
                    "qx",
                    "qy",
                    "qz",
                ]
            ].values

            # affinity_matrix = compute_affinity_matrix(
            #     dts_array[..., :3], gts_array[..., :3], cfg.affinity_type
            # )

            # print("affinity_matrix", affinity_matrix.shape)

            # # Get the GT label for each max-affinity GT label, detection pair.
            # idx_gts: NDArrayInt = affinity_matrix.argmax(axis=1)[None]

            # idx_gts_orig = idx_gts.copy()[0]

            # # The affinity matrix is an N by M matrix of the detections and ground truth labels respectively.
            # # We want to take the corresponding affinity for each of the initial assignments using gt_matches.
            # # The following line grabs the max affinity for each detection to a ground truth label.
            # affinities: NDArrayFloat = np.take_along_axis(
            #     affinity_matrix.transpose(), idx_gts, axis=0
            # )[0]

            # # Find the indices of the first detection assigned to each GT.
            # assignments: Tuple[NDArrayInt, NDArrayInt] = np.unique(
            #     idx_gts, return_index=True
            # )
            # idx_gts, idx_dts = assignments

            # print("idx_dts", idx_dts)

            # dts_is_tps = np.zeros((len(sweep_dts),), dtype=bool)
            # gts_is_tps = np.zeros((len(sweep_gts),), dtype=bool)

            # T, E = len(cfg.affinity_thresholds_m), 3
            # dts_metrics: NDArrayFloat = np.zeros((len(sweep_dts), T + E))
            # dts_metrics[:, 4:] = cfg.metrics_defaults[1:4]
            # gts_metrics: NDArrayFloat = np.zeros((len(sweep_gts), T + E))
            # gts_metrics[:, 4:] = cfg.metrics_defaults[1:4]
            # for i, threshold_m in enumerate(cfg.affinity_thresholds_m):
            #     is_tp: NDArrayBool = affinities[idx_dts] > -threshold_m

            #     dts_metrics[idx_dts[is_tp], i] = True
            #     gts_metrics[idx_gts, i] = True

            #     if threshold_m != cfg.tp_threshold_m:
            #         continue  # Skip if threshold isn't the true positive threshold.
            #     if not np.any(is_tp):
            #         continue  # Skip if no true positives exist.

            #     idx_tps_dts: NDArrayInt = idx_dts[is_tp]
            #     idx_tps_gts: NDArrayInt = idx_gts[is_tp]

            #     tps_dts = dts_array[idx_tps_dts]
            #     tps_gts = gts_array[idx_tps_gts]

            #     # keep track of tps
            #     dts_is_tps[idx_tps_dts] = True
            #     gts_is_tps[idx_tps_gts] = True

            #     translation_errors = distance(
            #         tps_dts[:, :3], tps_gts[:, :3], DistanceType.TRANSLATION
            #     )
            #     scale_errors = distance(
            #         tps_dts[:, 3:6], tps_gts[:, 3:6], DistanceType.SCALE
            #     )
            #     orientation_errors = distance(
            #         tps_dts[:, 6:10], tps_gts[:, 6:10], DistanceType.ORIENTATION
            #     )
            #     dts_metrics[idx_tps_dts, 4:] = np.stack(
            #         (translation_errors, scale_errors, orientation_errors), axis=-1
            #     )

            # # Create assignment pairs
            # assigned_pairs = []

            # # Time detection feature extraction
            # detection_count = 0

            # print(
            #     "dts_is_tps",
            #     dts_is_tps.shape,
            #     np.sum(dts_is_tps == 0),
            #     np.sum(dts_is_tps == 1),
            # )

            gt_lidar_boxes = argo2_box_to_lidar(gts_array).to(dtype=torch.float32)
            pred_lidar_boxes = argo2_box_to_lidar(dts_array).to(dtype=torch.float32)


            if len(gt_lidar_boxes) > 0 and len(pred_lidar_boxes) > 0:
                ious = rotate_iou_cpu_eval(gt_lidar_boxes, pred_lidar_boxes).reshape(gt_lidar_boxes.shape[0], pred_lidar_boxes.shape[0], 2)
                ious = ious[:, :, 0]
            else:
                ious = torch.zeros((len(gt_lidar_boxes), len(pred_lidar_boxes)), dtype=torch.float32)

            cost = torch.ones_like(ious) - ious
            cost = cost.numpy()

            gt_idxes, dt_idxes = linear_sum_assignment(cost)

            valid_mask = ious[gt_idxes, dt_idxes] > 0.3
            print(f'valid_mask', valid_mask.sum(), valid_mask.shape)

            gt_idxes = gt_idxes[valid_mask]
            dt_idxes = dt_idxes[valid_mask]

            # unmatched gts -> FN, unmatched_dts -> FP
            unmatched_gts = set(list(range(gt_lidar_boxes))).symmetric_difference(gt_idxes.tolist())
            unmatched_dts = set(list(range(pred_lidar_boxes))).symmetric_difference(dt_idxes.tolist())

            assert len(unmatched_gts) + len(gt_idxes) == len(gt_lidar_boxes)
            assert len(unmatched_dts) + len(dt_idxes) == len(pred_lidar_boxes)

            print(f"gt_idxes={len(gt_idxes)}, dt_idxes={len(dt_idxes)}, unmatched_dts={len(unmatched_dts)} unmatched_dts={len(unmatched_dts)}")

            assigned_pairs = []
            # add TPs
            for gt_idx, dt_idx in zip(gt_idxes, dt_idxes):
                assigned_pairs.append(("detection", dt_idx, gt_idx, "TP"))

            for dt_idx in unmatched_dts:
                assigned_pairs.append(("detection", dt_idx, None, "FP"))

            # Add unmatched GTs
            for gt_idx in unmatched_gts:
                assigned_pairs.append(("ground_truth", None, gt_idx, "FN"))

            # Add assigned DT-GT pairs
            # for dt_idx, dt in sweep_dts.iterrows():
            #     is_tp = dts_is_tps[dt_idx]
            #     gt_idx = idx_gts_orig[dt_idx]
            #     if is_tp:
            #         # Find which GT this DT is assigned to (you'll need to modify assign() to return this info)
            #         assigned_pairs.append(("detection", dt_idx, gt_idx, "TP"))

            #     else:
            #         assigned_pairs.append(("detection", dt_idx, None, "FP"))

            #     detection_count += 1

            # # Add unmatched GTs
            # for gt_idx, gt in sweep_gts.iterrows():
            #     is_matched = gts_metrics[gt_idx, : len(cfg.affinity_thresholds_m)].any()
            #     if not is_matched:
            #         assigned_pairs.append(("ground_truth", None, gt_idx, "FN"))

            # Replace both detection and GT loops with:
            for sample_type, dt_idx, gt_idx, outcome in assigned_pairs:

                if sample_type == "detection":
                    dt = sweep_dts.loc[dt_idx]
                    gt = sweep_gts.loc[gt_idx] if gt_idx is not None else None
                    features = self._extract_detection_pair_features(
                        dt, gt, sweep_dts, sweep_gts, cfg, threshold_col, ego_poses
                    )
                else:  # ground_truth
                    gt = sweep_gts.loc[gt_idx]
                    features = self._extract_single_gt_features(
                        gt, sweep_dts, sweep_gts, cfg, ego_poses
                    )

                features["sample_type"] = sample_type
                features["outcome"] = outcome

                if dt_idx in occlusions:
                    det_occlusions_dict = occlusions[dt_idx]
                    # features['dt_occlusion_overall'] = occlusion['overall_occlusion']
                    # features['dt_occlusion_lidar'] = occlusion['lidar_occlusion']
                    # features['dt_occlusion_bbox'] = occlusion['bbox_occlusion']
                    # features['dt_avg_ray_intersections'] = occlusion['avg_intersections_per_ray']
                    # features['dt_num_occluders'] = occlusion['num_occluder_candidates']

                    # log_id, timestamp_ns, occlusions = occlusion_result

                    keys_wanted = [
                        "overall_occlusion",
                        "lidar_occlusion",
                        "bbox_occlusion",
                        "avg_intersections_per_ray",
                        "num_occluder_candidates",
                        "distance_to_ego",
                        "confidence",
                        "shadow_point_ratio",
                        "points_near_ground",
                        "height_above_nearby_ground",
                        "path_obstruction_ratio",
                        "viewing_angle",
                    ]

                    pprint(dict(det_occlusions_dict=det_occlusions_dict))

                    assert all([x in det_occlusions_dict.keys() for x in keys_wanted])

                    for key in keys_wanted:
                        features[f"dt_occl_{key}"] = det_occlusions_dict[key]

                features_list.append(features)

            outcome_counter = Counter()
            outcome_counter.update([x[-1] for x in assigned_pairs])
            pprint(outcome_counter)

        outcome_counter = Counter()
        outcome_counter.update([x["outcome"] for x in features_list])
        pprint(outcome_counter)

        key_counter = Counter()
        for d in features_list:
            key_counter.update(d.keys())

        # print("Key counts:")
        for key, count in key_counter.items():
            if count != len(features_list):
                print(f"WARNING {key}: present in {count}/{len(features_list)} dicts")

        features_df = pd.DataFrame(features_list)
        print(
            f"Extracted {len(features_df)} feature vectors with {len(features_df.columns)} features"
        )

        features_df["gt_track_uuid"] = features_df["gt_track_uuid"].astype(str)

        for col in features_df.columns:
            if features_df[col].dtype == "object":
                types = features_df[col].map(type).value_counts()
                if len(types) > 1:
                    print(f"Column '{col}' has mixed types: {types.to_dict()}")

        return features_df

    def _get_default_gt_features(self) -> Dict:
        """Default GT features for FP cases."""
        return {
            "category": "NONE",
            # "num_interior_pts": 0,
            "length_m": 0,
            "width_m": 0,
            "height_m": 0,
            "tx_m": 0,
            "ty_m": 0,
            "tz_m": 0,
            "track_uuid": "",
            "qw": 0,
            "qx": 0,
            "qy": 0,
            "qz": 0,
            "distance_to_ego": 0,
            "0.5": 0,
            "1.0": 0,
            "2.0": 0,
            "4.0": 0,
        }

    def _get_default_det_features(self) -> Dict:
        """Default det features for FP cases."""
        return {
            "category": "NONE",
            "num_interior_pts": 0,
            "length_m": 0,
            "width_m": 0,
            "height_m": 0,
            "tx_m": 0,
            "ty_m": 0,
            "tz_m": 0,
            "distance_to_ego": 0,
            "score": 0.0,
            "distance_3d": 0.0,
            "azimuth": 0.0,
            "elevation": 0.0,
            "height_above_ground": 0.0,
            "volume": 0.0,
            "surface_area": 0.0,
            "compactness": 0.0,
            "log_id": "",
        }

    def _extract_detection_pair_features(
        self,
        detection: pd.Series,
        ground_truth: Optional[pd.Series],
        sweep_dts: pd.DataFrame,
        sweep_gts: pd.DataFrame,
        cfg: DetectionCfg,
        threshold_col: str,
        ego_poses: Optional[pd.DataFrame],
    ) -> Dict:
        """Extract features for a single detection."""
        features = {}

        features.update({f"dt_{k}": v for k, v in detection.items()})

        gt_box = argo2_box_to_lidar(ground_truth[
            [
                "tx_m",
                "ty_m",
                "tz_m",
                "length_m",
                "width_m",
                "height_m",
                "qw",
                "qx",
                "qy",
                "qz",
            ]
        ].values).to(dtype=torch.float32)

        dt_box = argo2_box_to_lidar(detection[
            [
                "tx_m",
                "ty_m",
                "tz_m",
                "length_m",
                "width_m",
                "height_m",
                "qw",
                "qx",
                "qy",
                "qz",
            ]
        ].values).to(dtype=torch.float32)

        print("dt_box", dt_box)
        print("gt_box", gt_box)

        iou3d, iou2d = rotate_iou_cpu_one(gt_box, dt_box)

        print("iou3d, iou2d", iou3d, iou2d)

        features['gt_iou3d'] = iou3d
        features['gt_iou2d'] = iou2d

        # Distance-based features
        features["dt_distance_to_ego"] = np.sqrt(
            detection["tx_m"] ** 2 + detection["ty_m"] ** 2
        )
        features["dt_distance_3d"] = np.sqrt(
            detection["tx_m"] ** 2 + detection["ty_m"] ** 2 + detection["tz_m"] ** 2
        )

        # Angular position relative to ego
        features["dt_azimuth"] = np.arctan2(detection["ty_m"], detection["tx_m"])
        features["dt_elevation"] = np.arctan2(
            detection["tz_m"], features["dt_distance_to_ego"]
        )

        # Height above ground (assuming ground is roughly at z=0)
        features["dt_height_above_ground"] = (
            detection["tz_m"] + detection["height_m"] / 2
        )

        # Volume and surface area
        features["dt_volume"] = (
            detection["length_m"] * detection["width_m"] * detection["height_m"]
        )
        features["dt_surface_area"] = 2 * (
            detection["length_m"] * detection["width_m"]
            + detection["length_m"] * detection["height_m"]
            + detection["width_m"] * detection["height_m"]
        )

        # Compactness - how "cube-like" is the object
        features["dt_compactness"] = features["dt_volume"] / (
            features["dt_surface_area"] ** 1.5
        )

        if ground_truth is not None:
            features.update({f"gt_{k}": v for k, v in ground_truth.items()})

            det_array = np.array(
                [
                    detection["tx_m"],
                    detection["ty_m"],
                    detection["tz_m"],
                    detection["length_m"],
                    detection["width_m"],
                    detection["height_m"],
                    detection["qw"],
                    detection["qx"],
                    detection["qy"],
                    detection["qz"],
                ]
            ).reshape(1, -1)

            gt_array = np.array(
                [
                    ground_truth["tx_m"],
                    ground_truth["ty_m"],
                    ground_truth["tz_m"],
                    ground_truth["length_m"],
                    ground_truth["width_m"],
                    ground_truth["height_m"],
                    ground_truth["qw"],
                    ground_truth["qx"],
                    ground_truth["qy"],
                    ground_truth["qz"],
                ]
            ).reshape(1, -1)

            # Calculate error metrics using AV2 distance functions
            features["ATE"] = distance(
                det_array[:, :3], gt_array[:, :3], DistanceType.TRANSLATION
            )[0]

            features["ASE"] = distance(
                det_array[:, 3:6], gt_array[:, 3:6], DistanceType.SCALE
            )[0]

            features["AOE"] = distance(
                det_array[:, 6:10], gt_array[:, 6:10], DistanceType.ORIENTATION
            )[0]

            # Category match
            features["category_match"] = (
                detection["category"] == ground_truth["category"]
            )

            # Raw size differences (in addition to normalized ASE)
            features["length_diff"] = abs(
                detection["length_m"] - ground_truth["length_m"]
            )
            features["width_diff"] = abs(detection["width_m"] - ground_truth["width_m"])
            features["height_diff"] = abs(
                detection["height_m"] - ground_truth["height_m"]
            )

            # Position differences (in addition to normalized ATE)
            features["tx_diff"] = abs(detection["tx_m"] - ground_truth["tx_m"])
            features["ty_diff"] = abs(detection["ty_m"] - ground_truth["ty_m"])
            features["tz_diff"] = abs(detection["tz_m"] - ground_truth["tz_m"])

            # Relative errors (useful for understanding scale-dependent patterns)
            features["relative_length_error"] = features["length_diff"] / max(
                ground_truth["length_m"], 0.01
            )
            features["relative_width_error"] = features["width_diff"] / max(
                ground_truth["width_m"], 0.01
            )
            features["relative_height_error"] = features["height_diff"] / max(
                ground_truth["height_m"], 0.01
            )

            features["gt_aspect_ratio_lw"] = ground_truth["length_m"] / max(
                ground_truth["width_m"], 0.01
            )
            features["gt_aspect_ratio_lh"] = ground_truth["length_m"] / max(
                ground_truth["height_m"], 0.01
            )
            features["gt_volume"] = (
                ground_truth["length_m"]
                * ground_truth["width_m"]
                * ground_truth["height_m"]
            )

            features["gt_distance_to_ego"] = np.sqrt(
                ground_truth["tx_m"] ** 2 + ground_truth["ty_m"] ** 2
            )

            # Contextual features
            features["dt_num_nearby_objects"] = self._count_nearby_objects(
                detection, sweep_dts, radius=5.0
            )
            features["dt_num_same_class_nearby"] = self._count_nearby_objects(
                detection,
                sweep_dts[sweep_dts["category"] == detection["category"]],
                radius=5.0,
            )
            features["dt_num_all_objects_nearby"] = self._count_nearby_objects(
                detection, sweep_dts, radius=5.0
            )

        else:
            default_gt_features = self._get_default_gt_features()
            features.update({f"gt_{k}": v for k, v in default_gt_features.items()})

            # Default pair features for FPs (no ground truth to compare against)
            features.update(
                {
                    "ATE": float("inf"),  # or MAX_TRANSLATION_ERROR if defined
                    "ASE": MAX_NORMALIZED_ASE,  # Use AV2 constant
                    "AOE": MAX_YAW_RAD_ERROR,  # Use AV2 constant
                    "category_match": False,
                    "length_diff": 10,
                    "width_diff": 10,
                    "height_diff": 10,
                    "tx_diff": 10,
                    "ty_diff": 10,
                    "tz_diff": 10,
                    "relative_length_error": 10,
                    "relative_width_error": 10,
                    "relative_height_error": 10,
                    "gt_aspect_ratio_lw": 0,
                    "gt_aspect_ratio_lh": 0,
                    "gt_volume": 0,
                    "gt_distance_to_ego": 0,
                }
            )

        return features

    # def _extract_single_detection_features_old(
    #     self,
    #     detection: pd.Series,
    #     sweep_dts: pd.DataFrame,
    #     sweep_gts: pd.DataFrame,
    #     cfg: DetectionCfg,
    #     threshold_col: str,
    #     ego_poses: Optional[pd.DataFrame],
    # ) -> Dict:
    #     """Extract features for a single detection."""
    #     features = {}

    #     # Basic object properties
    #     features["category"] = detection["category"]
    #     features["log_id"] = detection["log_id"]
    #     features["timestamp_ns"] = detection["timestamp_ns"]

    #     # Geometric features
    #     features["length_m"] = detection.get("length_m", 0)
    #     features["width_m"] = detection.get("width_m", 0)
    #     features["height_m"] = detection.get("height_m", 0)
    #     features["aspect_ratio_lw"] = features["length_m"] / max(
    #         features["width_m"], 0.01
    #     )
    #     features["aspect_ratio_lh"] = features["length_m"] / max(
    #         features["height_m"], 0.01
    #     )
    #     features["volume"] = (
    #         features["length_m"] * features["width_m"] * features["height_m"]
    #     )

    #     # Position and distance
    #     features["tx_m"] = detection.get("tx_m", 0)
    #     features["ty_m"] = detection.get("ty_m", 0)
    #     features["tz_m"] = detection.get("tz_m", 0)
    #     features["distance_to_ego"] = np.sqrt(
    #         features["tx_m"] ** 2 + features["ty_m"] ** 2
    #     )
    #     features["height_above_ground"] = features["tz_m"]

    #     # Orientation (if available)
    #     if "qw" in detection and "qz" in detection:
    #         # Convert quaternion to yaw angle
    #         qw, qx, qy, qz = (
    #             detection["qw"],
    #             detection.get("qx", 0),
    #             detection.get("qy", 0),
    #             detection["qz"],
    #         )
    #         features["orientation_yaw"] = 2 * np.arctan2(qz, qw)
    #     else:
    #         features["orientation_yaw"] = 0

    #     # Detection quality metrics
    #     features["detection_score"] = detection.get("score", 0)
    #     features["ATE"] = detection.get("translation_error", 0)
    #     features["ASE"] = detection.get("scale_error", 0)
    #     features["AOE"] = detection.get("orientation_error", 0)
    #     features["is_evaluated"] = detection.get("is_evaluated", False)

    #     # Contextual features
    #     # features['num_nearby_objects'] = self._count_nearby_objects(
    #     #     detection, sweep_dts, radius=5.0
    #     # )
    #     # features['num_same_class_nearby'] = self._count_nearby_objects(
    #     #     detection, sweep_dts[sweep_dts['category'] == detection['category']], radius=5.0
    #     # )
    #     # features['num_all_objects_nearby'] = self._count_nearby_objects(
    #     #     detection, pd.concat([sweep_dts, sweep_gts]), radius=5.0
    #     # )

    #     # Occlusion estimation
    #     features["occlusion_level_estimate"] = 0
    #     # features['occlusion_level_estimate'] = self._estimate_occlusion(
    #     #     detection, sweep_gts
    #     # )

    #     # Category frequency (how common is this class)
    #     all_categories = pd.concat([sweep_dts["category"], sweep_gts["category"]])
    #     features["category_freq"] = (
    #         all_categories == detection["category"]
    #     ).sum() / len(all_categories)

    #     # Time-based features
    #     features["time_of_day"] = self._extract_time_of_day(detection["timestamp_ns"])

    #     # Ego speed (if poses available)
    #     if ego_poses is not None:
    #         features["ego_speed"] = self._get_ego_speed(detection, ego_poses)
    #     else:
    #         features["ego_speed"] = 0

    #     # Detection consistency (simplified - would need tracking info)
    #     features["was_detected_last_frame"] = 0  # Placeholder

    #     return features

    def _extract_single_gt_features(
        self,
        ground_truth: pd.Series,
        sweep_dts: pd.DataFrame,
        sweep_gts: pd.DataFrame,
        cfg: DetectionCfg,
        ego_poses: Optional[pd.DataFrame],
    ) -> Dict:
        """Extract features for a missed ground truth (False Negative)."""
        features = {}

        features.update({f"gt_{k}": v for k, v in ground_truth.items()})
        dt_features = self._get_default_det_features()
        features.update({f"dt_{k}": v for k, v in dt_features.items()})

        features.update(
            {
                "category_match": False,
                "length_diff": 10,
                "width_diff": 10,
                "height_diff": 10,
                "tx_diff": 10,
                "ty_diff": 10,
                "tz_diff": 10,
                "gt_iou2d": 0.0,
                "gt_iou3d": 0.0,
                "relative_length_error": 10,
                "relative_width_error": 10,
                "relative_height_error": 10,
                "gt_aspect_ratio_lw": 0,
                "gt_aspect_ratio_lh": 0,
                "gt_volume": 0,
                "gt_distance_to_ego": 0,
            }
        )

        # features["num_nearby_dt_objects"] = self._count_nearby_objects(
        #     ground_truth, sweep_dts, radius=5.0
        # )
        # features["num_same_class_nearby"] = self._count_nearby_objects(
        #     ground_truth,
        #     sweep_gts[sweep_gts["category"] == ground_truth["category"]],
        #     radius=5.0,
        # )
        # features["num_all_objects_nearby"] = self._count_nearby_objects(
        #     ground_truth, sweep_gts, radius=5.0
        # )

        return features

    def _count_nearby_objects(
        self, obj: pd.Series, objects: pd.DataFrame, radius: float = 5.0
    ) -> int:
        """Count objects within radius of given object."""
        if len(objects) == 0:
            return 0

        obj_pos = np.array([obj.get("tx_m", 0), obj.get("ty_m", 0)])
        other_pos = objects[["tx_m", "ty_m"]].values

        distances = np.linalg.norm(other_pos - obj_pos, axis=1)
        return np.sum(
            (distances < radius) & (distances > 0.1)
        )  # Exclude self/very close

    def _estimate_occlusion(self, obj: pd.Series, other_objects: pd.DataFrame) -> float:
        """
        Estimate occlusion level using bounding box overlap heuristic.
        Returns value between 0 (no occlusion) and 1 (heavily occluded).
        """
        if len(other_objects) == 0:
            return 0.0

        # Simple 2D bounding box overlap approach
        obj_x, obj_y = obj.get("tx_m", 0), obj.get("ty_m", 0)
        obj_l, obj_w = obj.get("length_m", 1), obj.get("width_m", 1)

        # Count overlapping objects within 2m distance (closer objects more likely to occlude)
        close_objects = other_objects[
            np.sqrt(
                (other_objects["tx_m"] - obj_x) ** 2
                + (other_objects["ty_m"] - obj_y) ** 2
            )
            < 2.0
        ]

        if len(close_objects) == 0:
            return 0.0

        # Simple density-based occlusion estimate
        density = len(close_objects) / max(obj_l * obj_w, 0.1)
        return min(density / 3.0, 1.0)  # Normalize to [0,1]

    def _extract_time_of_day(self, timestamp_ns: int) -> float:
        """Extract time of day as hours (0-24) from nanosecond timestamp."""
        import datetime

        dt = datetime.datetime.fromtimestamp(timestamp_ns / 1e9)
        return dt.hour + dt.minute / 60.0

    def _get_ego_speed(self, obj: pd.Series, ego_poses: pd.DataFrame) -> float:
        """Get ego vehicle speed at object timestamp (simplified)."""
        # This would require proper ego pose interpolation in real implementation
        return 10.0  # Placeholder - assume 10 m/s

    def train_random_forest_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced training with better validation and analysis.
        """
        print("Training Enhanced Random Forest models...")
        results = {}

        # Enhanced feature preparation
        X_processed, feature_cols = self._prepare_features_enhanced(features_df)

        # 1. Enhanced TP vs FP classification
        detection_mask = features_df["sample_type"] == "detection"
        if detection_mask.sum() > self.config.min_samples_per_class:
            results["tp_vs_fp"] = self._train_classification_model_enhanced(
                X_processed[detection_mask],
                features_df.loc[detection_mask, "outcome"],
                "TP vs FP Classification",
                feature_cols,
                use_probability_calibration=True,
            )

        # 2. Multi-class detection analysis (TP/FP/FN)
        outcome_mapping = {"TP": 0, "FP": 1, "FN": 2}
        multi_class_labels = features_df["outcome"].map(outcome_mapping)
        results["multiclass_detection"] = self._train_classification_model_enhanced(
            X_processed, multi_class_labels, "Multi-class Detection", feature_cols
        )

        # 3. Enhanced regression analysis with multiple targets
        tp_mask = features_df["outcome"] == "TP"
        if tp_mask.sum() > self.config.min_samples_per_class:
            results["regression_models"] = self._train_multi_target_regression(
                X_processed[tp_mask], features_df[tp_mask], feature_cols
            )

        # 4. Distance-based analysis
        results["distance_analysis"] = self._analyze_by_distance(
            X_processed, features_df, feature_cols
        )

        # 5. Enhanced per-category analysis with cross-validation
        results["per_category"] = self._enhanced_category_analysis(
            X_processed, features_df, feature_cols
        )

        # 6. Failure mode analysis
        results["failure_modes"] = self._analyze_failure_modes(
            X_processed, features_df, feature_cols
        )

        return results

    def _prepare_features_enhanced(
        self, features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Enhanced feature preparation with better handling of edge cases."""

        # Select numeric features more carefully
        feature_cols = []
        for col in features_df.columns:
            if col in self.ignore_columns:
                continue
            if "gt_" in col:  # remove gt features
                continue
            if pd.api.types.is_numeric_dtype(features_df[col]):
                # Check for sufficient variance
                if features_df[col].var() > 1e-10:
                    feature_cols.append(col)

        print(f"Selected {feature_cols=} features with sufficient variance")

        X_processed = features_df[feature_cols].copy()

        # Enhanced categorical encoding
        categorical_cols = [
            col
            for col in feature_cols
            if "category" in col.lower() or col in ["outcome"]
        ]
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(
                    features_df[col].astype(str)
                )

        # Better handling of infinite/missing values
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)

        # Use different imputation strategies for different feature types
        for col in X_processed.columns:
            if X_processed[col].isna().sum() > 0:
                if "distance" in col.lower() or "depth" in col.lower():
                    # Use median for distance-like features
                    X_processed[col] = X_processed[col].fillna(
                        X_processed[col].median()
                    )
                else:
                    # Use mean for other continuous features
                    X_processed[col] = X_processed[col].fillna(X_processed[col].mean())

        # Robust scaling
        X_scaled = self.scaler.fit_transform(X_processed)
        X_df = pd.DataFrame(X_scaled, columns=feature_cols, index=features_df.index)

        return X_df, feature_cols

    def _train_classification_model_enhanced(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        feature_names: List[str],
        use_probability_calibration: bool = False,
        max_samples: int = 10000,
    ) -> Dict:
        """Enhanced classification with cross-validation and better metrics."""

        if len(y.unique()) < 2:
            return {"error": f"Insufficient class diversity for {model_name}"}

        # Sample data if too large
        original_size = len(X)
        if len(X) > max_samples:
            print(
                f"Sampling {max_samples} from {original_size} rows for {model_name} training"
            )

            # Stratified sampling to maintain class distribution
            sample_indices = []
            for class_label in y.unique():
                class_mask = y == class_label
                class_size = class_mask.sum()

                # Proportional sampling
                n_samples_class = int((class_size / len(y)) * max_samples)
                n_samples_class = max(
                    1, min(n_samples_class, class_size)
                )  # At least 1, at most available

                class_indices = y[class_mask].index
                sampled_class_indices = np.random.choice(
                    class_indices, size=n_samples_class, replace=False
                )
                sample_indices.extend(sampled_class_indices)

            # Apply sampling
            X = X.loc[sample_indices]
            y = y.loc[sample_indices]
            print(
                f"Sampled data: {len(X)} rows, class distribution: {y.value_counts().to_dict()}"
            )

        # Stratified split with larger test size for better evaluation
        test_size = max(0.3, self.config.test_size)  # At least 30% for test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.config.random_seed,
            stratify=y if len(y.unique()) > 1 else None,
        )

        # Reduced Random Forest for faster training on sampled data
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,  # Reduced from 200+ since we're sampling
            max_depth=self.config.max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=self.config.random_seed,
            class_weight="balanced",
            n_jobs=-1,  # Use all available cores
            verbose=0,  # Reduced verbosity
        )

        print(f"Training {model_name} on {len(X_train)} samples...")
        rf.fit(X_train, y_train)

        # Comprehensive predictions
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)

        # Enhanced metrics
        metrics = {
            "test_accuracy": rf.score(X_test, y_test),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "model_name": model_name,
            "n_samples_original": original_size,
            "n_samples_used": len(X),
            "n_features": len(feature_names),
            "sampling_ratio": len(X) / original_size,
        }

        pprint(metrics)

        # Add AUC for binary classification
        if len(y.unique()) == 2:
            unique_labels = sorted(y_test.unique())
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            y_test_binary = y_test.map(label_map)
            y_scores = y_pred_proba[:, 1]

            metrics["roc_auc"] = roc_auc_score(y_test_binary, y_scores)
            metrics["average_precision"] = average_precision_score(
                y_test_binary, y_scores
            )

        # Enhanced feature importance analysis
        metrics.update(
            self._analyze_feature_importance(rf, X_test, y_test, feature_names)
        )

        print("feature_importance", metrics["feature_importance"])

        return metrics

    def _analyze_feature_importance(self, model, X_test, y_test, feature_names) -> Dict:
        """Comprehensive feature importance analysis."""

        # Standard feature importance
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Permutation importance (more reliable)
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
            )

            perm_importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "perm_importance_mean": perm_importance.importances_mean,
                    "perm_importance_std": perm_importance.importances_std,
                }
            ).sort_values("perm_importance_mean", ascending=False)

        except Exception as e:
            print(f"Permutation importance failed: {e}")
            perm_importance_df = None

        return {
            "feature_importance": importance_df,
            "permutation_importance": perm_importance_df,
            "top_5_features": importance_df.head(5)["feature"].tolist(),
        }

    def _train_multi_target_regression(
        self, X: pd.DataFrame, features_df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict:
        """Train regression models for multiple error metrics."""

        regression_results = {}

        # Define regression targets with better validation
        targets = {
            "ATE": "gt_ATE",
            "ASE": "gt_ASE",
            "distance_error": (
                "distance_error" if "distance_error" in features_df.columns else None
            ),
            "iou_3d": "iou_3d" if "iou_3d" in features_df.columns else None,
        }

        for target_name, target_col in targets.items():
            if target_col is None or target_col not in features_df.columns:
                continue

            # Clean target values
            valid_mask = (
                features_df[target_col].replace([np.inf, -np.inf], np.nan).notna()
            )

            if valid_mask.sum() < 20:  # Need sufficient samples
                continue

            y_target = features_df.loc[valid_mask, target_col]
            X_target = X.loc[valid_mask]

            # Remove outliers (beyond 3 std)
            if target_name in ["ATE", "ASE"]:
                z_scores = np.abs((y_target - y_target.mean()) / y_target.std())
                outlier_mask = z_scores < 3
                y_target = y_target[outlier_mask]
                X_target = X_target.loc[y_target.index]

            regression_results[target_name] = self._train_single_regression(
                X_target, y_target, f"{target_name} Regression", feature_cols
            )

        return regression_results

    def _train_single_regression(
        self, X: pd.DataFrame, y: pd.Series, model_name: str, feature_names: List[str]
    ) -> Dict:
        """Train a single regression model with comprehensive evaluation."""

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.config.random_seed
        )

        rf_reg = RandomForestRegressor(
            n_estimators=200,
            max_depth=self.config.max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=self.config.random_seed,
            n_jobs=-1,
            verbose=0,
        )

        # Cross-validation
        # cv_scores = cross_val_score(
        #     rf_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
        # )

        rf_reg.fit(X_train, y_train)
        y_pred = rf_reg.predict(X_test)

        print(f"{model_name=} r2_score=", r2_score(y_test, y_pred))
        pprint(
            pd.DataFrame(
                {"feature": feature_names, "importance": rf_reg.feature_importances_}
            ).sort_values("importance", ascending=False)
        )

        return {
            # "model": rf_reg,
            "r2_score": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            # "cv_mse_mean": -cv_scores.mean(),
            # "cv_mse_std": cv_scores.std(),
            "model_name": model_name,
            "feature_importance": pd.DataFrame(
                {"feature": feature_names, "importance": rf_reg.feature_importances_}
            ).sort_values("importance", ascending=False),
        }

    def _analyze_by_distance(
        self, X: pd.DataFrame, features_df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict:
        """Analyze performance by distance ranges."""

        if "distance" not in features_df.columns:
            return {"error": "No distance column found"}

        # Define distance bins
        distances = features_df["distance"]
        distance_bins = [0, 20, 40, 60, 80, 100, np.inf]
        distance_labels = ["0-20m", "20-40m", "40-60m", "60-80m", "80-100m", "100m+"]

        features_df["distance_bin"] = pd.cut(
            distances, bins=distance_bins, labels=distance_labels, include_lowest=True
        )

        distance_results = {}

        for bin_label in distance_labels:
            bin_mask = features_df["distance_bin"] == bin_label
            if bin_mask.sum() < self.config.min_samples_per_class:
                continue

            bin_outcomes = features_df.loc[bin_mask, "outcome"].map(
                {"TP": 0, "FP": 1, "FN": 2}
            )

            if len(bin_outcomes.unique()) > 1:
                distance_results[bin_label] = self._train_classification_model_enhanced(
                    X[bin_mask], bin_outcomes, f"Detection at {bin_label}", feature_cols
                )

        return distance_results

    def _enhanced_category_analysis(
        self, X: pd.DataFrame, features_df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict:
        """Enhanced per-category analysis with statistical tests."""

        category_results = {}

        for category in features_df["dt_category"].unique():
            cat_mask = features_df["dt_category"] == category
            cat_data = features_df[cat_mask]

            if len(cat_data) < self.config.min_samples_per_class:
                continue

            # Calculate category-specific statistics
            category_stats = {
                "total_samples": len(cat_data),
                "tp_rate": (cat_data["outcome"] == "TP").mean(),
                "fp_rate": (cat_data["outcome"] == "FP").mean(),
                "fn_rate": (cat_data["outcome"] == "FN").mean(),
                "precision": len(cat_data[cat_data["outcome"] == "TP"])
                / max(1, len(cat_data[cat_data["outcome"].isin(["TP", "FP"])])),
                "recall": len(cat_data[cat_data["outcome"] == "TP"])
                / max(1, len(cat_data[cat_data["outcome"].isin(["TP", "FN"])])),
            }

            # Train category-specific model
            cat_outcomes = cat_data["outcome"].map({"TP": 0, "FP": 1, "FN": 2})
            if len(cat_outcomes.unique()) > 1:
                model_results = self._train_classification_model_enhanced(
                    X[cat_mask], cat_outcomes, f"{category} Detection", feature_cols
                )
                category_results[category] = {**category_stats, **model_results}
            else:
                category_results[category] = category_stats

        return category_results

    def _analyze_failure_modes(
        self, X: pd.DataFrame, features_df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict:
        """Analyze common failure modes and patterns."""

        failure_analysis = {}

        # False Positive analysis
        fp_mask = features_df["outcome"] == "FP"
        if fp_mask.sum() > 10:
            # What makes FPs different from TPs?
            tp_mask = features_df["outcome"] == "TP"

            # Compare feature distributions
            feature_comparison = {}
            for feature in feature_cols[:10]:  # Top 10 features
                if feature in features_df.columns:
                    tp_mean = features_df.loc[tp_mask, feature].mean()
                    fp_mean = features_df.loc[fp_mask, feature].mean()
                    feature_comparison[feature] = {
                        "tp_mean": tp_mean,
                        "fp_mean": fp_mean,
                        "difference": fp_mean - tp_mean,
                    }

            failure_analysis["fp_vs_tp_features"] = feature_comparison

        # False Negative analysis
        fn_mask = features_df["outcome"] == "FN"
        if fn_mask.sum() > 10:
            # Analyze what ground truth objects are being missed
            fn_categories = features_df.loc[fn_mask, "dt_category"].value_counts()
            failure_analysis["missed_categories"] = fn_categories.to_dict()

            if "distance" in features_df.columns:
                fn_distances = features_df.loc[fn_mask, "distance"].describe()
                failure_analysis["missed_distance_stats"] = fn_distances.to_dict()

        return failure_analysis

    def _train_classification_model(
        self, X: pd.DataFrame, y: pd.Series, model_name: str, feature_names: List[str]
    ) -> Dict:
        """Train a Random Forest classifier and extract insights."""
        if len(y.unique()) < 2:
            return {"error": f"Insufficient class diversity for {model_name}"}

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=y if len(y.unique()) > 1 else None,
        )

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_seed,
            class_weight="balanced",
            verbose=0,
        )

        rf.fit(X_train, y_train)

        # Predictions and metrics
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)

        # Feature importance
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        return {
            # "model": rf,
            "feature_importance": importance_df,
            "test_accuracy": rf.score(X_test, y_test),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "model_name": model_name,
            "n_samples": len(X),
            "n_features": len(feature_names),
        }

    def _train_regression_model(
        self, X: pd.DataFrame, y: pd.Series, model_name: str, feature_names: List[str]
    ) -> Dict:
        """Train a Random Forest regressor and extract insights."""
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_seed
        )

        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_seed,
            verbose=0,
        )

        rf.fit(X_train, y_train)

        # Predictions and metrics
        y_pred = rf.predict(X_test)

        # Feature importance
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        return {
            # "model": rf,
            "feature_importance": importance_df,
            "test_r2": rf.score(X_test, y_test),
            "test_rmse": np.sqrt(np.mean((y_test - y_pred) ** 2)),
            "model_name": model_name,
            "n_samples": len(X),
            "n_features": len(feature_names),
        }

    def perform_error_clustering(
        self, features_df: pd.DataFrame, max_samples: int = 10000
    ) -> Dict[str, Any]:
        """
        Perform clustering analysis to discover failure modes.
        """
        print("Performing error clustering analysis...")

        # Prepare feature matrix
        available_features = [
            col
            for col in features_df.columns
            if pd.api.types.is_numeric_dtype(features_df[col])
        ]

        if len(available_features) < 3:
            return {"error": "Insufficient features for clustering"}

        # Use only samples with finite error values (exclude FN with inf errors)
        finite_mask = np.isfinite(features_df["ATE"]) & np.isfinite(features_df["ASE"])
        for col in available_features:
            finite_mask &= np.isfinite(features_df[col])
        cluster_data = features_df[finite_mask].copy()

        if len(cluster_data) > max_samples:
            print(
                f"Sampling {max_samples} from {len(cluster_data)} rows for clustering"
            )
            cluster_data = cluster_data.sample(
                n=max_samples, random_state=self.config.random_seed
            )

        if len(cluster_data) < 50:
            return {"error": "Insufficient data for clustering"}

        # Prepare feature matrix
        X_cluster = cluster_data[available_features].copy()

        for col in X_cluster.columns:
            if X_cluster[col].dtype == "float64":
                X_cluster[col] = X_cluster[col].astype("float32")
            elif X_cluster[col].dtype == "int64":
                X_cluster[col] = X_cluster[col].astype("int32")

        # Encode categoricals
        if "category" in X_cluster.columns:
            le = LabelEncoder()
            X_cluster["category"] = le.fit_transform(X_cluster["category"].astype(str))

        # Scale features
        X_scaled = StandardScaler().fit_transform(X_cluster.fillna(X_cluster.median()))

        results = {}

        # DBSCAN clustering
        dbscan = DBSCAN(
            eps=self.config.dbscan_eps, min_samples=self.config.dbscan_min_samples
        )
        dbscan_labels = dbscan.fit_predict(X_scaled)

        results["dbscan"] = {
            "labels": dbscan_labels,
            "n_clusters": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            "n_noise": list(dbscan_labels).count(-1),
            "silhouette_score": self._safe_silhouette_score(X_scaled, dbscan_labels),
        }

        # Gaussian Mixture Model
        gmm = GaussianMixture(
            n_components=self.config.n_gmm_components,
            random_state=self.config.random_seed,
        )
        gmm_labels = gmm.fit_predict(X_scaled)
        del X_scaled  # If not needed later
        gc.collect()

        results["gmm"] = {
            "labels": gmm_labels,
            "n_clusters": self.config.n_gmm_components,
            "bic": gmm.bic(X_scaled),
            "aic": gmm.aic(X_scaled),
            "silhouette_score": self._safe_silhouette_score(X_scaled, gmm_labels),
        }

        # Analyze clusters
        results["cluster_analysis"] = self._analyze_clusters(
            cluster_data, dbscan_labels, "DBSCAN"
        )
        results["feature_names"] = available_features
        results["cluster_data_indices"] = cluster_data.index.tolist()

        return results

    def _safe_silhouette_score(self, X, labels):
        """Compute silhouette score safely."""
        try:
            from sklearn.metrics import silhouette_score

            unique_labels = set(labels)
            if len(unique_labels) > 1 and len(unique_labels) < len(X):
                return silhouette_score(X, labels)
        except:
            pass
        return np.nan

    def _analyze_clusters(
        self, data: pd.DataFrame, labels: np.ndarray, method: str
    ) -> Dict:
        """Analyze characteristics of discovered clusters."""
        analysis = {}

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise cluster

        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]

            if len(cluster_data) < 5:
                continue

            analysis[f"cluster_{cluster_id}"] = {
                "size": len(cluster_data),
                "mean_ate": (
                    cluster_data["ATE"].mean() if "ATE" in cluster_data else np.nan
                ),
                "mean_distance": (
                    cluster_data["gt_distance_to_ego"].mean()
                    if "gt_distance_to_ego" in cluster_data
                    else np.nan
                ),
                "dominant_category": (
                    cluster_data["dt_category"].mode().iloc[0]
                    if "dt_category" in cluster_data
                    else "unknown"
                ),
                "mean_occlusion": (
                    cluster_data["occlusion_level_estimate"].mean()
                    if "occlusion_level_estimate" in cluster_data
                    else np.nan
                ),
                "outcome_distribution": (
                    cluster_data["outcome"].value_counts().to_dict()
                    if "outcome" in cluster_data
                    else {}
                ),
            }

        return analysis

    def create_dimensionality_reduction_viz(
        self, features_df: pd.DataFrame, clustering_results: Dict = None
    ) -> Dict[str, Any]:
        """
        Create t-SNE and UMAP visualizations of the feature space.
        """
        print("Creating dimensionality reduction visualizations...")

        sampled_features = features_df.sample(n=1000)

        X_scaled, feature_cols = self._prepare_features_enhanced(sampled_features)

        print(f"{feature_cols=}")

        results = {}

        # t-SNE
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=min(self.config.tsne_perplexity, len(X_scaled) // 4),
                random_state=self.config.random_seed,
            )
            tsne_embedding = tsne.fit_transform(X_scaled)
            results["tsne"] = {
                "embedding": tsne_embedding,
                "feature_names": feature_cols,
            }
        except Exception as e:
            print(f"t-SNE failed: {e}")
            results["tsne"] = {"error": str(e)}

        # UMAP
        try:
            umap_reducer = umap.UMAP(
                n_neighbors=min(self.config.umap_n_neighbors, len(X_scaled) - 1),
                min_dist=self.config.umap_min_dist,
                random_state=self.config.random_seed,
            )
            umap_embedding = umap_reducer.fit_transform(X_scaled)
            results["umap"] = {
                "embedding": umap_embedding,
                "feature_names": feature_cols,
            }
        except Exception as e:
            print(f"UMAP failed: {e}")
            results["umap"] = {"error": str(e)}

        # outcomes = viz_data["outcome"].values

        print("X_scaled keys", X_scaled.keys())

        # Store metadata for coloring
        results["metadata"] = {
            "outcomes": sampled_features["outcome"].values,
            "categories": sampled_features["dt_category"].values,
            "distances": sampled_features["gt_distance_to_ego"].values,
            "scores": None,
            "indices": X_scaled.index.tolist(),
        }

        # Add clustering labels if available
        if clustering_results and "cluster_data_indices" in clustering_results:
            cluster_indices = clustering_results["cluster_data_indices"]
            # Match indices
            cluster_labels = np.full(len(X_scaled), -1)
            for i, idx in enumerate(X_scaled.index):
                if idx in cluster_indices:
                    pos = cluster_indices.index(idx)
                    cluster_labels[i] = clustering_results["dbscan"]["labels"][pos]
            results["metadata"]["cluster_labels"] = cluster_labels

        return results

    def train_interpretable_trees(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train simple decision trees to extract interpretable rules.
        """
        print("Training interpretable decision trees...")

        results = {}

        column_types = features_df.dtypes
        print(column_types)

        # Prepare feature matrix
        feature_cols = [
            col
            for col in features_df.columns
            if pd.api.types.is_numeric_dtype(features_df[col])
            and col not in self.ignore_columns
            and "gt_" not in col
        ]

        X_processed = features_df[feature_cols].copy()

        # Encode categoricals
        for col in ["category", "outcome"]:
            if col in X_processed.columns:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))

        # Handle infinite values
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        X_processed = X_processed.fillna(X_processed.median())

        # 1. Detection success tree
        detection_success = features_df["outcome"].map({"TP": 1, "FP": 0, "FN": 0})

        tree_clf = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=self.config.random_seed
        )
        tree_clf.fit(X_processed, detection_success)

        rules = export_text(tree_clf, feature_names=feature_cols, max_depth=5)
        results["detection_success_tree"] = {
            # "model": tree_clf,
            "rules": rules,
            "feature_importance": pd.DataFrame(
                {"feature": feature_cols, "importance": tree_clf.feature_importances_}
            ).sort_values("importance", ascending=False),
            "accuracy": tree_clf.score(X_processed, detection_success),
        }

        print("detection_success_tree rules", rules)
        print(
            "detection_success_tree accuracy",
            results["detection_success_tree"]["accuracy"],
        )
        print(
            "detection_success_tree feature_importance",
            results["detection_success_tree"]["feature_importance"],
        )

        # 2. TP vs FP tree (for detections only)
        detection_mask = features_df["sample_type"] == "detection"

        # Filter only TP and FP outcomes for detections
        tp_fp_mask = detection_mask & features_df["outcome"].isin(["TP", "FP"])

        if tp_fp_mask.sum() > 20:
            # Label: TP -> 1, FP -> 0
            tp_fp_labels = (features_df.loc[tp_fp_mask, "outcome"] == "TP").astype(int)

            tree_tp_fp = DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=15, random_state=self.config.random_seed
            )
            tree_tp_fp.fit(X_processed[tp_fp_mask], tp_fp_labels)

            rules = export_text(tree_tp_fp, feature_names=feature_cols)
            print("tp_vs_fp_tree rules", rules)

            results["tp_vs_fp_tree"] = {
                "rules": rules,
                "feature_importance": pd.DataFrame(
                    {
                        "feature": feature_cols,
                        "importance": tree_tp_fp.feature_importances_,
                    }
                ).sort_values("importance", ascending=False),
                "accuracy": tree_tp_fp.score(X_processed[tp_fp_mask], tp_fp_labels),
            }

            print("tp_vs_fp_tree accuracy", results["tp_vs_fp_tree"]["accuracy"])
            print(
                "tp_vs_fp_tree feature_importance",
                results["tp_vs_fp_tree"]["feature_importance"],
            )

        return results

    def create_comprehensive_visualizations(
        self,
        rf_results: Dict,
        clustering_results: Dict,
        dimred_results: Dict,
        tree_results: Dict,
        output_path: str,
    ):
        """Create comprehensive visualization plots as separate files."""
        output_dir = Path(output_path).parent

        # 1. Feature importance comparison
        self._create_feature_importance_plot(
            rf_results, output_dir / "feature_importance.png"
        )

        # 2. Dimensionality reduction plots
        if "tsne" in dimred_results and "error" not in dimred_results["tsne"]:
            self._create_tsne_plot(
                dimred_results, output_dir / "tsne_visualization.png"
            )
        else:
            print("tsne error?")
            print("dimred_results", dimred_results.keys())
            pprint(dimred_results["tsne"])

        if "umap" in dimred_results and "error" not in dimred_results["umap"]:
            self._create_umap_plot(
                dimred_results, output_dir / "umap_visualization.png"
            )
        else:
            print("umap error?")
            print("dimred_results", dimred_results.keys())
            pprint(dimred_results["umap"])

        # 3. Clustering analysis
        if "dbscan" in clustering_results:
            self._create_clustering_plot(
                clustering_results, output_dir / "clustering_analysis.png"
            )

        # 4. Model performance comparison
        self._create_performance_comparison(
            rf_results, output_dir / "model_performance.png"
        )

        # 5. Category-wise analysis
        # if "per_category" in rf_results:
        # self._create_category_analysis(rf_results["per_category"], output_dir / "category_performance.png")

        # 6. Decision tree visualization
        # if tree_results:
        # self._create_tree_visualization(tree_results, output_dir / "decision_trees.png")

        # 7. Error analysis plots
        self._create_error_analysis_plots(rf_results, output_dir)

        print(f"Visualizations saved to {output_dir}")

    def _create_feature_importance_plot(self, rf_results: Dict, output_path: str):
        """Create detailed feature importance comparison plot."""
        plt.figure(figsize=(15, 10))

        # Collect feature importance from classification models only
        importance_data = {}
        model_names = []

        for model_name, results in rf_results.items():
            if isinstance(results, dict) and "feature_importance" in results:
                model_names.append(model_name)
                top_features = results["feature_importance"].head(15)

                for _, row in top_features.iterrows():
                    feature = row["feature"]
                    importance = row["importance"]

                    if feature not in importance_data:
                        importance_data[feature] = {}
                    importance_data[feature][model_name] = importance

        if not importance_data:
            plt.text(
                0.5,
                0.5,
                "No feature importance data available",
                ha="center",
                va="center",
                fontsize=16,
            )
            plt.title("Feature Importance Analysis")
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return

        # Create DataFrame and plot
        importance_df = pd.DataFrame(importance_data).T.fillna(0)

        # Plot top 20 features
        top_features = importance_df.sum(axis=1).nlargest(20).index
        plot_data = importance_df.loc[top_features]

        ax = plot_data.plot(kind="barh", figsize=(12, 10), width=0.8)
        plt.title("Feature Importance Across Models", fontsize=16, fontweight="bold")
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _create_tsne_plot(self, dimred_results: Dict, output_path: str):
        """Create t-SNE visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        tsne_data = dimred_results["tsne"]["embedding"]
        metadata = dimred_results["metadata"]

        # Plot 1: Colored by outcome
        outcome_colors = {"TP": "green", "FP": "red", "FN": "blue"}
        colors = [
            outcome_colors.get(outcome, "gray") for outcome in metadata["outcomes"]
        ]

        axes[0].scatter(tsne_data[:, 0], tsne_data[:, 1], c=colors, alpha=0.6, s=30)
        axes[0].set_title("t-SNE: Colored by Outcome", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("t-SNE Dimension 1")
        axes[0].set_ylabel("t-SNE Dimension 2")

        # Create legend
        for outcome, color in outcome_colors.items():
            axes[0].scatter([], [], c=color, label=outcome, s=50)
        axes[0].legend()

        # Plot 2: Colored by distance
        scatter = axes[1].scatter(
            tsne_data[:, 0],
            tsne_data[:, 1],
            c=metadata["distances"],
            cmap="viridis",
            alpha=0.6,
            s=30,
        )
        axes[1].set_title(
            "t-SNE: Colored by Distance to Ego", fontsize=14, fontweight="bold"
        )
        axes[1].set_xlabel("t-SNE Dimension 1")
        axes[1].set_ylabel("t-SNE Dimension 2")
        plt.colorbar(scatter, ax=axes[1], label="Distance (m)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _create_umap_plot(self, dimred_results: Dict, output_path: str):
        """Create UMAP visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        umap_data = dimred_results["umap"]["embedding"]
        metadata = dimred_results["metadata"]

        # Plot 1: Colored by outcome
        outcome_colors = {"TP": "green", "FP": "red", "FN": "blue"}
        colors = [
            outcome_colors.get(outcome, "gray") for outcome in metadata["outcomes"]
        ]

        axes[0].scatter(umap_data[:, 0], umap_data[:, 1], c=colors, alpha=0.6, s=30)
        axes[0].set_title("UMAP: Colored by Outcome", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("UMAP Dimension 1")
        axes[0].set_ylabel("UMAP Dimension 2")

        # Create legend
        for outcome, color in outcome_colors.items():
            axes[0].scatter([], [], c=color, label=outcome, s=50)
        axes[0].legend()

        # Plot 2: Colored by category
        categories = metadata["categories"]
        unique_cats = list(set(categories))
        cat_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cats)))
        cat_color_map = dict(zip(unique_cats, cat_colors))
        colors = [cat_color_map[cat] for cat in categories]

        axes[1].scatter(umap_data[:, 0], umap_data[:, 1], c=colors, alpha=0.6, s=30)
        axes[1].set_title("UMAP: Colored by Category", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("UMAP Dimension 1")
        axes[1].set_ylabel("UMAP Dimension 2")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _create_clustering_plot(self, clustering_results: Dict, output_path: str):
        """Create clustering analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Cluster sizes
        if "cluster_analysis" in clustering_results:
            analysis = clustering_results["cluster_analysis"]

            cluster_ids = []
            cluster_sizes = []
            mean_ates = []

            for cluster_name, stats in analysis.items():
                if cluster_name.startswith("cluster_"):
                    cluster_ids.append(cluster_name.replace("cluster_", ""))
                    cluster_sizes.append(stats["size"])
                    mean_ates.append(stats.get("mean_ate", 0))

            if cluster_ids:
                axes[0, 0].bar(cluster_ids, cluster_sizes, alpha=0.7, color="skyblue")
                axes[0, 0].set_title("Cluster Sizes (DBSCAN)", fontweight="bold")
                axes[0, 0].set_xlabel("Cluster ID")
                axes[0, 0].set_ylabel("Number of Samples")

                # Plot 2: Mean ATE per cluster
                valid_ates = [
                    (cid, ate)
                    for cid, ate in zip(cluster_ids, mean_ates)
                    if not np.isnan(ate)
                ]
                if valid_ates:
                    cids, ates = zip(*valid_ates)
                    axes[0, 1].bar(cids, ates, alpha=0.7, color="lightcoral")
                    axes[0, 1].set_title("Mean ATE per Cluster", fontweight="bold")
                    axes[0, 1].set_xlabel("Cluster ID")
                    axes[0, 1].set_ylabel("Mean ATE (m)")

        # Plot 3: Clustering metrics
        if "dbscan" in clustering_results and "gmm" in clustering_results:
            dbscan_info = clustering_results["dbscan"]
            gmm_info = clustering_results["gmm"]

            methods = ["DBSCAN", "GMM"]
            n_clusters = [dbscan_info["n_clusters"], gmm_info["n_clusters"]]

            axes[1, 0].bar(methods, n_clusters, alpha=0.7, color=["orange", "green"])
            axes[1, 0].set_title("Number of Clusters Found", fontweight="bold")
            axes[1, 0].set_ylabel("Number of Clusters")

            # Silhouette scores
            sil_scores = []
            for method_results in [dbscan_info, gmm_info]:
                score = method_results.get("silhouette_score", np.nan)
                sil_scores.append(score if not np.isnan(score) else 0)

            axes[1, 1].bar(methods, sil_scores, alpha=0.7, color=["orange", "green"])
            axes[1, 1].set_title("Silhouette Scores", fontweight="bold")
            axes[1, 1].set_ylabel("Silhouette Score")
            axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _create_performance_comparison(self, rf_results: Dict, output_path: str):
        """Create model performance comparison."""
        plt.figure(figsize=(12, 8))

        models = []
        scores = []
        score_types = []

        for model_name, results in rf_results.items():
            if isinstance(results, dict):
                if "test_accuracy" in results:
                    models.append(model_name.replace("_", " ").title())
                    scores.append(results["test_accuracy"])
                    score_types.append("Accuracy")
                elif "test_r2" in results:
                    models.append(model_name.replace("_", " ").title())
                    scores.append(max(0, results["test_r2"]))
                    score_types.append("R² Score")

        if models:
            colors = [
                "skyblue" if st == "Accuracy" else "lightgreen" for st in score_types
            ]
            bars = plt.bar(models, scores, alpha=0.8, color=colors)

            plt.title("Model Performance Comparison", fontsize=16, fontweight="bold")
            plt.ylabel("Score", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.ylim(0, 1.1)

            # Add value labels on bars
            for bar, score, score_type in zip(bars, scores, score_types):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{score:.3f}\n({score_type})",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="skyblue", label="Accuracy"),
                Patch(facecolor="lightgreen", label="R² Score"),
            ]
            plt.legend(handles=legend_elements)

            plt.grid(axis="y", alpha=0.3)
        else:
            plt.text(
                0.5,
                0.5,
                "No performance data available",
                ha="center",
                va="center",
                fontsize=16,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _create_error_analysis_plots(self, rf_results: Dict, output_dir: Path):
        """Create error analysis plots."""
        # Failure mode analysis
        if "failure_modes" in rf_results:
            failure_modes = rf_results["failure_modes"]

            plt.figure(figsize=(14, 6))

            # Plot missed categories if available
            if "missed_categories" in failure_modes:
                plt.subplot(1, 2, 1)
                missed_cats = failure_modes["missed_categories"]
                if isinstance(missed_cats, dict) and missed_cats:
                    categories = list(missed_cats.keys())[:10]  # Top 10
                    counts = [missed_cats[cat] for cat in categories]

                    plt.barh(categories, counts, alpha=0.7, color="lightcoral")
                    plt.title("Most Frequently Missed Categories", fontweight="bold")
                    plt.xlabel("Number of Missed Objects")

            # Plot FP vs TP feature differences if available
            if "fp_vs_tp_features" in failure_modes:
                plt.subplot(1, 2, 2)
                feature_diffs = failure_modes["fp_vs_tp_features"]

                if isinstance(feature_diffs, dict) and feature_diffs:
                    features = list(feature_diffs.keys())[:10]
                    differences = [
                        feature_diffs[feat].get("difference", 0) for feat in features
                    ]

                    colors = ["red" if diff > 0 else "blue" for diff in differences]
                    plt.barh(features, differences, alpha=0.7, color=colors)
                    plt.title("FP vs TP Feature Differences", fontweight="bold")
                    plt.xlabel("Difference (FP - TP)")
                    plt.axvline(x=0, color="black", linestyle="-", alpha=0.5)

            plt.tight_layout()
            plt.savefig(
                output_dir / "failure_analysis.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

    def generate_comprehensive_report(
        self,
        rf_results: Dict,
        clustering_results: Dict,
        tree_results: Dict,
        features_df: pd.DataFrame,
    ) -> str:
        """Generate comprehensive feature importance analysis report."""

        report = "# Random Forest Feature Importance Analysis\n\n"

        # Executive Summary
        report += "## Executive Summary\n\n"

        total_samples = len(features_df)
        tp_count = (features_df["outcome"] == "TP").sum()
        fp_count = (features_df["outcome"] == "FP").sum()
        fn_count = (features_df["outcome"] == "FN").sum()

        report += f"**Dataset**: {total_samples} samples ({tp_count} TP, {fp_count} FP, {fn_count} FN)\n"
        report += f"**Categories analyzed**: {features_df['dt_category'].nunique()}\n"
        report += f"**Models trained**: {len([k for k in rf_results.keys() if 'error' not in rf_results.get(k, {})])}\n\n"

        # Key Findings from Random Forest Models
        report += "## Key Findings: Feature Importance\n\n"

        # Global feature importance across models
        if rf_results:
            all_features = {}
            for model_name, results in rf_results.items():
                if "feature_importance" in results:
                    for _, row in results["feature_importance"].head(5).iterrows():
                        feature = row["feature"]
                        importance = row["importance"]
                        if feature not in all_features:
                            all_features[feature] = []
                        all_features[feature].append((model_name, importance))

            # Rank features by frequency of appearance in top 5
            feature_ranks = [
                (feat, len(appearances)) for feat, appearances in all_features.items()
            ]
            feature_ranks.sort(key=lambda x: x[1], reverse=True)

            report += "### Most Consistent Important Features:\n"
            for feature, count in feature_ranks[:8]:
                avg_importance = np.mean([imp for _, imp in all_features[feature]])
                report += f"- **{feature}**: appears in top 5 of {count} models (avg importance: {avg_importance:.3f})\n"
            report += "\n"

        # Model-specific results
        for model_name, results in rf_results.items():
            if "error" in results:
                continue

            report += f"### {model_name}\n"
            if "test_accuracy" in results:
                report += f"**Accuracy**: {results['test_accuracy']:.3f}\n"
            elif "test_r2" in results:
                report += f"**R² Score**: {results['test_r2']:.3f} (RMSE: {results['test_rmse']:.3f})\n"

            report += f"**Samples**: {results.get('n_samples')}\n"
            report += "**Top Features**:\n"

            if "feature_importance" in results:
                for _, row in results["feature_importance"].head(5).iterrows():
                    report += f"  - {row['feature']}: {row['importance']:.3f}\n"
            else:
                print(f"feature_importance not in results: {results.keys()}")
            report += "\n"

        # Interpretable Rules from Decision Trees
        if tree_results:
            report += "## Interpretable Rules (Decision Trees)\n\n"

            for tree_name, results in tree_results.items():
                report += f"### {tree_name.replace('_', ' ').title()}\n"
                report += f"**Accuracy**: {results['accuracy']:.3f}\n\n"

                # Extract simple rules from tree text
                rules_text = results["rules"]
                rules_lines = rules_text.split("\n")

                # Find meaningful rules (simplified extraction)
                meaningful_rules = []
                for line in rules_lines:
                    if "|---" in line and ("class" in line or "value" in line):
                        # This is a simplified rule extraction
                        clean_line = line.replace("|---", "").strip()
                        if len(clean_line) > 10:  # Skip very short lines
                            meaningful_rules.append(clean_line)

                if meaningful_rules:
                    report += "**Key Rules**:\n"
                    for rule in meaningful_rules[:5]:  # Top 5 rules
                        report += f"- {rule}\n"
                else:
                    report += "*See full tree visualization for detailed rules*\n"
                report += "\n"

        # Clustering Analysis
        if clustering_results and "cluster_analysis" in clustering_results:
            report += "## Failure Mode Clustering\n\n"

            if "dbscan" in clustering_results:
                dbscan_info = clustering_results["dbscan"]
                report += (
                    f"**DBSCAN Results**: {dbscan_info['n_clusters']} clusters found\n"
                )
                report += f"**Noise samples**: {dbscan_info['n_noise']}\n"
                if not np.isnan(dbscan_info["silhouette_score"]):
                    report += (
                        f"**Silhouette Score**: {dbscan_info['silhouette_score']:.3f}\n"
                    )
                report += "\n"

            analysis = clustering_results["cluster_analysis"]
            report += "### Discovered Failure Modes:\n"

            for cluster_name, stats in analysis.items():
                if cluster_name.startswith("cluster_"):
                    cluster_id = cluster_name.replace("cluster_", "")
                    report += f"**Cluster {cluster_id}** ({stats['size']} samples):\n"
                    report += f"  - Dominant category: {stats['dominant_category']}\n"
                    if not np.isnan(stats["mean_ate"]):
                        report += f"  - Mean ATE: {stats['mean_ate']:.3f}m\n"
                    if not np.isnan(stats["mean_distance"]):
                        report += f"  - Mean distance: {stats['mean_distance']:.1f}m\n"
                    if not np.isnan(stats["mean_occlusion"]):
                        report += f"  - Mean occlusion: {stats['mean_occlusion']:.3f}\n"

                    # Outcome distribution
                    if stats["outcome_distribution"]:
                        outcomes = ", ".join(
                            [
                                f"{k}: {v}"
                                for k, v in stats["outcome_distribution"].items()
                            ]
                        )
                        report += f"  - Outcomes: {outcomes}\n"
                    report += "\n"

        # Actionable Recommendations
        report += "## Actionable Recommendations\n\n"

        # Based on top features
        if rf_results:
            # Find most important features across all models
            feature_votes = defaultdict(int)
            for model_name, results in rf_results.items():
                if "feature_importance" in results:
                    for _, row in results["feature_importance"].head(3).iterrows():
                        feature_votes[row["feature"]] += 1

            top_features = sorted(
                feature_votes.items(), key=lambda x: x[1], reverse=True
            )[:5]

            report += "### Based on Feature Importance:\n"
            for feature, votes in top_features:
                if "distance" in feature.lower():
                    report += f"🎯 **{feature}** is critical → Consider distance-adaptive thresholds or training data balancing\n"
                elif "occlusion" in feature.lower():
                    report += f"🎯 **{feature}** drives failures → Improve occlusion handling or multi-view fusion\n"
                elif "score" in feature.lower():
                    report += f"🎯 **{feature}** is predictive → Confidence calibration may help\n"
                elif "nearby" in feature.lower():
                    report += f"🎯 **{feature}** affects performance → Context modeling improvements needed\n"
                else:
                    report += f"🎯 **{feature}** is important → Investigate {feature.replace('_', ' ')} effects\n"
            report += "\n"

        # Performance-based recommendations
        if "detection_vs_miss" in rf_results:
            detection_acc = rf_results["detection_vs_miss"].get("test_accuracy", 0)
            if detection_acc < 0.8:
                report += (
                    "⚠️ **Low detection accuracy** → Focus on reducing false negatives\n"
                )

        if "tp_vs_fp" in rf_results:
            tp_fp_acc = rf_results["tp_vs_fp"].get("test_accuracy", 0)
            if tp_fp_acc < 0.75:
                report += "⚠️ **Poor TP/FP separation** → Improve confidence score calibration\n"

        # Category-specific recommendations
        if "per_category" in rf_results:
            worst_categories = []
            for category, results in rf_results["per_category"].items():
                if "test_accuracy" in results and results["test_accuracy"] < 0.7:
                    worst_categories.append((category, results["test_accuracy"]))

            if worst_categories:
                worst_categories.sort(key=lambda x: x[1])
                report += f"⚠️ **Challenging categories**: {', '.join([c for c, _ in worst_categories[:3]])} need targeted improvements\n"

        report += "\n### Next Steps:\n"
        report += "1. **Data Collection**: Gather more samples for challenging scenarios identified by clustering\n"
        report += "2. **Feature Engineering**: Focus on the top predictive features for model improvements\n"
        report += "3. **Model Architecture**: Consider specialized handling for distance/occlusion effects\n"
        report += (
            "4. **Evaluation**: Use decision tree rules as test cases for validation\n"
        )

        return report


def train_interpretable_tree(features_df: pd.DataFrame, target_col='gt_best_iou', 
                           max_depth=10, min_samples_leaf=10):
    """
    Train an interpretable decision tree and extract decision rules
    """
    
    ignore_cols = ['object_id', 'proto_id', 'frame_idx', 'timestamp_ns']
    
    feature_cols = []
    for col in features_df.columns:
        if col != target_col and col not in ignore_cols and pd.api.types.is_numeric_dtype(features_df[col]):
            # Check for sufficient variance
            if features_df[col].var() > 1e-10:
                feature_cols.append(col)

    print(f"Selected {len(feature_cols)} features with sufficient variance")

    X_processed = features_df[feature_cols].copy()
    X_processed = X_processed.replace([np.inf, -np.inf], np.nan)

    for col in X_processed.columns:
        if X_processed[col].isna().sum() > 0:
            X_processed[col] = X_processed[col].fillna(X_processed[col].mean())

    # For decision trees, scaling is not necessary but can be done for consistency
    # Uncomment if you want to scale:
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_processed)
    
    # Using unscaled features for better interpretability
    X_scaled = X_processed.values
    scaler = None

    y = features_df[target_col].copy()
    if y.isna().sum() > 0:
        print(f"Warning: {y.isna().sum()} missing values in target. Dropping these rows.")
        mask = ~y.isna()
        X_scaled = X_scaled[mask]
        y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Decision Tree with interpretability constraints
    dt_reg = DecisionTreeRegressor(
        max_depth=max_depth,           # Limit depth for interpretability
        min_samples_leaf=min_samples_leaf,  # Avoid overfitting
        min_samples_split=20,          # Require meaningful splits
        max_features=None,             # Use all features for full interpretability
        random_state=42
    )

    dt_reg.fit(X_train, y_train)
    y_pred = dt_reg.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Decision Tree R² score: {r2:.4f}")
    print(f"Decision Tree RMSE: {rmse:.4f}")
    print(f"Tree depth: {dt_reg.get_depth()}")
    print(f"Number of leaves: {dt_reg.get_n_leaves()}")
    
    return {
        'model': dt_reg,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'test_r2': r2,
        'test_rmse': rmse,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        "feature_importance": pd.DataFrame(
            {"feature": feature_cols, "importance": dt_reg.feature_importances_}
        ).sort_values("importance", ascending=False),
    }

def extract_decision_rules(model_result):
    """
    Extract human-readable decision rules from the trained tree
    """
    dt_model = model_result['model']
    feature_cols = model_result['feature_cols']
    
    # Method 1: Text representation of the tree
    tree_rules = export_text(dt_model, feature_names=feature_cols, 
                            max_depth=100, spacing=2)
    print("="*50)
    print("DECISION TREE RULES:")
    print("="*50)
    print(tree_rules)
    
    return tree_rules

def add_random_forest_analysis_av2(
    evaluator: StandaloneLongTailEvaluator,
    av2_results: Dict,
    ego_poses: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Add Random Forest feature importance analysis using AV2 evaluation results.

    Args:
        evaluator: The main evaluator instance
        av2_results: Results from run_av2_evaluation containing eval_dts, eval_gts, cfg
        ego_poses: Optional ego pose data for speed calculation

    Returns:
        Dictionary of Random Forest analysis results
    """
    print("Starting AV2-integrated Random Forest analysis...")

    analyzer = RandomForestFeatureAnalyzer()
    eval_dts = av2_results["eval_dts"]
    eval_gts = av2_results["eval_gts"]
    cfg = av2_results["cfg"]

    output_dir = Path(evaluator.output_dir)
    features_path = output_dir / "av2_features.feather"

    if features_path.exists():
        print(f"Reading features_df from {features_path=}")

        features_df = pd.read_feather(features_path)
    else:
        print(f"{features_path=} does not exist yet, creating...")

        # Extract comprehensive features
        features_df = analyzer.extract_av2_features(eval_dts, eval_gts, cfg, ego_poses)

        features_df.to_feather(features_path)

    if len(features_df) < 50:
        print("Insufficient data for Random Forest analysis")
        return {"error": "Insufficient data"}

    # rename cols
    original_cols = features_df.columns
    for col in original_cols:
        if col.startswith("det_"):
            dt_col = "dt_" + col[4:]  # Replace 'det_' with 'dt_'

            if dt_col in features_df.columns:
                # Merge into dt_col: fillna with det_col
                features_df[dt_col] = features_df[dt_col].combine_first(
                    features_df[col]
                )
                features_df.drop(columns=col, inplace=True)
            else:
                # Rename det_col to dt_col
                features_df.rename(columns={col: dt_col}, inplace=True)

    print("original_cols", original_cols)
    print(f"{features_df.columns=}")

    dt_result = train_interpretable_tree(features_df, max_depth=5)
    pprint(dt_result)

    extract_decision_rules(dt_result)

    dt_model = dt_result['model']
    feature_cols = dt_result['feature_cols']
    
    plt.figure(figsize=(20, 12))
    plot_tree(dt_model, 
              feature_names=feature_cols,
              filled=True,
              rounded=True,
              fontsize=10,
              max_depth=4)  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig(evaluator.output_dir / "decision_tree_visualisation.png")

    exit()

    # # Train Random Forest models
    rf_results = analyzer.train_random_forest_models(features_df)

    # # Perform clustering analysis
    clustering_results = analyzer.perform_error_clustering(features_df)

    # Create dimensionality reduction visualizations
    dimred_results = analyzer.create_dimensionality_reduction_viz(
        features_df, clustering_results
    )
    print("dimred_results", dimred_results.keys())

    # Train interpretable decision trees
    tree_results = analyzer.train_interpretable_trees(features_df)

    # pprint(rf_results)
    # pprint(clustering_results)
    # pprint(dimred_results)
    # pprint(tree_results)

    # Generate outputs
    if rf_results:
        # Create comprehensive visualizations
        output_path = evaluator.output_dir / "random_forest_analysis.png"
        analyzer.create_comprehensive_visualizations(
            rf_results,
            clustering_results,
            dimred_results,
            tree_results,
            str(output_path),
        )

        # Generate report
        report = analyzer.generate_comprehensive_report(
            rf_results, clustering_results, tree_results, features_df
        )
        report_path = evaluator.output_dir / "random_forest_analysis.md"
        with open(report_path, "w") as f:
            f.write(report)

        print(
            f"Random Forest analysis complete! Results saved to {evaluator.output_dir}"
        )

    return {
        "rf_results": rf_results,
        "clustering_results": clustering_results,
        "dimensionality_reduction": dimred_results,
        "decision_trees": tree_results,
        "features_df": features_df,
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")  # list of row dicts
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # convert numpy scalars to native Python types
        return super().default(obj)


# Example usage with main pipeline
if __name__ == "__main__":
    # This would be integrated into the main analysis

    config = EvaluationConfig(
        # predictions_path="../../lion/output/lion_models/lion_mamba_1f_1x_argo_128dim_sparse_v2/default/eval/epoch_2/val/default/processed_results.feather",
        predictions_path="/home/uqdetche/lidar_longtail_mining/lion/output/dataset_configs/cpd/waymo_unsupervised_cproto/default/0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2/0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2_outline_C_PROTO_argo.feather",
        ground_truth_path="../../lion/data/argo2/val_anno.feather",
        dataset_dir="../../lion/data/argo2/sensor/val",
        output_dir="./longtail_feature_analysis",
    )

    evaluator = StandaloneLongTailEvaluator(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    av2_results_path = output_dir / "av2_results.pkl"

    random_forest_results_path = output_dir / "random_forest_results.json"

    if av2_results_path.exists():
        with open(av2_results_path, "rb") as file:
            av2_results = pickle.load(file)
    else:
        # Run main analysis
        av2_results = evaluator.run_av2_eval()

        with open(av2_results_path, "wb") as file:
            pickle.dump(av2_results, file)

    cfg: DetectionCfg = av2_results["cfg"]
    cfg = replace(cfg, dataset_dir=Path(config.dataset_dir))

    av2_results["cfg"] = cfg

    # Add random forest analysis
    results = add_random_forest_analysis_av2(evaluator, av2_results)

    with open(random_forest_results_path, "w") as f:
        json.dump(results, f, cls=NumpyEncoder)

    print("analysis complete!")
