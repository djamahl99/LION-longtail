import copy
import cProfile
import io
import os
import pickle as pkl
import pstats
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_plotting_utils
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import trimesh
from av2.datasets.sensor.constants import RingCameras, StereoCameras
from av2.datasets.sensor.sensor_dataloader import (
    SensorDataloader,
    SynchronizedSensorData,
)
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.map.lane_segment import LaneSegment
from av2.map.map_api import ArgoverseStaticMap, GroundHeightLayer
from av2.rendering.color import ColorFormats, create_range_map
from av2.rendering.rasterize import draw_points_xy_in_img
from av2.structures.sweep import Sweep
from av2.structures.timestamped_image import TimestampedImage
from av2.utils.io import read_city_SE3_ego, read_ego_SE3_sensor, read_feather
from kornia.geometry.linalg import transform_points
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from tqdm import tqdm, trange

from lion.unsupervised_core.box_utils import argo2_box_to_lidar
from lion.unsupervised_core.convex_hull_tracker.alpha_shape_tracker import (
    AlphaShapeTracker,
)
from lion.unsupervised_core.convex_hull_tracker.alpha_shape_utils import AlphaShapeUtils
from lion.unsupervised_core.convex_hull_tracker.convex_hull_kalman_tracker import (
    ConvexHullKalmanTracker,
)
from lion.unsupervised_core.convex_hull_tracker.convex_hull_object import (
    ConvexHullObject,
)
from lion.unsupervised_core.convex_hull_tracker.convex_hull_track import (
    ConvexHullTrackState,
)
from lion.unsupervised_core.convex_hull_tracker.convex_hull_utils import (
    box_iou_3d,
    draw_ellipse_outline,
    point_in_triangle_vectorized,
    render_triangle_with_fillpoly_barycentric,
    save_depth_buffer_colorized,
    voxel_sampling_fast,
)
from lion.unsupervised_core.rotate_iou_cpu_eval import rotate_iou_cpu_eval
from lion.unsupervised_core.tracker.box_op import register_bbs

from .box_utils import *
from .file_utils import load_predictions_parallel
from .outline_utils import (
    OutlineFitter,
    TrackSmooth,
    points_rigid_transform,
)
from .owlvit_frustum_tracker import OWLViTFrustumTracker
from .trajectory_optimizer import (
    GlobalTrajectoryOptimizer,
    optimize_with_gtsam_timed,
    simple_pairwise_icp_refinement,
)

PROFILING = True

def iou_multipoint(hull1, hull2) -> float:
    """
    Compute IoU using cached geometric objects - much faster than original.

    Args:
        shape1, shape2: Alpha shape dictionaries with cached geometry

    Returns:
        IoU value between 0 and 1
    """
    # Handle degenerate cases (points, lines)
    if hull1.area == 0 or hull2.area == 0:
        return 0.0

    # Compute intersection and union
    intersection = hull1.intersection(hull2).area
    union = hull1.union(hull2).area

    # Return IoU
    return intersection / union if union > 0 else 0.0

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


class OWLViTAlphaShapeMFCF:
    """
    Hybrid vision-LiDAR alpha shape tracking with temporal consistency.

    Combines:
    - OWLViT 2D detections from all ring cameras
    - LiDAR clustering (traditional + vision-guided)
    - Alpha shapes in 3D and UV space
    - Temporal tracking with AlphaShapeTracker
    """

    def __init__(
        self,
        log_id: str,
        root_path: str,
        owlvit_predictions_dir: Path,
        config,
        debug: bool = False,
        split: str = "val",
    ):
        self.log_id = log_id
        self.root_path = root_path
        self.owlvit_predictions_dir = owlvit_predictions_dir
        self.dataset_cfg = config
        self.debug = debug
        self.split = split

        self.min_box_iou = 0.3
        self.lidar_connected_components_eps = 0.5
        self.min_component_points = 15

        self.nms_iou_threshold = 0.7
        self.nms_query_distance = 5.0 # metres
        self.nms_semantic_threshold = 0.9

        dataset_path = Path(
            "/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/"
        )
        self.dataloader = ModifiedSensorDataloader(dataset_path, with_annotations=False)

        # Ring cameras
        self.ring_cameras = [x.value for x in list(RingCameras)]

        # Initialize outline estimator (reuse from original MFCF)
        self.outline_estimator = OutlineFitter(
            sensor_height=config.GeneratorConfig.sensor_height,
            ground_min_threshold=config.GeneratorConfig.ground_min_threshold,
            ground_min_distance=config.GeneratorConfig.ground_min_distance,
            cluster_dis=config.GeneratorConfig.cluster_dis,
            cluster_min_points=config.GeneratorConfig.cluster_min_points,
            discard_max_height=config.GeneratorConfig.discard_max_height,
            min_box_volume=config.GeneratorConfig.min_box_volume,
            min_box_height=config.GeneratorConfig.min_box_height,
            max_box_volume=config.GeneratorConfig.max_box_volume,
            max_box_len=config.GeneratorConfig.max_box_len,
        )

    def _load_owlvit_predictions(
        self, target_timestamp_ns: int
    ) -> Dict[str, List[Dict]]:
        """
        Load all OWLViT predictions within temporal window for all cameras.

        Returns:
            Dict[camera_name, List[prediction_dicts]] for all cameras and timestamps
        """
        camera_predictions = {cam: [] for cam in self.ring_cameras}

        # Get all available prediction files
        pred_dir = self.owlvit_predictions_dir / self.log_id
        if not pred_dir.exists():
            print(f"Warning: OWLViT predictions not found for {self.log_id}")
            return camera_predictions

        pred_files = pred_dir.glob("*.pkl")

        camera_timestamps = {cam_name: [] for cam_name in self.ring_cameras}

        # Find all prediction files within temporal window
        for pred_file in pred_files:
            try:
                # Parse timestamp and camera from filename: {timestamp}_{camera}.pkl
                parts = pred_file.stem.split("_")
                if len(parts) < 2:
                    continue

                timestamp_ns = int(parts[0])
                camera_name = "_".join(parts[1:])  # Handle multi-part camera names

                camera_timestamps[camera_name].append(timestamp_ns)

            except (ValueError, IndexError) as e:
                if self.debug:
                    print(f"Skipping invalid prediction file {pred_file}: {e}")
                continue

        # Sort predictions by timestamp for each camera
        for camera_name in self.ring_cameras:
            timestamps_all = np.array(camera_timestamps[camera_name], dtype=int)
            timestamps_diffs = np.abs(timestamps_all - target_timestamp_ns)

            cam_timestamp_ns = timestamps_all[np.argmin(timestamps_diffs)]

            print(
                f"lowest timestamp diff for {camera_name=} : {timestamps_all[np.argmin(timestamps_diffs)]}"
            )

            pred_file = pred_dir / f"{cam_timestamp_ns}_{camera_name}.pkl"

            with open(pred_file, "rb") as f:
                prediction = pkl.load(f)
                prediction["cam_timestamp_ns"] = cam_timestamp_ns
                camera_predictions[camera_name].append(prediction)

        return camera_predictions

    def _load_camera_name_timestamps(self):
        split = self.split
        log_id = self.log_id

        # Ggather all cameras and timestamps
        log_dir = self.dataloader.dataset_dir / split / log_id
        sensor_dir = log_dir / "sensors"
        cameras_dir = sensor_dir / "cameras"

        # Find the lidar timestamps
        lidar_folder = sensor_dir / "lidar"
        lidar_timestamps = np.array(
            [int(x.stem) for x in lidar_folder.rglob("*.feather")], dtype=int
        )

        lidar_timestamps = np.sort(lidar_timestamps)

        owlvit_pred_dir = self.owlvit_predictions_dir / log_id

        # TODO
        src_sensor_name = "lidar"
        synchronization_cache_path = (
            Path.home() / ".cache" / "av2" / "synchronization_cache.feather"
        )
        self.synchronization_cache = read_feather(synchronization_cache_path)
        # Finally, create a MultiIndex set the sync records index and sort it.
        self.synchronization_cache.set_index(
            keys=["split", "log_id", "sensor_name"], inplace=True
        )
        self.synchronization_cache.sort_index(inplace=True)

        assert owlvit_pred_dir.exists()

        camera_name_timestamps: List[Tuple] = []

        pred_exists = 0
        pred_doesnt_exist = 0

        for sweep_timestamp_ns in lidar_timestamps:
            for camera_name in self.ring_cameras:
                camera_dir = cameras_dir / camera_name

                src_timedelta_ns = pd.Timedelta(sweep_timestamp_ns)
                src_to_target_records = self.synchronization_cache.loc[
                    (split, log_id, src_sensor_name)
                ].set_index(src_sensor_name)
                index = src_to_target_records.index
                if src_timedelta_ns not in index:
                    # This timestamp does not correspond to any lidar sweep.
                    continue

                # Grab the synchronization record.
                target_timestamp_ns = src_to_target_records.loc[
                    src_timedelta_ns, camera_name
                ]
                if pd.isna(target_timestamp_ns):
                    # No match was found within tolerance.
                    continue
                cam_timestamp_ns = target_timestamp_ns.asm8.item()
                cam_timestamp_ns_str = str(target_timestamp_ns.asm8.item())

                owlvit_pred_path = (
                    owlvit_pred_dir / f"{cam_timestamp_ns_str}_{camera_name}.pkl"
                )
                if owlvit_pred_path.exists():
                    camera_name_timestamps.append(
                        (camera_name, cam_timestamp_ns, sweep_timestamp_ns)
                    )
                    pred_exists += 1
                else:
                    pred_doesnt_exist += 1
                    print(f"{owlvit_pred_path=} doesnt exist")

        return camera_name_timestamps

    def _load_owlvit_predictions_exhaustive(
        self, target_timestamp_ns: int, temporal_window_ns: int = 1e9
    ) -> Dict[str, List[Dict]]:
        """
        Load all OWLViT predictions within temporal window for all cameras.

        Returns:
            Dict[camera_name, List[prediction_dicts]] for all cameras and timestamps
        """
        camera_predictions = {cam: [] for cam in self.ring_cameras}

        # Get all available prediction files
        pred_dir = self.owlvit_predictions_dir / self.log_id
        if not pred_dir.exists():
            print(f"Warning: OWLViT predictions not found for {self.log_id}")
            return camera_predictions

        pred_files = pred_dir.glob("*.pkl")

        timestamps = []

        camera_timestamps = {cam_name: [] for cam_name in self.ring_cameras}

        # Find all prediction files within temporal window
        for pred_file in pred_files:
            try:
                # Parse timestamp and camera from filename: {timestamp}_{camera}.pkl
                parts = pred_file.stem.split("_")
                if len(parts) < 2:
                    continue

                timestamp_ns = int(parts[0])
                camera_name = "_".join(parts[1:])  # Handle multi-part camera names

                timestamps.append(timestamp_ns)

                # Check if within temporal window and is ring camera
                if (
                    camera_name in self.ring_cameras
                    and abs(timestamp_ns - target_timestamp_ns) <= temporal_window_ns
                ):

                    with open(pred_file, "rb") as f:
                        prediction = pkl.load(f)
                        prediction["timestamp_ns"] = timestamp_ns
                        camera_predictions[camera_name].append(prediction)
                # elif abs(timestamp_ns - target_timestamp_ns) > temporal_window_ns:
                # print(f"skipped {pred_file.stem} as {abs(timestamp_ns - target_timestamp_ns)} > {temporal_window_ns=}")

                # else:
                # print(f"skipped {pred_file.stem} as not in ring_cameras?")

            except (ValueError, IndexError) as e:
                if self.debug:
                    print(f"Skipping invalid prediction file {pred_file}: {e}")
                continue

        print("unique timestamps", len(set(timestamps)))

        timestamps = np.array(
            [x / 1e9 for x in timestamps], dtype=float
        )  # in seconds now
        print(
            "timestamps range (seconds)",
            timestamps.max() - timestamps.min(),
            timestamps.shape,
        )

        # Sort predictions by timestamp for each camera
        for cam in self.ring_cameras:
            camera_predictions[cam].sort(key=lambda x: x["timestamp_ns"])

        return camera_predictions

    def _load_temporal_lidar_window(
        self, frame_idx: int, infos: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and aggregate LiDAR points from temporal window (same as AlphaShapeMFCF)."""
        pose_i = np.linalg.inv(infos[frame_idx]["pose"])
        all_points = []
        all_H = []
        cur_points = None

        win_size = self.dataset_cfg.GeneratorConfig.frame_num
        inte = self.dataset_cfg.GeneratorConfig.frame_interval
        thresh = self.dataset_cfg.GeneratorConfig.ppscore_thresh

        for j in range(
            max(frame_idx - win_size, 0), min(frame_idx + win_size, len(infos)), inte
        ):
            info_path = str(j).zfill(4) + ".npy"
            lidar_path = os.path.join(self.root_path, self.log_id, info_path)
            if not os.path.exists(lidar_path):
                continue

            pose_j = infos[j]["pose"]
            lidar_points = np.load(lidar_path)[:, 0:3]
            if j == frame_idx:
                cur_points = lidar_points

            lidar_points = points_rigid_transform(lidar_points, pose_j)
            H_path = os.path.join(self.root_path, self.log_id, "ppscore", info_path)
            if os.path.exists(H_path):
                H = np.load(H_path)
                all_H.append(H)
            else:
                all_H.append(np.ones(len(lidar_points)))  # Default to all points
            all_points.append(lidar_points)

        all_points = np.concatenate(all_points)
        all_points = points_rigid_transform(all_points, pose_i)
        all_H = np.concatenate(all_H)
        all_points = all_points[all_H > thresh]
        new_box_points = np.concatenate([all_points, cur_points])
        new_box_points = voxel_sampling_fast(new_box_points)

        return new_box_points, cur_points

    def _get_traditional_clusters(self, points: np.ndarray) -> List[np.ndarray]:
        """Get traditional LiDAR clusters (same as AlphaShapeMFCF)."""
        non_ground_points = self.outline_estimator.remove_ground(points)
        clusters, labels = self.outline_estimator.clustering(non_ground_points)

        # Filter clusters by size and volume constraints
        valid_clusters = []
        for cluster in clusters:
            if len(cluster) >= 3:
                cluster_dims = cluster.max(axis=0) - cluster.min(axis=0)
                cluster_vol = np.prod(cluster_dims)

                if (
                    not np.any(cluster_dims > 10)
                    and not np.any(
                        cluster_dims
                        < getattr(
                            self.dataset_cfg.GeneratorConfig, "min_box_volume", 0.1
                        )
                    )
                    and cluster_vol
                    >= getattr(self.dataset_cfg.GeneratorConfig, "min_box_volume", 0.1)
                    and cluster_vol
                    <= getattr(self.dataset_cfg.GeneratorConfig, "max_box_volume", 200)
                ):
                    valid_clusters.append(cluster)

        return valid_clusters

    def _get_box_frustum_corners_in_ego(
        self, box: np.ndarray, camera_model: PinholeCamera, near_clip=1.0, far_clip=25.0
    ):
        """
        Get the corners of a the frustum for a box in world coordinates.
        """
        x1, y1, x2, y2 = box
        box_corners = np.array(
            [[x1, y1], [x1, y2], [x2, y1], [x2, y2]], dtype=np.float32
        )
        ray_dirs_cam = camera_model.compute_pixel_ray_directions(box_corners)

        # Create near and far points along each ray in camera frame
        near_points_cam = ray_dirs_cam * near_clip
        far_points_cam = ray_dirs_cam * far_clip

        # Stack to get 8 corners in camera frame
        corners_cam = np.vstack([near_points_cam, far_points_cam])

        # Transform to ego frame
        corners_ego = camera_model.ego_SE3_cam.transform_point_cloud(corners_cam)

        return corners_ego

    def _track_owlvit_predictions(self, log_id: str, split: str = "val"):

        # Step 1 -> gather all cameras and timestamps
        log_dir = self.dataloader.dataset_dir / split / log_id
        sensor_dir = log_dir / "sensors"
        cameras_dir = sensor_dir / "cameras"

        # Find the lidar timestamps
        lidar_folder = sensor_dir / "lidar"
        lidar_timestamps = np.array(
            [int(x.stem) for x in lidar_folder.rglob("*.feather")], dtype=int
        )

        # get ego dict
        timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_dir)

        owlvit_pred_dir = self.owlvit_predictions_dir / log_id

        assert owlvit_pred_dir.exists()

        camera_name_timestamps: List[Tuple] = []

        pred_exists = 0
        pred_doesnt_exist = 0

        min_timestamp = float("inf")

        for camera_name in self.ring_cameras:
            camera_dir = cameras_dir / camera_name

            assert camera_dir.exists(), f"{camera_dir=} doesnt exist"

            images = camera_dir.rglob("*.jpg")
            timestamps = [int(x.stem) for x in images]

            for timestamp in timestamps:
                if timestamp < min_timestamp:
                    min_timestamp = timestamp

                owlvit_pred_path = owlvit_pred_dir / f"{timestamp}_{camera_name}.pkl"
                if owlvit_pred_path.exists():
                    camera_name_timestamps.append((camera_name, timestamp))
                    pred_exists += 1
                else:
                    pred_doesnt_exist += 1
                    print(f"{owlvit_pred_path=} doesnt exist")

        total = pred_exists + pred_doesnt_exist
        print(
            f"{pred_exists=} ({(100*pred_exists/total):.2f}%) {pred_doesnt_exist=} ({100*pred_doesnt_exist/total:.2f}%)"
        )

        # Step 2: Generate frustums

        # Load camera models for each
        camera_models = {
            cam_name: PinholeCamera.from_feather(log_dir=log_dir, cam_name=cam_name)
            for cam_name in self.ring_cameras
        }

        raster_ground_height_layer = GroundHeightLayer.from_file(log_dir / "map")

        # Step 3: add all frustums
        frustums = []
        for camera_name, cam_timestamp_ns in camera_name_timestamps:  # TODO remove :30
            owlvit_pred_path = owlvit_pred_dir / f"{cam_timestamp_ns}_{camera_name}.pkl"

            if self.debug:  # only do the first second for debugging
                if abs(cam_timestamp_ns - min_timestamp) > 1e9:
                    continue

            with open(owlvit_pred_path, "rb") as f:
                prediction = pkl.load(f)

            # find the closest lidar timestamp
            timestamp_diffs = np.abs(lidar_timestamps - cam_timestamp_ns)
            sweep_timestamp_ns = lidar_timestamps[np.argmin(timestamp_diffs)]

            lidar_feather_path = lidar_folder / f"{sweep_timestamp_ns}.feather"
            lidar = read_feather(lidar_feather_path)
            pcl_ego = lidar.loc[:, ["x", "y", "z"]].to_numpy().astype(float)

            # Load city SE3 ego transformations
            timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_dir)

            # ego transformation (ego -> city)
            city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam_timestamp_ns]
            city_SE3_ego_lidar_t = timestamp_city_SE3_ego_dict[sweep_timestamp_ns]

            pcl_city_1 = city_SE3_ego_lidar_t.transform_point_cloud(pcl_ego)
            # is_ground = avm.get_ground_points_boolean(pcl_city_1).astype(bool)

            is_ground = raster_ground_height_layer.get_ground_points_boolean(
                pcl_city_1
            ).astype(bool)

            print("is_ground", is_ground.shape, is_ground.sum())

            is_not_ground = ~is_ground

            lidar_xyz = pcl_ego[is_not_ground]

            camera_model = camera_models[camera_name]

            # project lidar to this camera
            (
                uv_points,
                points_cam,
                is_valid_points,
            ) = camera_model.project_ego_to_img_motion_compensated(
                lidar_xyz,
                city_SE3_ego_cam_t=city_SE3_ego_cam_t,
                city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
            )

            if not np.any(is_valid_points):  # shouldn't happen...
                continue

            valid_uv = uv_points[is_valid_points]
            valid_points_cam = points_cam[is_valid_points]
            valid_3d_points = lidar_xyz[is_valid_points]

            pred_boxes = prediction["pred_boxes"]
            image_class_embeds = prediction["image_class_embeds"]
            objectness_scores = prediction["objectness_scores"]

            for box, image_class_embed, objectness_score in zip(
                pred_boxes, image_class_embeds, objectness_scores
            ):
                x1, y1, x2, y2 = box

                # Find points within this box
                in_box_mask = (
                    (valid_uv[:, 0] >= x1)
                    & (valid_uv[:, 0] <= x2)
                    & (valid_uv[:, 1] >= y1)
                    & (valid_uv[:, 1] <= y2)
                )

                if np.sum(in_box_mask) < 3:  # Need at least 3 points
                    continue

                box_uv_points = valid_uv[in_box_mask]
                box_cam_points = valid_points_cam[in_box_mask]
                box_depths = box_cam_points[:, 2]
                box_3d_points = valid_3d_points[in_box_mask]

                # connected components
                component_labels, n_components = find_connected_components_lidar(
                    box_3d_points
                )
                assert (
                    component_labels.min() >= 0
                    and component_labels.max() < n_components
                ), f"{n_components=} {component_labels.min()} {component_labels.max()}"

                component_boxes = np.stack(
                    [
                        np.concatenate(
                            [
                                box_uv_points[(component_labels == i), :2].min(axis=0),
                                box_uv_points[(component_labels == i), :2].max(axis=0),
                            ]
                        )
                        for i in range(n_components)
                    ],
                    axis=0,
                )

                # print("component_boxes", component_boxes.shape)

                ious = box_iou(box.reshape(1, 4), component_boxes.reshape(-1, 4))

                # print("ious", ious.shape)
                ious = ious[0, :]

                best_component = np.argmax(ious)
                best_component_iou = ious[best_component]

                if ious[best_component] < 0.1:  # TODO: change
                    continue

                # revise with this component
                in_box_mask = component_labels == best_component

                if np.sum(in_box_mask) < 3:  # Need at least 3 points
                    continue

                # print(f"{best_component_iou=}")

                box_uv_points = box_uv_points[in_box_mask]
                box_cam_points = box_cam_points[in_box_mask]
                box_depths = box_cam_points[:, 2]
                box_3d_points = box_3d_points[in_box_mask]

                # use quantiles?
                min_depth = np.quantile(box_depths, 0.1)
                max_depth = np.quantile(box_depths, 0.9)

                # corners in ego
                corners_ego = self._get_box_frustum_corners_in_ego(
                    box,
                    camera_model,
                    near_clip=float(min_depth),
                    far_clip=float(max_depth),
                )

                # print('corners ego', corners_ego.shape, corners_ego.min(axis=0), corners_ego.max(axis=0))

                corners_city = city_SE3_ego_cam_t.transform_point_cloud(corners_ego)
                # print('corners_city', corners_city.shape, corners_city.min(axis=0), corners_city.max(axis=0))

                frustums.append(
                    {
                        "box": box,
                        "corners_city": corners_city,
                        "corners_ego": corners_ego,
                        "sweep_timestamp_ns": sweep_timestamp_ns,
                        "cam_timestamp_ns": cam_timestamp_ns,
                        "semantic_features": image_class_embed,
                        "objectness_score": objectness_score,
                        "pose": city_SE3_ego_cam_t.transform_matrix,
                    }
                )

        tracker = OWLViTFrustumTracker(
            log_id, self.dataset_cfg.GeneratorConfig, debug=self.debug
        )
        tracker.track_frustums(frustums)

        exit()

    def _get_instance_buffers(self, component_meshes: List[trimesh.Trimesh], camera_models: Dict[str, PinholeCamera], timestamp_city_SE3_ego_dict, sweep_camera_name_timestamps: List[Tuple], sweep_timestamp_ns: int):
        instance_buffers = {}

        all_centroids = np.stack([x.centroid for x in component_meshes], axis=0)
        all_dims = np.stack([x.vertices.max(axis=0) - x.vertices.min(axis=0) for x in component_meshes], axis=0)
        all_vols = np.prod(all_dims, axis=1)

        all_dists = np.linalg.norm(all_centroids, axis=1)

        cost = (all_dists / all_dists.max()) * (all_vols / all_vols.max())

        ordered = np.argsort(cost, axis=0)

        cmap = plt.get_cmap("tab20", len(component_meshes))

        for camera_name, cam_timestamp_ns in sweep_camera_name_timestamps:
            city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam_timestamp_ns]
            city_SE3_ego_lidar_t = timestamp_city_SE3_ego_dict[sweep_timestamp_ns]
            camera_model = camera_models[camera_name]

            height = camera_model.height_px
            width = camera_model.width_px

            depth_buffer = np.full((height, width), np.inf)  # Z-buffer for occlusion handling
            instance_buffer = np.full((height, width), -1)
            # rendered_image = np.zeros((height, width, 3), int)

            for idx in ordered:
                mesh = component_meshes[idx]

                (
                    uv_points,
                    points_cam,
                    is_valid_points,
                ) = camera_model.project_ego_to_img_motion_compensated(
                    mesh.vertices,
                    city_SE3_ego_cam_t=city_SE3_ego_cam_t,
                    city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
                )

                color = cmap(idx)
                color = tuple((np.array(color)[:3]*255).astype(int).tolist())

                uvs = uv_points[:, :2]
                depths = points_cam[:, 2]
                cam_mask = is_valid_points

                if not np.any(cam_mask):
                    continue

                # Render each triangular face
                for face in mesh.faces:
                    # Get the vertices of the triangle
                    tri_verts_2d = uvs[face]  # (3, 2): Triangle in 2D space
                    tri_depths = depths[face]  # (3,): Depth of the triangle vertices
                    tri_mask = cam_mask[face]

                    if np.any(tri_depths < 0) or not np.any(tri_mask):
                        continue

                    # Compute the average depth of the triangle
                    avg_depth = np.mean(tri_depths)

                    # # Create a mask for the triangle
                    # tri_mask = np.zeros((height, width), dtype=np.uint8)
                    # tri_verts_2d = tri_verts_2d.astype(int)
                    # cv2.fillConvexPoly(tri_mask, tri_verts_2d, 1)
                    
                    # # Find pixels where this triangle is visible (closer than current depth buffer)
                    # visible_pixels = (tri_mask > 0) & (avg_depth < depth_buffer)
                    
                    # Compute bounding box
                    min_x = max(0, int(np.floor(tri_verts_2d[:, 0].min())))
                    max_x = min(width - 1, int(np.ceil(tri_verts_2d[:, 0].max())))
                    min_y = max(0, int(np.floor(tri_verts_2d[:, 1].min())))
                    max_y = min(height - 1, int(np.ceil(tri_verts_2d[:, 1].max())))
                    
                    if min_x >= max_x or min_y >= max_y:
                        continue
                    
                    # Pre-compute for point-in-triangle test
                    v0, v1, v2 = tri_verts_2d
                    
                    # Use vectorized point-in-triangle test
                    y_coords, x_coords = np.mgrid[min_y:max_y+1, min_x:max_x+1]
                    points = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1).astype(np.float32)
                    
                    # Fast point-in-triangle test using cross products
                    inside_mask = point_in_triangle_vectorized(points, v0, v1, v2)
                    
                    if not np.any(inside_mask):
                        continue
                    
                    # Get valid pixel coordinates
                    valid_points = points[inside_mask].astype(int)
                    valid_y, valid_x = valid_points[:, 1], valid_points[:, 0]
                    
                    # Update depth buffer where triangle is closer
                    closer_mask = avg_depth < depth_buffer[valid_y, valid_x]
                    
                    if np.any(closer_mask):
                        update_y = valid_y[closer_mask]
                        update_x = valid_x[closer_mask]
                        
                        depth_buffer[update_y, update_x] = avg_depth
                        # rendered_image[update_y, update_x] = color
                        instance_buffer[update_y, update_x] = idx

            instance_buffers[camera_name] = instance_buffer
            # save_depth_buffer_colorized(depth_buffer, f"depth_buffer_{camera_name}.jpg")
            # cv2.imwrite(f"rendered_image_{camera_name}.jpg", rendered_image)

        # pr.disable()

        # s = io.StringIO()
        # sortby = "cumtime"
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # with open(
        #     "depth_buffer_analysis.txt", "w"
        # ) as f:
        #     f.write(s.getvalue())

        return instance_buffers

    def _get_vision_guided_clusters(
        self,
        points: np.ndarray,
        traditional_clusters,
        camera_name_timestamps: List[Tuple],
        timestamp_ns: int,
        camera_models: Dict[str, PinholeCamera],
        timestamp_city_SE3_ego_dict,
        tracker: ConvexHullKalmanTracker,
        lidar_tree: cKDTree,
        search_radius:float =100.0
    ) -> List[Dict]:
        """
        Get vision-guided clusters by projecting LiDAR to cameras and filtering by OWLViT boxes.

        Returns:
            List of cluster dicts with 3D points, camera info, UV alpha shapes, and semantic features
        """
        vision_clusters = []
        # ego transformation (ego -> city)
        city_SE3_ego_lidar_t = timestamp_city_SE3_ego_dict[timestamp_ns]

        # points_orig_shape = points.shape
        # points = self.outline_estimator.remove_ground(points.copy())
        # is_ground = raster_ground_height_layer.get_ground_points_boolean(points).astype(bool)
        # is_not_ground = ~is_ground

        # points = points[is_not_ground]
        # print(f"before ground removal: {points_orig_shape} after: {points.shape}")

        # pre run connected components?
        # lidar_connected_components_labels, lidar_n_components = fast_connected_components(points, eps=self.lidar_connected_components_eps)

        lidar_connected_components_infos = {}

        # turn them into meshes?
        lidar_meshes = []
        all_cmp_vertices = []
        # all_cmp_labels = []

        num_no_box = 0
        num_w_box = 0

        lidar_cmp_indices = set()
        track_cmp_indices = set()
        n_vertices_per_cmp = {}

        component_meshes = []
        component_boxes = []
        component_vols = []
        component_densities = []
        component_original_points = []

        cmp_lbl = 0
        for eps in [0.5]:
            components_labels, n_components = fast_connected_components(points, eps=eps)

            for cmp_lbl_ in range(n_components):
                mask = (components_labels == cmp_lbl_)

                cmp_points = points[mask]

                dims_mins = cmp_points.min(axis=0)
                dims_maxes = cmp_points.max(axis=0)

                lwh = dims_maxes - dims_mins

                # if np.any(lwh > 15) or np.any(lwh < 0.1):
                    # print(f"ConvexHullObject: Cluster too large! {lwh=}")
                    # return None

                volume = np.prod(lwh)
                sorted_dims = np.sort(lwh)  # [smallest, medium, largest]

                if volume > 200: # TODO: add these to config
                    continue
                elif volume < 0.0001:  # Very small volume (0.1 liter)
                    continue
                elif sorted_dims[0] < 0.005:  # 5mm absolute minimum 
                    continue
                elif sorted_dims[1] < 0.05:  # Second dimension must be at least 5cm
                    continue
                elif sorted_dims[2] > 20: 
                    continue

                if mask.sum() > self.min_component_points:
                    mesh = trimesh.convex.convex_hull(cmp_points)
                    centre = mesh.centroid

                    if np.linalg.norm(centre[:2]) > 75:
                        continue # TODO: config -> too far

                    component_original_points.append(cmp_points)
                    component_meshes.append(mesh)
                    component_boxes.append(ConvexHullObject.points_to_bounding_box(mesh.vertices, mesh.centroid))
                    component_vols.append(volume)

                    density = mask.sum() / volume
                    component_densities.append(density)

                    vertices = mesh.vertices
                    lidar_cmp_indices.add(cmp_lbl)
                    all_cmp_vertices.append(vertices)
                    # all_cmp_labels.append(np.full((len(vertices),), fill_value=cmp_lbl))
                    n_vertices_per_cmp[cmp_lbl] = len(vertices)

                    cmp_lbl += 1


        # component nms
        component_densities = np.array(component_densities, float)
        ordered = np.argsort(-1.0*component_densities)
        idx2order = {x: i for i, x in enumerate(ordered)}

        print("component_densities", component_densities.shape)
        print("component_densities ordered", component_densities[ordered])

        # ordered_positions = np.stack([component_boxes[idx][:3] for idx in ordered], axis=0)
        positions = np.stack([box[:3] for box in component_boxes], axis=0)

        iou_thresh = 0.5
        suppressed = set()
        keep_indices = set()

        components_tree = cKDTree(positions)

        for i in range(len(ordered)):
            idx1 = ordered[i]
            if idx1 in suppressed:
                continue

            keep_indices.add(idx1)

            indices = components_tree.query_ball_point(positions[idx1], self.nms_query_distance)

            for idx2 in indices:
                j = idx2order[idx2]
                if idx2 in suppressed or j in keep_indices:
                    continue

                iou = box_iou_3d(component_boxes[idx1], component_boxes[idx2])

                if iou > iou_thresh:
                    suppressed.add(idx2)

        keep_indices = set(keep_indices) # fast lookup

        # have to be in the same order as all_cmp_labels
        component_boxes = [x for i, x in enumerate(component_boxes) if i in keep_indices]
        component_densities = [x for i, x in enumerate(component_densities) if i in keep_indices]
        component_vols = [x for i, x in enumerate(component_vols) if i in keep_indices]
        component_meshes = [x for i, x in enumerate(component_meshes) if i in keep_indices]
        component_original_points = [x for i, x in enumerate(component_original_points) if i in keep_indices]

        # prune all_cmp_labels with keep indices...
        # all_cmp_vertices = [x for i, x in enumerate(all_cmp_vertices) if i in keep_indices]
        # all_cmp_labels = [x for i, x in enumerate(all_cmp_labels) if i in keep_indices]
        all_cmp_labels = []
        all_cmp_vertices_ = []
        for cmp_lbl, idx in enumerate(list(sorted(keep_indices))):
            vertices = all_cmp_vertices[idx]
            all_cmp_vertices_.append(vertices)
            all_cmp_labels.append(np.full((len(vertices),), fill_value=cmp_lbl))

        all_cmp_vertices = all_cmp_vertices_

        lidar_n_components = len(keep_indices)

        print(f"{len(keep_indices)=} {len(suppressed)=}")

        # filter the camera images for ones with this sweep
        sweep_camera_name_timestamps = [
            (camera_name, cam_timestamp_ns)
            for camera_name, cam_timestamp_ns, sweep_timestamp_ns in camera_name_timestamps
            if sweep_timestamp_ns == timestamp_ns
        ]


        instance_buffers = self._get_instance_buffers(component_meshes, camera_models, timestamp_city_SE3_ego_dict, sweep_camera_name_timestamps, timestamp_ns)


        # add confirmed tracklets
        track_ids = []
        label_to_track_id = {}
        track_label = lidar_n_components
        world_to_ego = np.linalg.inv(city_SE3_ego_lidar_t.transform_matrix)
        for track_idx, track in enumerate(tracker.tracks):
            # if not track.is_confirmed():
            #     continue

            assert track_idx == track.track_id

            vertices = get_rotated_3d_box_corners(track.to_box())
            vertices_ego = points_rigid_transform(vertices, world_to_ego)
            track_ids.append(track.track_id)

            label_to_track_id[track_label] = track.track_id

            all_cmp_vertices.append(vertices_ego)
            all_cmp_labels.append(np.full((len(vertices),), fill_value=track_label))

            track_cmp_indices.add(track_label)
            n_vertices_per_cmp[track_label] = len(vertices)

            track_label += 1

        assert track_cmp_indices.isdisjoint(lidar_cmp_indices), f"overlaps {len(track_cmp_indices.intersection(lidar_cmp_indices))} {len(track_cmp_indices)} {len(lidar_cmp_indices)}"

        n_tracks = len(track_ids)

        n_components = lidar_n_components + n_tracks

        all_cmp_vertices = np.concatenate(all_cmp_vertices, axis=0)
        all_cmp_labels = np.concatenate(all_cmp_labels)

        owlvit_pred_dir = self.owlvit_predictions_dir / self.log_id

        results = load_predictions_parallel(sweep_camera_name_timestamps, owlvit_pred_dir, num_workers=4)

        # count the number of boxes total
        total_boxes = sum(len(result['pred_boxes']) for result in results)
        print("total_boxes", total_boxes)

        if total_boxes == 0:
            with open('log_no_boxes.csv', 'a') as f:
                f.write(f'{self.log_id},{timestamp_ns}')
            print(f"WARNING: NO BOXES FOR {self.log_id=},{timestamp_ns=}")
            print("sweep_camera_name_timestamps", sweep_camera_name_timestamps)
            return []
        result_box_ids = []
        start = 0
        for result in results:
            n_boxes = len(result['pred_boxes'])

            result_box_ids.append(np.arange(start=start, stop=start+n_boxes))

            start += n_boxes

        iou_matrix = np.zeros((total_boxes, n_components), dtype=np.float32)
        mask_iou_matrix = np.zeros((total_boxes, n_components), dtype=np.float32)
        objectness_matrix = np.zeros((total_boxes, 1), dtype=np.float32)
        dist_matrix = np.full((total_boxes, n_components), fill_value=1000, dtype=np.float32)
        box_infos = {i: {} for i in range(total_boxes)}

        for box_ids, result in tqdm(
            zip(result_box_ids, results),
            desc="Lifting 2D boxes to 3D",
        ):
            if result['success']:
                camera_name = result['camera_name']
                cam_timestamp_ns = result['cam_timestamp_ns']
                pred_boxes = result['pred_boxes']
                image_class_embeds = result['image_class_embeds']
                objectness_scores = result['objectness_scores']
            assert len(pred_boxes) == len(image_class_embeds) == len(objectness_scores) == len(box_ids)

            if len(pred_boxes) == 0:
                continue

            instance_buffer = instance_buffers[camera_name]

            objectness_matrix[box_ids, 0] = objectness_scores

            camera_model = camera_models[camera_name]
            city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam_timestamp_ns]

            if city_SE3_ego_cam_t is None or city_SE3_ego_lidar_t is None:
                continue

            (
                uv_points,
                _,
                is_valid_points,
            ) = camera_model.project_ego_to_img_motion_compensated(
                all_cmp_vertices,
                city_SE3_ego_cam_t=city_SE3_ego_cam_t,
                city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
            )

            if not np.any(is_valid_points):
                continue

            uv_points[:, 0] = np.clip(uv_points[:, 0], 0, camera_model.width_px)
            uv_points[:, 1] = np.clip(uv_points[:, 1], 0, camera_model.height_px)

            valid_cmp_labels = all_cmp_labels[is_valid_points]
            valid_cmp_labels = np.unique(valid_cmp_labels)

            component_boxes = []
            component_centres = []
            component_hulls = []
            for cmp_label in valid_cmp_labels:
                mask = (all_cmp_labels == cmp_label) & is_valid_points

                if mask.sum() == 0:
                    component_boxes.append(np.zeros((4,), dtype=np.float32))
                    continue

                component_uv_points = uv_points[mask]

                xy1 = component_uv_points.min(axis=0)
                xy2 = component_uv_points.max(axis=0)

                xyc = (xy1 + xy2) / 2.0

                component_boxes.append(
                    np.concatenate([xy1, xy2], axis=0).astype(np.float32)
                )
                component_centres.append(xyc)
                component_hulls.append(MultiPoint(component_uv_points).convex_hull)

            component_boxes = np.stack(component_boxes, axis=0)
            component_centres = np.stack(component_centres, axis=0)

            box_ious = box_iou(pred_boxes, component_boxes)


            for i, box_idx in enumerate(box_ids):
                box_xyxy = pred_boxes[i]
                box_xc = (box_xyxy[:2] + box_xyxy[2:]) / 2.0
                x1, y1, x2, y2 = box_xyxy.astype(int)
                # width = x2 - x1
                # height = y2 - y1
                # center = [(x1 + x2) / 2, (y1 + y2) / 2]

                # Create as a 2D path
                box_shape = Polygon.from_bounds(x1, y1, x2, y2)

                mask_in_box = (uv_points[:, 0] >= x1) & (uv_points[:, 1] >= y1) & (uv_points[:, 0] <= x2) & (uv_points[:, 1] <= y2) 

                buffer_in_box = instance_buffer[y1:y2, x1:x2]

                box_area = (x2-x1) * (y2-y1)
                if buffer_in_box.sum() > 0:
                    values, counts = np.unique(buffer_in_box, return_counts=True)

                    for value, count in zip(values, counts):
                        mask_iou_matrix[box_idx, value] = count / box_area


                for j, cmp_label in enumerate(valid_cmp_labels):
                    mask = (all_cmp_labels == cmp_label) & is_valid_points & mask_in_box
                    cmp_n_vertices = n_vertices_per_cmp[cmp_label]
                    
                    in_box_prop = mask.sum() / max(1, cmp_n_vertices)

                    iou_box = box_ious[i, j]
                    iou_mesh = iou_multipoint(box_shape, component_hulls[j])
                    iou_matrix[box_idx, cmp_label] = (iou_box + iou_mesh + in_box_prop) / 3.0

                    dist = np.linalg.norm((box_xc - component_centres[j]))                    
                    dist_matrix[box_idx, cmp_label] = dist

            # For each OWLViT detection box
            for box_idx, box, semantic_features, objectness_score in zip(
                box_ids,
                pred_boxes,
                image_class_embeds,
                objectness_scores,
            ):
                box_infos[box_idx] = {
                    "camera_name": camera_name,
                    "camera_timestamp": cam_timestamp_ns,
                    "box_2d": box,
                    "box_idx": box_idx,
                    "source": "vision_guided",
                    "semantic_features": semantic_features,  # CLIP embeddings from OWLViT
                    "objectness_score": objectness_score,  # Confidence score
                    "has_semantic_features": semantic_features is not None,
                }
                
        # normalize by the dist
        dist_matrix_normalized = dist_matrix.copy() / dist_matrix.max(axis=1, keepdims=True)

        cost_matrix = dist_matrix_normalized + (1.0 - iou_matrix) + (1.0 - objectness_matrix) + (1.0 - mask_iou_matrix)

        # Greedy assignment
        # clusters_assigned = np.argmin(cost_matrix, axis=1)
        box_assigned, cluster_assigned = linear_sum_assignment(cost_matrix)

        # keep track of which clusters have been assigned.
        clusters_assigned_set = set()

        # for box_idx, cmp_label in enumerate(clusters_assigned):
        for box_idx, cmp_label in zip(box_assigned, cluster_assigned):
            if iou_matrix[box_idx, cmp_label] < self.min_box_iou:
                continue

            if cmp_label >= lidar_n_components:
                print(f"TODO: match with track...  IoU: {iou_matrix[box_idx, cmp_label]:.2f} Distance: {dist_matrix[box_idx, cmp_label]:.2f}")

                assert cmp_label in track_cmp_indices

                box_info = box_infos[box_idx]
                track_idx = cmp_label - lidar_n_components
                track_id = track_ids[track_idx]

                track = tracker.tracks[track_id]

                box = track.to_box()
                pos = box[:3]
                radius = np.linalg.norm(box[3:6]*0.5)

                lidar_indices = lidar_tree.query_ball_point(pos, radius)
                lidar_indices = np.array(lidar_indices, int)
                
                if len(lidar_indices) < self.min_component_points:
                    continue

                world_points = lidar_tree.data[lidar_indices]
                world_points = ConvexHullObject.points_in_box(box, world_points.copy())

                if len(world_points) < self.min_component_points:
                    continue

                obj = ConvexHullObject(
                    original_points=world_points,
                    confidence=(box_info['objectness_score']+iou_matrix[box_idx, cmp_label])/2.0,
                    iou_2d=iou_matrix[box_idx, cmp_label],
                    objectness_score=box_info['objectness_score'],
                    feature=box_info['semantic_features'],
                    timestamp=timestamp_ns,
                    source="vision_guided"
                )

                if obj.original_points is not None:
                    vision_clusters.append(obj)

                continue

            box_info = box_infos[box_idx]
            x1, y1, x2, y2 = box_info['box_2d']
            camera_name = box_info['camera_name']
            cam_timestamp_ns = box_info['camera_timestamp']

            camera_model = camera_models[camera_name]
            city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam_timestamp_ns]

            if city_SE3_ego_cam_t is None or city_SE3_ego_lidar_t is None:
                continue

            needs_filtering = False

            # print(f"MATCH: Box {box_idx} <-> Component {cmp_label}: IoU: {iou_matrix[box_idx, cmp_label]:.2f} Distance: {dist_matrix[box_idx, cmp_label]:.2f}")
            # component_mask = (lidar_connected_components_labels == cmp_label)
            # component_3d_points = points[component_mask].copy()
            component_3d_points = component_original_points[cmp_label]

            num_w_box += 1

            component_3d_points_city = city_SE3_ego_lidar_t.transform_point_cloud(
                component_3d_points
            )

            full_obj = ConvexHullObject(
                original_points=component_3d_points_city,
                confidence=(box_info['objectness_score']+iou_matrix[box_idx, cmp_label])/2.0,
                iou_2d=iou_matrix[box_idx, cmp_label],
                objectness_score=box_info['objectness_score'],
                feature=box_info['semantic_features'],
                timestamp=timestamp_ns,
                source="vision_guided"
            )

            if full_obj.original_points is not None:
                vision_clusters.append(full_obj)

            (
                uv_points,
                _,
                is_valid_points,
            ) = camera_model.project_ego_to_img_motion_compensated(
                component_3d_points,
                city_SE3_ego_cam_t=city_SE3_ego_cam_t,
                city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
            )

            # if is_valid_points.sum() < self.min_component_points:
                # continue

            component_3d_points = component_3d_points[is_valid_points]
            valid_uv = uv_points[is_valid_points]

            in_box_mask = (
                (valid_uv[:, 0] >= x1)
                & (valid_uv[:, 0] <= x2)
                & (valid_uv[:, 1] >= y1)
                & (valid_uv[:, 1] <= y2)
            )

            if in_box_mask.sum() >= self.min_component_points:
                component_3d_points = component_3d_points[in_box_mask].copy()

            # TODO: potentially do filtering with boxes again.....
            in_box_prop = in_box_mask.sum() / max(1, len(in_box_mask))

            clusters_assigned_set.add(cmp_label)

            num_w_box += 1

            # if in_box_prop > self.nms_iou_threshold:
            #     # don't add subsequent if mostly aligns?
            #     continue
            
            component_3d_points_city = city_SE3_ego_lidar_t.transform_point_cloud(
                component_3d_points
            )

            obj = ConvexHullObject(
                original_points=component_3d_points_city,
                confidence=(box_info['objectness_score']+iou_matrix[box_idx, cmp_label])/2.0,
                iou_2d=iou_matrix[box_idx, cmp_label],
                objectness_score=box_info['objectness_score'],
                feature=box_info['semantic_features'],
                timestamp=timestamp_ns,
                source="vision_guided"
            )

            if obj.original_points is not None:
                vision_clusters.append(obj)


            # if obj.original_points is not None and in_box_prop < self.nms_iou_threshold:
            #     vision_clusters.append(obj)
            # elif full_obj.original_points is not None:
            #     vision_clusters.append(full_obj)

        # find non assigned clusters
        non_assigned_clusters = set(cmp_lbl for cmp_lbl in range(lidar_n_components)).difference(clusters_assigned_set)

        print(f"{len(non_assigned_clusters)=}")

        uniform_semantic_features = np.ones((768,), dtype=np.float32)
        uniform_semantic_features = uniform_semantic_features / np.linalg.norm(uniform_semantic_features)

        # for cmp_label in non_assigned_clusters:
        #     mask = (lidar_connected_components_labels == cmp_label)

        #     component_3d_points = points[mask].copy()

        #     if len(component_3d_points) < self.min_component_points:
        #         continue

        #     component_3d_points_city = city_SE3_ego_lidar_t.transform_point_cloud(
        #         component_3d_points
        #     )

        #     obj = ConvexHullObject(
        #         original_points=component_3d_points_city,
        #         confidence=0.0,
        #         iou_2d=0.0,
        #         objectness_score=0.0,
        #         feature=uniform_semantic_features,
        #         timestamp=timestamp_ns,
        #         source="cluster"
        #     )

        #     if obj.original_points is not None:
        #         num_no_box += 1
        #         vision_clusters.append(obj)
            

        total = num_w_box + num_no_box
        print(f"{num_no_box=} ({num_no_box/total}) {num_w_box=} {num_w_box/max(total,1)}")

        return vision_clusters

    def _compute_bev_iou_alpha_shapes(
        self, shape1_points: np.ndarray, shape2_points: np.ndarray
    ) -> float:
        """
        Compute IoU between two alpha shapes in bird's eye view (XY plane).

        This is a key component for removing duplicate detections that might arise
        from both traditional LiDAR clustering and vision-guided clustering detecting
        the same physical object.
        """
        try:

            # Project to BEV (XY plane only)
            shape1_bev = shape1_points[:, :2]  # Take only X,Y coordinates
            shape2_bev = shape2_points[:, :2]

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

        except Exception as e:
            if self.debug:
                print(f"Error computing BEV IoU: {e}")
            return 0.0

    def _apply_hybrid_nms(
        self, all_clusters: List[Dict]
    ) -> List[Dict]:
        """
        Apply non-maximum suppression to remove duplicate detections.

        The strategy prioritizes detections with semantic features (from OWLViT) over
        purely geometric detections (from traditional clustering). This makes sense
        because vision-guided detections have additional semantic information that
        can be valuable for tracking.

        Args:
            all_clusters: Combined list of traditional and vision-guided clusters
            iou_threshold: BEV IoU threshold for considering detections as duplicates

        Returns:
            Filtered list of clusters after NMS
        """
        if len(all_clusters) <= 1:
            return all_clusters

        # Sort clusters by priority: vision-guided with features > vision-guided without > traditional
        # Within each category, sort by objectness score or point count
        def get_priority_score(cluster):
            if cluster["source"] == "vision_guided" and cluster.get(
                "has_semantic_features", False
            ):
                return (
                    2,
                    cluster.get("objectness_score", 0.0),
                    -1.0 * len(cluster.get("original_points", [])),
                )  # Highest priority
            elif cluster["source"] == "vision_guided":
                return (
                    1,
                    cluster.get("objectness_score", 0.0),
                    -1.0 * len(cluster.get("original_points", [])),
                )  # Medium priority
            else:
                return (
                    0,
                    0.0,
                    -1.0 * len(cluster.get("original_points", [])),
                )  # Lowest priority, sort by cluster size

        # # Create a list of pairs to compare
        # pairs_to_compare = []
        # for i in range(len(all_clusters)):
        #     for j in range(i + 1, len(all_clusters)):
        #         pairs_to_compare.append((all_clusters[i], all_clusters[j], i, j))

        # iou_matrix = np.zeros((len(all_clusters), len(all_clusters)))

        # with ProcessPoolExecutor() as executor:
        #     # Submit all IoU calculations to the process pool
        #     future_to_pair = {
        #         executor.submit(AlphaShapeUtils.convex_hull_iou_trimesh, cluster_i, cluster_j): (i, j) 
        #         for (cluster_i, cluster_j, i, j) in pairs_to_compare
        #     }
            
        #     for future in as_completed(future_to_pair):
        #         pair = future_to_pair[future]
        #         try:
        #             iou = future.result()
        #             # Store the result, perhaps keyed by the indices of the shapes
        #             # For demonstration, we'll use object ids
        #             # pair_key = (id(pair[0]), id(pair[1]))
        #             # print("pair", pair)
        #             i, j = pair
        #             iou_matrix[i, j] = iou
        #             iou_matrix[j, i] = iou
        #         except Exception as exc:
        #             print(f'Generated an exception: {exc}')

        # def get_priority_score(cluster
        sorted_clusters = sorted(all_clusters, key=get_priority_score, reverse=True)

        sorted_clusters_centres = np.stack([x['centroid_3d'] for x in sorted_clusters], axis=0)

        print("sorted_clusters_centres", sorted_clusters_centres.shape)

        clusters_tree = cKDTree(sorted_clusters_centres)


        # Apply NMS
        keep_indices = []
        suppressed = set()


        semantic_overlaps = []

        # for i, cluster_i in enumerate(sorted_clusters):
        num_clusters = len(sorted_clusters)
        for i in range(num_clusters):
            if i in suppressed:
                continue

            keep_indices.append(i)

            # Check all remaining clusters for overlap
            # for j in range(i+1, num_clusters):
            indices = clusters_tree.query_ball_point(
                sorted_clusters_centres[i], self.nms_query_distance
            )
            for j in indices:
                if j in suppressed or j <= i:
                    continue

                # Compute BEV IoU between the two clusters
                # iou = self._compute_bev_iou_alpha_shapes(
                #     cluster_i['original_points'],
                #     cluster_j['original_points']
                # )
                # print("cluster_i", cluster_i)
                # print("cluster_j", cluster_j)
                # iou = AlphaShapeUtils.alpha_shape_3d_iou(
                #     cluster_i["alpha_shape"], cluster_j["alpha_shape"]
                # )

                # iou = AlphaShapeUtils.voxel_iou_from_sets(
                #     cluster_i["voxel_set"], cluster_j["voxel_set"]
                # )

                # iou = iou_matrix[i, j]

                cluster_i = sorted_clusters[i]
                cluster_j = sorted_clusters[j]

                # t0 = time.time()
                # iou = AlphaShapeUtils.convex_hull_iou_voxelized(cluster_i, cluster_j)
                # t1 = time.time()

                iou = AlphaShapeUtils.convex_hull_iou_trimesh(cluster_i, cluster_j)

                semantic_overlap = np.dot(cluster_i['semantic_features'], cluster_j['semantic_features'])
                # t2 = time.time()
                semantic_overlaps.append(semantic_overlap)

                # voxelised_time = t1 - t0
                # mesh_time = t2 - t1

                # print(f"mesh_time={mesh_time:.2f} secs voxelised_time={voxelised_time:.2f} secs")


                # Suppress the lower-priority cluster if IoU is high
                if iou > self.nms_iou_threshold and semantic_overlap > self.nms_semantic_threshold:
                    suppressed.add(j)
                    # if self.debug:
                    #     print(
                    #         f"NMS: Suppressed {cluster_j['source']} cluster (IoU={iou:.3f}) with {len(cluster_j['original_points'])} points "
                    #         f"in favor of {cluster_i['source']} cluster with {len(cluster_i['original_points'])} points"
                    #     )

        # Return the kept clusters in original order (not sorted order)
        kept_clusters = [sorted_clusters[i] for i in keep_indices]

        if self.debug:
            original_count = len(all_clusters)
            final_count = len(kept_clusters)
            print(
                f"NMS: Kept {final_count}/{original_count} clusters "
                f"(removed {original_count - final_count} duplicates)"
            )

            print(f"NMS: semantic_overlaps {np.min(semantic_overlaps)} {np.mean(semantic_overlaps)} {np.max(semantic_overlaps)}")

        return kept_clusters

    def _fit_hybrid_alpha_shapes(
        self, traditional_clusters: List[np.ndarray], vision_clusters: List[Dict]
    ) -> List[Dict]:
        """
        Fit alpha shapes for both traditional and vision-guided clusters, then apply NMS.

        This method represents the core fusion logic of our hybrid approach. It processes
        both types of clusters (traditional LiDAR-only and vision-guided), creates alpha
        shapes for each, and then intelligently removes duplicates through non-maximum
        suppression that favors detections with semantic information.

        The output maintains compatibility with the existing AlphaShapeTracker while
        adding semantic features for enhanced tracking when available.
        """
        all_candidate_clusters = []

        # Process traditional LiDAR clusters
        # These clusters come from purely geometric analysis of the LiDAR point cloud
        # They don't have semantic information but capture the full 3D structure
        for cluster in traditional_clusters:
            new_points = voxel_sampling_fast(cluster)
            alpha_shape = AlphaShapeUtils.compute_alpha_shape(new_points)

            if alpha_shape is not None:
                all_candidate_clusters.append(
                    {
                        "source": "traditional",
                        "has_semantic_features": False,
                        "semantic_features": None,  # No CLIP embeddings for pure LiDAR
                        "objectness_score": 0.0,  # Default score for traditional clusters
                        **alpha_shape,
                    }
                )

        # Process vision-guided clusters
        # These clusters come from LiDAR points filtered through OWLViT 2D detections
        # They have rich semantic features but may miss objects not visible in cameras
        for cluster_info in vision_clusters:
            # if alpha_shape_3d is not None:
            # new_points = voxel_sampling(cluster_info["original_points"])
            new_points = cluster_info["original_points"]

            alpha_shape = AlphaShapeUtils.compute_alpha_shape(new_points)

            if alpha_shape is not None:
                all_candidate_clusters.append(
                    {
                        **cluster_info,
                        **alpha_shape,
                    }
                )

        # return all_candidate_clusters

        # frame_alpha_shapes = all_candidate_clusters

        # Apply non-maximum suppression to remove duplicate detections
        # This is crucial because the same physical object might be detected by both
        # traditional clustering and vision-guided clustering. NMS ensures we keep
        # the most informative detection (preferring those with semantic features).
        frame_alpha_shapes = self._apply_hybrid_nms(all_candidate_clusters)



        # Convert to final format with alpha shapes computed
        # This step ensures compatibility with the existing tracker interface while
        # preserving all the semantic information we've gathered
        # frame_alpha_shapes = filtered_clusters
        # frame_alpha_shapes = []
        # for cluster_data in filtered_clusters:
        #     alpha_shape_3d = AlphaShapeUtils.compute_alpha_shape_2d(cluster_data['points_3d'])

        #     if alpha_shape_3d is not None:
        #         # Create the final alpha shape object with all metadata
        #         alpha_shape_object = {
        #             'alpha_shape_3d': alpha_shape_3d,
        #             'points_3d': cluster_data['points_3d'],
        #             'source': cluster_data['source'],
        #             'has_semantic_features': cluster_data['has_semantic_features'],
        #             'semantic_features': cluster_data['semantic_features'],
        #             'objectness_score': cluster_data['objectness_score'],
        #         }

        #         # Add UV-specific information for vision-guided clusters
        #         if cluster_data['source'] == 'vision_guided':
        #             alpha_shape_object['alpha_shape_uv'] = cluster_data.get('alpha_shape_uv', None)
        #             alpha_shape_object['points_uv'] = cluster_data.get('points_uv', None)

        #         frame_alpha_shapes.append(alpha_shape_object)

        if self.debug:
            traditional_count = sum(
                1 for s in frame_alpha_shapes if s["source"] == "traditional"
            )
            vision_count = sum(
                1 for s in frame_alpha_shapes if s["source"] == "vision_guided"
            )
            semantic_count = sum(
                1 for s in frame_alpha_shapes if s["has_semantic_features"]
            )
            print(
                f"Final clusters: {traditional_count} traditional, {vision_count} vision-guided, "
                f"{semantic_count} with semantic features"
            )

        return frame_alpha_shapes

    def generate_hybrid_alpha_shapes(self):
        """Generate hybrid alpha shapes combining traditional and vision-guided clustering."""
        input_pkl_path = os.path.join(self.root_path, self.log_id, self.log_id + ".pkl")
        method_name = f"{self.dataset_cfg.InitLabelGenerator}_owlvit_hybrid"
        output_pkl_path = os.path.join(
            self.root_path,
            self.log_id,
            self.log_id + "_alpha_shapes_" + str(method_name) + ".pkl",
        )

        if os.path.exists(output_pkl_path):
            with open(output_pkl_path, "rb") as f:
                infos = pkl.load(f)
            return infos

        with open(input_pkl_path, "rb") as f:
            infos = pkl.load(f)

        # TODO: revisit
        # self._track_owlvit_predictions(self.log_id)
        # exit()

        camera_name_timestamps: List[Tuple] = self._load_camera_name_timestamps()

        log_dir = self.dataloader.dataset_dir / self.split / self.log_id
        sensor_dir = log_dir / "sensors"
        cameras_dir = sensor_dir / "cameras"

        # Load camera models for each
        camera_models = {
            cam_name: PinholeCamera.from_feather(log_dir=log_dir, cam_name=cam_name)
            for cam_name in self.ring_cameras
        }

        raster_ground_height_layer = GroundHeightLayer.from_file(log_dir / "map")

        all_alpha_shapes = []
        all_pose = []
        all_timestamps = []

        tracker = ConvexHullKalmanTracker(debug=True)

        class_features = torch.load("/home/uqdetche/lidar_longtail_mining/lion/tools/class_features.pt")
        # print(class_features)
        class_features_names = list(class_features.keys())
        class_features_array = np.stack([x.detach().numpy() for x in class_features.values()], axis=0)

        class_features_array = class_features_array / np.linalg.norm(class_features_array, axis=1, keepdims=True)

        print(f"{class_features_array.shape=}")

        # load annotations for recall printing
        vis_annotations_feather = Path(
            f"/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val/{self.log_id}/annotations.feather"
        )
        assert vis_annotations_feather.exists()

        gts = pd.read_feather(vis_annotations_feather)
        gts = gts.assign(
            log_id=pd.Series([self.log_id for _ in range(len(gts))], dtype="string").values
        )
        gts = gts.set_index(["log_id", "timestamp_ns"], drop=False).sort_values("category")

        for i in trange(min(500, len(infos)), desc=f"Generating hybrid alpha shapes"):
            # 1. Load temporal LiDAR window
            aggregated_points = None
            # aggregated_points, cur_points = self._load_temporal_lidar_window(i, infos)

            pose = infos[i]['pose']

            # 2. Load OWLViT predictions for all cameras (exhaustive)
            timestamp_ns = infos[i].get(
                "timestamp_ns", None
            )  # Fallback if no timestamp
            assert timestamp_ns is not None, infos[i].keys()
            all_timestamps.append(timestamp_ns)

            gt_frame = gts.loc[[(self.log_id, int(timestamp_ns))]]
            gt_lidar_boxes = argo2_box_to_lidar(
                gt_frame[
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
            ).to(dtype=torch.float32)

            gt_categories = gt_frame["category"].values

            # 3. Generate both cluster types
            # traditional_clusters = self._get_traditional_clusters(aggregated_points)
            traditional_clusters = []  # TODO: for testing
            if i == 0 and PROFILING:
                pr = cProfile.Profile()
                pr.enable()
            tracker.predict(timestamp_ns)


            # Load city SE3 ego transformations
            timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_dir)

            # ego transformation (ego -> city)
            city_SE3_ego_lidar_t = timestamp_city_SE3_ego_dict[timestamp_ns]

            # Find the lidar timestamps
            lidar_folder = sensor_dir / "lidar"

            if aggregated_points is None:
                lidar_feather_path = lidar_folder / f"{timestamp_ns}.feather"
                lidar = read_feather(lidar_feather_path)
                pcl_ego = lidar.loc[:, ["x", "y", "z"]].to_numpy().astype(float)
                pcl_city_1 = city_SE3_ego_lidar_t.transform_point_cloud(pcl_ego)
            else:
                pcl_ego = aggregated_points
                pcl_city_1 = city_SE3_ego_lidar_t.transform_point_cloud(pcl_ego)

            # TODO cache
            is_ground = raster_ground_height_layer.get_ground_points_boolean(
                pcl_city_1
            ).astype(bool)
            is_not_ground = ~is_ground

            pcl_city_1 = pcl_city_1[is_not_ground]
            pcl_ego = pcl_ego[is_not_ground]

            # create lidar tree
            lidar_tree = cKDTree(pcl_city_1)

            # TODO: project tracks and see if align with boxes...
            vision_guided_clusters = self._get_vision_guided_clusters(
                pcl_ego,
                traditional_clusters,
                camera_name_timestamps,
                timestamp_ns,
                camera_models,
                timestamp_city_SE3_ego_dict,
                tracker,
                lidar_tree
            )

            if i == 0 and PROFILING:
                pr.disable()

                s = io.StringIO()
                sortby = "cumtime"
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                ps.print_stats()
                with open("_get_vision_guided_clusters_cprofile.txt", "w") as f:
                    f.write(s.getvalue())

            if self.debug:
                print(
                    f"Frame {i}: {len(traditional_clusters)} traditional, {len(vision_guided_clusters)} vision-guided clusters"
                )

            if i > 0 and PROFILING:
                pr = cProfile.Profile()
                pr.enable()
            tracker.update(vision_guided_clusters, infos[i]['pose'], lidar_tree)
            if i > 0 and PROFILING:
                pr.disable()

                s = io.StringIO()
                sortby = "cumtime"
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                ps.print_stats()
                with open("_convex_hull_kalman_tracker_update_cprofile.txt", "w") as f:
                    f.write(s.getvalue())

            #################################################################
            info_path = str(i).zfill(4) + ".npy"
            lidar_path = os.path.join(self.root_path, self.log_id, info_path)

            all_boxes = []
            all_labels = []
            all_box_types = []


            fig, ax = plt.subplots(figsize=(8,8))

            ax.scatter(
                pcl_city_1[:, 0],
                pcl_city_1[:, 1],
                s=1,
                c="blue",
                label="Lidar Points",
                alpha=0.5,
            )

            ego_pos = pose[:3, 3]

            tracks = tracker.tracks

            for obj in vision_guided_clusters:
                box = obj.box
                center_xy = box[:2]
                length = box[3]
                width = box[4]
                yaw = box[6]

                # all_boxes.append(box)
                # all_labels.append("vision_cluster")
                # all_box_types.append("box_cluster")

                # Get rotated box corners
                corners = get_rotated_box(center_xy, length, width, yaw)

                color = "black"

                obj_polygon = patches.Polygon(
                    corners,
                    linewidth=1,
                    edgecolor=color,
                    facecolor="none",
                    alpha=1.0,
                    linestyle="-",
                )
                ax.add_patch(obj_polygon)

            for track in tracks:
                if track.is_deleted():
                    continue
                box = track.to_box()
                # box = track.last_box
                center_xy = box[:2]
                length = box[3]
                width = box[4]
                yaw = box[6]


                all_boxes.append(box)
                all_labels.append(f"track_{track.track_id}")
                all_box_types.append("track.to_box()")

                # Get rotated box corners
                corners = get_rotated_box(center_xy, length, width, yaw)

                color = "yellow"
                alpha = 0.5
                if track.is_confirmed():
                    color = "green"
                    alpha = 0.7
                elif track.is_deleted():
                    color = "grey"
                    alpha = 0.1

                track_polygon = patches.Polygon(
                    corners,
                    linewidth=2,
                    edgecolor="brown",
                    facecolor="none",
                    alpha=alpha,
                    linestyle="-",
                )
                ax.add_patch(track_polygon)

                if track.spline_boxes is not None:
                    for box in track.spline_boxes[-1:]:
                        all_boxes.append(box)
                        all_labels.append(f"spline_box_{track.track_id}")
                        all_box_types.append("spline_box")

                        center_xy = box[:2]
                        length = box[3]
                        width = box[4]
                        yaw = box[6]

                        # Get rotated box corners
                        corners = get_rotated_box(center_xy, length, width, yaw)
                        track_polygon = patches.Polygon(
                            corners,
                            linewidth=1,
                            edgecolor="red",
                            facecolor="none",
                            alpha=0.7,
                            linestyle="-",
                        )
                        ax.add_patch(track_polygon)

                if track.optimized_boxes is not None:
                    for box in track.optimized_boxes[-1:]:
                        center_xy = box[:2]
                        length = box[3]
                        width = box[4]
                        yaw = box[6]

                        all_boxes.append(box)
                        all_labels.append(f"optimized_track_{track.track_id}")
                        all_box_types.append("optimized_box")

                        # Get rotated box corners
                        corners = get_rotated_box(center_xy, length, width, yaw)
                        track_polygon = patches.Polygon(
                            corners,
                            linewidth=1,
                            edgecolor=color,
                            facecolor="none",
                            alpha=0.7,
                            linestyle="--",
                        )
                        ax.add_patch(track_polygon)

                # if track.spline_boxes is not None:
                #     for box in track.spline_boxes[-1:]:
                #         center_xy = box[:2]
                #         length = box[3]
                #         width = box[4]
                #         yaw = box[6]

                #         # Get rotated box corners
                #         corners = get_rotated_box(center_xy, length, width, yaw)
                #         track_polygon = patches.Polygon(
                #             corners,
                #             linewidth=1,
                #             edgecolor="blue",
                #             facecolor="none",
                #             alpha=0.5,
                #             linestyle="dotted"
                #         )
                #         ax.add_patch(track_polygon)

                if track.is_confirmed():
                    next_timestamp_ns = timestamp_ns + 1e+8
                    
                    boxes = track.extrapolate_box([timestamp_ns]) # use the track timestamps rather than all...
                    for box in boxes:
                        center_xy = box[:2]
                        length = box[3]
                        width = box[4]
                        yaw = box[6]

                        
                        all_boxes.append(box)
                        all_labels.append(f"extrapolate_box_track_{track.track_id}")
                        all_box_types.append("extrapolated_box")

                        # Get rotated box corners
                        corners = get_rotated_box(center_xy, length, width, yaw)
                        track_polygon = patches.Polygon(
                            corners,
                            linewidth=1,
                            edgecolor="purple",
                            facecolor="none",
                            alpha=0.5,
                            linestyle="--",
                        )
                        ax.add_patch(track_polygon)

                    # if track.path_box is not None:
                    #     box = track.path_box
                    #     center_xy = box[:2]
                    #     length = box[3]
                    #     width = box[4]
                    #     yaw = box[6]


                    #     all_boxes.append(box)
                    #     all_labels.append(f"path_track_{track.track_id}")
                    #     all_box_types.append("path_box")

                    #     # Get rotated box corners
                    #     corners = get_rotated_box(center_xy, length, width, yaw)
                    #     track_polygon = patches.Polygon(
                    #         corners,
                    #         linewidth=1,
                    #         edgecolor="pink",
                    #         facecolor="none",
                    #         alpha=0.5,
                    #         linestyle="dotted",
                    #     )
                    #     ax.add_patch(track_polygon)

                    for ellipse_params in track.ellipses[-1:]:
                        lw = np.linalg.norm(track.lwh) # big enough to cover?

                        cx, cy, cz, a, b, h, theta = ellipse_params

                        ellipse_box = ellipse_params.copy()
                        ellipse_box[3:5] *= 2

                        all_boxes.append(ellipse_box)
                        all_labels.append(f"ellipse_track_{track.track_id}")
                        all_box_types.append("ellipse_box")

                        x_outline, y_outline = draw_ellipse_outline(cx, cy, a, b, theta)

                        plt.plot(x_outline, y_outline, 'b-', linewidth=1, alpha=0.3)

                    optimized_motion_box = track.optimized_motion_box()
                    history_motion_box = track.optimized_motion_box(history=True)

                    if optimized_motion_box is not None:
                        all_boxes.append(optimized_motion_box)
                        all_labels.append(f"optimized_motion_box_{track.track_id}")
                        all_box_types.append("optimized_motion_box")


                    if history_motion_box is not None:
                        all_boxes.append(history_motion_box)
                        all_labels.append(f"history_motion_box_{track.track_id}")
                        all_box_types.append("history_motion_box")

                        box = history_motion_box
                        center_xy = box[:2]
                        length = box[3]
                        width = box[4]
                        yaw = box[6]

                        # Get rotated box corners
                        corners = get_rotated_box(center_xy, length, width, yaw)
                        track_polygon = patches.Polygon(
                            corners,
                            linewidth=1,
                            edgecolor="orange",
                            facecolor="none",
                            alpha=0.7,
                            linestyle="dotted",
                        )                        


                    # for ellipse_params in track.predict_ellipse_motion_model(track.timestamps):
                    #     cx, cy, cz, a, b, h, theta = ellipse_params

                    #     x_outline, y_outline = draw_ellipse_outline(cx, cy, a, b, theta)

                    #     plt.plot(x_outline, y_outline, 'g-', linewidth=2, alpha=0.5)


                # if track.is_confirmed():
                #     next_timestamp_ns = timestamp_ns + 1e+8
                #     box = track.extrapolate_kalman_box(next_timestamp_ns)
                #     center_xy = box[:2]
                #     length = box[3]
                #     width = box[4]
                #     yaw = box[6]

                #     # Get rotated box corners
                #     corners = get_rotated_box(center_xy, length, width, yaw)
                #     track_polygon = patches.Polygon(
                #         corners,
                #         linewidth=1,
                #         edgecolor="purple",
                #         facecolor="none",
                #         alpha=1.0,
                #         linestyle="--",
                #     )
                #     ax.add_patch(track_polygon)



                # center_xy = track.to_box()[:2]
                # xc, yc = center_xy[0], center_xy[1]
                # dist = np.linalg.norm(center_xy[:2]-ego_pos[:2])
                # if track.last_avg_velocity is not None and dist < 50:
                #     vel_2d = track.last_avg_velocity[:2]
                #     last_yaw = track.last_yaw
                #     box_yaw = track.to_box()[6]

                #     ax.text(xc ,yc, f"{vel_2d[0]:.2f} {vel_2d[1]:.2f} {last_yaw:.2f} {box_yaw:.2f}", fontsize=12)


                # center_xy = track.to_box()[:2]
                # xc, yc = center_xy[0], center_xy[1]
                # dist = np.linalg.norm(center_xy[:2]-ego_pos[:2])
                # if dist < 50 and track.is_confirmed():
                #     track_features = track.features[-1]
                #     dots = class_features_array @ track_features
                #     print(f"dots {dots.shape} {dots.min()} {dots.max()}")
                #     best_cls_idx = np.argmax(dots)
                #     class_name = class_features_names[best_cls_idx]
                #     ax.text(xc ,yc, f"{class_name} {dots[best_cls_idx]:.3f}", fontsize=12)
                #     # ax.text(xc ,yc, f"id:{track.track_id} age:{track.age} hits:{track.hits} iou:{track.last_iou}", fontsize=12)


                # if track.last_mesh is not None and track.is_confirmed():
                if track.last_mesh is not None and track.is_confirmed():
                    vertices_3d = track.last_mesh.vertices
                    vertices_2d = vertices_3d[:, :2]

                    hull = ConvexHull(vertices_2d)
                    vertices_2d = vertices_2d[hull.vertices]

                    # Create polygon patch for alpha shape
                    polygon = patches.Polygon(
                        vertices_2d,
                        linewidth=1,
                        edgecolor=color,
                        facecolor="none",
                        alpha=0.5,
                    )
                    ax.add_patch(polygon)                    

                positions = np.stack(track.positions, axis=0)
                ax.plot(positions[:, 0], positions[:, 1], alpha=alpha, linewidth=1, color=color)

            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("X (meters)", fontsize=12)
            ax.set_ylabel("Y (meters)", fontsize=12)

            # Now zoom in
            xc, yc = ego_pos[:2]
            plt.xlim(xc + -50, xc + 50)
            plt.ylim(yc + -50, yc + 50)

            plt.tight_layout()

            save_folder = Path("./convex_hull_kalman_tracker")
            save_folder.mkdir(exist_ok=True)
            save_path = save_folder / f"frame_{i}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            world_to_ego = np.linalg.inv(infos[i]['pose'])

            all_boxes = np.stack(all_boxes, axis=0)
            print("all_boxes", all_boxes.shape)
            all_boxes = register_bbs(all_boxes, world_to_ego)

            all_boxes = all_boxes.astype(float)

            try:
                all_boxes = torch.from_numpy(all_boxes)
            except Exception as e: 
                print("all_boxes", all_boxes)
                raise e

            if len(gt_lidar_boxes) > 0 and len(all_boxes) > 0:
                ious = rotate_iou_cpu_eval(gt_lidar_boxes, all_boxes).reshape(
                    gt_lidar_boxes.shape[0], all_boxes.shape[0], 2
                )
                ious = ious[:, :, 0]

                idx = np.argmax(ious, axis=1)
                votes = []
                for i, j in enumerate(idx):
                    if ious[i, j] > 0.1:
                        gt_cat = gt_categories[i]
                        print(f"GT {i} ({gt_cat}) matches with {all_labels[int(j)]} @ {ious[i, j]:.3f}")
                        votes.append(all_box_types[j])

                    for jj in range(len(all_boxes)):
                        if jj == j:
                            continue
                        if ious[i, jj] > 0.1:
                            print(f"    ALSO matches with {all_labels[int(jj)]} @ {ious[i, jj]:.3f}")

                idx = np.argmax(ious, axis=0)
                type_ious = {x: [] for x in set(all_box_types)}
                for j, i in enumerate(idx):
                    if ious[i, j] > 0.1: # only include those with some gt (e.g. we might find other objects)
                        box_type = all_box_types[int(j)]
                        type_ious[box_type].append(ious[i, j])

                for box_type, box_type_ious in type_ious.items():
                    box_type_ious = np.array(box_type_ious, float)
                    if len(box_type_ious) > 0:
                        print(f"Box type {box_type}: {box_type_ious.min():.3f} {box_type_ious.mean():.3f} {box_type_ious.max():.3f}")
                    else:
                        print(f"Box type {box_type} had no matches")

                from collections import Counter
                counter = Counter(votes)
                print("best gt matches", counter.most_common(4))
                
            else:
                print(f"{gt_lidar_boxes.shape=} {all_boxes.shape=}")

            

            # image = np.zeros((2000, 2000, 3), dtype=np.uint8)
            # new_image = image.copy()
            # text_size = int(image.shape[1] / 1000)
            # text_pixel = int(image.shape[1]/400)
            # max_shape = max(image.shape)
            # previous_label_boxes = []
            # text_infos = []

            # color_map = {
            #     "yellow": (0, 255, 255),
            #     "green": (0, 255, 0), 
            #     "red": (0, 0, 255)
            # }

            # # Define world coordinate bounds (-50 to +50 meters relative to ego)
            # world_x_min, world_x_max = ego_pos[0] - 50, ego_pos[0] + 50
            # world_y_min, world_y_max = ego_pos[1] - 50, ego_pos[1] + 50

            # print("world_x_range", (world_x_min, world_x_max))
            # print("world_y_range", (world_y_min, world_y_max))

            # def to_image_coords(points):
            #     """Convert world coordinates to image pixel coordinates"""
            #     # Handle both single point [x, y] and multiple points [[x1, y1], [x2, y2], ...]
            #     points = np.array(points)
            #     if points.ndim == 1:
            #         points = points.reshape(1, -1)
                
            #     # Extract x, y coordinates
            #     x_world = points[:, 0]
            #     y_world = points[:, 1]
                
            #     # Normalize to [0, 1] range
            #     x_norm = (x_world - world_x_min) / (world_x_max - world_x_min)
            #     y_norm = (y_world - world_y_min) / (world_y_max - world_y_min)
                
            #     # Convert to pixel coordinates
            #     # Note: cv2 coordinate system has (0,0) at top-left
            #     # x increases rightward (same as world)
            #     # y increases downward (opposite of typical world coordinates)
            #     x_pixel = x_norm * 2000
            #     y_pixel = (1 - y_norm) * 2000  # Flip y-axis for cv2
                
            #     # Convert to integers and clip to image bounds
            #     x_pixel = np.clip(x_pixel, 0, 1999).astype(np.int32)
            #     y_pixel = np.clip(y_pixel, 0, 1999).astype(np.int32)
                
            #     # Return as integer coordinates
            #     coords = np.column_stack([x_pixel, y_pixel])
                
            #     # If input was single point, return single point
            #     if coords.shape[0] == 1:
            #         return tuple(coords[0])
            #     else:
            #         return coords



            # for track in tracks:
            #     box = track.to_box()
            #     center_xy = box[:2]
            #     length = box[3]
            #     width = box[4]
            #     yaw = box[6]

            #     # Get rotated box corners
            #     corners = get_rotated_box(center_xy, length, width, yaw)
                
            #     color_name = "yellow"
            #     alpha = 0.3
            #     if track.is_confirmed():
            #         color_name = "green"
            #         alpha = 0.7
            #     elif track.is_deleted():
            #         color_name = "red"

            #     color_bgr = color_map[color_name]
                
            #     # Convert corners to integer pixel coordinates
            #     corners = to_image_coords(corners)
            #     corners_int = np.array(corners, dtype=np.int32)

            #     dist = np.linalg.norm(center_xy[:2]-ego_pos[:2])
            #     # print(f"x, y = {x}, {y}")
                
            #     if dist >= 50:
            #         continue


            #     x, y = to_image_coords(center_xy)

            #     # Create overlay for alpha blending
            #     overlay = image.copy()
                
            #     # Draw solid polygon outline
            #     cv2.polylines(overlay, [corners_int], isClosed=True, color=color_bgr, thickness=2)
                
            #     # Blend with original image for alpha effect
            #     cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            #     continue

            #     s= f"id:{track.track_id} age:{track.age} hits:{track.hits} iou:{track.last_iou}"

            #     (label_width, label_height), baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_pixel)
            #     original_top_left = (int(x), int(y) - label_height - baseline*2)
            #     top_left = original_top_left

            #     # clip the top left so that the label is not off the image
            #     top_left = (np.clip(top_left[0], 0, max_shape - label_width), np.clip(top_left[1], 0, max_shape - label_height - baseline))

            #     # base the bottom right on the new top_left
            #     bottom_right = (top_left[0] + label_width, top_left[1] + label_height + baseline*2)
            #     curr_box = np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]])

            #     # check if the current label overlaps with another, then move it by it's height
            #     for prev_box in previous_label_boxes:
            #         if box_iou(curr_box[None, :], prev_box[None, :]) > 0.1:
            #             curr_box[[1, 3]] += label_height + baseline * 2

            #     previous_label_boxes.append(curr_box)
            #     top_left, bottom_right = curr_box[0:2], curr_box[2:]

            #     # Draw arrow if text was moved significantly from original position
            #     original_center = (int(x), int(y))
            #     final_text_center = (int(top_left[0] + label_width // 2), int(top_left[1] + label_height // 2))
            #     distance_moved = np.sqrt((final_text_center[0] - original_center[0])**2 + 
            #                             (final_text_center[1] - original_center[1])**2)

            #     # Draw arrow if moved more than threshold (e.g., 10 pixels)
            #     if distance_moved > 10:
            #         # Draw arrow from final text position to original position
            #         arrow_color = (128, 128, 128)  # Gray color for arrow
            #         arrow_thickness = 1
                    
            #         # Calculate arrow start point (edge of text box closest to original position)
            #         dx = original_center[0] - final_text_center[0]
            #         dy = original_center[1] - final_text_center[1]
                    
            #         # Normalize direction and scale to text box edge
            #         if abs(dx) > abs(dy):
            #             # Arrow starts from left/right edge of text
            #             arrow_start_x = int(top_left[0] if dx > 0 else bottom_right[0])
            #             arrow_start_y = final_text_center[1]
            #         else:
            #             # Arrow starts from top/bottom edge of text
            #             arrow_start_x = final_text_center[0]
            #             arrow_start_y = int(top_left[1] if dy > 0 else bottom_right[1])
                    
            #         arrow_start = (arrow_start_x, arrow_start_y)
                    
            #         # Draw the arrow line
            #         cv2.arrowedLine(image, arrow_start, original_center, 
            #                         arrow_color, arrow_thickness, tipLength=0.3)

            #     # Draw the label background rectangle
            #     new_image = cv2.rectangle(
            #         new_image, tuple(int(x) for x in top_left), tuple(int(x) for x in bottom_right), color_bgr, -1)

            #     # Add text info for later rendering
            #     text_infos.append([s, (int(top_left[0]), int(top_left[1]) + baseline + label_height), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, text_size, color_bgr, text_pixel, cv2.LINE_AA])

            # image = cv2.addWeighted(new_image, alpha, image, 1 - alpha, 0)

            # for text_info in text_infos:
            #     print("text_info", text_info)
            #     cv2.putText(
            #         image, *text_info
            #     )

            # cv2.imwrite(str(save_folder / f"frame_{i}_cv.png"), image)


            #################################################################

            
            # # 4. Fit hybrid alpha shapes
            # if i == 0:
            #     pr = cProfile.Profile()
            #     pr.enable()
            # # frame_alpha_shapes = self._fit_hybrid_alpha_shapes(
            # #     traditional_clusters, vision_guided_clusters
            # # )
            # if i == 0:
            #     pr.disable()

            #     s = io.StringIO()
            #     sortby = "cumtime"
            #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            #     ps.print_stats()
            #     with open("_fit_hybrid_alpha_shapes_cprofile.txt", "w") as f:
            #         f.write(s.getvalue())

            # frame_objects = []
            # for track in tracker.tracks:

            # all_alpha_shapes.append(frame_alpha_shapes)
            # all_pose.append(infos[i]["pose"])
            # all_timestamps.append(target_timestamp)

        # Track alpha shapes with AlphaShapeTracker
        print("Tracking hybrid alpha shapes with AlphaShapeTracker")

        # pr = cProfile.Profile()
        # pr.enable()

        # alpha_shape_tracker_time = 0
        # t0 = time.time()
        # tracker = AlphaShapeTracker(self.dataset_cfg.GeneratorConfig, debug=self.debug)
        # tracker.track_alpha_shapes(all_alpha_shapes, all_pose, all_timestamps)
        # t1 = time.time()

        # tracker.enhanced_forward_backward_consistency()

        non_null_sum = len([track for track in tracker.tracks if track.optimized_boxes is not None])

        print(f'tracks with non None optimized_boxes {non_null_sum=} ')

        if non_null_sum == 0:
            for track in tracker.tracks:
                print(f'Track {track.track_id} {track.optimized_boxes=} {track.optimized_poses=} {track.merged_mesh=}')
        


        timestamp_to_idx = {infos[i]['timestamp_ns']: i for i in range(len(infos))}

        for i in range(len(infos)):
            infos[i]["outline_box"] = []
            infos[i]["outline_ids"] = []
            infos[i]["outline_cls"] = []
            infos[i]["outline_dif"] = []
            infos[i]["alpha_shapes"] = []   
            infos[i]["outline_poses"] = []


        non_none = set()
        for track in tqdm(tracker.tracks, desc='Adding tracks to output'):
            if track.optimized_boxes is None or track.optimized_poses is None or track.timestamps is None:
                print(f"optimized_boxes is none for track_id: {track.track_id}")
                print(f"{track.optimized_boxes=} {track.optimized_poses=} {track.timestamps=}")
                continue

            non_none.add(track.track_id)

            for timestamp_ns, box, pose in zip(track.timestamps, track.optimized_boxes, track.optimized_poses):
                frame_id = timestamp_to_idx[timestamp_ns]

                ego_pose = infos[frame_id]['pose']
                world_to_ego = np.linalg.inv(ego_pose)

                world_points = points_rigid_transform(track.object_points, pose)
                ego_points = points_rigid_transform(world_points, world_to_ego)

                mesh = trimesh.convex.convex_hull(ego_points)

                track_box = track.to_box()
                print("optimized box", box)
                print("track_box", track_box)
                box = track_box

                # box = apply_pose_to_box(world_to_ego, box)
                print("world box", box)
                box = register_bbs(box.reshape(1, 7), world_to_ego)[0]
                print("ego box", box)

                infos[frame_id]["outline_box"].append(box)
                infos[frame_id]["outline_ids"].append(track.track_id)
                infos[frame_id]["outline_cls"].append(track.source)
                infos[frame_id]["outline_dif"].append(1)
                infos[frame_id]["alpha_shapes"].append({'vertices_3d': mesh.vertices})
                infos[frame_id]["outline_poses"].append(pose)

        print(f"{non_none=}")

        for i in range(len(infos)):
            if len(infos[i]["outline_box"]) > 0:
                infos[i]["outline_box"] = np.stack(infos[i]["outline_box"], axis=0)
            else:
                infos[i]["outline_box"] = np.zeros((0, 7))



        # print(f"Alpha shape tracker {(t1-t0):.2f} secs")
        # alpha_shape_tracker_time = t1 - t0


        # pr.disable()
        # s = io.StringIO()
        # sortby = "cumtime"
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # with open("_owlvit_alphashapetracker_cprofile.txt", "w") as f:
        #     f.write(s.getvalue())

        
        # nn_budget = None
        # max_cosine_distance = 0.2
        # metric = nn_matching.NearestNeighborDistanceMetric(
        #     "cosine", max_cosine_distance, nn_budget)
        # tracker = Tracker(metric)

        # uniform_semantic_features = np.ones((768,), dtype=np.float32)
        # uniform_semantic_features = uniform_semantic_features / np.linalg.norm(uniform_semantic_features)

        # tracker_total_time = 0
        
        # for frame_id, (alpha_shapes, pose, timestamp) in enumerate(zip(all_alpha_shapes, all_pose, all_timestamps)):
        #     objects = []
        #     t0 = time.time()
        #     for alpha_shape in alpha_shapes:
        #         world_points = points_rigid_transform(alpha_shape['original_points'], pose)
        #         alpha_shape_2d = AlphaShapeUtils.compute_alpha_shape_2d(world_points)
        #         box = AlphaShapeToBoxConverter.alpha_shape_to_box(alpha_shape_2d)
        #         objects.append(Object(box, alpha_shape.get('objectness_score', 0.5), alpha_shape.get('semantic_features', uniform_semantic_features)))

        #     # Update tracker.
        #     tracker.predict()
        #     tracker.update(objects)
        #     t1 = time.time()

        #     tracker_total_time += (t1 - t0)

        #     info_path = str(frame_id).zfill(4) + ".npy"
        #     lidar_path = os.path.join(self.root_path, self.log_id, info_path)


        #     fig, ax = plt.subplots(figsize=(8,8))

        #     lidar_points = np.load(lidar_path)[:, 0:3]
        #     lidar_points = points_rigid_transform(lidar_points, pose)

        #     ax.scatter(
        #         lidar_points[:, 0],
        #         lidar_points[:, 1],
        #         s=1,
        #         c="blue",
        #         label="Lidar Points",
        #         alpha=0.5,
        #     )

        #     ego_pos = pose[:3, 3]

        #     tracks = tracker.tracks

        #     for obj in objects:
        #         box = obj.box
        #         center_xy = box[:2]
        #         length = box[3]
        #         width = box[4]
        #         yaw = box[6]

        #         # Get rotated box corners
        #         corners = get_rotated_box(center_xy, length, width, yaw)

        #         det_polygon = patches.Polygon(
        #             corners,
        #             linewidth=5,
        #             edgecolor="red",
        #             facecolor="none",
        #             alpha=0.8,
        #             linestyle="--",
        #         )
        #         ax.add_patch(det_polygon)

        #     boxes = []
        #     ids = []
        #     cls_labels = []
        #     difficulties = []
        #     alpha_shapes = []
        #     for track in tracks:
        #         box = track.to_box()
        #         center_xy = box[:2]
        #         length = box[3]
        #         width = box[4]
        #         yaw = box[6]

        #         # Get rotated box corners
        #         corners = get_rotated_box(center_xy, length, width, yaw)

        #         corners3d = get_rotated_3d_box_corners(box)

        #         alpha_shapes.append(AlphaShapeUtils.compute_alpha_shape(corners3d))
        #         boxes.append(box)
        #         ids.append(track.track_id)
        #         difficulties.append(1)
        #         cls_labels.append("deepsort")


        #         track_polygon = patches.Polygon(
        #             corners,
        #             linewidth=3,
        #             edgecolor="green",
        #             facecolor="none",
        #             alpha=0.8,
        #             linestyle="-",
        #         )
        #         ax.add_patch(track_polygon)

        #     ax.set_aspect("equal")
        #     ax.grid(True, alpha=0.3)
        #     ax.set_xlabel("X (meters)", fontsize=12)
        #     ax.set_ylabel("Y (meters)", fontsize=12)

        #     # Now zoom in
        #     xc, yc = ego_pos[:2]
        #     plt.xlim(xc + -50, xc + 50)
        #     plt.ylim(yc + -50, yc + 50)

        #     plt.tight_layout()

        #     save_folder = Path("./deepsort_tracking")
        #     save_folder.mkdir(exist_ok=True)
        #     save_path = save_folder / f"frame_{frame_id}.png"
        #     plt.savefig(save_path, dpi=300, bbox_inches="tight")

        #     # boxes = np.stack(boxes, axis=0)

        #     # infos[i]["outline_box"] = boxes
        #     # infos[i]["outline_ids"] = ids
        #     # infos[i]["outline_cls"] = cls_labels
        #     # infos[i]["outline_dif"] = difficulties
        #     # infos[i]["alpha_shapes"] = alpha_shapes
        #     # # infos[i]["tracked_objects"] = tracker.tracked_objects # redundant..


        # print("saved as ", output_pkl_path)
        # # Save results
        # with open(output_pkl_path, "wb") as f:
        #     pkl.dump(infos, f)

        # return infos

        # print(f"Alpha shape tracker {(alpha_shape_tracker_time):.2f} secs")
        # print(f"kalman box tracker {(tracker_total_time):.2f} secs")




        # for i, target_timestamp in tqdm(
        #     enumerate(all_timestamps),
        #     desc="Generating output infos from OWLViTAlphaShapeMFCF",
        # ):
        #     target_timestamp = infos[i]["timestamp_ns"]

        #     track_ids = tracker.frame_alpha_shapes[target_timestamp]

        #     boxes = []
        #     cls_labels = []
        #     difficulties = []
        #     alpha_shapes = []
        #     ids = []

        #     for track_id in track_ids:
        #         if track_id not in tracker.tracked_objects:
        #             continue
        #         tracked_obj = tracker.tracked_objects[track_id]
        #         cur_traj = tracked_obj["trajectory"][target_timestamp]

        #         alpha_shape = cur_traj["alpha_shape"]
        #         oriented_box = cur_traj["oriented_box"]

        #         cls_labels.append(alpha_shape.get("source", "vision_guided"))
        #         difficulties.append(1)
        #         boxes.append(oriented_box)
        #         alpha_shapes.append(alpha_shape)
        #         ids.append(track_id)

        #     boxes = np.stack(boxes, axis=0)

        #     assert (
        #         len(boxes)
        #         == len(ids)
        #         == len(cls_labels)
        #         == len(difficulties)
        #         == len(alpha_shapes)
        #     ), f"{len(boxes)} {len(ids)} {len(cls_labels)} {len(difficulties)} {len(alpha_shapes)}"
        #     infos[i]["outline_box"] = boxes
        #     infos[i]["outline_ids"] = ids
        #     infos[i]["outline_cls"] = cls_labels
        #     infos[i]["outline_dif"] = difficulties
        #     infos[i]["alpha_shapes"] = alpha_shapes
        #     infos[i]["tracked_objects"] = tracker.tracked_objects # redundant..

        print("saved as ", output_pkl_path)
        # Save results
        with open(output_pkl_path, "wb") as f:
            pkl.dump(infos, f)

        # # Save debug statistics
        # if self.debug:
        #     debug_stats = tracker.debug_stats
        #     debug_output_path = output_pkl_path.replace(".pkl", "_debug_stats.pkl")
        #     with open(debug_output_path, "wb") as f:
        #         pkl.dump(debug_stats, f)

        #     # Print summary
        #     print(f"\n=== HYBRID DEBUG SUMMARY ===")
        #     traditional_count = sum(
        #         len([s for s in shapes if s.get("source") == "traditional"])
        #         for shapes in all_alpha_shapes
        #     )
        #     vision_count = sum(
        #         len([s for s in shapes if s.get("source") == "vision_guided"])
        #         for shapes in all_alpha_shapes
        #     )
        #     print(f"Traditional clusters: {traditional_count}")
        #     print(f"Vision-guided clusters: {vision_count}")
        #     print(f"Total clusters: {traditional_count + vision_count}")

        return infos

    def __call__(self):
        return self.generate_hybrid_alpha_shapes()


class AlphaShapeMFCF:
    """
    Alpha shape-based MFCF implementation.

    Usage with debug mode:
        mfcf = AlphaShapeMFCF(seq_name, root_path, config, debug=True)
        result = mfcf()

    Debug mode provides detailed tracking information including:
    - Frame-by-frame processing stats
    - IoU values during matching
    - New vs updated tracklet counts
    - Track length statistics
    - Forward-backward consistency updates
    """

    def __init__(self, seq_name: str, root_path: str, config, debug: bool = False):
        self.seq_name = seq_name
        self.root_path = root_path
        self.dataset_cfg = config
        self.debug = debug

        # Initialize outline estimator (reuse from original MFCF)
        self.outline_estimator = OutlineFitter(
            sensor_height=config.GeneratorConfig.sensor_height,
            ground_min_threshold=config.GeneratorConfig.ground_min_threshold,
            ground_min_distance=config.GeneratorConfig.ground_min_distance,
            cluster_dis=config.GeneratorConfig.cluster_dis,
            cluster_min_points=config.GeneratorConfig.cluster_min_points,
            discard_max_height=config.GeneratorConfig.discard_max_height,
            min_box_volume=config.GeneratorConfig.min_box_volume,
            min_box_height=config.GeneratorConfig.min_box_height,
            max_box_volume=config.GeneratorConfig.max_box_volume,
            max_box_len=config.GeneratorConfig.max_box_len,
        )

    def generate_alpha_shapes(self):
        """Generate alpha shapes for the sequence."""
        seq_name, root_path, dataset_cfg = (
            self.seq_name,
            self.root_path,
            self.dataset_cfg,
        )

        input_pkl_path = os.path.join(root_path, seq_name, seq_name + ".pkl")
        method_name = dataset_cfg.InitLabelGenerator
        output_pkl_path = os.path.join(
            root_path, seq_name, seq_name + "_alpha_shapes_" + str(method_name) + ".pkl"
        )

        if os.path.exists(output_pkl_path):
            with open(output_pkl_path, "rb") as f:
                infos = pkl.load(f)
            return infos

        with open(input_pkl_path, "rb") as f:
            infos = pkl.load(f)

        all_alpha_shapes = []
        all_pose = []
        win_size = self.dataset_cfg.GeneratorConfig.frame_num
        inte = self.dataset_cfg.GeneratorConfig.frame_interval
        thresh = self.dataset_cfg.GeneratorConfig.ppscore_thresh

        for i in trange(
            0, len(infos), desc=f"generate_alpha_shapes: iterating over infos"
        ):
            pose_i = np.linalg.inv(infos[i]["pose"])
            all_points = []
            all_H = []
            cur_points = None

            # Collect points from temporal window (same as original MFCF)
            for j in range(max(i - win_size, 0), min(i + win_size, len(infos)), inte):
                info_path = str(j).zfill(4) + ".npy"
                lidar_path = os.path.join(root_path, seq_name, info_path)
                if not os.path.exists(lidar_path):
                    print(f"{lidar_path=} doesn't exist")
                    continue

                pose_j = infos[j]["pose"]
                lidar_points = np.load(lidar_path)[:, 0:3]
                if j == i:
                    cur_points = lidar_points

                lidar_points = points_rigid_transform(lidar_points, pose_j)
                H_path = os.path.join(root_path, seq_name, "ppscore", info_path)
                H = np.load(H_path)
                all_points.append(lidar_points)
                all_H.append(H)

            all_points = np.concatenate(all_points)
            all_points = points_rigid_transform(all_points, pose_i)
            all_H = np.concatenate(all_H)
            all_points = all_points[all_H > thresh]
            new_box_points = np.concatenate([all_points, cur_points])
            new_box_points = voxel_sampling_fast(new_box_points)

            # Remove ground and cluster (same as original)
            non_ground_points = self.outline_estimator.remove_ground(new_box_points)
            clusters, labels = self.outline_estimator.clustering(non_ground_points)

            # Fit alpha shapes instead of boxes
            frame_alpha_shapes = []
            for cluster in clusters:
                if len(cluster) >= 3:  # Need at least 3 points for 2D convex hull

                    cluster_mins = cluster.min(axis=0)
                    cluster_maxes = cluster.max(axis=0)

                    cluster_dims = cluster_maxes - cluster_mins

                    cluster_vol = np.prod(cluster_dims)

                    if np.any(cluster_dims > 10):
                        print(f"Cluster dims too big! {cluster_dims=}")
                        continue

                    if np.any(
                        cluster_dims
                        < getattr(
                            self.dataset_cfg.GeneratorConfig, "min_box_volume", 0.1
                        )
                    ):
                        print(f"Cluster dims too small! {cluster_dims=}")
                        continue

                    if np.any(
                        cluster_vol
                        < getattr(
                            self.dataset_cfg.GeneratorConfig, "min_box_volume", 0.1
                        )
                    ):
                        print(f"Cluster vol too small! {cluster_vol=}")
                        continue

                    if np.any(
                        cluster_vol
                        > getattr(
                            self.dataset_cfg.GeneratorConfig, "max_box_volume", 200
                        )
                    ):
                        print(f"Cluster vol too large! {cluster_vol=}")
                        continue

                    alpha_shape = AlphaShapeUtils.compute_alpha_shape(cluster)
                    if alpha_shape is not None:
                        frame_alpha_shapes.append(alpha_shape)

            all_alpha_shapes.append(frame_alpha_shapes)
            all_pose.append(infos[i]["pose"])

        print("Tracking alpha shapes with AlphaShapeTracker")
        tracker = AlphaShapeTracker(self.dataset_cfg.GeneratorConfig, debug=self.debug)
        tracker.track_alpha_shapes(all_alpha_shapes, all_pose)

        # Convert tracked alpha shapes to boxes for output compatibility
        converter = AlphaShapeToBoxConverter()

        for i in trange(
            0, len(infos), desc="Generating output infos from AlphaShapeMFCF"
        ):
            # Get current frame objects
            alpha_shapes, ids, alpha_shape_data = (
                tracker.get_current_frame_alpha_shapes(i)
            )

            # Convert to boxes
            if len(alpha_shapes) > 0:
                boxes = converter.alpha_shapes_to_boxes(alpha_shapes)
                cls_labels = ["alpha_shape"] * len(boxes)
                difficulties = [1] * len(boxes)  # Default difficulty
            else:
                boxes = np.empty((0, 7))
                cls_labels = []
                difficulties = []

            infos[i]["outline_box"] = boxes
            infos[i]["outline_ids"] = ids
            infos[i]["outline_cls"] = cls_labels
            infos[i]["outline_dif"] = difficulties
            infos[i]["alpha_shapes"] = alpha_shapes  # Store original alpha shapes
            infos[i]["alpha_shape_data"] = alpha_shape_data  # Additional metadata

        with open(output_pkl_path, "wb") as f:
            pkl.dump(infos, f)

        # Save debug statistics if debug mode is enabled
        if self.debug:
            debug_stats = tracker.debug_stats
            debug_output_path = output_pkl_path.replace(".pkl", "_debug_stats.pkl")
            with open(debug_output_path, "wb") as f:
                pkl.dump(debug_stats, f)
            print(f"\nDebug statistics saved to: {debug_output_path}")

            # Print summary of debug statistics
            print(f"\n=== FINAL DEBUG SUMMARY ===")
            if "iou_statistics" in debug_stats:
                iou_stats = debug_stats["iou_statistics"]
                print(
                    f"IoU Statistics: Mean={iou_stats['mean']:.3f}, Std={iou_stats['std']:.3f}, Range=[{iou_stats['min']:.3f}, {iou_stats['max']:.3f}], Count={iou_stats['count']}"
                )

            if "track_statistics" in debug_stats:
                track_stats = debug_stats["track_statistics"]
                print(
                    f"Track Statistics: Count={track_stats['count']}, Mean Length={track_stats['mean_length']:.1f}, Range=[{track_stats['min_length']}, {track_stats['max_length']}]"
                )

        return infos

    def __call__(self):
        return self.generate_alpha_shapes()




class AlphaShapeToBoxConverter:
    """Convert alpha shapes to bounding boxes."""

    @staticmethod
    def alpha_shapes_to_boxes(alpha_shapes: List[Dict]) -> np.ndarray:
        """Convert list of alpha shapes to bounding boxes."""
        if not alpha_shapes:
            return np.empty((0, 7))

        boxes = []

        for alpha_shape in alpha_shapes:
            box = AlphaShapeToBoxConverter.alpha_shape_to_box(alpha_shape)
            if box is not None:
                boxes.append(box)

        if not boxes:
            return np.empty((0, 7))

        return np.array(boxes)

    @staticmethod
    def alpha_shape_to_box(alpha_shape: Dict) -> Optional[np.ndarray]:
        """Convert single alpha shape to bounding box [x, y, z, length, width, height, yaw]."""
        try:
            vertices_2d = alpha_shape["vertices_2d"]
            z_min = alpha_shape["z_min"]
            z_max = alpha_shape["z_max"]

            # Center
            center_2d = np.mean(vertices_2d, axis=0)
            center_x, center_y = center_2d
            center_z = (z_min + z_max) / 2
            height = z_max - z_min

            # Estimate orientation using PCA
            try:
                centered_vertices = vertices_2d - center_2d
                cov_matrix = np.cov(centered_vertices.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # Sort by eigenvalue (largest first)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Primary direction (largest eigenvalue) and secondary direction
                primary_direction = eigenvectors[:, 0]  # Length direction
                secondary_direction = eigenvectors[:, 1]  # Width direction
                
                # Calculate yaw from primary direction
                yaw = np.arctan2(primary_direction[1], primary_direction[0])
                
                # Project vertices onto principal directions to get length and width
                projections_length = np.dot(centered_vertices, primary_direction)
                projections_width = np.dot(centered_vertices, secondary_direction)
                
                # Length and width are ranges of projections
                length = np.max(projections_length) - np.min(projections_length)
                width = np.max(projections_width) - np.min(projections_width)

            except:
                # Fallback to axis-aligned approach
                min_x, min_y = np.min(vertices_2d, axis=0)
                max_x, max_y = np.max(vertices_2d, axis=0)
                length = max_x - min_x
                width = max_y - min_y
                yaw = 0.0

            return np.array([center_x, center_y, center_z, length, width, height, yaw])

        except Exception as e:
            print(f"Error converting alpha shape to box: {e}")
            return None
