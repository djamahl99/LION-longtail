import cProfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
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


from av2.map.map_api import ArgoverseStaticMap, GroundHeightLayer

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

from lion.unsupervised_core.convex_hull_tracker.alpha_shape_tracker import AlphaShapeTracker
from lion.unsupervised_core.convex_hull_tracker.alpha_shape_utils import AlphaShapeUtils
from lion.unsupervised_core.convex_hull_tracker.convex_hull_kalman_tracker import ConvexHullKalmanTracker
from lion.unsupervised_core.convex_hull_tracker.convex_hull_object import ConvexHullObject
from lion.unsupervised_core.convex_hull_tracker.convex_hull_track import ConvexHullTrackState
from lion.unsupervised_core.tracker.box_op import register_bbs


from .outline_utils import (
    OutlineFitter,
    TrackSmooth,
    voxel_sampling,
    points_rigid_transform,
)
from .trajectory_optimizer import (
    GlobalTrajectoryOptimizer,
    optimize_with_gtsam_timed,
    simple_pairwise_icp_refinement,
)

from .box_utils import *
from .file_utils import load_predictions_parallel
from .owlvit_frustum_tracker import OWLViTFrustumTracker
from sklearn.cluster import DBSCAN


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

        self.min_box_iou = 0.1
        self.lidar_connected_components_eps = 0.5
        self.connected_components_eps = 0.75
        self.min_component_points = 10

        self.nms_iou_threshold = 0.7
        self.nms_query_distance = 5.0 # metres
        self.nms_semantic_threshold = 0.7

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
        new_box_points = voxel_sampling(new_box_points)

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

    def _get_vision_guided_clusters(
        self,
        aggregated_points: np.ndarray,
        traditional_clusters,
        camera_name_timestamps: List[Tuple],
        timestamp_ns: int,
        camera_models: Dict[str, PinholeCamera],
        raster_ground_height_layer: GroundHeightLayer,
    ) -> List[Dict]:
        """
        Get vision-guided clusters by projecting LiDAR to cameras and filtering by OWLViT boxes.

        Returns:
            List of cluster dicts with 3D points, camera info, UV alpha shapes, and semantic features
        """
        vision_clusters = []

        log_dir = self.dataloader.dataset_dir / self.split / self.log_id
        sensor_dir = log_dir / "sensors"
        cameras_dir = sensor_dir / "cameras"

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


        is_ground = raster_ground_height_layer.get_ground_points_boolean(
            pcl_city_1
        ).astype(bool)
        print("is_ground", is_ground.shape, is_ground.sum())
        is_not_ground = ~is_ground

        pcl_city_1 = pcl_city_1[is_not_ground]
        pcl_ego = pcl_ego[is_not_ground]

        points = pcl_ego

        # points_orig_shape = points.shape
        # points = self.outline_estimator.remove_ground(points.copy())
        # is_ground = raster_ground_height_layer.get_ground_points_boolean(points).astype(bool)
        # is_not_ground = ~is_ground

        # points = points[is_not_ground]
        # print(f"before ground removal: {points_orig_shape} after: {points.shape}")

        # pre run connected components?
        lidar_connected_components_labels, lidar_n_components = fast_connected_components(points, eps=self.lidar_connected_components_eps)

        lidar_connected_components_infos = {}

        # turn them into meshes?
        lidar_meshes = []
        all_cmp_vertices = []
        all_cmp_labels = []

        num_no_box = 0
        num_w_box = 0

        for cmp_lbl in range(lidar_n_components):
            mask = (lidar_connected_components_labels == cmp_lbl)

            if mask.sum() > 25:
                cmp_points = points[mask]
                mesh = trimesh.convex.convex_hull(cmp_points)
                vertices = mesh.vertices
            elif mask.sum() >= self.min_component_points:
                vertices = points[mask]
            else:
                vertices = np.zeros((0, 3), dtype=np.float32)

            # lidar_meshes.append(mesh)
            all_cmp_vertices.append(vertices)
            all_cmp_labels.append(np.full((len(vertices),), fill_value=cmp_lbl))

        all_cmp_vertices = np.concatenate(all_cmp_vertices, axis=0)
        all_cmp_labels = np.concatenate(all_cmp_labels)

        print("all_cmp_vertices", all_cmp_vertices.shape)
        print("all_cmp_labels", all_cmp_labels.shape)
        print("points", points.shape)


        # filter the camera images for ones with this sweep
        sweep_camera_name_timestamps = [
            (camera_name, cam_timestamp_ns)
            for camera_name, cam_timestamp_ns, sweep_timestamp_ns in camera_name_timestamps
            if sweep_timestamp_ns == timestamp_ns
        ]

        owlvit_pred_dir = self.owlvit_predictions_dir / self.log_id

        results = load_predictions_parallel(sweep_camera_name_timestamps, owlvit_pred_dir, num_workers=4)

        # count the number of boxes total
        total_boxes = sum(len(result['pred_boxes']) for result in results)
        print("total_boxes", total_boxes)

        result_box_ids = []
        start = 0
        for result in results:
            n_boxes = len(result['pred_boxes'])

            result_box_ids.append(np.arange(start=start, stop=start+n_boxes))

            start += n_boxes
        # print("result_box_ids", result_box_ids)

        iou_matrix = np.zeros((total_boxes, lidar_n_components), dtype=np.float32)
        dist_matrix = np.full((total_boxes, lidar_n_components), fill_value=1000, dtype=np.float32)
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
            assert len(pred_boxes) == len(image_class_embeds) == len(objectness_scores)

            if len(pred_boxes) == 0:
                continue

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

            valid_cmp_labels = all_cmp_labels[is_valid_points]
            valid_cmp_labels = np.unique(valid_cmp_labels)

            component_boxes = []
            component_centres = []
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

            component_boxes = np.stack(component_boxes, axis=0)
            component_centres = np.stack(component_centres, axis=0)

            box_ious = box_iou(pred_boxes, component_boxes)

            for i, box_idx in enumerate(box_ids):
                box_xyxy = pred_boxes[i]
                box_xc = (box_xyxy[:2] + box_xyxy[2:]) / 2.0
                for j, cmp_label in enumerate(valid_cmp_labels):
                    iou_matrix[box_idx, cmp_label] = box_ious[i, j]

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

        cost_matrix = dist_matrix_normalized + (1.0 - iou_matrix)

        # Greedy assignment
        clusters_assigned = np.argmin(cost_matrix, axis=1)

        matches_per_component = {i: (clusters_assigned == i).sum() for i in range(lidar_n_components)}
        print(f"matches_per_component", [(k, v) for k, v in matches_per_component.items() if v > 0])

        # for cmp_lbl, num_matched in matches_per_component.items():
        #     if num_matched == 0:
        #         continue

        #     box_ids = np.where(clusters_assigned == cmp_lbl)[0]

        #     ious = iou_matrix[box_ids, cmp_lbl]

        #     print(f"Component: {cmp_lbl} has IoUs {np.min(ious)} {np.mean(ious)} {np.max(ious)}")

        # keep track of which clusters have been assigned.
        clusters_assigned_set = set()

        for box_idx, cmp_label in enumerate(clusters_assigned):
            if iou_matrix[box_idx, cmp_label] < self.min_box_iou:
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

            print(f"MATCH: Box {box_idx} <-> Component {cmp_label}: IoU: {iou_matrix[box_idx, cmp_label]:.2f} Distance: {dist_matrix[box_idx, cmp_label]:.2f}")
            component_mask = (lidar_connected_components_labels == cmp_label)
            component_3d_points = points[component_mask].copy()

            num_w_box += 1
            # add first with entire cluster
            # alpha_shape = AlphaShapeUtils.compute_alpha_shape(component_3d_points)
            # vision_clusters.append(
            #     {
            #         "original_points": component_3d_points,
            #         **box_info,
            #         **alpha_shape
            #     }
            # )

            component_3d_points_city = city_SE3_ego_lidar_t.transform_point_cloud(
                component_3d_points
            )

            obj = ConvexHullObject(
                original_points=component_3d_points_city,
                confidence=box_info['objectness_score'],
                feature=box_info['semantic_features'],
                timestamp=timestamp_ns,
                source="vision_guided"
            )

            if obj.original_points is not None:
                vision_clusters.append(obj)

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

            clusters_assigned_set.add(cmp_label)

            alpha_shape = AlphaShapeUtils.compute_alpha_shape(component_3d_points)

            if alpha_shape is None:
                continue

            num_w_box += 1
            # add second time with constrained part of cluster to bbox
            # vision_clusters.append(
            #     {
            #         "original_points": component_3d_points,
            #         **box_info,
            #         **alpha_shape
            #     }
            # )
            component_3d_points_city = city_SE3_ego_lidar_t.transform_point_cloud(
                component_3d_points
            )

            # print(f"component_3d_points[0]={component_3d_points[0]}")
            # print(f"component_3d_points_city[0]={component_3d_points_city[0]}")

            obj = ConvexHullObject(
                original_points=component_3d_points_city,
                confidence=box_info['objectness_score'],
                feature=box_info['semantic_features'],
                timestamp=timestamp_ns,
                source="vision_guided"
            )

            if obj.original_points is not None:
                vision_clusters.append(obj)
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

        #     dims_mins = np.min(component_3d_points, axis=0)
        #     dims_maxes = np.max(component_3d_points, axis=0)

        #     lwh = dims_maxes - dims_mins

        #     if (
        #         np.any(lwh > 30.0)
        #         or np.any(lwh < 0.05)
        #     ):
        #         continue            

        #     alpha_shape = AlphaShapeUtils.compute_alpha_shape(component_3d_points)

        #     if alpha_shape is None:
        #         continue

        #     centroid_3d = alpha_shape['centroid_3d']

        #     # print(f"{cmp_label=} {centroid_3d=}")

        #     num_no_box += 1
        #     vision_clusters.append(
        #         {
        #             "source": "traditional",
        #             "has_semantic_features": True,
        #             "semantic_features": uniform_semantic_features,
        #             "objectness_score": 0.0, 
        #             **alpha_shape,
        #         }
        #     )
            

        total = num_w_box + num_no_box
        print(f"{num_no_box=} ({num_no_box/total}) {num_w_box=} {num_w_box/total}")

        return vision_clusters

    def _get_vision_guided_clusters_separate(
        self,
        aggregated_points: np.ndarray,
        traditional_clusters,
        camera_name_timestamps: List[Tuple],
        timestamp_ns: int,
        camera_models: Dict[str, PinholeCamera],
        raster_ground_height_layer: GroundHeightLayer,
    ) -> List[Dict]:
        """
        Get vision-guided clusters by projecting LiDAR to cameras and filtering by OWLViT boxes.

        Returns:
            List of cluster dicts with 3D points, camera info, UV alpha shapes, and semantic features
        """
        vision_clusters = []

        log_dir = self.dataloader.dataset_dir / self.split / self.log_id
        sensor_dir = log_dir / "sensors"
        cameras_dir = sensor_dir / "cameras"

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


        is_ground = raster_ground_height_layer.get_ground_points_boolean(
            pcl_city_1
        ).astype(bool)
        print("is_ground", is_ground.shape, is_ground.sum())
        is_not_ground = ~is_ground

        pcl_city_1 = pcl_city_1[is_not_ground]
        pcl_ego = pcl_ego[is_not_ground]

        points = pcl_ego

        # points_orig_shape = points.shape
        # points = self.outline_estimator.remove_ground(points.copy())
        # is_ground = raster_ground_height_layer.get_ground_points_boolean(points).astype(bool)
        # is_not_ground = ~is_ground

        # points = points[is_not_ground]
        # print(f"before ground removal: {points_orig_shape} after: {points.shape}")

        # filter the camera images for ones with this sweep
        sweep_camera_name_timestamps = [
            (camera_name, cam_timestamp_ns)
            for camera_name, cam_timestamp_ns, sweep_timestamp_ns in camera_name_timestamps
            if sweep_timestamp_ns == timestamp_ns
        ]

        owlvit_pred_dir = self.owlvit_predictions_dir / self.log_id

        for camera_name, cam_timestamp_ns in tqdm(
            sweep_camera_name_timestamps,
            desc="_get_vision_guided_clusters: iterating over camera_predictions",
        ):  # TODO remove :30
            owlvit_pred_path = owlvit_pred_dir / f"{cam_timestamp_ns}_{camera_name}.pkl"

            with open(owlvit_pred_path, "rb") as f:
                prediction = pkl.load(f)

            pred_boxes = prediction["pred_boxes"]
            image_class_embeds = prediction["image_class_embeds"]
            objectness_scores = prediction["objectness_scores"]
            assert len(pred_boxes) == len(image_class_embeds) == len(objectness_scores)

            if len(pred_boxes) == 0:
                continue

            camera_model = camera_models[camera_name]
            city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam_timestamp_ns]

            if city_SE3_ego_cam_t is None or city_SE3_ego_lidar_t is None:
                continue

            (
                uv_points,
                _,
                is_valid_points,
            ) = camera_model.project_ego_to_img_motion_compensated(
                points,
                city_SE3_ego_cam_t=city_SE3_ego_cam_t,
                city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
            )

            if not np.any(is_valid_points):
                continue

            valid_uv = uv_points[is_valid_points]
            # valid_points_cam = points_cam[is_valid_points]
            valid_3d_points = points[is_valid_points]
            # valid_city_points = pcl_city_1[is_valid_points]

            # For each OWLViT detection box
            for box_idx, box, semantic_features, objectness_score in zip(
                np.arange(len(pred_boxes)),
                pred_boxes,
                image_class_embeds,
                objectness_scores,
            ):
                x1, y1, x2, y2 = box

                # Find points within this box
                in_box_mask = (
                    (valid_uv[:, 0] >= x1)
                    & (valid_uv[:, 0] <= x2)
                    & (valid_uv[:, 1] >= y1)
                    & (valid_uv[:, 1] <= y2)
                )

                if np.sum(in_box_mask) >= self.min_component_points:  # Need at least 3 points
                    box_3d_points = valid_3d_points[in_box_mask]
                    # box_city_points = valid_city_points[in_box_mask]
                    box_uv_points = valid_uv[in_box_mask]
                    # box_cam_points = valid_points_cam[in_box_mask]

                    # connected components
                    # t0 = time.time()
                    # component_labels, n_components = find_connected_components_lidar(box_3d_points)
                    # t1 = time.time()
                    # assert component_labels.min() >= 0 and component_labels.max() < n_components, f"{n_components=} {component_labels.min()} {component_labels.max()}"

                    # bench0 = t1 - t0

                    # filtered_component_labels = component_labels
                    unique_uv = box_uv_points[:, :2]

                    component_labels, n_components = fast_connected_components(
                        box_3d_points, eps=self.connected_components_eps
                    )
                    filtered_component_labels = component_labels

                    # rounded_uv = np.round(box_uv_points[:, :2]).astype(int)

                    # # Find unique (u,v) coordinates and their indices
                    # unique_uv, unique_indices = np.unique(
                    #     rounded_uv, axis=0, return_index=True
                    # )

                    # # Apply mask to keep only unique points
                    # filtered_component_labels = component_labels[unique_indices]

                    component_boxes = []
                    for i in range(n_components):
                        mask = filtered_component_labels == i

                        if mask.sum() == 0:
                            component_boxes.append(np.zeros((4,), dtype=np.float32))
                            continue

                        component_uv_points = unique_uv[mask]

                        xy1 = component_uv_points.min(axis=0)
                        xy2 = component_uv_points.max(axis=0)

                        component_boxes.append(
                            np.concatenate([xy1, xy2], axis=0).astype(np.float32)
                        )

                    component_boxes = np.stack(component_boxes, axis=0)

                    ious = box_iou(box.reshape(1, 4), component_boxes.reshape(-1, 4))
                    ious = ious[0, :]

                    valid_components = np.where(ious >= self.min_box_iou)[0]

                    for component in valid_components:
                        # revise with this component
                        in_component_mask = component_labels == component

                        if np.sum(in_box_mask) < self.min_component_points:  # Need at least 3 points
                            continue

                        component_3d_points = box_3d_points[in_component_mask].copy()

                        dims_mins = np.min(component_3d_points, axis=0)
                        dims_maxes = np.max(component_3d_points, axis=0)

                        lwh = dims_maxes - dims_mins

                        # if (
                        #     np.any(lwh > 30.0)
                        #     or np.prod(lwh) > 200
                        #     or np.prod(lwh) < 0.1
                        #     or np.any(lwh < 0.05)
                        # ):
                        #     continue

                        vision_clusters.append(
                            {
                                "original_points": component_3d_points,
                                # "uv_points": box_uv_points,
                                "camera_name": camera_name,
                                "camera_timestamp": timestamp_ns,
                                "box_2d": box,
                                "box_idx": box_idx,
                                "source": "vision_guided",
                                "semantic_features": semantic_features,  # CLIP embeddings from OWLViT
                                "objectness_score": objectness_score,  # Confidence score
                                "has_semantic_features": semantic_features is not None,
                            }
                        )

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
            new_points = voxel_sampling(cluster)
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

        for i in trange(min(1000, len(infos)), desc=f"Generating hybrid alpha shapes"):
            # 1. Load temporal LiDAR window
            aggregated_points = None
            aggregated_points, cur_points = self._load_temporal_lidar_window(i, infos)

            pose = infos[i]['pose']

            # 2. Load OWLViT predictions for all cameras (exhaustive)
            target_timestamp = infos[i].get(
                "timestamp_ns", None
            )  # Fallback if no timestamp
            assert target_timestamp is not None, infos[i].keys()

            # 3. Generate both cluster types
            # traditional_clusters = self._get_traditional_clusters(aggregated_points)
            traditional_clusters = []  # TODO: for testing
            if i == 0:
                pr = cProfile.Profile()
                pr.enable()
            vision_guided_clusters = self._get_vision_guided_clusters(
                aggregated_points,
                traditional_clusters,
                camera_name_timestamps,
                target_timestamp,
                camera_models,
                raster_ground_height_layer,
            )

            if i == 0:
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

            if i > 0:
                pr = cProfile.Profile()
                pr.enable()
            tracker.predict(target_timestamp)
            tracker.update(vision_guided_clusters, infos[i]['pose'], target_timestamp)
            if i > 0:
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


            fig, ax = plt.subplots(figsize=(8,8))

            lidar_points = np.load(lidar_path)[:, 0:3]
            lidar_points = points_rigid_transform(lidar_points, pose)

            ax.scatter(
                lidar_points[:, 0],
                lidar_points[:, 1],
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

                # Get rotated box corners
                corners = get_rotated_box(center_xy, length, width, yaw)

                color = "black"

                obj_polygon = patches.Polygon(
                    corners,
                    linewidth=1,
                    edgecolor=color,
                    facecolor="none",
                    alpha=0.5,
                    linestyle="-",
                )
                ax.add_patch(obj_polygon)

            for track in tracks:
                box = track.to_box()
                center_xy = box[:2]
                length = box[3]
                width = box[4]
                yaw = box[6]

                # Get rotated box corners
                corners = get_rotated_box(center_xy, length, width, yaw)

                corners3d = get_rotated_3d_box_corners(box)

                color = "yellow"
                alpha = 0.3
                if track.is_confirmed():
                    color = "green"
                    alpha = 0.7
                elif track.is_deleted():
                    color = "red"

                track_polygon = patches.Polygon(
                    corners,
                    linewidth=3,
                    edgecolor=color,
                    facecolor="none",
                    alpha=alpha,
                    linestyle="-",
                )
                ax.add_patch(track_polygon)

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
            infos[i]["tracked_objects"] = copy.deepcopy(tracker.tracks)


        non_none = set()
        for track in tqdm(tracker.tracks, desc='Adding tracks to output'):
            if track.optimized_boxes is None:
                print(f"optimized_boxes is none for track_id: {track.track_id}")
                continue

            non_none.add(track.track_id)

            for timestamp_ns, box, pose in zip(track.timestamps, track.optimized_boxes, track.optimized_poses):
                frame_id = timestamp_to_idx[timestamp_ns]

                ego_pose = infos[frame_id]['pose']
                world_to_ego = np.linalg.inv(ego_pose)

                world_points = points_rigid_transform(track.object_points, pose)
                ego_points = points_rigid_transform(world_points, world_to_ego)

                mesh = trimesh.convex.convex_hull(ego_points)

                # box = apply_pose_to_box(world_to_ego, box)
                print("world box", box)
                box = register_bbs(box.reshape(1, 7), world_to_ego)[0]
                print("ego box", box)

                infos[frame_id]["outline_box"].append(box)
                infos[frame_id]["outline_ids"].append(track.track_id)
                infos[frame_id]["outline_cls"].append(track.source)
                infos[frame_id]["outline_dif"].append(1)
                infos[frame_id]["alpha_shapes"].append({'vertices_3d': mesh.vertices})

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
            new_box_points = voxel_sampling(new_box_points)

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
