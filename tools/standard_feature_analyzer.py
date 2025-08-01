"""
Random Forest Feature Importance Analysis for Long-tail Detection - AV2 Integration
Analyzes which features best predict detection success/failure modes using interpretable ML
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from tqdm import tqdm
import multiprocessing as mp

from pprint import pprint

# ML imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import umap

import cProfile
import io
import pstats
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Import AV2 utilities
from av2.evaluation.detection.utils import (
    accumulate, assign, compute_affinity_matrix, distance,
    DetectionCfg, AffinityType, DistanceType
)
from av2.datasets.sensor.constants import RingCameras, StereoCameras
from av2.datasets.sensor.sensor_dataloader import SensorDataloader, SynchronizedSensorData
from av2.rendering.color import ColorFormats, create_range_map
from av2.rendering.rasterize import draw_points_xy_in_img
from av2.structures.sweep import Sweep
from av2.utils.io import read_city_SE3_ego
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.timestamped_image import TimestampedImage

from standalone_analyze_longtail import EvaluationConfig, StandaloneLongTailEvaluator


class ModifiedSensorDataloader(SensorDataloader):
    """Extended SensorDataloader with direct access methods."""
    
    def get_sensor_data(self, log_id: str, split: str, timestamp_ns: int, cam_names: Optional[List[str]] = None) -> SynchronizedSensorData:
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
            print(f"Warning: Log {log_id} not found in sensor cache. Using default sweep number.")
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
    test_size: float = 0.2
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
    center = np.array([row['tx_m'], row['ty_m'], row['tz_m']])
    dims = np.array([row['length_m'], row['width_m'], row['height_m']])
    
    # Extract quaternion and convert to rotation matrix
    quat = np.array([row['qx'], row['qy'], row['qz'], row['qw']])  # scipy expects x,y,z,w
    rotation = Rotation.from_quat(quat).as_matrix()
    
    # Create local bounding box corners (centered at origin)
    l, w, h = dims / 2
    corners_local = np.array([
        [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],  # bottom face
        [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]       # top face
    ])
    
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
        return {'occupied_voxels': set(), 'bounds': None, 'voxel_size': voxel_size}
    
    # Calculate voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Create set of occupied voxels for fast lookup
    occupied_voxels = set(map(tuple, voxel_indices))
    
    # Calculate bounds
    min_bounds = np.min(voxel_indices, axis=0) * voxel_size
    max_bounds = np.max(voxel_indices, axis=0) * voxel_size
    
    return {
        'occupied_voxels': occupied_voxels,
        'bounds': (min_bounds, max_bounds),
        'voxel_size': voxel_size
    }

def raycast_through_voxels(start: np.ndarray, end: np.ndarray, voxel_map: Dict) -> Tuple[bool, int]:
    """
    Cast a ray through voxel grid and count intersections.
    
    Args:
        start: Ray start point (3,)
        end: Ray end point (3,)
        voxel_map: Voxel occupancy map
        
    Returns:
        (is_occluded, num_intersections)
    """
    if not voxel_map['occupied_voxels']:
        return False, 0
    
    voxel_size = voxel_map['voxel_size']
    occupied_voxels = voxel_map['occupied_voxels']
    
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

def check_bbox_ray_intersection(ray_start: np.ndarray, ray_end: np.ndarray, 
                               bbox_corners: np.ndarray, margin: float = 0.1) -> bool:
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

def estimate_occlusion_raycasting_batch(dts: pd.DataFrame,
                                       ego_pos: np.ndarray,
                                       lidar_points: np.ndarray,
                                       camera_positions: Optional[Dict[str, np.ndarray]] = None,
                                       num_rays: int = 16,
                                       voxel_size: float = 0.15,
                                       min_distance_threshold: float = 2.0) -> Dict[int, Dict[str, float]]:
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
    object_centers = dts[['tx_m', 'ty_m', 'tz_m']].values
    object_bboxes = {}
    
    for idx, row in dts.iterrows():
        object_bboxes[idx] = create_bbox_from_dts_row(row)
    
    results = {}
    
    # Process each target object
    for target_idx, target_row in dts.iterrows():
        target_center = np.array([target_row['tx_m'], target_row['ty_m'], target_row['tz_m']])
        
        # Find potential occluding objects (closer to ego and within reasonable distance)
        target_distance = np.linalg.norm(target_center - ego_pos)
        
        # Filter for objects that could occlude this target
        occluder_candidates = []
        for other_idx, other_row in dts.iterrows():
            if other_idx == target_idx:
                continue
                
            other_center = np.array([other_row['tx_m'], other_row['ty_m'], other_row['tz_m']])
            other_distance = np.linalg.norm(other_center - ego_pos)
            center_distance = np.linalg.norm(other_center - target_center)
            
            # Only consider objects that are closer to ego and within reasonable distance of target
            if (other_distance < target_distance and 
                center_distance < min_distance_threshold * max(target_row['length_m'], target_row['width_m'])):
                occluder_candidates.append(object_bboxes[other_idx])
        
        # Generate ray endpoints around target object
        ray_endpoints = []
        
        # Primary ray to object center
        ray_endpoints.append(target_center)
        
        # Additional rays around object center (small sphere sampling)
        target_size = max(target_row['length_m'], target_row['width_m'], target_row['height_m'])
        ray_radius = target_size * 0.3  # Sample within 30% of object size
        
        for i in range(num_rays - 1):
            theta = 2 * np.pi * i / (num_rays - 1)
            phi = np.pi * (i % 4) / 8  # Vary elevation
            
            offset = ray_radius * np.array([
                np.cos(theta) * np.cos(phi),
                np.sin(theta) * np.cos(phi),
                np.sin(phi)
            ])
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
        camera_occlusion = {}
        if camera_positions:
            for cam_name, cam_pos in camera_positions.items():
                cam_occluded = 0
                for endpoint in ray_endpoints:
                    lidar_occ, _ = raycast_through_voxels(cam_pos, endpoint, voxel_map)
                    bbox_occ = any(check_bbox_ray_intersection(cam_pos, endpoint, bbox) 
                                  for bbox in occluder_candidates)
                    if lidar_occ or bbox_occ:
                        cam_occluded += 1
                camera_occlusion[cam_name] = cam_occluded / total_rays
        
        print("target_idx", target_idx)
        results[target_idx] = {
            'overall_occlusion': occlusion_ratio,
            'lidar_occlusion': lidar_occlusion_ratio,
            'bbox_occlusion': bbox_occlusion_ratio,
            'camera_occlusion': camera_occlusion,
            'avg_intersections_per_ray': total_intersections / total_rays,
            'num_occluder_candidates': len(occluder_candidates),
            'distance_to_ego': target_distance,
            'confidence': min(1.0, len(lidar_points) / 1000.0)
        }
    
    return results

def estimate_occlusion_simple_batch(dts: pd.DataFrame, 
                                   target_datum,
                                   num_rays: int = 12,
                                   voxel_size: float = 0.25) -> Dict[int, float]:
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
    camera_positions = extract_camera_positions(target_datum)
    
    analysis_results = estimate_occlusion_raycasting_batch(
        dts=dts,
        ego_pos=ego_pos,
        lidar_points=lidar_points,
        camera_positions=camera_positions,
        num_rays=num_rays,
        voxel_size=voxel_size
    )
    
    # Extract just the overall occlusion values
    return {idx: result['overall_occlusion'] for idx, result in analysis_results.items()}

def _load_sweep_data_and_occlusions(args: Tuple[str, int, pd.DataFrame, Path]) -> Tuple[str, int, Dict[int, float]]:
    """
    Load sweep data and compute occlusions for a single sweep.
    This runs in a separate process to parallelize I/O and computation.
    """
    log_id, timestamp_ns, sweep_dts, dataset_dir = args
    
    # # Recreate dataloader in this process
    # dataloader = ModifiedSensorDataloader(dataset_dir=dataset_dir.parent)
    # split = dataset_dir.name
    
    # # Load sensor data and compute occlusions
    # target_datum = dataloader.get_sensor_data(log_id, split, timestamp_ns, cam_names=[])
    # occlusions = estimate_occlusion_simple_batch(sweep_dts, target_datum)
    
    occlusions = {}
    for target_idx, target_row in sweep_dts.iterrows():
        occlusions[target_idx] = 0

    return log_id, timestamp_ns, occlusions


class RandomForestFeatureAnalyzer:
    """
    Random Forest-based feature importance analysis for AV2 detection results.
    
    Analyzes which object and scene features are most predictive of:
    - True Positives vs False Positives
    - Detection vs Miss
    - Error magnitude (ATE, ASE, AOE)
    - Class-specific failure modes
    """
    
    def __init__(self, config: FeatureAnalysisConfig = None):
        self.config = config or FeatureAnalysisConfig()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def extract_av2_features(self, 
                        eval_dts: pd.DataFrame, 
                        eval_gts: pd.DataFrame,
                        cfg: DetectionCfg,
                        ego_poses: Optional[pd.DataFrame] = None,
                        num_processes: Optional[int] = 1,
                        enable_profiling: bool = True,
                        benchmark_every: int = 100) -> pd.DataFrame:
        """
        Optimized and benchmarked AV2 feature extraction with performance monitoring.
        
        Args:
            enable_profiling: If True, runs cProfile on feature extraction
            benchmark_every: Print timing stats every N iterations
        """
        print("Extracting comprehensive features from AV2 results (optimized + benchmarked)...")
        
        primary_threshold_idx = len(cfg.affinity_thresholds_m) // 2
        primary_threshold = cfg.affinity_thresholds_m[primary_threshold_idx]
        threshold_col = str(primary_threshold)

        assert cfg.dataset_dir is not None
        split = cfg.dataset_dir.name
        assert split in ["train", "val", "test"], f"{split=} is not valid!"

        # Optimized grouping using MultiIndex
        print("Creating optimized indexes...")
        eval_dts_indexed = eval_dts.set_index(['log_id', 'timestamp_ns'])
        eval_gts_indexed = eval_gts.set_index(['log_id', 'timestamp_ns'])
        unique_keys = eval_dts_indexed.index.unique()

        # lets remove some aha #################################################
        valid_uuids_gts = list(set(eval_gts_indexed.index.tolist()))[:150]

        eval_gts_indexed = eval_gts_indexed.loc[list(valid_uuids_gts)].sort_index()
        eval_dts_indexed = eval_dts_indexed.loc[list(valid_uuids_gts)].sort_index()

        unique_keys = eval_dts_indexed.index.unique()
        ########################################################################
        
        # Prepare arguments
        sweep_args = []
        sweep_lookup = {}
        
        print("Building sweep arguments with indexed lookup...")
        for key in tqdm(unique_keys, desc="Extracting sweep args"):
            sweep_dts = eval_dts_indexed.loc[[key]] if key in eval_dts_indexed.index else pd.DataFrame()
            sweep_gts = eval_gts_indexed.loc[[key]] if key in eval_gts_indexed.index else pd.DataFrame()
            
            log_id, timestamp_ns = key
            sweep_args.append((log_id, timestamp_ns, sweep_dts.reset_index(), cfg.dataset_dir))
            sweep_lookup[key] = (sweep_dts.reset_index(), sweep_gts.reset_index())

        print(f"Loading sensor data and computing occlusions for {len(sweep_args)} sweeps using {num_processes or mp.cpu_count()} processes...")

        if num_processes is None:
            num_processes = mp.cpu_count()
        
        with mp.Pool(processes=num_processes) as pool:
            occlusion_results = pool.map(_load_sweep_data_and_occlusions, sweep_args)
        
        occlusions_by_sweep = {(log_id, timestamp_ns): occlusions 
                            for log_id, timestamp_ns, occlusions in occlusion_results}

        print("Extracting features from loaded data...")
        
        # Benchmarking setup
        timing_stats = defaultdict(list)
        features_list = []
        
        total_start_time = time.time()
        
        # Optional profiling
        if enable_profiling:
            profiler = cProfile.Profile()
            profiler.enable()
        
        for i, (key, (sweep_dts, sweep_gts)) in enumerate(tqdm(sweep_lookup.items(), desc="Extracting features")):
            sweep_start_time = time.time()
            
            log_id, timestamp_ns = key
            
            # Time occlusion lookup
            occlusion_start = time.time()
            occlusions = occlusions_by_sweep[(log_id, timestamp_ns)]
            timing_stats['occlusion_lookup'].append(time.time() - occlusion_start)
            
            # Time detection feature extraction
            detection_start = time.time()
            detection_count = 0
            for idx, dt in sweep_dts.iterrows():
                single_det_start = time.time()
                
                features = self._extract_single_detection_features(
                    dt, sweep_dts, sweep_gts, cfg, threshold_col, ego_poses
                )
                features['sample_type'] = 'detection'
                features['outcome'] = 'TP' if (threshold_col in dt and dt[threshold_col] == 1) else 'FP'
                
                # Optimize index lookup
                detection_idx = detection_count  # Use counter instead of list.index()
                features['occlusion_level_estimate'] = occlusions.get(detection_idx, 0.0)
                
                features_list.append(features)
                detection_count += 1
                
                timing_stats['single_detection'].append(time.time() - single_det_start)
            
            timing_stats['all_detections'].append(time.time() - detection_start)
            
            # Time ground truth feature extraction
            gt_start = time.time()
            missed_gts = sweep_gts[sweep_gts.get('is_matched', pd.Series(False, index=sweep_gts.index)) != True]
            
            gt_filter_time = time.time() - gt_start
            timing_stats['gt_filtering'].append(gt_filter_time)
            
            gt_extraction_start = time.time()
            for idx, gt in missed_gts.iterrows():
                single_gt_start = time.time()
                
                features = self._extract_single_gt_features(
                    gt, sweep_dts, sweep_gts, cfg, ego_poses
                )
                features['sample_type'] = 'ground_truth'  
                features['outcome'] = 'FN'
                features_list.append(features)
                
                timing_stats['single_gt'].append(time.time() - single_gt_start)
            
            timing_stats['all_gts'].append(time.time() - gt_extraction_start)
            timing_stats['sweep_total'].append(time.time() - sweep_start_time)
            
            # Print benchmark stats periodically
            if (i + 1) % benchmark_every == 0:
                current_time = time.time()
                elapsed = current_time - total_start_time
                rate = (i + 1) / elapsed
                eta_seconds = (len(sweep_lookup) - (i + 1)) / rate
                
                print(f"\n=== BENCHMARK STATS (after {i+1}/{len(sweep_lookup)} sweeps) ===")
                print(f"Current rate: {rate:.2f} sweeps/sec")
                print(f"ETA: {eta_seconds/3600:.1f} hours ({eta_seconds/60:.1f} minutes)")
                
                # Average timings per operation
                if timing_stats['single_detection']:
                    avg_det = sum(timing_stats['single_detection']) / len(timing_stats['single_detection'])
                    print(f"Avg detection feature extraction: {avg_det*1000:.2f}ms")
                
                if timing_stats['single_gt']:
                    avg_gt = sum(timing_stats['single_gt']) / len(timing_stats['single_gt'])
                    print(f"Avg GT feature extraction: {avg_gt*1000:.2f}ms")
                
                if timing_stats['sweep_total']:
                    avg_sweep = sum(timing_stats['sweep_total']) / len(timing_stats['sweep_total'])
                    print(f"Avg sweep total time: {avg_sweep*1000:.2f}ms")
                
                # Identify bottlenecks
                total_det_time = sum(timing_stats['all_detections'])
                total_gt_time = sum(timing_stats['all_gts'])
                total_occlusion_time = sum(timing_stats['occlusion_lookup'])
                
                print(f"Time breakdown: Detection={total_det_time:.1f}s, GT={total_gt_time:.1f}s, Occlusion={total_occlusion_time:.1f}s")
                print("=" * 60)

                # Final profiling results
                if enable_profiling:
                    profiler.disable()
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                    ps.print_stats(20)  # Top 20 functions
                    print("\n=== PROFILING RESULTS ===")
                    print(s.getvalue())

                    profiler.enable()

            else:
                print(f"{i=} {benchmark_every=}")
        
        # Final profiling results
        if enable_profiling:
            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            print("\n=== PROFILING RESULTS ===")
            print(s.getvalue())
        
        # Final timing summary
        total_time = time.time() - total_start_time
        print(f"\n=== FINAL TIMING SUMMARY ===")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average rate: {len(sweep_lookup)/total_time:.2f} sweeps/sec")
        
        if timing_stats['single_detection']:
            total_detections = len(timing_stats['single_detection'])
            avg_det_time = sum(timing_stats['single_detection']) / total_detections
            print(f"Processed {total_detections} detections, avg {avg_det_time*1000:.2f}ms each")
        
        if timing_stats['single_gt']:
            total_gts = len(timing_stats['single_gt'])
            avg_gt_time = sum(timing_stats['single_gt']) / total_gts
            print(f"Processed {total_gts} ground truths, avg {avg_gt_time*1000:.2f}ms each")
        
        features_df = pd.DataFrame(features_list)
        print(f"Extracted {len(features_df)} feature vectors with {len(features_df.columns)} features")
        
        return features_df
    
    def _extract_single_detection_features(self, 
                                         detection: pd.Series,
                                         sweep_dts: pd.DataFrame,
                                         sweep_gts: pd.DataFrame,
                                         cfg: DetectionCfg,
                                         threshold_col: str,
                                         ego_poses: Optional[pd.DataFrame]) -> Dict:
        """Extract features for a single detection."""
        features = {}

        profiler = cProfile.Profile()
        profiler.enable()
        
        # Basic object properties
        features['category'] = detection['category']
        features['log_id'] = detection['log_id'] 
        features['timestamp_ns'] = detection['timestamp_ns']
        
        # Geometric features
        features['length_m'] = detection.get('length_m', 0)
        features['width_m'] = detection.get('width_m', 0)
        features['height_m'] = detection.get('height_m', 0)
        features['aspect_ratio_lw'] = features['length_m'] / max(features['width_m'], 0.01)
        features['aspect_ratio_lh'] = features['length_m'] / max(features['height_m'], 0.01)
        features['volume'] = features['length_m'] * features['width_m'] * features['height_m']
        
        # Position and distance
        features['tx_m'] = detection.get('tx_m', 0)
        features['ty_m'] = detection.get('ty_m', 0) 
        features['tz_m'] = detection.get('tz_m', 0)
        features['distance_to_ego'] = np.sqrt(features['tx_m']**2 + features['ty_m']**2)
        features['height_above_ground'] = features['tz_m']
        
        # Orientation (if available)
        if 'qw' in detection and 'qz' in detection:
            # Convert quaternion to yaw angle
            qw, qx, qy, qz = detection['qw'], detection.get('qx', 0), detection.get('qy', 0), detection['qz']
            features['orientation_yaw'] = 2 * np.arctan2(qz, qw)
        else:
            features['orientation_yaw'] = 0
            
        # Detection quality metrics
        features['detection_score'] = detection.get('score', 0)
        features['ATE'] = detection.get('translation_error', 0)
        features['ASE'] = detection.get('scale_error', 0) 
        features['AOE'] = detection.get('orientation_error', 0)
        features['is_evaluated'] = detection.get('is_evaluated', False)
        
        # Contextual features
        # features['num_nearby_objects'] = self._count_nearby_objects(
        #     detection, sweep_dts, radius=5.0
        # )
        # features['num_same_class_nearby'] = self._count_nearby_objects(
        #     detection, sweep_dts[sweep_dts['category'] == detection['category']], radius=5.0
        # )
        # features['num_all_objects_nearby'] = self._count_nearby_objects(
        #     detection, pd.concat([sweep_dts, sweep_gts]), radius=5.0
        # )
        
        # Occlusion estimation
        features['occlusion_level_estimate'] = 0
        # features['occlusion_level_estimate'] = self._estimate_occlusion(
        #     detection, sweep_gts
        # )
        
        # Category frequency (how common is this class)
        all_categories = pd.concat([sweep_dts['category'], sweep_gts['category']])
        features['category_freq'] = (all_categories == detection['category']).sum() / len(all_categories)
        
        # Time-based features  
        features['time_of_day'] = self._extract_time_of_day(detection['timestamp_ns'])
        
        # Ego speed (if poses available)
        if ego_poses is not None:
            features['ego_speed'] = self._get_ego_speed(detection, ego_poses)
        else:
            features['ego_speed'] = 0
            
        # Detection consistency (simplified - would need tracking info)
        features['was_detected_last_frame'] = 0  # Placeholder
        
        return features
    
    def _extract_single_gt_features(self,
                                   ground_truth: pd.Series,
                                   sweep_dts: pd.DataFrame, 
                                   sweep_gts: pd.DataFrame,
                                   cfg: DetectionCfg,
                                   ego_poses: Optional[pd.DataFrame]) -> Dict:
        """Extract features for a missed ground truth (False Negative)."""
        features = {}
        
        # Basic object properties
        features['category'] = ground_truth['category']
        features['log_id'] = ground_truth['log_id']
        features['timestamp_ns'] = ground_truth['timestamp_ns']
        
        # Geometric features
        features['length_m'] = ground_truth.get('length_m', 0)
        features['width_m'] = ground_truth.get('width_m', 0)
        features['height_m'] = ground_truth.get('height_m', 0)
        features['aspect_ratio_lw'] = features['length_m'] / max(features['width_m'], 0.01)
        features['aspect_ratio_lh'] = features['length_m'] / max(features['height_m'], 0.01)
        features['volume'] = features['length_m'] * features['width_m'] * features['height_m']
        
        # Position and distance  
        features['tx_m'] = ground_truth.get('tx_m', 0)
        features['ty_m'] = ground_truth.get('ty_m', 0)
        features['tz_m'] = ground_truth.get('tz_m', 0)
        features['distance_to_ego'] = np.sqrt(features['tx_m']**2 + features['ty_m']**2)
        features['height_above_ground'] = features['tz_m']
        
        # Orientation
        if 'qw' in ground_truth and 'qz' in ground_truth:
            qw, qx, qy, qz = ground_truth['qw'], ground_truth.get('qx', 0), ground_truth.get('qy', 0), ground_truth['qz']
            features['orientation_yaw'] = 2 * np.arctan2(qz, qw)
        else:
            features['orientation_yaw'] = 0
            
        # No detection quality metrics for FN (they weren't detected)
        features['detection_score'] = 0
        features['ATE'] = np.inf  # Infinite error since not detected
        features['ASE'] = np.inf
        features['AOE'] = np.inf
        features['is_evaluated'] = ground_truth.get('is_evaluated', False)
        
        # Contextual features
        # features['num_nearby_objects'] = self._count_nearby_objects(
        #     ground_truth, sweep_gts, radius=5.0
        # )
        # features['num_same_class_nearby'] = self._count_nearby_objects(
        #     ground_truth, sweep_gts[sweep_gts['category'] == ground_truth['category']], radius=5.0
        # )
        # features['num_all_objects_nearby'] = self._count_nearby_objects(
        #     ground_truth, pd.concat([sweep_dts, sweep_gts]), radius=5.0
        # )
        
        # Occlusion estimation
        # features['occlusion_level_estimate'] = self._estimate_occlusion(
        #     ground_truth, sweep_gts
        # )
        
        # Category frequency
        all_categories = pd.concat([sweep_dts['category'], sweep_gts['category']])
        features['category_freq'] = (all_categories == ground_truth['category']).sum() / len(all_categories)
        
        # Time-based features
        features['time_of_day'] = self._extract_time_of_day(ground_truth['timestamp_ns'])
        
        # Ego speed
        if ego_poses is not None:
            features['ego_speed'] = self._get_ego_speed(ground_truth, ego_poses)
        else:
            features['ego_speed'] = 0
            
        features['was_detected_last_frame'] = 0  # Placeholder
        
        return features
    
    def _count_nearby_objects(self, obj: pd.Series, objects: pd.DataFrame, radius: float = 5.0) -> int:
        """Count objects within radius of given object."""
        if len(objects) == 0:
            return 0
            
        obj_pos = np.array([obj.get('tx_m', 0), obj.get('ty_m', 0)])
        other_pos = objects[['tx_m', 'ty_m']].values
        
        distances = np.linalg.norm(other_pos - obj_pos, axis=1)
        return np.sum((distances < radius) & (distances > 0.1))  # Exclude self/very close
    
    def _estimate_occlusion(self, obj: pd.Series, other_objects: pd.DataFrame) -> float:
        """
        Estimate occlusion level using bounding box overlap heuristic.
        Returns value between 0 (no occlusion) and 1 (heavily occluded).
        """
        if len(other_objects) == 0:
            return 0.0
            
        # Simple 2D bounding box overlap approach
        obj_x, obj_y = obj.get('tx_m', 0), obj.get('ty_m', 0)
        obj_l, obj_w = obj.get('length_m', 1), obj.get('width_m', 1)
        
        # Count overlapping objects within 2m distance (closer objects more likely to occlude)
        close_objects = other_objects[
            np.sqrt((other_objects['tx_m'] - obj_x)**2 + (other_objects['ty_m'] - obj_y)**2) < 2.0
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
        Train Random Forest models for different prediction tasks.
        
        Returns dictionary of trained models and feature importance results.
        """
        print("Training Random Forest models...")
        
        results = {}
        
        # Prepare feature matrix
        feature_cols = [col for col in features_df.columns 
                       if col not in ['sample_type', 'outcome', 'log_id', 'timestamp_ns', 'category']]
        
        # Encode categorical features
        X_processed = features_df[feature_cols].copy()
        categorical_cols = ['category'] if 'category' in feature_cols else []
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
            else:
                X_processed[col] = self.label_encoders[col].transform(features_df[col].astype(str))
        
        # Handle infinite values (from FN samples)
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        X_processed = X_processed.fillna(X_processed.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_processed)
        X_df = pd.DataFrame(X_scaled, columns=feature_cols, index=features_df.index)
        
        # 1. TP vs FP classification (only for detections)
        detection_mask = features_df['sample_type'] == 'detection'
        if detection_mask.sum() > self.config.min_samples_per_class:
            results['tp_vs_fp'] = self._train_classification_model(
                X_df[detection_mask], 
                features_df.loc[detection_mask, 'outcome'],
                'TP vs FP Classification',
                feature_cols
            )
        
        # 2. Detection vs Miss (TP+FP vs FN)
        detection_success = features_df['outcome'].map({'TP': 1, 'FP': 1, 'FN': 0})
        results['detection_vs_miss'] = self._train_classification_model(
            X_df, detection_success, 'Detection vs Miss', feature_cols
        )
        
        # 3. Error magnitude regression (only for TPs)
        tp_mask = features_df['outcome'] == 'TP'
        if tp_mask.sum() > self.config.min_samples_per_class:
            # ATE regression
            valid_ate = features_df.loc[tp_mask, 'ATE'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_ate) > 10:
                results['ate_regression'] = self._train_regression_model(
                    X_df.loc[valid_ate.index], valid_ate, 'ATE Regression', feature_cols
                )
            
            # ASE regression 
            valid_ase = features_df.loc[tp_mask, 'ASE'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_ase) > 10:
                results['ase_regression'] = self._train_regression_model(
                    X_df.loc[valid_ase.index], valid_ase, 'ASE Regression', feature_cols
                )
        
        # 4. Per-category analysis
        results['per_category'] = {}
        for category in features_df['category'].unique():
            cat_mask = features_df['category'] == category
            cat_data = features_df[cat_mask]
            
            if len(cat_data) < self.config.min_samples_per_class:
                continue
                
            # Category-specific detection success
            cat_success = cat_data['outcome'].map({'TP': 1, 'FP': 1, 'FN': 0})
            if len(cat_success.unique()) > 1:  # Need both classes
                results['per_category'][category] = self._train_classification_model(
                    X_df[cat_mask], cat_success, f'{category} Detection Success', feature_cols
                )
        
        return results
    
    def _train_classification_model(self, X: pd.DataFrame, y: pd.Series, 
                                  model_name: str, feature_names: List[str]) -> Dict:
        """Train a Random Forest classifier and extract insights."""
        if len(y.unique()) < 2:
            return {'error': f'Insufficient class diversity for {model_name}'}
            
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_seed,
            stratify=y if len(y.unique()) > 1 else None
        )
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_seed,
            class_weight='balanced'
        )
        
        rf.fit(X_train, y_train)
        
        # Predictions and metrics
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': rf,
            'feature_importance': importance_df,
            'test_accuracy': rf.score(X_test, y_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model_name': model_name,
            'n_samples': len(X),
            'n_features': len(feature_names)
        }
    
    def _train_regression_model(self, X: pd.DataFrame, y: pd.Series,
                              model_name: str, feature_names: List[str]) -> Dict:
        """Train a Random Forest regressor and extract insights."""
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_seed
        )
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_seed
        )
        
        rf.fit(X_train, y_train)
        
        # Predictions and metrics
        y_pred = rf.predict(X_test)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': rf,
            'feature_importance': importance_df,
            'test_r2': rf.score(X_test, y_test),
            'test_rmse': np.sqrt(np.mean((y_test - y_pred) ** 2)),
            'model_name': model_name,
            'n_samples': len(X),
            'n_features': len(feature_names)
        }
    
    def perform_error_clustering(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform clustering analysis to discover failure modes.
        """
        print("Performing error clustering analysis...")
        
        # Prepare error embedding features 
        error_features = ['ATE', 'ASE', 'AOE', 'distance_to_ego', 'occlusion_level_estimate',
                         'volume', 'num_nearby_objects', 'category_freq']
        
        available_features = [f for f in error_features if f in features_df.columns]
        
        if len(available_features) < 3:
            return {'error': 'Insufficient features for clustering'}
        
        # Use only samples with finite error values (exclude FN with inf errors)
        finite_mask = np.isfinite(features_df['ATE']) & np.isfinite(features_df['ASE'])
        cluster_data = features_df[finite_mask].copy()
        
        if len(cluster_data) < 50:
            return {'error': 'Insufficient data for clustering'}
        
        # Prepare feature matrix
        X_cluster = cluster_data[available_features].copy()
        
        # Encode categoricals
        if 'category' in X_cluster.columns:
            le = LabelEncoder()
            X_cluster['category'] = le.fit_transform(X_cluster['category'].astype(str))
        
        # Scale features
        X_scaled = StandardScaler().fit_transform(X_cluster.fillna(X_cluster.median()))
        
        results = {}
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=self.config.dbscan_eps, min_samples=self.config.dbscan_min_samples)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        results['dbscan'] = {
            'labels': dbscan_labels,
            'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'n_noise': list(dbscan_labels).count(-1),
            'silhouette_score': self._safe_silhouette_score(X_scaled, dbscan_labels)
        }
        
        # Gaussian Mixture Model
        gmm = GaussianMixture(n_components=self.config.n_gmm_components, random_state=self.config.random_seed)
        gmm_labels = gmm.fit_predict(X_scaled)
        
        results['gmm'] = {
            'labels': gmm_labels,
            'n_clusters': self.config.n_gmm_components,
            'bic': gmm.bic(X_scaled),
            'aic': gmm.aic(X_scaled),
            'silhouette_score': self._safe_silhouette_score(X_scaled, gmm_labels)
        }
        
        # Analyze clusters
        results['cluster_analysis'] = self._analyze_clusters(cluster_data, dbscan_labels, 'DBSCAN')
        results['feature_names'] = available_features
        results['cluster_data_indices'] = cluster_data.index.tolist()
        
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
    
    def _analyze_clusters(self, data: pd.DataFrame, labels: np.ndarray, method: str) -> Dict:
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
                
            analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'mean_ate': cluster_data['ATE'].mean() if 'ATE' in cluster_data else np.nan,
                'mean_distance': cluster_data['distance_to_ego'].mean() if 'distance_to_ego' in cluster_data else np.nan,
                'dominant_category': cluster_data['category'].mode().iloc[0] if 'category' in cluster_data else 'unknown',
                'mean_occlusion': cluster_data['occlusion_level_estimate'].mean() if 'occlusion_level_estimate' in cluster_data else np.nan,
                'outcome_distribution': cluster_data['outcome'].value_counts().to_dict() if 'outcome' in cluster_data else {}
            }
        
        return analysis
    
    def create_dimensionality_reduction_viz(self, features_df: pd.DataFrame, 
                                          clustering_results: Dict = None) -> Dict[str, Any]:
        """
        Create t-SNE and UMAP visualizations of the feature space.
        """
        print("Creating dimensionality reduction visualizations...")
        
        # Prepare features for visualization
        viz_features = ['ATE', 'ASE', 'AOE', 'distance_to_ego', 'occlusion_level_estimate',
                       'volume', 'num_nearby_objects', 'detection_score']
        
        available_features = [f for f in viz_features if f in features_df.columns]
        
        # Use finite data only
        finite_mask = np.isfinite(features_df['ATE']) & np.isfinite(features_df['ASE'])
        viz_data = features_df[finite_mask].copy()
        
        if len(viz_data) < 50:
            return {'error': 'Insufficient data for visualization'}
        
        # Prepare feature matrix
        X_viz = viz_data[available_features].fillna(viz_data[available_features].median())
        X_scaled = StandardScaler().fit_transform(X_viz)
        
        results = {}
        
        # t-SNE
        try:
            tsne = TSNE(n_components=2, perplexity=min(self.config.tsne_perplexity, len(X_scaled)//4),
                       random_state=self.config.random_seed)
            tsne_embedding = tsne.fit_transform(X_scaled)
            results['tsne'] = {
                'embedding': tsne_embedding,
                'feature_names': available_features
            }
        except Exception as e:
            print(f"t-SNE failed: {e}")
            results['tsne'] = {'error': str(e)}
        
        # UMAP
        try:
            umap_reducer = umap.UMAP(
                n_neighbors=min(self.config.umap_n_neighbors, len(X_scaled)-1),
                min_dist=self.config.umap_min_dist,
                random_state=self.config.random_seed
            )
            umap_embedding = umap_reducer.fit_transform(X_scaled)
            results['umap'] = {
                'embedding': umap_embedding,
                'feature_names': available_features
            }
        except Exception as e:
            print(f"UMAP failed: {e}")
            results['umap'] = {'error': str(e)}
        
        outcomes = viz_data['outcome'].values
        print('outcomes', outcomes)
        print('outcomes', outcomes.shape)
        print('outcomes', outcomes.min(), outcomes.max())
        # Store metadata for coloring
        results['metadata'] = {
            'outcomes': viz_data['outcome'].values,
            'categories': viz_data['category'].values,
            'distances': viz_data['distance_to_ego'].values,
            'scores': viz_data.get('detection_score', pd.Series(0, index=viz_data.index)).values,
            'indices': viz_data.index.tolist()
        }
        
        # Add clustering labels if available
        if clustering_results and 'cluster_data_indices' in clustering_results:
            cluster_indices = clustering_results['cluster_data_indices']
            # Match indices
            cluster_labels = np.full(len(viz_data), -1)
            for i, idx in enumerate(viz_data.index):
                if idx in cluster_indices:
                    pos = cluster_indices.index(idx)
                    cluster_labels[i] = clustering_results['dbscan']['labels'][pos]
            results['metadata']['cluster_labels'] = cluster_labels
        
        return results
    
    def train_interpretable_trees(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train simple decision trees to extract interpretable rules.
        """
        print("Training interpretable decision trees...")
        
        results = {}
        
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['sample_type', 'outcome', 'log_id', 'timestamp_ns']]
        
        X_processed = features_df[feature_cols].copy()
        
        # Encode categoricals
        for col in ['category']:
            if col in X_processed.columns:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Handle infinite values
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        X_processed = X_processed.fillna(X_processed.median())
        
        # 1. Detection success tree
        detection_success = features_df['outcome'].map({'TP': 1, 'FP': 1, 'FN': 0})
        
        tree_clf = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=self.config.random_seed
        )
        tree_clf.fit(X_processed, detection_success)
        
        results['detection_success_tree'] = {
            'model': tree_clf,
            'rules': export_text(tree_clf, feature_names=feature_cols, max_depth=5),
            'feature_importance': pd.DataFrame({
                'feature': feature_cols,
                'importance': tree_clf.feature_importances_
            }).sort_values('importance', ascending=False),
            'accuracy': tree_clf.score(X_processed, detection_success)
        }
        
        # 2. TP vs FP tree (for detections only)
        detection_mask = features_df['sample_type'] == 'detection'
        if detection_mask.sum() > 20:
            tp_fp_labels = (features_df.loc[detection_mask, 'outcome'] == 'TP').astype(int)
            
            tree_tp_fp = DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=15, random_state=self.config.random_seed
            )
            tree_tp_fp.fit(X_processed[detection_mask], tp_fp_labels)
            
            results['tp_vs_fp_tree'] = {
                'model': tree_tp_fp,
                'rules': export_text(tree_tp_fp, feature_names=feature_cols, max_depth=4),
                'feature_importance': pd.DataFrame({
                    'feature': feature_cols,
                    'importance': tree_tp_fp.feature_importances_
                }).sort_values('importance', ascending=False),
                'accuracy': tree_tp_fp.score(X_processed[detection_mask], tp_fp_labels)
            }
        
        return results
    
    def create_comprehensive_visualizations(self, 
                                          rf_results: Dict,
                                          clustering_results: Dict,
                                          dimred_results: Dict,
                                          tree_results: Dict,
                                          output_path: str):
        """Create comprehensive visualization plots."""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Feature importance comparison (top subplot)
        ax1 = plt.subplot(3, 4, (1, 2))
        self._plot_feature_importance_comparison(rf_results, ax1)
        
        # 2. t-SNE visualization
        ax2 = plt.subplot(3, 4, 3)
        if 'tsne' in dimred_results and 'error' not in dimred_results['tsne']:
            self._plot_tsne(dimred_results, ax2)
        
        # 3. UMAP visualization  
        ax3 = plt.subplot(3, 4, 4)
        if 'umap' in dimred_results and 'error' not in dimred_results['umap']:
            self._plot_umap(dimred_results, ax3)
            
        # 4. Clustering analysis
        ax4 = plt.subplot(3, 4, (5, 6))
        if 'dbscan' in clustering_results:
            self._plot_clustering_analysis(clustering_results, ax4)
        
        # 5. Error distribution by distance
        ax5 = plt.subplot(3, 4, 7)
        self._plot_error_by_distance(rf_results, ax5)
        
        # 6. Category-wise performance
        ax6 = plt.subplot(3, 4, 8)
        if 'per_category' in rf_results:
            self._plot_category_performance(rf_results['per_category'], ax6)
        
        # 7-8. Decision tree visualization
        ax7 = plt.subplot(3, 4, (9, 10))
        if 'detection_success_tree' in tree_results:
            self._plot_tree_importance(tree_results['detection_success_tree'], ax7)
            
        # 9-10. Model performance comparison
        ax8 = plt.subplot(3, 4, (11, 12))
        self._plot_model_performance_comparison(rf_results, ax8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance_comparison(self, rf_results: Dict, ax):
        """Plot feature importance comparison across models."""
        if not rf_results:
            ax.text(0.5, 0.5, 'No RF results available', ha='center', va='center')
            return
            
        # Collect top features from each model
        all_importance = {}
        
        for model_name, results in rf_results.items():
            if 'feature_importance' in results:
                top_features = results['feature_importance'].head(10)
                for _, row in top_features.iterrows():
                    feature = row['feature']
                    importance = row['importance']
                    if feature not in all_importance:
                        all_importance[feature] = {}
                    all_importance[feature][model_name] = importance
        
        # Create comparison plot
        if all_importance:
            feature_df = pd.DataFrame(all_importance).T.fillna(0)
            feature_df.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Feature Importance Comparison')
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_tsne(self, dimred_results: Dict, ax):
        """Plot t-SNE embedding."""
        tsne_data = dimred_results['tsne']['embedding']
        metadata = dimred_results['metadata']
        
        scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1], 
                           c=metadata['distances'], cmap='viridis', alpha=0.6, s=20)
        ax.set_title('t-SNE (colored by distance)')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax, label='Distance to Ego')
    
    def _plot_umap(self, dimred_results: Dict, ax):
        """Plot UMAP embedding."""
        umap_data = dimred_results['umap']['embedding']
        metadata = dimred_results['metadata']
        
        # Color by outcome
        outcome_colors = {'TP': 'green', 'FP': 'red', 'FN': 'blue'}
        colors = [outcome_colors.get(outcome, 'gray') for outcome in metadata['outcomes']]
        
        ax.scatter(umap_data[:, 0], umap_data[:, 1], c=colors, alpha=0.6, s=20)
        ax.set_title('UMAP (colored by outcome)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        # Legend
        for outcome, color in outcome_colors.items():
            ax.scatter([], [], c=color, label=outcome)
        ax.legend()
    
    def _plot_clustering_analysis(self, clustering_results: Dict, ax):
        """Plot clustering analysis."""
        if 'cluster_analysis' in clustering_results:
            analysis = clustering_results['cluster_analysis']
            
            cluster_ids = []
            cluster_sizes = []
            mean_ates = []
            
            for cluster_name, stats in analysis.items():
                if cluster_name.startswith('cluster_'):
                    cluster_ids.append(cluster_name)
                    cluster_sizes.append(stats['size'])
                    mean_ates.append(stats.get('mean_ate', 0))
            
            if cluster_ids:
                bars = ax.bar(range(len(cluster_ids)), cluster_sizes, alpha=0.7)
                ax.set_title('Cluster Sizes (DBSCAN)')
                ax.set_xlabel('Cluster ID')
                ax.set_ylabel('Number of Samples')
                ax.set_xticks(range(len(cluster_ids)))
                ax.set_xticklabels([c.replace('cluster_', '') for c in cluster_ids])
    
    def _plot_error_by_distance(self, rf_results: Dict, ax):
        """Plot error trends by distance."""
        ax.text(0.5, 0.5, 'Error by Distance\n(requires implementation)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Error Trends by Distance')
    
    def _plot_category_performance(self, category_results: Dict, ax):
        """Plot per-category model performance."""
        categories = []
        accuracies = []
        
        for category, results in category_results.items():
            if 'test_accuracy' in results:
                categories.append(category)
                accuracies.append(results['test_accuracy'])
        
        if categories:
            ax.bar(categories, accuracies, alpha=0.7)
            ax.set_title('Per-Category Detection Accuracy')
            ax.set_ylabel('Accuracy')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No category results', ha='center', va='center')
    
    def _plot_tree_importance(self, tree_results: Dict, ax):
        """Plot decision tree feature importance."""
        importance_df = tree_results['feature_importance'].head(8)
        
        ax.barh(importance_df['feature'], importance_df['importance'], alpha=0.7)
        ax.set_title(f'Decision Tree Importance\n(Acc: {tree_results["accuracy"]:.3f})')
        ax.set_xlabel('Importance')
    
    def _plot_model_performance_comparison(self, rf_results: Dict, ax):
        """Plot model performance comparison."""
        models = []
        scores = []
        
        for model_name, results in rf_results.items():
            if 'test_accuracy' in results:
                models.append(model_name.replace('_', ' ').title())
                scores.append(results['test_accuracy'])
            elif 'test_r2' in results:
                models.append(model_name.replace('_', ' ').title())
                scores.append(max(0, results['test_r2']))  # Clip negative R²
        
        if models:
            bars = ax.bar(models, scores, alpha=0.7)
            ax.set_title('Model Performance Comparison')
            ax.set_ylabel('Score')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
    
    def generate_comprehensive_report(self, 
                                    rf_results: Dict,
                                    clustering_results: Dict, 
                                    tree_results: Dict,
                                    features_df: pd.DataFrame) -> str:
        """Generate comprehensive feature importance analysis report."""
        
        report = "# Random Forest Feature Importance Analysis\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        
        total_samples = len(features_df)
        tp_count = (features_df['outcome'] == 'TP').sum()
        fp_count = (features_df['outcome'] == 'FP').sum() 
        fn_count = (features_df['outcome'] == 'FN').sum()
        
        report += f"**Dataset**: {total_samples} samples ({tp_count} TP, {fp_count} FP, {fn_count} FN)\n"
        report += f"**Categories analyzed**: {features_df['category'].nunique()}\n"
        report += f"**Models trained**: {len([k for k in rf_results.keys() if 'error' not in rf_results.get(k, {})])}\n\n"
        
        # Key Findings from Random Forest Models
        report += "## Key Findings: Feature Importance\n\n"
        
        # Global feature importance across models
        if rf_results:
            all_features = {}
            for model_name, results in rf_results.items():
                if 'feature_importance' in results:
                    for _, row in results['feature_importance'].head(5).iterrows():
                        feature = row['feature']
                        importance = row['importance']
                        if feature not in all_features:
                            all_features[feature] = []
                        all_features[feature].append((model_name, importance))
            
            # Rank features by frequency of appearance in top 5
            feature_ranks = [(feat, len(appearances)) for feat, appearances in all_features.items()]
            feature_ranks.sort(key=lambda x: x[1], reverse=True)
            
            report += "### Most Consistent Important Features:\n"
            for feature, count in feature_ranks[:8]:
                avg_importance = np.mean([imp for _, imp in all_features[feature]])
                report += f"- **{feature}**: appears in top 5 of {count} models (avg importance: {avg_importance:.3f})\n"
            report += "\n"
        
        # Model-specific results
        for model_name, results in rf_results.items():
            if 'error' in results:
                continue
                
            report += f"### {model_name}\n"
            if 'test_accuracy' in results:
                report += f"**Accuracy**: {results['test_accuracy']:.3f}\n"
            elif 'test_r2' in results:
                report += f"**R² Score**: {results['test_r2']:.3f} (RMSE: {results['test_rmse']:.3f})\n"
            
            report += f"**Samples**: {results.get('n_samples')}\n"
            report += "**Top Features**:\n"
            
            if 'feature_importance' in results:
                for _, row in results['feature_importance'].head(5).iterrows():
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
                rules_text = results['rules']
                rules_lines = rules_text.split('\n')
                
                # Find meaningful rules (simplified extraction)
                meaningful_rules = []
                for line in rules_lines:
                    if '|---' in line and ('class' in line or 'value' in line):
                        # This is a simplified rule extraction
                        clean_line = line.replace('|---', '').strip()
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
        if clustering_results and 'cluster_analysis' in clustering_results:
            report += "## Failure Mode Clustering\n\n"
            
            if 'dbscan' in clustering_results:
                dbscan_info = clustering_results['dbscan']
                report += f"**DBSCAN Results**: {dbscan_info['n_clusters']} clusters found\n"
                report += f"**Noise samples**: {dbscan_info['n_noise']}\n"
                if not np.isnan(dbscan_info['silhouette_score']):
                    report += f"**Silhouette Score**: {dbscan_info['silhouette_score']:.3f}\n"
                report += "\n"
            
            analysis = clustering_results['cluster_analysis']
            report += "### Discovered Failure Modes:\n"
            
            for cluster_name, stats in analysis.items():
                if cluster_name.startswith('cluster_'):
                    cluster_id = cluster_name.replace('cluster_', '')
                    report += f"**Cluster {cluster_id}** ({stats['size']} samples):\n"
                    report += f"  - Dominant category: {stats['dominant_category']}\n"
                    if not np.isnan(stats['mean_ate']):
                        report += f"  - Mean ATE: {stats['mean_ate']:.3f}m\n"
                    if not np.isnan(stats['mean_distance']):
                        report += f"  - Mean distance: {stats['mean_distance']:.1f}m\n"
                    if not np.isnan(stats['mean_occlusion']):
                        report += f"  - Mean occlusion: {stats['mean_occlusion']:.3f}\n"
                    
                    # Outcome distribution
                    if stats['outcome_distribution']:
                        outcomes = ', '.join([f"{k}: {v}" for k, v in stats['outcome_distribution'].items()])
                        report += f"  - Outcomes: {outcomes}\n"
                    report += "\n"
        
        # Actionable Recommendations
        report += "## Actionable Recommendations\n\n"
        
        # Based on top features
        if rf_results:
            # Find most important features across all models
            feature_votes = defaultdict(int)
            for model_name, results in rf_results.items():
                if 'feature_importance' in results:
                    for _, row in results['feature_importance'].head(3).iterrows():
                        feature_votes[row['feature']] += 1
            
            top_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)[:5]
            
            report += "### Based on Feature Importance:\n"
            for feature, votes in top_features:
                if 'distance' in feature.lower():
                    report += f"🎯 **{feature}** is critical → Consider distance-adaptive thresholds or training data balancing\n"
                elif 'occlusion' in feature.lower():
                    report += f"🎯 **{feature}** drives failures → Improve occlusion handling or multi-view fusion\n"
                elif 'score' in feature.lower():
                    report += f"🎯 **{feature}** is predictive → Confidence calibration may help\n"
                elif 'nearby' in feature.lower():
                    report += f"🎯 **{feature}** affects performance → Context modeling improvements needed\n"
                else:
                    report += f"🎯 **{feature}** is important → Investigate {feature.replace('_', ' ')} effects\n"
            report += "\n"
        
        # Performance-based recommendations
        if 'detection_vs_miss' in rf_results:
            detection_acc = rf_results['detection_vs_miss'].get('test_accuracy', 0)
            if detection_acc < 0.8:
                report += "⚠️ **Low detection accuracy** → Focus on reducing false negatives\n"
            
        if 'tp_vs_fp' in rf_results:
            tp_fp_acc = rf_results['tp_vs_fp'].get('test_accuracy', 0)
            if tp_fp_acc < 0.75:
                report += "⚠️ **Poor TP/FP separation** → Improve confidence score calibration\n"
        
        # Category-specific recommendations
        if 'per_category' in rf_results:
            worst_categories = []
            for category, results in rf_results['per_category'].items():
                if 'test_accuracy' in results and results['test_accuracy'] < 0.7:
                    worst_categories.append((category, results['test_accuracy']))
            
            if worst_categories:
                worst_categories.sort(key=lambda x: x[1])
                report += f"⚠️ **Challenging categories**: {', '.join([c for c, _ in worst_categories[:3]])} need targeted improvements\n"
        
        report += "\n### Next Steps:\n"
        report += "1. **Data Collection**: Gather more samples for challenging scenarios identified by clustering\n"
        report += "2. **Feature Engineering**: Focus on the top predictive features for model improvements\n"
        report += "3. **Model Architecture**: Consider specialized handling for distance/occlusion effects\n"
        report += "4. **Evaluation**: Use decision tree rules as test cases for validation\n"
        
        return report


def add_random_forest_analysis_av2(evaluator: StandaloneLongTailEvaluator, av2_results: Dict, 
                                  ego_poses: Optional[pd.DataFrame] = None) -> Dict:
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
    eval_dts = av2_results['eval_dts']
    eval_gts = av2_results['eval_gts']
    cfg = av2_results['cfg']
    
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
        return {'error': 'Insufficient data'}
    
    # Train Random Forest models
    rf_results = analyzer.train_random_forest_models(features_df)
    
    # Perform clustering analysis
    clustering_results = analyzer.perform_error_clustering(features_df)
    
    # Create dimensionality reduction visualizations
    dimred_results = analyzer.create_dimensionality_reduction_viz(features_df, clustering_results)
    
    # Train interpretable decision trees
    tree_results = analyzer.train_interpretable_trees(features_df)

    pprint(rf_results)
    pprint(clustering_results)
    pprint(dimred_results)
    pprint(tree_results)
    
    # Generate outputs
    if rf_results:
        # Create comprehensive visualizations
        output_path = evaluator.output_dir / "random_forest_analysis.png"
        analyzer.create_comprehensive_visualizations(
            rf_results, clustering_results, dimred_results, tree_results, str(output_path)
        )
        
        # Generate report
        report = analyzer.generate_comprehensive_report(
            rf_results, clustering_results, tree_results, features_df
        )
        report_path = evaluator.output_dir / "random_forest_analysis.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Random Forest analysis complete! Results saved to {evaluator.output_dir}")
    
    return {
        'rf_results': rf_results,
        'clustering_results': clustering_results,
        'dimensionality_reduction': dimred_results,
        'decision_trees': tree_results,
        'features_df': features_df
    }

# Example usage with main pipeline
if __name__ == "__main__":
    # This would be integrated into the main analysis


    config = EvaluationConfig(
        # predictions_path="/scratch/project_mnt/S0202/uqdetche/lidar-longtail-mining/lion/output/lion_models/lion_mamba_1f_1x_argo_128dim_sparse_v2/default/eval/epoch_2/val/default/result.pkl",
        predictions_path="../../lion/output/lion_models/lion_mamba_1f_1x_argo_128dim_sparse_v2/default/eval/epoch_2/val/default/processed_results.feather",
        ground_truth_path="../../lion/data/argo2/val_anno.feather",
        dataset_dir="../../lion/data/argo2/sensor/val",
        output_dir="./longtail_feature_analysis"
    )
    
    evaluator = StandaloneLongTailEvaluator(config)
    
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    av2_results_path = output_dir / "av2_results.pkl"

    random_forest_results_path = output_dir / "random_forest_results.json"


    if av2_results_path.exists():
        with open(av2_results_path, 'rb') as file:
            av2_results = pickle.load(file)
    else:
        # Run main analysis
        av2_results = evaluator.run_av2_eval()

        with open(av2_results_path, 'wb') as file:
            pickle.dump(av2_results, file)

    # Add random forest analysis
    results = add_random_forest_analysis_av2(evaluator, av2_results)
    
    with open(random_forest_results_path, 'w') as f:
        json.dump(results, f)

    print("analysis complete!")