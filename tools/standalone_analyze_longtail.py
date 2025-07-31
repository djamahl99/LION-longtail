"""
Standalone Long-tail Evaluation Pipeline for Argoverse 2
Independent of OpenPCDet, compatible with Python 3.9+
"""

import gc
import pickle
import json
from pprint import pprint
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pyarrow.feather import read_feather
import torch

# AV2 imports (Python 3.9+)
from av2.evaluation.detection.eval import evaluate, evaluate_hierarchy
from av2.evaluation import SensorCompetitionCategories
from av2.evaluation.detection.utils import DetectionCfg
from av2.structures.cuboid import Cuboid, CuboidList
from av2.utils.typing import NDArrayFloat

import psutil
import os
import time
import threading

CLASSES = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG')


@dataclass
class EvaluationConfig:
    """Configuration for standalone evaluation."""
    predictions_path: str
    ground_truth_path: str
    dataset_dir: str
    output_dir: str = "./standalone_longtail_analysis"
    evaluate_range: float = 150.0
    eval_only_roi: bool = True
    
    # Long-tail specific
    longtail_classes: List[str] = None
    iou_thresholds: List[float] = None
    
    def __post_init__(self):
        if self.longtail_classes is None:
            self.longtail_classes = [
                'ARTICULATED_BUS', 'LARGE_VEHICLE', 'MESSAGE_BOARD_TRAILER',
                'SIGN', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
                'WHEELCHAIR', 'WHEELED_RIDER'
            ]
        if self.iou_thresholds is None:
            self.iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class PredictionAdapter:
    """Adapter to convert OpenPCDet predictions to AV2 format."""
    
    @staticmethod
    def load_openpcdet_predictions(pkl_path: str) -> List[Dict]:
        """Load predictions from OpenPCDet pickle file."""
        with open(pkl_path, 'rb') as f:
            predictions = pickle.load(f)
        return predictions
    
    @staticmethod
    def convert_to_av2_format(predictions: List[Dict]) -> pd.DataFrame:
        """
        Convert OpenPCDet predictions to AV2 evaluation format.
        
        Expected OpenPCDet format:
        [
            {
                'frame_id': str,
                'boxes_lidar': np.ndarray (N, 7) [x, y, z, dx, dy, dz, heading],
                'score': np.ndarray (N,),
                'name': np.ndarray (N,) of str
            },
            ...
        ]
        
        AV2 format columns:
        - log_id, timestamp_ns
        - tx_m, ty_m, tz_m (center)
        - length_m, width_m, height_m
        - qw, qx, qy, qz (quaternion)
        - score, category
        """
        rows = []
        
        for pred in predictions:
            frame_id = pred['frame_id']
            # Parse frame_id to get log_id and timestamp
            # Format: "{log_id}_{timestamp_ns}"
            parts = frame_id.split('_')
            log_id = '_'.join(parts[:-1])  # Handle log_ids with underscores
            timestamp_ns = int(parts[-1])
            
            if 'boxes_lidar' not in pred or len(pred['boxes_lidar']) == 0:
                continue
                
            boxes = pred['boxes_lidar']  # (N, 7)
            scores = pred['score']
            names = pred['name']
            
            for i in range(len(boxes)):
                box = boxes[i]
                # Extract box parameters
                x, y, z, dx, dy, dz, heading = box
                
                # Convert heading to quaternion
                qw, qx, qy, qz = heading_to_quaternion(heading)
                
                # Map class names to AV2 format (uppercase)
                category = names[i].upper().replace('_', '_')
                
                rows.append({
                    'log_id': log_id,
                    'timestamp_ns': timestamp_ns,
                    'tx_m': x,
                    'ty_m': y,
                    'tz_m': z,
                    'length_m': dx,
                    'width_m': dy,
                    'height_m': dz,
                    'qw': qw,
                    'qx': qx,
                    'qy': qy,
                    'qz': qz,
                    'score': scores[i],
                    'category': category
                })
        
        return pd.DataFrame(rows)

        
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import gc
from pathlib import Path

# Import the existing AV2 evaluation functions
from av2.evaluation.detection.utils import (
    accumulate_hierarchy,
    is_evaluated,
    load_mapped_avm_and_egoposes
)
from av2.evaluation.detection.constants import (
    HIERARCHY, 
    LCA, 
    LCA_COLUMNS,
    NUM_DECIMALS
)
from av2.evaluation import SensorCompetitionCategories
from av2.evaluation.detection.eval import DetectionCfg
from tqdm import tqdm, trange

class MemoryEfficientEvaluator:
    """Memory-efficient version of AV2 evaluation using existing AV2 functions."""
    
    def __init__(self, cfg: DetectionCfg):
        self.cfg = cfg
    
    def evaluate_hierarchy_memory_efficient(
        self, 
        dts: pd.DataFrame, 
        gts: pd.DataFrame,
        chunk_size: int = 500
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Memory-efficient evaluation that processes data in chunks using AV2 functions.
        
        Args:
            dts: Detection dataframe
            gts: Ground truth dataframe  
            chunk_size: Number of UUIDs to process at once
            
        Returns:
            Tuple of (eval_dts, eval_gts, metrics) matching original evaluate_hierarchy
        """
        print("Starting memory-efficient hierarchical evaluation...")
        
        # Sort both dataframes for consistent processing
        UUID_COLUMNS = ['log_id', 'timestamp_ns']
        dts = dts.sort_values(UUID_COLUMNS)
        gts = gts.sort_values(UUID_COLUMNS)
        
        # Get unique UUIDs that exist in both datasets
        dts_uuids = set(zip(dts['log_id'], dts['timestamp_ns']))
        if hasattr(gts.index, 'tolist'):
            gts_uuids = set(gts.index.tolist()) 
            print("using index?")
        else:
            gts_uuids = set(zip(gts['log_id'], gts['timestamp_ns']))
            print("using zip?")

        print(f"dts_uuids={len(dts_uuids)}")
        print(f"gts_uuids={len(gts_uuids)}")
        print(sorted(list(dts_uuids))[:10])
        print(sorted(list(gts_uuids))[:10])

        common_uuids = sorted(list(dts_uuids & gts_uuids))
        print(f"Processing {len(common_uuids)} common UUIDs in chunks of {chunk_size}")
        
        # Load maps and egoposes if needed (same as original)
        log_id_to_avm = None
        log_id_to_timestamped_poses = None
        if self.cfg.eval_only_roi_instances and self.cfg.dataset_dir is not None:
            print("Loading maps and egoposes...")
            log_ids = list(set([uuid[0] for uuid in common_uuids]))
            log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(
                log_ids, self.cfg.dataset_dir
            )
        
        # Process in chunks
        all_processed_outputs = []
        
        for i in trange(0, len(common_uuids), chunk_size, desc="Processing Chunks"):
            chunk_uuids = common_uuids[i:i+chunk_size]
            # print(f"Processing chunk {i//chunk_size + 1}/{(len(common_uuids)-1)//chunk_size + 1}")
            
            # Process this chunk
            chunk_outputs = self._process_chunk(
                dts, gts, chunk_uuids, log_id_to_avm, log_id_to_timestamped_poses
            )
            all_processed_outputs.extend(chunk_outputs)
            
            # Force garbage collection
            gc.collect()
        
        print("Concatenating results...")
        
        # Concatenate all chunk results
        if all_processed_outputs:
            all_sweep_dts = [output[0] for output in all_processed_outputs]
            all_sweep_gts = [output[1] for output in all_processed_outputs]
            all_sweep_dts_cats = [output[2] for output in all_processed_outputs]
            all_sweep_gts_cats = [output[3] for output in all_processed_outputs]
            all_uuids = [output[4] for output in all_processed_outputs]
            
            # Concatenate numpy arrays
            dts_npy = np.concatenate(all_sweep_dts).astype(np.float64)
            gts_npy = np.concatenate(all_sweep_gts).astype(np.float64)
            dts_categories_npy = np.concatenate(all_sweep_dts_cats).astype(np.object_)
            gts_categories_npy = np.concatenate(all_sweep_gts_cats).astype(np.object_)
            
            # Create UUID arrays
            dts_uuids_npy = np.array([uuid for outputs in all_processed_outputs 
                                    for uuid in [outputs[4]] * len(outputs[0])])
            gts_uuids_npy = np.array([uuid for outputs in all_processed_outputs 
                                    for uuid in [outputs[4]] * len(outputs[1])])
        else:
            # Handle empty case
            print("empty case", f"{all_processed_outputs=}")
            dts_npy = np.zeros((0, 10))
            gts_npy = np.zeros((0, 10))
            dts_categories_npy = np.zeros((0, 1), dtype=np.object_)
            gts_categories_npy = np.zeros((0, 1), dtype=np.object_)
            dts_uuids_npy = np.array([])
            gts_uuids_npy = np.array([])
        
        print("Computing hierarchical metrics...")
        # exit()
        
        # Now use the original accumulate_hierarchy function for each category/hierarchy combination
        accumulate_hierarchy_args_list = []
        for category in self.cfg.categories:
            index = HIERARCHY["FINEGRAIN"].index(category)
            for super_category, categories in HIERARCHY.items():
                    print(f"{super_category=}, {categories=}, {category=}")

                    lca_category = LCA[categories[index]]
                    pprint(dict(
                        dts_npy=dts_npy,
                        gts_npy=gts_npy,
                        dts_categories_npy=dts_categories_npy,
                        gts_categories_npy=gts_categories_npy,
                        dts_uuids_npy=dts_uuids_npy,
                        gts_uuids_npy=gts_uuids_npy,
                        category=category,
                        lca_category=lca_category,
                        super_category=super_category,
                        cfg=self.cfg,
                    ))
                    accumulate_hierarchy_args_list.append((
                        dts_npy,
                        gts_npy,
                        dts_categories_npy,
                        gts_categories_npy,
                        dts_uuids_npy,
                        gts_uuids_npy,
                        category,
                        lca_category,
                        super_category,
                        self.cfg,
                    ))
        
        # Compute metrics using the original AV2 function
        accumulate_outputs = []
        for accumulate_args in tqdm(accumulate_hierarchy_args_list, desc="Computing hierarchical AP"):
            accumulate_outputs.append(accumulate_hierarchy(*accumulate_args))
        
        # Format results like the original function
        super_categories = list(HIERARCHY.keys())
        metrics = np.zeros((len(self.cfg.categories), len(HIERARCHY.keys())))
        for ap, category, super_category in accumulate_outputs:
            if category in self.cfg.categories:
                category_index = self.cfg.categories.index(category)
                super_category_index = super_categories.index(super_category)
                metrics[category_index][super_category_index] = round(ap, NUM_DECIMALS)
        
        metrics_df = pd.DataFrame(metrics, columns=LCA_COLUMNS, index=self.cfg.categories)
        
        return dts_npy, gts_npy, metrics_df
    
    def _process_chunk(
        self, 
        dts: pd.DataFrame, 
        gts: pd.DataFrame, 
        chunk_uuids: List[Tuple[str, int]],
        log_id_to_avm: Optional[Dict] = None,
        log_id_to_timestamped_poses: Optional[Dict] = None
    ) -> List[Tuple]:
        """Process a chunk of UUIDs using the existing AV2 is_evaluated function."""
        
        chunk_outputs = []
        
        for uuid in chunk_uuids:
            log_id, timestamp_ns = uuid
            
            # Extract data for this UUID
            sweep_dts = self._extract_sweep_data(dts, uuid)
            sweep_gts = self._extract_sweep_data(gts, uuid)
            
            # Convert to numpy arrays (same format as original)
            if not sweep_dts.empty:
                sweep_dts_npy, sweep_dts_cats = self._convert_dts_to_numpy(sweep_dts)
            else:
                sweep_dts_npy = np.zeros((0, 10))
                sweep_dts_cats = np.zeros((0, 1), dtype=np.object_)
            
            if not sweep_gts.empty:
                sweep_gts_npy, sweep_gts_cats = self._convert_gts_to_numpy(sweep_gts)
            else:
                sweep_gts_npy = np.zeros((0, 10))
                sweep_gts_cats = np.zeros((0, 1), dtype=np.object_)
            
            # Get map and pose for this log if needed
            avm = log_id_to_avm[log_id] if log_id_to_avm else None
            city_SE3_ego = None
            if log_id_to_timestamped_poses:
                city_SE3_ego = log_id_to_timestamped_poses[log_id][int(timestamp_ns)]
            
            # Use the original AV2 is_evaluated function to filter
            filtered_dts, filtered_gts, filtered_dts_cats, filtered_gts_cats, _ = is_evaluated(
                sweep_dts_npy,
                sweep_gts_npy,
                sweep_dts_cats,
                sweep_gts_cats,
                uuid,
                self.cfg,
                avm,
                city_SE3_ego
            )
            
            chunk_outputs.append((
                filtered_dts,
                filtered_gts,
                filtered_dts_cats,
                filtered_gts_cats,
                uuid
            ))
        
        return chunk_outputs
    
    def _extract_sweep_data(self, df: pd.DataFrame, uuid: Tuple[str, int]) -> pd.DataFrame:
        """Extract data for a specific UUID from dataframe."""
        log_id, timestamp_ns = uuid
        
        if hasattr(df.index, 'tolist'):
            # Multi-index case (ground truth)
            if uuid in df.index:
                return df.loc[[uuid]].copy()
            else:
                return pd.DataFrame()
        else:
            # Regular case (detections)
            mask = (df['log_id'] == log_id) & (df['timestamp_ns'] == timestamp_ns)
            return df[mask].copy()
    
    def _convert_dts_to_numpy(self, sweep_dts: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert detection dataframe to numpy format expected by AV2."""
        # Columns expected: tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz
        dts_columns = ['tx_m', 'ty_m', 'tz_m', 'length_m', 'width_m', 'height_m', 'qw', 'qx', 'qy', 'qz']
        
        # Extract numeric data
        dts_data = sweep_dts[dts_columns].values.astype(np.float64)
        
        # Extract categories
        dts_cats = sweep_dts['category'].values.reshape(-1, 1).astype(np.object_)
        
        return dts_data, dts_cats
    
    def _convert_gts_to_numpy(self, sweep_gts: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert ground truth dataframe to numpy format expected by AV2."""
        # Columns expected: tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz
        gts_columns = ['tx_m', 'ty_m', 'tz_m', 'length_m', 'width_m', 'height_m', 'qw', 'qx', 'qy', 'qz']
        
        # Extract numeric data
        gts_data = sweep_gts[gts_columns].values.astype(np.float64)
        
        # Extract categories
        gts_cats = sweep_gts['category'].values.reshape(-1, 1).astype(np.object_)
        
        return gts_data, gts_cats

def heading_to_quaternion(heading: float) -> Tuple[float, float, float, float]:
    """Convert heading angle to quaternion (qw, qx, qy, qz)."""
    # Heading is rotation around z-axis
    cy = np.cos(heading * 0.5)
    sy = np.sin(heading * 0.5)
    
    qw = cy
    qx = 0.0
    qy = 0.0
    qz = sy
    
    return qw, qx, qy, qz


def quaternion_to_heading(qw: float, qx: float, qy: float, qz: float) -> float:
    """Convert quaternion to heading angle."""
    heading = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return heading


class StandaloneLongTailEvaluator:
    """Standalone evaluator independent of OpenPCDet."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load predictions and ground truth."""
        # Load predictions
        if self.config.predictions_path.endswith('.pkl'):
            raw_preds = PredictionAdapter.load_openpcdet_predictions(
                self.config.predictions_path
            )
            dts = PredictionAdapter.convert_to_av2_format(raw_preds)
        elif self.config.predictions_path.endswith('.feather'):
            dts = read_feather(self.config.predictions_path)
        else:
            raise ValueError(f"Unsupported prediction format: {self.config.predictions_path}")
            
        # Load ground truth
        gts = read_feather(self.config.ground_truth_path)
        
        # Set indices for efficient lookup
        gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")
        
        return dts, gts
    
    def run_av2_evaluation(self, dts: pd.DataFrame, gts: pd.DataFrame) -> Dict:
        """Run official AV2 evaluation with hierarchy."""
        # Setup detection config
        
        # Run evaluation
        print("Running AV2 evaluation...")

        dts = dts.set_index(["log_id", "timestamp_ns"]).sort_index()

        valid_uuids_gts = gts.index.tolist()
        valid_uuids_dts = dts.index.tolist()
        valid_uuids_gts = set(valid_uuids_gts)
        valid_uuids_dts = set(valid_uuids_dts)

        print("valid_uuids_gts", len(valid_uuids_gts))
        print("valid_uuids_dts", len(valid_uuids_dts))

        valid_uuids = valid_uuids_gts & valid_uuids_dts
        gts = gts.loc[list(valid_uuids)].sort_index()

        # Setup categories and config
        categories = set(x.value for x in SensorCompetitionCategories)
        categories &= set(gts["category"].unique().tolist())
        
        cfg = DetectionCfg(
            dataset_dir=Path(self.config.dataset_dir),
            categories=tuple(sorted(categories)),
            max_range_m=self.config.evaluate_range,
            eval_only_roi_instances=True,
        )

        # print("dts[0]", dts.iloc[0])
        # print("gts[0]", gts.iloc[0])
        eval_dts, eval_gts, default_metrics = evaluate(dts.reset_index(), gts.reset_index(), cfg, n_jobs=8)
        metrics = dict(
            default_metrics=default_metrics, 
            hierarchy_metrics=dict(),
        )

        return {
            'eval_dts': eval_dts,
            'eval_gts': eval_gts, 
            'metrics': metrics,
            'cfg': cfg
        }
    
    def compute_iou_3d(self, box1: Dict, box2: Dict) -> float:
        """Compute 3D IoU between two boxes."""
        # Convert to numpy arrays
        corners1 = self.box_to_corners(box1)
        corners2 = self.box_to_corners(box2)
        
        # Use Cuboid for IoU computation
        cuboid1 = Cuboid(
            dst_SE3_object=np.eye(4),  # Will set translation/rotation separately
            length_m=box1['length_m'],
            width_m=box1['width_m'], 
            height_m=box1['height_m'],
            category=box1.get('category', 'UNKNOWN')
        )
        
        cuboid2 = Cuboid(
            dst_SE3_object=np.eye(4),
            length_m=box2['length_m'],
            width_m=box2['width_m'],
            height_m=box2['height_m'],
            category=box2.get('category', 'UNKNOWN')
        )
        
        # Set positions
        cuboid1.dst_SE3_object[:3, 3] = [box1['tx_m'], box1['ty_m'], box1['tz_m']]
        cuboid2.dst_SE3_object[:3, 3] = [box2['tx_m'], box2['ty_m'], box2['tz_m']]
        
        # Set rotations from quaternions
        from scipy.spatial.transform import Rotation
        
        quat1 = [box1['qx'], box1['qy'], box1['qz'], box1['qw']]  # scipy format
        quat2 = [box2['qx'], box2['qy'], box2['qz'], box2['qw']]
        
        cuboid1.dst_SE3_object[:3, :3] = Rotation.from_quat(quat1).as_matrix()
        cuboid2.dst_SE3_object[:3, :3] = Rotation.from_quat(quat2).as_matrix()
        
        # Compute IoU
        iou = cuboid1.intersection_over_union(cuboid2)
        
        return iou
    
    def box_to_corners(self, box: Dict) -> np.ndarray:
        """Convert box parameters to 8 corners."""
        # This is a simplified version - you may want to use the actual
        # transformation with quaternions for accurate corners
        x, y, z = box['tx_m'], box['ty_m'], box['tz_m']
        l, w, h = box['length_m'], box['width_m'], box['height_m']
        
        # Simple axis-aligned corners (before rotation)
        corners = np.array([
            [-l/2, -w/2, -h/2],
            [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2],
            [l/2, -w/2, h/2],
            [l/2, w/2, h/2],
            [-l/2, w/2, h/2]
        ])
        
        # Apply rotation
        heading = quaternion_to_heading(
            box['qw'], box['qx'], box['qy'], box['qz']
        )
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        rot_matrix = np.array([
            [cos_h, -sin_h, 0],
            [sin_h, cos_h, 0],
            [0, 0, 1]
        ])
        
        corners = corners @ rot_matrix.T
        corners += np.array([x, y, z])
        
        return corners
    
    def find_closest_box(self, 
                        query_box: pd.Series, 
                        candidate_boxes: pd.DataFrame,
                        max_distance: float = 5.0) -> Optional[Tuple[int, float]]:
        """Find closest box and return (index, iou)."""
        if len(candidate_boxes) == 0:
            return None
            
        best_iou = 0.0
        best_idx = None
        
        # First filter by distance for efficiency
        distances = np.sqrt(
            (candidate_boxes['tx_m'] - query_box['tx_m'])**2 +
            (candidate_boxes['ty_m'] - query_box['ty_m'])**2 +
            (candidate_boxes['tz_m'] - query_box['tz_m'])**2
        )

        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        return (best_idx, best_distance) if best_distance <= max_distance else None

        # nearby_mask = distances < max_distance
        # nearby_boxes = candidate_boxes[nearby_mask]
        
        # for idx, cand_box in nearby_boxes.iterrows():
        #     print("query_box", query_box)
        #     print("cand_box", cand_box)

        #     # distance(tps_dts[:, 3:6], tps_gts[:, 3:6], DistanceType.SCALE)
        #     iou = self.compute_iou_3d(
        #         query_box.to_dict(),
        #         cand_box.to_dict()
        #     )
        #     if iou > best_iou:
        #         best_iou = iou
        #         best_idx = idx
                
        return (best_idx, best_iou) if best_idx is not None else None
    
    def analyze_detections_per_class(self, 
                                   class_name: str,
                                   dts: pd.DataFrame,
                                   gts: pd.DataFrame) -> Dict:
        """Comprehensive analysis for a single class."""
        # Filter for this class
        class_dts = dts[dts['category'] == class_name].copy()
        class_gts = gts[gts['category'] == class_name].copy()
        
        analysis = {
            'class_name': class_name,
            'num_gt': len(class_gts),
            'num_det': len(class_dts),
            'true_positives': [],
            'false_positives': [],
            'false_negatives': []
        }
        
        # Track matched GTs
        matched_gts = set()
        
        # Analyze each detection
        for dt_idx, dt in class_dts.iterrows():
            # Find scene GTs
            scene_key = (dt['log_id'], dt['timestamp_ns'])
            if scene_key in gts.index:
                scene_gts = gts.loc[[scene_key]]
                scene_class_gts = scene_gts[scene_gts['category'] == class_name]
            else:
                scene_class_gts = pd.DataFrame()
            
            # Find best matching GT
            match_result = self.find_closest_box(dt, scene_class_gts)

            print("match_result", match_result)
            print("scene_class_gts", scene_class_gts)
            
            if match_result and match_result[1] >= 0.5:  # TP threshold
                gt_idx, iou = match_result
                matched_gts.add((scene_key, gt_idx))
                
                analysis['true_positives'].append({
                    'det_idx': dt_idx,
                    'gt_idx': gt_idx,
                    'iou': iou,
                    'det_data': dt.to_dict(),
                    'gt_data': scene_class_gts.loc[gt_idx].to_dict()
                })
            else:
                # False positive - analyze error type
                fp_info = {
                    'det_idx': dt_idx,
                    'det_data': dt.to_dict(),
                    'error_type': 'hallucination',
                    'closest_gt': None
                }
                
                # Check all GTs in scene (any class)
                if scene_key in gts.index:
                    all_scene_gts = gts.loc[[scene_key]]
                    closest = self.find_closest_box(dt, all_scene_gts)
                    
                    if closest:
                        gt_idx, iou = closest
                        closest_gt = all_scene_gts.loc[gt_idx]
                        
                        if iou > 0.1:  # Some overlap
                            if closest_gt['category'] != class_name:
                                fp_info['error_type'] = 'wrong_class'
                                fp_info['predicted_as'] = closest_gt['category']
                            else:
                                fp_info['error_type'] = 'low_iou'
                            
                            fp_info['closest_gt'] = {
                                'category': closest_gt['category'],
                                'iou': iou,
                                'data': closest_gt.to_dict()
                            }
                
                analysis['false_positives'].append(fp_info)
        
        # Find false negatives
        for gt_key, gt_group in class_gts.groupby(level=[0, 1]):
            for gt_idx, gt in gt_group.iterrows():
                if (gt_key, gt_idx) not in matched_gts:
                    # This GT was missed
                    fn_info = {
                        'gt_idx': gt_idx,
                        'gt_data': gt.to_dict(),
                        'miss_reason': 'not_detected',
                        'closest_det': None
                    }
                    
                    # Find closest detection
                    scene_dts = dts[
                        (dts['log_id'] == gt_key[0]) & 
                        (dts['timestamp_ns'] == gt_key[1])
                    ]
                    
                    closest = self.find_closest_box(gt, scene_dts)
                    if closest:
                        det_idx, iou = closest
                        closest_det = scene_dts.loc[det_idx]
                        
                        if closest_det['category'] != class_name:
                            fn_info['miss_reason'] = 'detected_as_wrong_class'
                        elif iou < 0.5:
                            fn_info['miss_reason'] = 'low_confidence_or_poor_localization'
                            
                        fn_info['closest_det'] = {
                            'category': closest_det['category'],
                            'score': closest_det['score'],
                            'iou': iou,
                            'data': closest_det.to_dict()
                        }
                    
                    analysis['false_negatives'].append(fn_info)
        
        return analysis
    
    def compute_parameter_distributions(self, analysis: Dict) -> Dict:
        """Compute distributions of predicted parameters."""
        distributions = {}
        
        # True Positive distributions
        if analysis['true_positives']:
            tp_data = pd.DataFrame([
                {
                    'iou': tp['iou'],
                    'score': tp['det_data']['score'],
                    'size_error': np.sqrt(
                        (tp['det_data']['length_m'] - tp['gt_data']['length_m'])**2 +
                        (tp['det_data']['width_m'] - tp['gt_data']['width_m'])**2 +
                        (tp['det_data']['height_m'] - tp['gt_data']['height_m'])**2
                    ),
                    'position_error': np.sqrt(
                        (tp['det_data']['tx_m'] - tp['gt_data']['tx_m'])**2 +
                        (tp['det_data']['ty_m'] - tp['gt_data']['ty_m'])**2 +
                        (tp['det_data']['tz_m'] - tp['gt_data']['tz_m'])**2
                    ),
                    'heading_error': np.abs(
                        quaternion_to_heading(
                            tp['det_data']['qw'], tp['det_data']['qx'],
                            tp['det_data']['qy'], tp['det_data']['qz']
                        ) -
                        quaternion_to_heading(
                            tp['gt_data']['qw'], tp['gt_data']['qx'],
                            tp['gt_data']['qy'], tp['gt_data']['qz']
                        )
                    )
                }
                for tp in analysis['true_positives']
            ])
            
            distributions['tp'] = {
                'iou': tp_data['iou'].describe().to_dict(),
                'score': tp_data['score'].describe().to_dict(),
                'size_error': tp_data['size_error'].describe().to_dict(),
                'position_error': tp_data['position_error'].describe().to_dict(),
                'heading_error': tp_data['heading_error'].describe().to_dict()
            }
        
        # False Positive distributions
        if analysis['false_positives']:
            fp_scores = [fp['det_data']['score'] for fp in analysis['false_positives']]
            fp_types = pd.Series([fp['error_type'] for fp in analysis['false_positives']])
            
            distributions['fp'] = {
                'score': pd.Series(fp_scores).describe().to_dict(),
                'error_types': fp_types.value_counts().to_dict()
            }
        
        return distributions
    
    def create_visualizations(self, analysis: Dict, output_path: Path):
        """Create visualization plots for the analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Analysis for {analysis['class_name']}", fontsize=16)
        
        # 1. IoU distribution for TPs
        if analysis['true_positives']:
            ious = [tp['iou'] for tp in analysis['true_positives']]
            axes[0, 0].hist(ious, bins=20, edgecolor='black', alpha=0.7)
            axes[0, 0].set_xlabel('IoU')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('True Positive IoU Distribution')
            axes[0, 0].axvline(0.5, color='red', linestyle='--', label='IoU=0.5')
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, 'No True Positives', ha='center', va='center')
            axes[0, 0].set_title('True Positive IoU Distribution')
        
        # 2. Score distributions
        if analysis['true_positives'] or analysis['false_positives']:
            tp_scores = [tp['det_data']['score'] for tp in analysis['true_positives']]
            fp_scores = [fp['det_data']['score'] for fp in analysis['false_positives']]
            
            axes[0, 1].hist(tp_scores, bins=20, alpha=0.5, label='TP', edgecolor='black')
            axes[0, 1].hist(fp_scores, bins=20, alpha=0.5, label='FP', edgecolor='black')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Score Distributions')
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, 'No Detections', ha='center', va='center')
            axes[0, 1].set_title('Score Distributions')
        
        # 3. Error type breakdown
        error_counts = {
            'TP': len(analysis['true_positives']),
            'FP': len(analysis['false_positives']),
            'FN': len(analysis['false_negatives'])
        }
        
        axes[1, 0].bar(error_counts.keys(), error_counts.values())
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Detection Outcome Counts')
        
        # Add text labels
        for i, (k, v) in enumerate(error_counts.items()):
            axes[1, 0].text(i, v + 0.5, str(v), ha='center')
        
        # 4. FP error types
        if analysis['false_positives']:
            fp_types = [fp['error_type'] for fp in analysis['false_positives']]
            fp_type_counts = pd.Series(fp_types).value_counts()
            
            axes[1, 1].pie(fp_type_counts.values, labels=fp_type_counts.index,
                          autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('False Positive Error Types')
        else:
            axes[1, 1].text(0.5, 0.5, 'No False Positives', ha='center', va='center')
            axes[1, 1].set_title('False Positive Error Types')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_markdown_report(self, analysis: Dict, distributions: Dict) -> str:
        """Generate a markdown report for a class."""
        report = f"# Analysis Report: {analysis['class_name']}\n\n"
        
        # Summary statistics
        report += "## Summary Statistics\n\n"
        report += f"- **Ground Truth Instances**: {analysis['num_gt']}\n"
        report += f"- **Detections**: {analysis['num_det']}\n"
        report += f"- **True Positives**: {len(analysis['true_positives'])}\n"
        report += f"- **False Positives**: {len(analysis['false_positives'])}\n"
        report += f"- **False Negatives**: {len(analysis['false_negatives'])}\n\n"
        
        # Compute metrics
        tp = len(analysis['true_positives'])
        fp = len(analysis['false_positives'])
        fn = len(analysis['false_negatives'])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        report += f"- **Precision**: {precision:.3f}\n"
        report += f"- **Recall**: {recall:.3f}\n"
        report += f"- **F1 Score**: {f1:.3f}\n\n"
        
        # True Positive Analysis
        report += "## True Positive Analysis\n\n"
        if 'tp' in distributions:
            report += "### IoU Distribution\n"
            iou_stats = distributions['tp']['iou']
            report += f"- Mean: {iou_stats['mean']:.3f}\n"
            report += f"- Std: {iou_stats['std']:.3f}\n"
            report += f"- Min: {iou_stats['min']:.3f}\n"
            report += f"- Max: {iou_stats['max']:.3f}\n\n"
            
            report += "### Position Error (meters)\n"
            pos_stats = distributions['tp']['position_error']
            report += f"- Mean: {pos_stats['mean']:.3f}\n"
            report += f"- Std: {pos_stats['std']:.3f}\n"
            report += f"- Median: {pos_stats['50%']:.3f}\n\n"
        else:
            report += "*No true positives found*\n\n"
        
        # False Positive Analysis
        report += "## False Positive Analysis\n\n"
        if 'fp' in distributions:
            report += "### Error Type Breakdown\n"
            for error_type, count in distributions['fp']['error_types'].items():
                percentage = count / len(analysis['false_positives']) * 100
                report += f"- {error_type}: {count} ({percentage:.1f}%)\n"
            report += "\n"
            
            report += "### Confidence Score Statistics\n"
            score_stats = distributions['fp']['score']
            report += f"- Mean: {score_stats['mean']:.3f}\n"
            report += f"- Std: {score_stats['std']:.3f}\n"
            report += f"- Min: {score_stats['min']:.3f}\n"
            report += f"- Max: {score_stats['max']:.3f}\n\n"
        else:
            report += "*No false positives found*\n\n"
        
        # False Negative Analysis
        report += "## False Negative Analysis\n\n"
        if analysis['false_negatives']:
            miss_reasons = pd.Series([fn['miss_reason'] for fn in analysis['false_negatives']])
            miss_counts = miss_reasons.value_counts()
            
            report += "### Miss Reason Breakdown\n"
            for reason, count in miss_counts.items():
                percentage = count / len(analysis['false_negatives']) * 100
                report += f"- {reason}: {count} ({percentage:.1f}%)\n"
        else:
            report += "*No false negatives found*\n\n"
        
        return report
    
    def run_av2_eval(self):
        """Run the complete analysis pipeline."""
        # Load data
        print("Loading data...")
        dts, gts = self.load_data()
        
        # Run AV2 evaluation
        return self.run_av2_evaluation(dts, gts)

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        # Load data
        print("Loading data...")
        dts, gts = self.load_data()
        
        # Run AV2 evaluation
        print("Running AV2 evaluation...")
        eval_results = self.run_av2_evaluation(dts, gts)
        
        # Save overall metrics
        metrics_path = self.output_dir / "av2_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(
                eval_results['metrics'].to_dict() if hasattr(eval_results['metrics'], 'to_dict') else eval_results['metrics'],
                f, 
                indent=2,
                default=str
            )
        
        # Per-class analysis
        class_results = {}
        
        for class_name in self.config.longtail_classes:
            print(f"\nAnalyzing class: {class_name}")
            
            # Run analysis
            analysis = self.analyze_detections_per_class(class_name, dts, gts)
            distributions = self.compute_parameter_distributions(analysis)
            
            # Create visualizations
            viz_path = self.output_dir / f"{class_name.lower()}_analysis.png"
            self.create_visualizations(analysis, viz_path)
            
            # Generate report
            report = self.generate_markdown_report(analysis, distributions)
            report_path = self.output_dir / f"{class_name.lower()}_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            
            # Store results
            class_results[class_name] = {
                'analysis': analysis,
                'distributions': distributions
            }
            
            print(f"  - TP: {len(analysis['true_positives'])}")
            print(f"  - FP: {len(analysis['false_positives'])}")
            print(f"  - FN: {len(analysis['false_negatives'])}")
        
        # Save complete results
        results_path = self.output_dir / "complete_analysis.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(class_results, f)
        
        print(f"\nAnalysis complete! Results saved to {self.output_dir}")
        
        return class_results




def log_memory_usage(log_interval=5, output_file="memory_log.txt"):
    pid = os.getpid()
    process = psutil.Process(pid)
    with open(output_file, "a") as f:
        while True:
            mem_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)
            f.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"RSS: {mem_info.rss / 1024 ** 2:.2f} MB, "
                f"CPU: {cpu_percent:.2f}%\n"
            )
            f.flush()
            time.sleep(log_interval)


# Example usage
if __name__ == "__main__":
    # Start the logging thread
    threading.Thread(target=log_memory_usage, daemon=True).start()

    config = EvaluationConfig(
        predictions_path="/scratch/project_mnt/S0202/uqdetche/lidar-longtail-mining/lion/output/lion_models/lion_mamba_1f_1x_argo_128dim_sparse_v2/default/eval/epoch_2/val/default/processed_results.feather",
        ground_truth_path="/scratch/project_mnt/S0202/uqdetche/lidar-longtail-mining/lion/data/argo2/val_anno.feather",
        dataset_dir="/scratch/project_mnt/S0202/uqdetche/lidar-longtail-mining/lion/data/argo2/sensor/val",
        output_dir="./longtail_analysis_results"
    )
    
    evaluator = StandaloneLongTailEvaluator(config)
    results = evaluator.run_complete_analysis()