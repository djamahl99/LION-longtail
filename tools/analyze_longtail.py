#!/usr/bin/env python3
"""
analyze_longtail.py - Long-tail class analysis for LION predictions on AV2

This script analyzes LION model performance on long-tail classes in Argoverse 2,
focusing on failure modes and detailed metrics.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import json

# PCDet imports
from pcdet.datasets.argo2.argo2_dataset import Argo2Dataset
from pcdet.config import cfg, cfg_from_yaml_file

# AV2 evaluation imports
from av2.evaluation.detection.constants import CompetitionCategories
from av2.evaluation.detection.utils import DetectionCfg
from av2.evaluation.detection.eval import evaluate
from av2.utils.io import read_feather
from pcdet.ops.iou3d_nms import iou3d_nms_utils


class LongTailAnalyzer:
    """Analyzer for long-tail class performance in AV2."""
    
    def __init__(self, 
                 argo2_root: str,
                 predictions_path: str,
                 config_path: str,
                 output_dir: str = "./longtail_analysis"):
        """
        Initialize the analyzer.
        
        Args:
            argo2_root: Root path to AV2 dataset
            predictions_path: Path to LION predictions (pickle file)
            config_path: Path to config file
            output_dir: Directory to save analysis results
        """
        self.argo2_root = Path(argo2_root)
        self.predictions_path = predictions_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Long-tail classes of interest
        self.longtail_classes = [
            'articulated_bus', 'large_vehicle', 'message_board_trailer', 
            'sign', 'truck', 'truck_cab', 'vehicular_trailer', 
            'wheelchair', 'wheeled_rider'
        ]
        
        # Initialize dataset
        cfg_from_yaml_file(config_path, cfg)
        self.dataset = Argo2Dataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=Path(argo2_root),
            logger=self._setup_logger()
        )
        
        # IoU thresholds for analysis
        self.iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('LongTailAnalyzer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def load_predictions(self) -> List[Dict]:
        """Load LION predictions from pickle file."""
        self.dataset.logger.info(f"Loading predictions from {self.predictions_path}")
        
        if self.predictions_path.endswith('.pkl'):
            with open(self.predictions_path, 'rb') as f:
                predictions = pickle.load(f)
        elif self.predictions_path.endswith('.json'):
            with open(self.predictions_path, 'r') as f:
                predictions = json.load(f)
        else:
            raise ValueError(f"Unsupported prediction file format: {self.predictions_path}")
            
        self.dataset.logger.info(f"Loaded {len(predictions)} predictions")
        return predictions
        
    def filter_longtail_classes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data for long-tail classes only."""
        return data[data['category'].isin(self.longtail_classes)]
        
    def compute_detailed_metrics(self, 
                               eval_dts: pd.DataFrame, 
                               eval_gts: pd.DataFrame,
                               cfg: DetectionCfg) -> Dict:
        """Compute detailed metrics for long-tail classes."""
        
        # Filter for long-tail classes
        longtail_dts = self.filter_longtail_classes(eval_dts)
        longtail_gts = self.filter_longtail_classes(eval_gts)
        
        self.dataset.logger.info(f"Analyzing {len(longtail_dts)} detections and {len(longtail_gts)} ground truth instances")
        
        # Compute metrics per class
        detailed_metrics = {}
        
        for class_name in self.longtail_classes:
            class_dts = longtail_dts[longtail_dts['category'] == class_name]
            class_gts = longtail_gts[longtail_gts['category'] == class_name]
            
            if len(class_gts) == 0:
                self.dataset.logger.warning(f"No ground truth instances for class {class_name}")
                continue
                
            # Compute metrics at different IoU thresholds
            class_metrics = self._compute_class_metrics(class_dts, class_gts, class_name)
            detailed_metrics[class_name] = class_metrics
            
        return detailed_metrics
        
    def _compute_class_metrics(self, 
                              class_dts: pd.DataFrame, 
                              class_gts: pd.DataFrame,
                              class_name: str) -> Dict:
        """Compute metrics for a specific class."""
        
        metrics = {
            'class_name': class_name,
            'num_gt': len(class_gts),
            'num_det': len(class_dts),
            'iou_metrics': {}
        }
        
        for iou_thresh in self.iou_thresholds:
            # Compute precision, recall at this IoU threshold
            tp, fp, fn = self._compute_tp_fp_fn(class_dts, class_gts, iou_thresh)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics['iou_metrics'][iou_thresh] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
        return metrics
        
    def _compute_tp_fp_fn(self, 
                         dts: pd.DataFrame, 
                         gts: pd.DataFrame, 
                         iou_threshold: float) -> Tuple[int, int, int]:
        """Compute true positives, false positives, false negatives."""
        
        # Group by log_id and timestamp for matching
        tp = fp = fn = 0
        
        # Get unique scenes
        scenes = set(gts.index.get_level_values(0).tolist()) | set(dts.index.get_level_values(0).tolist())
        
        for scene in scenes:
            scene_gts = gts.loc[gts.index.get_level_values(0) == scene] if scene in gts.index.get_level_values(0) else pd.DataFrame()
            scene_dts = dts.loc[dts.index.get_level_values(0) == scene] if scene in dts.index.get_level_values(0) else pd.DataFrame()
            
            if len(scene_gts) == 0:
                fp += len(scene_dts)
                continue
                
            if len(scene_dts) == 0:
                fn += len(scene_gts)
                continue
                
            # Compute IoU matrix and find matches
            scene_tp, scene_fp, scene_fn = self._match_boxes_in_scene(scene_dts, scene_gts, iou_threshold)
            tp += scene_tp
            fp += scene_fp
            fn += scene_fn
            
        return tp, fp, fn
        
    def _match_boxes_in_scene(self, 
                             scene_dts: pd.DataFrame, 
                             scene_gts: pd.DataFrame,
                             iou_threshold: float) -> Tuple[int, int, int]:
        """Match detection and ground truth boxes in a scene."""
        
        # Simple matching based on overlap (simplified version)
        # In practice, you'd compute 3D IoU here
        
        matched_gts = set()
        matched_dts = set()

        iou_min = 1.0
        iou_max = 0.0
        
        for dt_idx, dt_row in scene_dts.iterrows():
            best_iou = 0
            best_gt_idx = None
            
            for gt_idx, gt_row in scene_gts.iterrows():
                if gt_idx in matched_gts:
                    continue
                    
                # Compute IoU (simplified - you'd use actual 3D IoU computation)
                iou = self._compute_3d_iou(dt_row, gt_row)
                iou_min = min(iou, iou_min)
                iou_max = max(iou, iou_max)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    
            if best_iou >= iou_threshold:
                matched_gts.add(best_gt_idx)
                matched_dts.add(dt_idx)
                
        tp = len(matched_dts)
        fp = len(scene_dts) - len(matched_dts)
        fn = len(scene_gts) - len(matched_gts)

        print(f"{iou_min=} {iou_max=}")
        print(f"{tp=} {fp=} {fn=}")
        
        return tp, fp, fn
        
    def _compute_3d_iou(self, det_row: pd.Series, gt_row: pd.Series) -> float:
        """Compute 3D IoU between detection and ground truth."""
        import torch
        from pcdet.ops.iou3d_nms import iou3d_nms_utils
        
        # Extract box parameters from detection
        det_box = self._extract_box_params(det_row)
        
        # Extract box parameters from ground truth  
        gt_box = self._extract_box_params(gt_row)
        
        # Convert to tensors (reshape to (1, 7) for single boxes)
        det_tensor = torch.tensor(det_box, dtype=torch.float32).cuda().unsqueeze(0)  # (1, 7)
        gt_tensor = torch.tensor(gt_box, dtype=torch.float32).cuda().unsqueeze(0)    # (1, 7)
        
        # Compute 3D IoU
        iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(det_tensor, gt_tensor)  # (1, 1)
        
        # Extract the single IoU value
        iou_value = iou_matrix[0, 0].cpu().item()
        
        return iou_value
    
    def _extract_box_params(self, row: pd.Series) -> List[float]:
        """Extract box parameters in format [x, y, z, dx, dy, dz, heading]."""
        
        # AV2 format typically has these columns:
        # tx_m, ty_m, tz_m for translation (center coordinates)
        # length_m, width_m, height_m for dimensions
        # qw, qx, qy, qz for quaternion rotation
        
        try:
            # Extract translation (center coordinates)
            x = row['tx_m']
            y = row['ty_m'] 
            z = row['tz_m']
            
            # Extract dimensions
            dx = row['length_m']  # length
            dy = row['width_m']   # width  
            dz = row['height_m']  # height
            
            # Convert quaternion to heading (yaw angle)
            # AV2 uses quaternion [qw, qx, qy, qz]
            qw = row['qw']
            qx = row['qx']
            qy = row['qy'] 
            qz = row['qz']
            
            # Convert quaternion to yaw angle (heading)
            heading = self._quaternion_to_yaw(qw, qx, qy, qz)
            
            return [x, y, z, dx, dy, dz, heading]
            
        except KeyError as e:
            # Fallback for different column naming conventions
            self.dataset.logger.warning(f"Missing expected column {e}, trying alternative names")
            
            # Try alternative column names
            try:
                x = row.get('x', row.get('center_x', row.get('tx_m', 0)))
                y = row.get('y', row.get('center_y', row.get('ty_m', 0)))
                z = row.get('z', row.get('center_z', row.get('tz_m', 0)))
                
                dx = row.get('length', row.get('l', row.get('length_m', 1)))
                dy = row.get('width', row.get('w', row.get('width_m', 1)))
                dz = row.get('height', row.get('h', row.get('height_m', 1)))
                
                # Try to get heading directly or compute from quaternion
                if 'heading' in row:
                    heading = row['heading']
                elif 'yaw' in row:
                    heading = row['yaw']
                elif all(q in row for q in ['qw', 'qx', 'qy', 'qz']):
                    heading = self._quaternion_to_yaw(row['qw'], row['qx'], row['qy'], row['qz'])
                else:
                    heading = 0.0  # Default heading
                    
                return [x, y, z, dx, dy, dz, heading]
                
            except Exception as e2:
                self.dataset.logger.error(f"Could not extract box parameters: {e2}")
                self.dataset.logger.error(f"Available columns: {list(row.index)}")
                # Return a default box to avoid crashes
                return [0, 0, 0, 1, 1, 1, 0]
    
    def _quaternion_to_yaw(self, qw: float, qx: float, qy: float, qz: float) -> float:
        """Convert quaternion to yaw angle (heading)."""
        # Convert quaternion to yaw angle using atan2
        # For a quaternion [qw, qx, qy, qz], yaw is:
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return yaw
        
    def analyze_failure_modes(self, 
                             eval_dts: pd.DataFrame, 
                             eval_gts: pd.DataFrame) -> Dict:
        """Analyze failure modes: missed detections vs false positives."""
        
        failure_analysis = {}
        
        print("eval_dts", eval_dts)
        print("eval_gts", eval_gts)

        for class_name in self.longtail_classes:
            class_dts = eval_dts[eval_dts['category'] == class_name.upper()]
            class_gts = eval_gts[eval_gts['category'] == class_name.upper()]

            gt_cats = set(eval_gts['category'])
            
            if len(class_gts) == 0:
                print(f"{class_name=} no gt?")
                print(f"{gt_cats=}")
                continue
                
            # Analyze missed detections (false negatives)
            missed_detections = self._find_missed_detections(class_dts, class_gts)
            
            # Analyze false positives
            false_positives = self._find_false_positives(class_dts, class_gts)
            
            failure_analysis[class_name] = {
                'missed_detections': missed_detections,
                'false_positives': false_positives,
                'missed_count': len(missed_detections),
                'fp_count': len(false_positives)
            }


        # Save failure analysis summary
        failure_summary = {}
        for class_name, failures in failure_analysis.items():
            failure_summary[class_name] = {
                'missed_count': failures['missed_count'],
                'fp_count': failures['fp_count'],
                'total_gt': len(failures['missed_detections']) + failures['missed_count']
            }
            
        summary_file = self.output_dir / "failure_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(failure_summary, f, indent=2)
            
        return failure_analysis
        
    def _find_missed_detections(self, 
                               class_dts: pd.DataFrame, 
                               class_gts: pd.DataFrame) -> List[Dict]:
        """Find ground truth instances that were missed."""
        
        missed = []
        
        # Group by scene
        for scene_id in class_gts.index.get_level_values(0).unique():
            scene_gts = class_gts.loc[class_gts.index.get_level_values(0) == scene_id]
            scene_dts = class_dts.loc[class_dts.index.get_level_values(0) == scene_id] if scene_id in class_dts.index.get_level_values(0) else pd.DataFrame()
            
            for gt_idx, gt_row in scene_gts.iterrows():
                # Check if this GT has a matching detection
                if not self._has_matching_detection(gt_row, scene_dts):
                    missed.append({
                        'log_id': gt_idx[0],
                        'timestamp_ns': gt_idx[1],
                        'gt_data': gt_row.to_dict()
                    })
                    
        return missed
        
    def _find_false_positives(self, 
                             class_dts: pd.DataFrame, 
                             class_gts: pd.DataFrame) -> List[Dict]:
        """Find detections that are false positives."""
        
        false_positives = []
        
        # Group by scene
        for scene_id in class_dts.index.get_level_values(0).unique():
            scene_dts = class_dts.loc[class_dts.index.get_level_values(0) == scene_id]
            scene_gts = class_gts.loc[class_gts.index.get_level_values(0) == scene_id] if scene_id in class_gts.index.get_level_values(0) else pd.DataFrame()
            
            for dt_idx, dt_row in scene_dts.iterrows():
                # Check if this detection has a matching GT
                # print("dt_idx, dt_row", dt_idx, dt_row)
                if not self._has_matching_gt(dt_row, scene_gts):
                    false_positives.append({
                        'log_id': dt_row['log_id'],
                        'timestamp_ns': dt_row['timestamp_ns'],
                        'det_data': dt_row.to_dict()
                    })
                    
        return false_positives
        
    def _has_matching_detection(self, gt_row: pd.Series, scene_dts: pd.DataFrame, iou_thresh: float = 0.5) -> bool:
        """Check if ground truth has a matching detection."""
        for _, dt_row in scene_dts.iterrows():
            if self._compute_3d_iou(dt_row, gt_row) >= iou_thresh:
                print(f"matched {dt_row=} {gt_row=}")
                return True
        return False
        
    def _has_matching_gt(self, dt_row: pd.Series, scene_gts: pd.DataFrame, iou_thresh: float = 0.5) -> bool:
        """Check if detection has a matching ground truth."""
        for _, gt_row in scene_gts.iterrows():
            if self._compute_3d_iou(dt_row, gt_row) >= iou_thresh:
                print(f"matched {dt_row=} {gt_row=}")
                return True
        return False
        
    def save_failure_examples(self, failure_analysis: Dict) -> None:
        """Save examples of failures for visualization."""
        
        examples_dir = self.output_dir / "failure_examples"
        examples_dir.mkdir(exist_ok=True)
        
        for class_name, failures in failure_analysis.items():
            class_dir = examples_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Save missed detections
            missed_file = class_dir / "missed_detections.json"
            with open(missed_file, 'w') as f:
                json.dump(failures['missed_detections'][:50], f, indent=2, default=str)  # Limit to 50 examples
                
            # Save false positives
            fp_file = class_dir / "false_positives.json"
            with open(fp_file, 'w') as f:
                json.dump(failures['false_positives'][:50], f, indent=2, default=str)  # Limit to 50 examples
                
        self.dataset.logger.info(f"Saved failure examples to {examples_dir}")
        
    def run_analysis(self) -> None:
        """Run the complete long-tail analysis."""
        
        self.dataset.logger.info("Starting long-tail analysis")
        
        # Load predictions
        predictions = self.load_predictions()
        
        # Format predictions using dataset method
        dts = self.dataset.format_results(predictions, self.dataset.class_names)
        
        # Load ground truth
        val_anno_path = self.argo2_root / 'val_anno.feather'
        gts = read_feather(Path(val_anno_path))
        gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")

        # Get valid UUIDs from ground truth
        valid_uuid_set = set(gts.index.tolist())
        print(f"Found {len(valid_uuid_set)} valid UUID pairs in ground truth")
        
        # Method 1: If dts is already in memory, filter it directly but more efficiently
        if isinstance(dts, pd.DataFrame):
            print("Filtering detections to valid UUIDs...")
            # Create a combined key for faster lookup
            dts['_uuid_key'] = dts['log_id'].astype(str) + '_' + dts['timestamp_ns'].astype(str)
            gts_keys = set([f"{log_id}_{ts}" for log_id, ts in valid_uuid_set])
            
            # Filter using isin (more efficient than list comprehension)
            dts_filtered = dts[dts['_uuid_key'].isin(gts_keys)].drop(columns=['_uuid_key'])
            print(f"Filtered detections from {len(dts)} to {len(dts_filtered)}")
        else:
            # If dts is a file path, we need a different approach
            print("Loading and filtering detections from file...")
            # For now, just load it all - we'll optimize this if needed
            dts = pd.read_feather(dts)
            dts['_uuid_key'] = dts['log_id'].astype(str) + '_' + dts['timestamp_ns'].astype(str)
            gts_keys = set([f"{log_id}_{ts}" for log_id, ts in valid_uuid_set])
            dts_filtered = dts[dts['_uuid_key'].isin(gts_keys)].drop(columns=['_uuid_key'])
        
        # Filter gts to only include UUIDs that are in both
        dts_uuid_set = set(zip(dts_filtered['log_id'], dts_filtered['timestamp_ns']))
        valid_uuids = list(valid_uuid_set & dts_uuid_set)
        gts = gts.loc[valid_uuids].sort_index()
        
        # Set up evaluation config
        from av2.evaluation.detection.constants import CompetitionCategories
        categories = set(x.value for x in CompetitionCategories)
        categories &= set(gts["category"].unique().tolist())
        
        dataset_dir = self.argo2_root / 'sensor' / 'val'
        cfg = DetectionCfg(
            dataset_dir=dataset_dir,
            categories=tuple(sorted(categories)),
            max_range_m=self.dataset.evaluate_range,
            eval_only_roi_instances=True,
        )
        
        # Run evaluation
        # self.dataset.logger.info("Running evaluation")
        # eval_dts, eval_gts, metrics = evaluate(dts.reset_index(), gts.reset_index(), cfg)
        # eval_dts, eval_gts, metrics = evaluate(
        #     dts, gts.reset_index(), cfg, n_jobs=1
        # )

        # Compute detailed metrics for long-tail classes
        # self.dataset.logger.info("Computing detailed metrics")
        # detailed_metrics = self.compute_detailed_metrics(eval_dts, eval_gts, cfg)
        
        # Analyze failure modes
        self.dataset.logger.info("Analyzing failure modes")
        # failure_analysis = self.analyze_failure_modes(eval_dts, eval_gts)
        failure_analysis = self.analyze_failure_modes(dts, gts)
        
        # Save results
        # self._save_results(detailed_metrics, failure_analysis, metrics)
        
        # Save failure examples
        self.save_failure_examples(failure_analysis)
        
        self.dataset.logger.info("Analysis complete!")
        
    def _save_results(self, detailed_metrics: Dict, failure_analysis: Dict, overall_metrics: pd.DataFrame) -> None:
        """Save analysis results."""
        
        # Save detailed metrics
        metrics_file = self.output_dir / "longtail_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2, default=str)
            
        # Save failure analysis summary
        failure_summary = {}
        for class_name, failures in failure_analysis.items():
            failure_summary[class_name] = {
                'missed_count': failures['missed_count'],
                'fp_count': failures['fp_count'],
                'total_gt': len(failures['missed_detections']) + failures['missed_count']
            }
            
        summary_file = self.output_dir / "failure_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(failure_summary, f, indent=2)
            
        # Save overall metrics
        overall_file = self.output_dir / "overall_metrics.csv"
        overall_metrics.to_csv(overall_file)
        
        # Create summary report
        self._create_summary_report(detailed_metrics, failure_summary)
        
    def _create_summary_report(self, detailed_metrics: Dict, failure_summary: Dict) -> None:
        """Create a human-readable summary report."""
        
        report_file = self.output_dir / "summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("LONG-TAIL CLASS ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            for class_name in self.longtail_classes:
                if class_name not in detailed_metrics:
                    f.write(f"{class_name.upper()}: No data available\n\n")
                    continue
                    
                metrics = detailed_metrics[class_name]
                failures = failure_summary.get(class_name, {})
                
                f.write(f"{class_name.upper()}:\n")
                f.write(f"  Ground Truth Instances: {metrics['num_gt']}\n")
                f.write(f"  Detections: {metrics['num_det']}\n")
                f.write(f"  Missed Detections: {failures.get('missed_count', 0)}\n")
                f.write(f"  False Positives: {failures.get('fp_count', 0)}\n")
                
                f.write(f"  Performance at IoU=0.5:\n")
                if 0.5 in metrics['iou_metrics']:
                    iou_50 = metrics['iou_metrics'][0.5]
                    f.write(f"    Precision: {iou_50['precision']:.3f}\n")
                    f.write(f"    Recall: {iou_50['recall']:.3f}\n")
                    f.write(f"    F1: {iou_50['f1']:.3f}\n")
                
                f.write("\n")
                
        self.dataset.logger.info(f"Summary report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze long-tail class performance')
    parser.add_argument('--argo2_root', type=str, required=True,
                        help='Root path to AV2 dataset')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to LION predictions file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./longtail_analysis',
                        help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = LongTailAnalyzer(
        argo2_root=args.argo2_root,
        predictions_path=args.predictions,
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    analyzer.run_analysis()


if __name__ == "__main__":
    main()