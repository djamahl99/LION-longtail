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

from lion.unsupervised_core.convex_hull_tracker.alpha_shape_utils import AlphaShapeUtils
from lion.unsupervised_core.convex_hull_tracker.convex_hull_utils import relative_object_pose, rigid_icp
from lion.unsupervised_core.convex_hull_tracker.pose_kalman_filter import PoseKalmanFilter
from lion.unsupervised_core.convex_hull_tracker.convex_hull_track import ConvexHullTrack, ConvexHullTrackState
from lion.unsupervised_core.convex_hull_tracker import nn_matching


from lion.unsupervised_core.outline_utils import (
    OutlineFitter,
    voxel_sampling,
    points_rigid_transform,
)
from lion.unsupervised_core.tracker.box_op import register_bbs
from lion.unsupervised_core.trajectory_optimizer import (
    GlobalTrajectoryOptimizer,
    optimize_with_gtsam_timed,
    simple_pairwise_icp_refinement,
)

from .convex_hull_object import ConvexHullObject

from lion.unsupervised_core.convex_hull_tracker import linear_assignment


from lion.unsupervised_core.box_utils import *
from lion.unsupervised_core.file_utils import load_predictions_parallel
from sklearn.cluster import DBSCAN

np.set_printoptions(suppress=True, precision=2)

class ConvexHullKalmanTracker:

    def __init__(self, config=None, debug: bool = False):
        self.config = config
        self.debug = debug
        self.tracks: List[ConvexHullTrack] = [] # id -> track

        nn_budget = None
        max_cosine_distance = 0.2
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)

        self.kf = PoseKalmanFilter(dt=0.1)

        self._next_id = 0
        self.all_pose = None
        self.frame_tracks: Dict[int, List[int]] = {} # timestamp_ns -> tracks
        self.icp_fail_max = 3
        self.ppscore_thresh = 0.7
        self.track_query_eps = 3.0  # metres
        self.max_iou_distance = 0.9

        self.min_semantic_threshold = 0.7
        self.min_iou_threshold = 0.1
        self.min_box_iou = 0.1
        self.icp_max_dist = 1.0

        self.nms_iou_threshold = 0.7
        self.nms_semantic_threshold: float = 0.9
        self.nms_query_distance: float = 3.0


        self.n_init = 3
        self.max_age = 10 # 10 frames -> 1 second
        


        self.updates_per_track = {}

        # Debug statistics
        if self.debug:
            self.debug_stats = {
                "total_new_tracks": 0,
                "total_updates": 0,
                "frame_stats": {},
                "iou_history": [],
            }

    def predict(self, timestamp: int):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf, timestamp)

    def update(self, convex_hulls: List[ConvexHullObject], pose: np.ndarray, timestamp_ns: int):
        # Run matching cascade.
        # matches, unmatched_tracks, unmatched_detections = \
        #     self._match(convex_hulls, pose)

        matches, unmatched_tracks, unmatched_detections = \
            self._match_full(convex_hulls, pose)

        print(f"Made {len(matches)} matches")
        print(f"Made {len(unmatched_tracks)} unmatched tracks")
        print(f"Made {len(unmatched_detections)} unmatched dets")

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, convex_hulls[detection_idx])
            track_id = self.tracks[track_idx].track_id
            self.updates_per_track[track_id] += 1
            # print(f"self.updates_per_track[track_id]", self.updates_per_track[track_id])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # tracks_tree = cKDTree([x.to_box()[:3] for x in self.tracks if not x.is_deleted()])
        for detection_idx in unmatched_detections:
            # ious = np.array([self._box_iou_3d(convex_hulls[detection_idx].box, x.to_box()) for x in self.tracks])
            convex_hull_box = convex_hulls[detection_idx].box
            convex_hull_feature = convex_hulls[detection_idx].feature
            found_match = False
            for track in self.tracks:
                iou = self._box_iou_3d(convex_hull_box, track.to_box())
                semantic_overlap = np.dot(convex_hull_feature, track.features[-1])
                if iou > self.nms_iou_threshold and semantic_overlap > self.nms_semantic_threshold:
                    found_match = True
                    break

            if not found_match:
                self._initiate_track(convex_hulls[detection_idx])

        self.track_nms()

        # self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # # Update distance metric.
        # active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # features, targets = [], []
        # for track in self.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     features += track.features
        #     targets += [track.track_id for _ in track.features]
        #     track.features = []
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)
            
        # # # self.frame_tracks[timestamp_ns] = frame_assignments

    def track_nms(self):
        def get_priority_score(track: ConvexHullTrack):
            avg_confidence = np.mean([x.confidence for x in track.history])
            return (len(track.history), avg_confidence)

        sorted_tracks = sorted([x for x in self.tracks if not x.is_deleted()], key=get_priority_score, reverse=True)

        sorted_boxes = np.stack([x.to_box() for x in sorted_tracks], axis=0)
        sorted_positions = sorted_boxes[:, :3]
        sorted_features = np.stack([np.mean(np.stack(x.features, axis=0), axis=0) for x in sorted_tracks], axis=0)
        sorted_track_ids = np.array([x.track_id for x in sorted_tracks], dtype=int)

        sorted_features = sorted_features / (np.linalg.norm(sorted_features, axis=1, keepdims=True) + 1e-6)

        print(f"sorted_features={sorted_features.shape} {sorted_boxes.shape} {sorted_positions.shape} {sorted_track_ids.shape}")

        tracks_tree = cKDTree(sorted_positions)

        keep_indices = []
        suppressed = set()

        semantic_overlaps = sorted_features @ sorted_features.T

        print(f"{semantic_overlaps.shape=} {semantic_overlaps.min()} {semantic_overlaps.max()}")

        num_tracks = len(sorted_tracks)
        for i in range(num_tracks):
            if i in suppressed:
                continue

            keep_indices.append(i)

            # Check all remaining clusters for overlap
            # for j in range(i+1, num_clusters):
            indices = tracks_tree.query_ball_point(
                sorted_positions[i], self.nms_query_distance
            )
            for j in indices:
                if j in suppressed or j <= i:
                    continue

                iou = self._box_iou_3d(sorted_boxes[i], sorted_boxes[j])
                semantic_overlap = semantic_overlaps[i, j]

                # Suppress the lower-priority cluster if IoU is high
                if iou > self.nms_iou_threshold and semantic_overlap > self.nms_semantic_threshold:
                    suppressed.add(j)

        # Return the kept clusters in original order (not sorted order)
        for idx in suppressed:
            track_id = sorted_track_ids[idx]

            self.tracks[track_id].state = ConvexHullTrackState.Deleted

    def _match_full(self, detections, pose):
        n_tracks = len(self.tracks)
        n_dets = len(detections)

        if n_tracks == 0 or n_dets == 0:
            tracks_unmatched = set([i for i in range(n_tracks)])
            dets_unmatched = set([i for i in range(n_dets)])

            return [], list(tracks_unmatched), list(dets_unmatched) 
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((n_tracks, n_dets))
        semantic_matrix = np.zeros((n_tracks, n_dets))
        cost_matrix = np.full((n_tracks, n_dets), 100.0)
        icp_matrix = np.full((n_tracks, n_dets), 100.0)

        icp_matrix_max = 0.0


        # Prepare for geometric analysis
        detection_centroids = []

        for det_idx, detection in enumerate(detections):
            detection_centroids.append(detection.centroid_3d)
        
        # Build KDTree for spatial queries
        detections_tree = cKDTree(detection_centroids)

        # Predict track positions and shapes using Kalman + pose transformation
        predicted_centroids = []
        predicted_shapes = []
        predicted_boxes = []
        
        for track in self.tracks:
            # Get predicted bounding box from Kalman filter
            predicted_center = track.mean[:3]
            
            predicted_boxes.append(track.to_box())

            predicted_shape = track.to_shape_dict()

            predicted_centroids.append(predicted_center)
            predicted_shapes.append(predicted_shape)

        for track_idx, track in enumerate(self.tracks):
            if track.is_deleted():
                continue

            close_detection_indices = detections_tree.query_ball_point(
                predicted_centroids[track_idx], self.track_query_eps
            )

            predicted_shape = predicted_shapes[track_idx]


            for local_det_idx in close_detection_indices:
                # Compute geometric costs

                semantic_iou = np.dot(detections[local_det_idx].feature, track.features[-1])

                if semantic_iou < self.min_semantic_threshold:
                    continue

                # IoU cost
                iou = 0.0
                # orig_iou = self._convex_hull_iou_trimesh(predicted_shapes[track_idx]['mesh'], detections[local_det_idx].mesh)
                # iou = self._box_iou_3d(predicted_boxes[track_idx], detections[local_det_idx].box)

                icp_err = 0.0
                # ICP cost (simplified - you may want to use your existing ICP function
                R, t, _, _, icp_err = rigid_icp(predicted_shape['original_points'], detections[local_det_idx].original_points, max_iterations=5, debug=False, relative=False)
                # R, t, icp_err = icp(predicted_shape['original_points'], detections[local_det_idx].original_points, max_iterations=5, ret_err=True)

                transform = np.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = t

                # mesh_projected: trimesh.Trimesh = predicted_shapes[track_idx]['mesh'].copy()
                # mesh_projected = mesh_projected.apply_transform(transform)
                # iou = self._convex_hull_iou_trimesh(mesh_projected, detections[local_det_idx].mesh)

                # box iou
                box = np.copy(predicted_boxes[track_idx])
                box_transformed = register_bbs(box.copy().reshape(1, 7), transform)[0]
                box_diff = box_transformed - box

                if np.linalg.norm(box_diff) > 0.1:
                    print(f"box before", np.round(predicted_boxes[track_idx], 2), "box after", np.round(box_transformed, 2), "box_diff", np.round(box_diff, 2))
                
                iou = self._box_iou_3d(box_transformed, detections[local_det_idx].box)

                
                # iou = self._convex_hull_iou_trimesh(predicted_shapes[track_idx]['mesh'], detections[local_det_idx].mesh)

                # box = register_bbs(predicted_boxes[track_idx].reshape(1, 7), transform)[0]
                # iou = self._box_iou_3d(box, detections[local_det_idx].box)

                # distance = np.linalg.norm(t)

                # if distance > self.track_query_eps:
                #     continue

                # print(f"distance={distance:.2f} iou={iou:.2f} icp_err={icp_err:.2f} semantic_iou={semantic_iou:.2f}")
                
                # cost = (
                #     (1.0 - iou) + (1.0 - semantic_iou) + icp_err
                # )
                # cost_matrix[track_idx, local_det_idx] = cost
                iou_matrix[track_idx, local_det_idx] = iou
                semantic_matrix[track_idx, local_det_idx] = semantic_iou
                icp_matrix[track_idx, local_det_idx] = icp_err
                icp_matrix_max = max(icp_err, icp_matrix_max)

        print("icp_matrix", icp_matrix.shape, icp_matrix.min(), icp_matrix.max())
        icp_matrix_normed = icp_matrix / max(1.0, icp_matrix_max)
        print(f"icp_matrix_normed", icp_matrix_normed.min(), icp_matrix_normed.mean(), icp_matrix_normed.max())

        cost_matrix = (1.0 - iou_matrix) + (1.0 - semantic_matrix) #+ icp_matrix_normed

        # track_matches = np.argmin(cost_matrix, axis=1)
        # track_matches = [i for i, x in enumerate(track_matches) if (iou_matrix[i, x] > (1.0 - self.max_iou_distance)) and (semantic_matrix[i, x] > self.min_semantic_threshold)]
        # print(f"default matcehes {len(track_matches)}")

        # # Apply Kalman filter gating to semantic costs
        # cost_matrix = linear_assignment.gate_cost_matrix(
        #     self.kf, cost_matrix, self.tracks, detections, np.arange(n_tracks),
        #     np.arange(n_dets))

        # compute matches
        track_matches = np.argmin(cost_matrix, axis=1)
        # track_matches = [i for i, x in enumerate(track_matches) if (iou_matrix[i, x] > (1.0 - self.max_iou_distance)) and (semantic_matrix[i, x] > self.min_semantic_threshold)]

        
        track_set = set([i for i in range(n_tracks)])
        det_set = set([i for i in range(n_dets)])

        matched_tracks = set()
        matched_dets = set()

        matches = []
        for track_idx, matched_det in enumerate(track_matches):
            print(f"mach {track_idx=} {matched_det=} iou={iou_matrix[track_idx, matched_det]} semantic={semantic_matrix[track_idx, matched_det]}")
            print(f"icp_err={icp_matrix[track_idx, matched_det]}")
            if iou_matrix[track_idx, matched_det] >= self.min_iou_threshold and semantic_matrix[track_idx, matched_det] > self.min_semantic_threshold:
            # if icp_matrix[track_idx, matched_det] < self.icp_max_dist and semantic_matrix[track_idx, matched_det] > self.min_semantic_threshold:
                matches.append((track_idx, matched_det))

                matched_tracks.add(track_idx)
                matched_dets.add(matched_det)


        tracks_unmatched = track_set.difference(matched_tracks)
        dets_unmatched = det_set.difference(matched_dets)

        print(f"valid gated matches {len(matches)}")

        print(f"matched_tracks: {len(matched_tracks)} tracks_unmatched {len(tracks_unmatched)}")
        print(f"matched_tracks: {len(matched_dets)} dets_unmatched {len(dets_unmatched)}")



        return matches, list(tracks_unmatched), list(dets_unmatched)

    def _match(self, detections, pose):
        """Hybrid matching that combines semantic features, Kalman filtering, and geometric analysis."""
        
        def enhanced_gated_metric(tracks, dets, track_indices, detection_indices):
            """Enhanced gating that combines semantic, spatial, and geometric costs."""
            # Original semantic distance matrix
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            semantic_cost_matrix = self.metric.distance(features, targets)
            
            # Apply Kalman filter gating to semantic costs
            semantic_cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, semantic_cost_matrix, tracks, dets, track_indices,
                detection_indices)
            
            # If no valid detections after gating, return semantic costs
            if len(detection_indices) == 0 or len(track_indices) == 0:
                return semantic_cost_matrix
            
            # Prepare for geometric analysis
            detection_centroids = []
            
            for det_idx in detection_indices:
                detection = dets[det_idx]
                detection_centroids.append(detection.centroid_3d)
            
            # Build KDTree for spatial queries
            if len(detection_centroids) > 0:
                detections_tree = cKDTree(detection_centroids)
            
            # Predict track positions and shapes using Kalman + pose transformation
            predicted_centroids = []
            predicted_alpha_shapes = []
            
            for track_idx in track_indices:
                track = tracks[track_idx]
                
                # Get predicted bounding box from Kalman filter
                predicted_center = track.mean[:3]
                
                predicted_shape = track.to_shape_dict()

                predicted_centroids.append(predicted_center)
                predicted_alpha_shapes.append(predicted_shape)
            
            # Initialize combined cost matrix with semantic costs
            combined_cost_matrix = semantic_cost_matrix.copy()
            
            for i, track_idx in enumerate(track_indices):
                if predicted_centroids[i] is None:
                    continue
                    
                # Find spatially close detections
                if len(detection_centroids) > 0:
                    close_detection_indices = detections_tree.query_ball_point(
                        predicted_centroids[i], self.track_query_eps
                    )
                    
                    for local_det_idx in close_detection_indices:
                        j = local_det_idx  # Index in detection_indices
                        if j >= len(detection_indices):
                            continue
                        
                        # Skip if semantic cost is already too high (gated out)
                        # if semantic_cost_matrix[i, j] > self.metric.matching_threshold:
                            # continue
                        
                        # Compute geometric costs
                        predicted_shape = predicted_alpha_shapes[i]
                        
                        # IoU cost
                        iou = self._convex_hull_iou_trimesh(predicted_shape['mesh'], detections[j].mesh)
                        # iou =
                            
                        # ICP cost (simplified - you may want to use your existing ICP function
                        R, t, _, _, icp_cost = rigid_icp(predicted_shape['original_points'], detections[j].original_points, max_iterations=5, debug=False)
                        # R, t, icp_cost = icp(predicted_shape['original_points'], detections[j].original_points, max_iterations=5, ret_err=True)
                        

                        # Combine costs: semantic + geometric
                        geometric_cost = icp_cost + (1.0 - iou)
                        
                        # Weight combination (adjust weights as needed)
                        semantic_weight = 0.4
                        geometric_weight = 0.6
                        
                        combined_cost = (
                            semantic_weight * semantic_cost_matrix[i, j] +
                            geometric_weight * geometric_cost
                        )
                        
                        combined_cost_matrix[i, j] = combined_cost
            
            return combined_cost_matrix
        
        def geometric_iou_metric(tracks, dets, track_indices, detection_indices):
            """Pure geometric matching for unconfirmed tracks."""
            cost_matrix = np.ones((len(track_indices), len(detection_indices)))
            
            for i, track_idx in enumerate(track_indices):
                track = tracks[track_idx]
                # predicted_box = track.to_box()
                predicted_shape = track.to_shape_dict()
                
                for j, det_idx in enumerate(detection_indices):
                    detection = dets[det_idx]
                    
                    # Simple 3D box IoU as fallback
                    # iou = self._box_iou_3d(predicted_box, detection.box)
                    iou = self._convex_hull_iou_trimesh(predicted_shape['mesh'], detection.mesh)
                    cost_matrix[i, j] = 1.0 - iou
                    
            return cost_matrix

        def icp_metric(tracks, dets, track_indices, detection_indices):
            """Pure geometric matching for unconfirmed tracks."""
            cost_matrix = np.ones((len(track_indices), len(detection_indices)))
            
            for i, track_idx in enumerate(track_indices):
                track = tracks[track_idx]
                # predicted_box = track.to_box()
                predicted_shape = track.to_shape_dict()
                
                for j, det_idx in enumerate(detection_indices):
                    detection = dets[det_idx]
                    
                    # Simple 3D box IoU as fallback
                    # iou = self._box_iou_3d(predicted_box, detection.box)
                    R, t, _, _, icp_err = rigid_icp(predicted_shape['original_points'], detection.original_points, max_iterations=5, debug=False)
                    
                    # R, t, icp_err = icp(predicted_shape['original_points'], detection.original_points, max_iterations=5, ret_err=True)


                    cost_matrix[i, j] = icp_err
                    
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using enhanced gated metric (semantic + geometric)
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                enhanced_gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks with unconfirmed tracks using geometric IoU
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                icp_metric, self.icp_max_dist, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _convex_hull_iou_trimesh(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> float:
        """
        Compute IoU using cached geometric objects - much faster than original.

        Args:
            shape1, shape2: Alpha shape dictionaries with cached geometry

        Returns:
            IoU value between 0 and 1
        """

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

    def _box_iou_3d(self, box1: np.ndarray, box2: np.ndarray):
        """
        Calculate 3D IoU between two rotated bounding boxes.
        
        Args:
            box1: [x, y, z, l, w, h, yaw] - center coordinates, dimensions, and rotation
            box2: [x, y, z, l, w, h, yaw] - center coordinates, dimensions, and rotation
        
        Returns:
            float: 3D IoU value between 0 and 1
        """
        
        def get_box_corners_2d(box):
            """Get 2D corners of rotated rectangle in XY plane."""
            x, y, z, l, w, h, yaw = box
            
            # Half dimensions
            half_l, half_w = l / 2, w / 2
            
            # Corner offsets relative to center (before rotation)
            corners = np.array([
                [-half_l, -half_w],
                [half_l, -half_w], 
                [half_l, half_w],
                [-half_l, half_w]
            ])
            
            # Rotation matrix for yaw
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])
            
            # Rotate corners and translate to center
            rotated_corners = corners @ rotation_matrix.T
            rotated_corners[:, 0] += x
            rotated_corners[:, 1] += y
            
            return rotated_corners
        
        def calculate_z_intersection(box1, box2):
            """Calculate intersection in Z dimension."""
            z1, h1 = box1[2], box1[5]
            z2, h2 = box2[2], box2[5]
            
            # Z bounds for each box
            z1_min, z1_max = z1 - h1/2, z1 + h1/2
            z2_min, z2_max = z2 - h2/2, z2 + h2/2
            
            # Intersection bounds
            z_min = max(z1_min, z2_min)
            z_max = min(z1_max, z2_max)
            
            # Intersection height (0 if no overlap)
            z_intersection = max(0, z_max - z_min)
            return z_intersection
        
        def calculate_xy_intersection_area(box1, box2):
            """Calculate intersection area in XY plane using Shapely."""
            try:
                corners1 = get_box_corners_2d(box1)
                corners2 = get_box_corners_2d(box2)
                
                # Create polygons
                poly1 = Polygon(corners1)
                poly2 = Polygon(corners2)
                
                # Ensure polygons are valid
                if not poly1.is_valid:
                    poly1 = poly1.buffer(0)
                if not poly2.is_valid:
                    poly2 = poly2.buffer(0)
                
                # Calculate intersection
                intersection = poly1.intersection(poly2)
                
                # Return intersection area
                if intersection.is_empty:
                    return 0.0
                else:
                    return intersection.area
                    
            except Exception:
                # Fallback: return 0 if any geometric operation fails
                return 0.0
        
        # Calculate intersection volume
        z_intersection = calculate_z_intersection(box1, box2)
        if z_intersection == 0:
            return 0.0
        
        xy_intersection_area = calculate_xy_intersection_area(box1, box2)
        if xy_intersection_area == 0:
            return 0.0
        
        intersection_volume = xy_intersection_area * z_intersection
        
        # Calculate individual volumes
        volume1 = box1[3] * box1[4] * box1[5]  # l * w * h
        volume2 = box2[3] * box2[4] * box2[5]  # l * w * h
        
        # Calculate union volume
        union_volume = volume1 + volume2 - intersection_volume
        
        # Avoid division by zero
        if union_volume == 0:
            return 0.0
        
        # Calculate IoU
        iou = intersection_volume / union_volume
        
        return iou

    def _initiate_track(self, convex_hull: ConvexHullObject):
        mean, covariance = self.kf.box_initiate(convex_hull.box)
        self.tracks.append(ConvexHullTrack(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            convex_hull.feature, convex_hull))
        self.updates_per_track[self._next_id] = 1
        self._next_id += 1

    def enhanced_forward_backward_consistency(self):
        del_ids = []
        for track_idx, track in enumerate(self.tracks):
            # if not track.is_confirmed():
            #     print(f"skipping {track.track_id=} as not confirmed... {track.state=} {track.hits=}")
            #     continue

            print(f"Enhancing {track.track_id=}")

            timestamps = track.timestamps

            object_points_per_timestamp = []
            world_centers = []

            # Collect world points and centers
            for i, timestamp_ns in enumerate(timestamps):
                obj: ConvexHullObject = track.history[i]
                world_points = obj.original_points
                object_points_per_timestamp.append(world_points.copy())

                center = obj.centroid_3d
                world_centers.append(center)

            world_centers = np.array(world_centers)

            # Compute initial heading from velocity
            if len(world_centers) > 1:
                initial_velocity = world_centers[1] - world_centers[0]
                speed = np.linalg.norm(initial_velocity) / (
                    timestamps[1] - timestamps[0]
                )
                if speed > 0.1:  # Has significant motion
                    heading = initial_velocity / speed
                else:
                    heading = np.array([1, 0, 0])  # Default heading
            else:
                heading = np.array([1, 0, 0])

            # Build first object pose with aligned heading
            first_object_pose = self._create_pose_from_heading(
                world_centers[0], heading
            )
            initial_poses = [first_object_pose]

            relative_poses = []
            confidences = []
            cumulative_pose = first_object_pose.copy()
            for i in range(1, len(timestamps)):
                undo_pose = np.linalg.inv(cumulative_pose)

                prev_points = points_rigid_transform(
                    object_points_per_timestamp[i - 1], undo_pose
                )
                cur_points = points_rigid_transform(
                    object_points_per_timestamp[i], undo_pose
                )

                # transform, error, result = icp_open3d_robust(
                #     prev_points,
                #     cur_points,
                #     initial_alignment="ransac",
                #     return_full_result=True,
                # )

                # R, t, _ = icp(prev_points, cur_points, max_iterations=5)
                # R, t, _, _, _ = rigid_icp(prev_points, cur_points, max_iterations=5, debug=False, relative=False)
                R, t, _, _, icp_cost = relative_object_pose(prev_points, cur_points, max_iterations=5, debug=False)
                # R, t, icp_err = icp(prev_points, cur_points, max_iterations=5, ret_err=True)

                transform = np.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = t

                relative_poses.append(transform)
                confidences.append(1.0)

                cumulative_pose = cumulative_pose @ transform
                initial_poses.append(cumulative_pose)

            # Create constraints for optimization
            constraints = []
            for i, (relative_pose, confidence) in enumerate(
                zip(relative_poses, confidences)
            ):
                constraints.append(
                    {
                        "frame_i": i,
                        "frame_j": i + 1,
                        "relative_pose": relative_pose,
                        "confidence": confidence,
                    }
                )

            assert len(initial_poses) == len(
                timestamps
            ), "Must have one initial pose per timestamp"
            assert (
                len(constraints) == len(timestamps) - 1
            ), "Should have N-1 constraints for N poses"

            optimized_poses, marginals, quality = optimize_with_gtsam_timed(
                initial_poses, constraints, timestamps
            )

            # Update tracked objects
            # for i, timestamp_ns in enumerate(timestamps):
            #     trajectory[timestamp_ns]["optimized_pose"] = optimized_poses[i]

            track.optimized_poses = optimized_poses

            # After optimization, compute object-centric representation
            object_poses, object_points = self.compute_object_centric_transforms(track)

            # Merge all points in object frame
            # merged_object_points = self.merge_object_centric_points(object_points)
            merged_object_points = np.vstack(object_points)

            # find the best frame
            # num_kept_per_frame = [0 for _ in range(len(object_points))]
            # best_num_kept = -1
            # best_points = None
            # for i in range(len(object_points)):
            #     cur_points = object_points[i]

            #     # was using ppscore...
            #     num_kept = len(cur_points)
            #     num_kept_per_frame[i] = num_kept

            #     if num_kept > best_num_kept:
            #         best_points = cur_points
            #         best_num_kept = num_kept

            # print("num_kept_per_frame", num_kept_per_frame)

            # merged_object_points = best_points


            merged_object_points = voxel_sampling(merged_object_points, 0.05, 0.05, 0.05)

            mesh = trimesh.convex.convex_hull(merged_object_points)
            merged_vertices = mesh.vertices

            track.merged_mesh = mesh

            dims_mins = merged_vertices.min(axis=0)
            dims_maxes = merged_vertices.max(axis=0)

            old_lwh = track.lwh
            track.lwh = dims_maxes - dims_mins
            print(f"old_lwh:{np.round(old_lwh, 2)} track.lwh: {np.round(track.lwh, 2)}")

            # for i, timestamp_ns in enumerate(timestamps):
            #     object_cur_pose = optimized_poses[i]

            #     # move the merged_vertices to this object_pose
            #     cur_vertices = points_rigid_transform(
            #         merged_vertices.copy(), object_cur_pose
            #     )

            #     track["trajectory"][timestamp_ns]["alpha_shape"] = {
            #         "vertices_3d": cur_vertices_ego,
            #         "centroid_3d": cur_vertices_ego.mean(axis=0),
            #         "mesh": None,  # don't need anymore?
            #         "original_points": cur_vertices_ego,
            #     }

            object_boxes = self._compute_oriented_boxes(timestamps, optimized_poses, merged_object_points)

            # Store the canonical object representation
            track.merged_mesh = mesh
            track.optimized_poses = object_poses
            track.optimized_boxes = object_boxes

            self.tracks[track_idx] = track


    def compute_object_centric_transforms(self, track: ConvexHullTrack):
        """
        Compute object-to-world transforms for each timestamp.
        Returns:
            object_poses: List of 4x4 matrices that transform from object-centric to world
            object_points: List of points in object-centric coordinates
        """
        object_to_world_poses = track.optimized_poses
        history: List[ConvexHullObject] = track.history

        # Now transform points to object-centric coordinates
        object_centric_points = []
        for obj_pose, convex_hull_object in zip(object_to_world_poses, history):
            # Get world points
            world_points = convex_hull_object.original_points

            # Transform to object-centric: multiply by inverse of object pose
            world_to_object = np.linalg.inv(obj_pose)
            object_points = points_rigid_transform(world_points, world_to_object)
            object_centric_points.append(object_points)

        return object_to_world_poses, object_centric_points

    def _create_pose_from_heading(self, position, heading_vector):
        """Create a pose matrix with given position and heading direction."""
        pose = np.eye(4)
        pose[:3, 3] = position

        # Build rotation matrix where x-axis aligns with heading
        forward = heading_vector / np.linalg.norm(heading_vector)

        # Choose up vector (z-axis up in world)
        up_world = np.array([0, 0, 1])

        # Compute right vector
        right = np.cross(forward, up_world)
        if np.linalg.norm(right) > 0.01:
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)

            # Rotation matrix [forward, right, up]
            pose[:3, :3] = np.column_stack([forward, right, up])
        else:
            # Handle case where heading is vertical
            if abs(forward[0]) < 0.9:
                right = np.array([1, 0, 0])
            else:
                right = np.array([0, 1, 0])
            right = right - np.dot(right, forward) * forward
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            pose[:3, :3] = np.column_stack([forward, right, up])

        return pose

    def _compute_oriented_boxes(self, timestamps: List[int], optimized_poses: List[np.ndarray], merged_object_points: np.ndarray):
        """Compute oriented bounding boxes based on object motion direction"""
        boxes = []
        for i, cur_pose in enumerate(optimized_poses):
            # Determine orientation from motion direction
            if i > 0:
                # Use motion direction between consecutive poses
                prev_pos = optimized_poses[i - 1][:3, 3]
                curr_pos = optimized_poses[i][:3, 3]
                motion_dir = curr_pos - prev_pos
                if np.linalg.norm(motion_dir) > 0.01:
                    motion_dir = motion_dir / np.linalg.norm(motion_dir)
                    yaw = np.arctan2(motion_dir[1], motion_dir[0])
                else:
                    yaw = 0.0
            else:
                # For first frame, use next frame direction
                if len(optimized_poses) > 1:
                    next_pos = optimized_poses[1][:3, 3]
                    curr_pos = optimized_poses[0][:3, 3]
                    motion_dir = next_pos - curr_pos
                    if np.linalg.norm(motion_dir) > 0.01:
                        motion_dir = motion_dir / np.linalg.norm(motion_dir)
                        yaw = np.arctan2(motion_dir[1], motion_dir[0])
                    else:
                        yaw = 0.0
                else:
                    yaw = 0.0

            cur_points = points_rigid_transform(merged_object_points, cur_pose)

            # Create oriented bounding box from 3D vertices
            box = self._vertices_to_oriented_box(cur_points, yaw)
            boxes.append(box)

        return boxes

    @staticmethod
    def _vertices_to_oriented_box(vertices_3d, yaw):
        """Convert 3D vertices to oriented bounding box [x, y, z, l, w, h, yaw]"""
        if len(vertices_3d) < 3:
            return None

        # Rotate vertices to align with yaw=0
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        rotated_xy = vertices_3d[:, :2] @ rotation_matrix.T

        # Compute bounding box in rotated frame
        min_x, max_x = np.min(rotated_xy[:, 0]), np.max(rotated_xy[:, 0])
        min_y, max_y = np.min(rotated_xy[:, 1]), np.max(rotated_xy[:, 1])
        min_z, max_z = np.min(vertices_3d[:, 2]), np.max(vertices_3d[:, 2])

        # Box center and dimensions
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2

        # Rotate center back to original frame
        center_rotated = np.array([center_x, center_y]) @ rotation_matrix

        length = max_x - min_x
        width = max_y - min_y
        height = max_z - min_z

        return np.array(
            [center_rotated[0], center_rotated[1], center_z, length, width, height, yaw]
        )

    def get_current_frame_alpha_shapes(
        self, frame_id: int
    ) -> Tuple[List[Dict], List[int]]:
        """Get alpha shapes for a specific frame."""
        if frame_id not in self.frame_alpha_shapes:
            return [], []

        frame_data = self.frame_alpha_shapes[frame_id]
        alpha_shapes = []
        ids = []

        for obj_id in frame_data:
            # Use merged shape if available
            track = self.tracked_objects.get(obj_id, None)

            if track is None:
                continue

            alpha_shape = track["trajectory"][frame_id]["alpha_shape"]

            alpha_shapes.append(alpha_shape)
            ids.append(obj_id)

        return alpha_shapes, ids