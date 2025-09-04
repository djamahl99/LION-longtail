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
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from tqdm import tqdm, trange

from lion.unsupervised_core.box_utils import *
from lion.unsupervised_core.convex_hull_tracker import linear_assignment, nn_matching
from lion.unsupervised_core.convex_hull_tracker.alpha_shape_utils import AlphaShapeUtils
from lion.unsupervised_core.convex_hull_tracker.convex_hull_track import (
    ConvexHullTrack,
    ConvexHullTrackState,
)
from lion.unsupervised_core.convex_hull_tracker.convex_hull_utils import (
    relative_object_pose,
    rigid_icp,
)
from lion.unsupervised_core.convex_hull_tracker.pose_kalman_filter import (
    PoseKalmanFilter,
)
from lion.unsupervised_core.file_utils import load_predictions_parallel
from lion.unsupervised_core.outline_utils import (
    OutlineFitter,
    points_rigid_transform,
    voxel_sampling,
)
from lion.unsupervised_core.tracker.box_op import register_bbs
from lion.unsupervised_core.trajectory_optimizer import (
    GlobalTrajectoryOptimizer,
    optimize_with_gtsam_timed,
    simple_pairwise_icp_refinement,
)

from .convex_hull_object import ConvexHullObject

np.set_printoptions(suppress=True, precision=2)


class ConvexHullKalmanTracker:

    def __init__(self, config=None, debug: bool = False):
        self.config = config
        self.debug = debug
        self.tracks: List[ConvexHullTrack] = []  # id -> track

        nn_budget = None
        max_cosine_distance = 0.2
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )

        self.kf = PoseKalmanFilter(dt=0.1)

        self._next_id = 0
        self.all_pose = None
        self.frame_tracks: Dict[int, List[int]] = {}  # timestamp_ns -> tracks
        self.ppscore_thresh = 0.7
        self.track_query_eps = 3.0  # metres
        self.max_iou_distance = 0.9

        self.min_semantic_threshold = 0.7
        self.min_iou_threshold = 0.1
        self.min_box_iou = 0.1

        self.icp_max_dist = 1.0
        self.icp_max_iterations = 10

        self.nms_iou_threshold = 0.5
        self.nms_semantic_threshold: float = 0.7
        self.nms_query_distance: float = 1.0
        self.n_points_err_thresh = 0.3

        self.min_component_points = 10

        self.n_init = 3
        self.max_age = 10  # 10 frames -> 1 second

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

    def update(
        self,
        convex_hulls: List[ConvexHullObject],
        pose: np.ndarray,
        lidar_tree: cKDTree,
    ):
        # Run matching cascade.
        # matches, unmatched_tracks, unmatched_detections = \
        #     self._match(convex_hulls, pose)

        matches, unmatched_tracks, unmatched_detections = self._match_full(
            convex_hulls, pose
        )

        print(f"Made {len(matches)} matches")
        print(f"Made {len(unmatched_tracks)} unmatched tracks")
        print(f"Made {len(unmatched_detections)} unmatched dets")

        matched_tracks = set()

        # Update track set.
        for track_idx, detection_idx in matches:
            matched_tracks.add(track_idx)
            self.tracks[track_idx].update(self.kf, convex_hulls[detection_idx])
            # track_id = self.tracks[track_idx].track_id
            # print(f"self.updates_per_track[track_id]", self.updates_per_track[track_id])

        # TODO: use lidar to find if unmatched_tracks are still viable (reasonable points within and icp cost low)

        unmatched_and_missed = set()
        unmatched_and_found = set()

        unmatched_ious = []
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]

            if track.is_deleted():
                continue

            pos = track.mean[:3]
            radius = np.linalg.norm(track.lwh*0.5)

            lidar_indices = lidar_tree.query_ball_point(pos, radius)
            lidar_indices = np.array(lidar_indices, int)

            iou = 0.0
            if len(lidar_indices) > self.min_component_points:
                # track_mesh = track.to_shape_dict()['mesh']
                # points = lidar_tree.data[lidar_indices]
                # mesh_points = voxel_sampling(points)
                # cur_mesh = trimesh.convex.convex_hull(mesh_points)

                # iou = self._convex_hull_iou_trimesh(cur_mesh, track_mesh)

                track_box = track.to_box()
                points = lidar_tree.data[lidar_indices]

                # get points in the bbox
                n_orig_points = len(points)
                points = ConvexHullObject.points_in_box(track_box, points.copy())
                n_in_box = len(points)

                if n_in_box < self.min_component_points:
                    continue

                # print(f"points in box {n_in_box} ({(n_in_box/n_orig_points)})")
                
                cur_box = ConvexHullObject.points_to_bounding_box(points, track.mean[:3])

                iou = self._box_iou_3d(cur_box, track_box)

            unmatched_ious.append(iou)

            # if err > self.n_points_err_thresh:
            if iou < self.min_iou_threshold:
                self.tracks[track_idx].mark_missed()
                unmatched_and_missed.add(track_idx)
            else:
                # self.tracks[track_idx].mark_hit()
                points = lidar_tree.data[lidar_indices]
                self.tracks[track_idx].update_raw_lidar(self.kf, points)
                unmatched_and_found.add(track_idx)

        # for track_idx in unmatched_tracks:
        #     # self.tracks[track_idx].mark_missed()
        #     unmatched_and_missed.add(track_idx)

        if len(unmatched_ious) > 0:
            unmatched_ious = np.array(unmatched_ious)
            print(f"unmatched_ious {unmatched_ious.min()} {unmatched_ious.mean()} {unmatched_ious.max()}")

        # tracks_tree = cKDTree([x.to_box()[:3] for x in self.tracks if not x.is_deleted()])
        for detection_idx in unmatched_detections:
            # ious = np.array([self._box_iou_3d(convex_hulls[detection_idx].box, x.to_box()) for x in self.tracks])
            convex_hull_box = convex_hulls[detection_idx].box
            convex_hull_feature = convex_hulls[detection_idx].feature
            found_match = False
            for track in self.tracks:
                iou = self._box_iou_3d(convex_hull_box, track.to_box())
                semantic_overlap = np.dot(convex_hull_feature, track.features[-1])
                if (
                    iou > self.nms_iou_threshold
                    and semantic_overlap > self.nms_semantic_threshold
                ):
                    found_match = True
                    break

            if not found_match:
                self._initiate_track(convex_hulls[detection_idx])

        suppressed_tracks = self.track_nms()
        
        print(f"Tracks updated: matches: {len(matched_tracks)}")
        print(
            f"            unmatched_and_found: {len(unmatched_and_found)} unmatched_and_missed: {(len(unmatched_and_missed))}"
        )
        print(f"            suppressed_tracks: {len(suppressed_tracks)}")

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

    def track_nms(self) -> set:
        def get_priority_score(track: ConvexHullTrack):
            avg_confidence = np.mean([x.confidence for x in track.history])
            return (track.hits, avg_confidence)

        sorted_tracks = sorted(
            [x for x in self.tracks if not x.is_deleted()],
            key=get_priority_score,
            reverse=True,
        )

        sorted_boxes = np.stack([x.to_box() for x in sorted_tracks], axis=0)
        sorted_positions = sorted_boxes[:, :3]
        sorted_features = np.stack(
            [np.mean(np.stack(x.features, axis=0), axis=0) for x in sorted_tracks],
            axis=0,
        )
        sorted_track_ids = np.array([x.track_id for x in sorted_tracks], dtype=int)

        sorted_features = sorted_features / (
            np.linalg.norm(sorted_features, axis=1, keepdims=True) + 1e-6
        )

        print(
            f"sorted_features={sorted_features.shape} {sorted_boxes.shape} {sorted_positions.shape} {sorted_track_ids.shape}"
        )

        tracks_tree = cKDTree(sorted_positions)

        keep_indices = []
        suppressed = set()

        semantic_overlaps = sorted_features @ sorted_features.T

        print(
            f"{semantic_overlaps.shape=} {semantic_overlaps.min()} {semantic_overlaps.max()}"
        )

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
                if (
                    iou > self.nms_iou_threshold
                    and semantic_overlap > self.nms_semantic_threshold
                ):
                    suppressed.add(j)

        # Return the kept clusters in original order (not sorted order)
        for idx in suppressed:
            track_id = sorted_track_ids[idx]

            self.tracks[track_id].state = ConvexHullTrackState.Deleted

        return suppressed

    def _match_full(self, detections, pose):
        n_dets = len(detections)

        track_idxes = [i for i, track in enumerate(self.tracks) if not track.is_deleted()]
        n_tracks = len(track_idxes)

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

        for track_idx in track_idxes:
            track = self.tracks[track_idx]
            # Get predicted bounding box from Kalman filter
            predicted_center = track.mean[:3]

            predicted_boxes.append(track.to_box())

            predicted_shape = track.to_shape_dict()

            predicted_centroids.append(predicted_center)
            predicted_shapes.append(predicted_shape)

        for i, track_idx in enumerate(track_idxes):
            track = self.tracks[track_idx]
            if track.is_deleted():
                print(f"Track {track_idx} is deleted. {track.hits=} {track.age=} {track.time_since_update=}")
                continue

            close_detection_indices = detections_tree.query_ball_point(
                predicted_centroids[i], self.track_query_eps
            )

            predicted_shape = predicted_shapes[i]

            for local_det_idx in close_detection_indices:
                # Compute geometric costs

                semantic_iou = np.dot(
                    detections[local_det_idx].feature, track.features[-1]
                )

                if semantic_iou < self.min_semantic_threshold:
                    continue

                # IoU cost
                iou = 0.0
                # orig_iou = self._convex_hull_iou_trimesh(predicted_shapes[track_idx]['mesh'], detections[local_det_idx].mesh)
                # iou = self._box_iou_3d(predicted_boxes[track_idx], detections[local_det_idx].box)

                icp_err = 0.0
                # ICP cost (simplified - you may want to use your existing ICP function
                # R, t, _, _, icp_err = rigid_icp(predicted_shape['original_points'], detections[local_det_idx].original_points, max_iterations=5, debug=False, relative=False)
                # R, t, _, _, icp_err = rigid_icp(predicted_shape['original_points'], detections[local_det_idx].original_points, max_iterations=5, debug=False, relative=False)
                # R, t, icp_err = icp(predicted_shape['original_points'], detections[local_det_idx].original_points, max_iterations=5, ret_err=True)

                # best
                    # R, t, A_inliers, B_inliers, icp_err = relative_object_pose(
                    #     predicted_shape["mesh"].vertices,
                    #     detections[local_det_idx].mesh.vertices,
                    #     max_iterations=self.icp_max_iterations,
                    # )

                    # transform = np.eye(4)
                    # transform[:3, :3] = R
                    # transform[:3, 3] = t

                # mesh_projected: trimesh.Trimesh = predicted_shapes[track_idx]['mesh'].copy()
                # mesh_projected = mesh_projected.apply_transform(transform)
                # iou = self._convex_hull_iou_trimesh(mesh_projected, detections[local_det_idx].mesh)

                # box iou
                # box = np.copy(predicted_boxes[track_idx])
                # box_transformed = register_bbs(box.copy().reshape(1, 7), transform)[0]
                # box_diff = box_transformed - box

                # iou = self._box_iou_3d(box_transformed, detections[local_det_idx].box)

                iou = self._box_iou_3d(predicted_boxes[i], detections[local_det_idx].box)

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
                iou_matrix[i, local_det_idx] = iou
                semantic_matrix[i, local_det_idx] = semantic_iou
                icp_matrix[i, local_det_idx] = icp_err
                icp_matrix_max = max(icp_err, icp_matrix_max)

        print("icp_matrix", icp_matrix.shape, icp_matrix.min(), icp_matrix.max())
        icp_matrix_normed = icp_matrix / (icp_matrix_max + 1e-6)
        print(
            f"icp_matrix_normed",
            icp_matrix_normed.min(),
            icp_matrix_normed.mean(),
            icp_matrix_normed.max(),
        )

        cost_matrix = (1.0 - iou_matrix) + (
            1.0 - semantic_matrix
        )  # + icp_matrix_normed

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

        track_set = set(track_idxes)
        det_set = set([i for i in range(n_dets)])

        matched_tracks = set()
        matched_dets = set()

        matches = []
        for track_i, matched_det in enumerate(track_matches):
            track_idx = track_idxes[track_i]
            # print(f"mach {track_idx=} {matched_det=} iou={iou_matrix[track_idx, matched_det]} semantic={semantic_matrix[track_idx, matched_det]}")
            # print(f"icp_err={icp_matrix[track_idx, matched_det]}")
            if (
                iou_matrix[track_i, matched_det] >= self.min_iou_threshold
                and semantic_matrix[track_i, matched_det]
                > self.min_semantic_threshold
            ):
                matches.append((track_idx, matched_det))

                matched_tracks.add(track_idx)
                matched_dets.add(matched_det)

        tracks_unmatched = track_set.difference(matched_tracks)
        dets_unmatched = det_set.difference(matched_dets)

        print(
            f"matched_tracks: {len(matched_tracks)} tracks_unmatched {len(tracks_unmatched)}"
        )
        print(f"matched_dets: {len(matched_dets)} dets_unmatched {len(dets_unmatched)}")

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
                self.kf,
                semantic_cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
            )

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
                        iou = self._convex_hull_iou_trimesh(
                            predicted_shape["mesh"], detections[j].mesh
                        )
                        # iou =

                        # ICP cost (simplified - you may want to use your existing ICP function
                        R, t, _, _, icp_cost = rigid_icp(
                            predicted_shape["original_points"],
                            detections[j].original_points,
                            max_iterations=5,
                            debug=False,
                        )
                        # R, t, icp_cost = icp(predicted_shape['original_points'], detections[j].original_points, max_iterations=5, ret_err=True)

                        # Combine costs: semantic + geometric
                        geometric_cost = icp_cost + (1.0 - iou)

                        # Weight combination (adjust weights as needed)
                        semantic_weight = 0.4
                        geometric_weight = 0.6

                        combined_cost = (
                            semantic_weight * semantic_cost_matrix[i, j]
                            + geometric_weight * geometric_cost
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
                    iou = self._convex_hull_iou_trimesh(
                        predicted_shape["mesh"], detection.mesh
                    )
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
                    R, t, _, _, icp_err = rigid_icp(
                        predicted_shape["original_points"],
                        detection.original_points,
                        max_iterations=5,
                        debug=False,
                    )

                    # R, t, icp_err = icp(predicted_shape['original_points'], detection.original_points, max_iterations=5, ret_err=True)

                    cost_matrix[i, j] = icp_err

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # Associate confirmed tracks using enhanced gated metric (semantic + geometric)
        matches_a, unmatched_tracks_a, unmatched_detections = (
            linear_assignment.matching_cascade(
                enhanced_gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                confirmed_tracks,
            )
        )

        # Associate remaining tracks with unconfirmed tracks using geometric IoU
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = (
            linear_assignment.min_cost_matching(
                icp_metric,
                self.icp_max_dist,
                self.tracks,
                detections,
                iou_track_candidates,
                unmatched_detections,
            )
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _convex_hull_iou_trimesh(
        self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh
    ) -> float:
        """
        Compute IoU using cached geometric objects - much faster than original.

        Args:
            shape1, shape2: Alpha shape dictionaries with cached geometry

        Returns:
            IoU value between 0 and 1
        """

        try:
            # Using the 'manifold' engine for boolean operations
            intersection_mesh = mesh1.intersection(mesh2, engine="manifold")
            # The union volume can be calculated from the individual volumes and the intersection volume
            # union_volume = mesh1.volume + mesh2.volume - intersection_mesh.volume
            # Or by performing the union operation directly, which might be more robust
            union_mesh = mesh1.union(mesh2, engine="manifold")

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
            corners = np.array(
                [
                    [-half_l, -half_w],
                    [half_l, -half_w],
                    [half_l, half_w],
                    [-half_l, half_w],
                ]
            )

            # Rotation matrix for yaw
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

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
            z1_min, z1_max = z1 - h1 / 2, z1 + h1 / 2
            z2_min, z2_max = z2 - h2 / 2, z2 + h2 / 2

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
        self.tracks.append(
            ConvexHullTrack(
                mean,
                covariance,
                self._next_id,
                self.n_init,
                self.max_age,
                convex_hull.feature,
                convex_hull,
            )
        )
        self.updates_per_track[self._next_id] = 1
        self._next_id += 1

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
