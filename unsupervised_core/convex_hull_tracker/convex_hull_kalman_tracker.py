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
    box_iou_3d,
    relative_object_pose,
    rigid_icp,
    voxel_sampling_fast,
)
from lion.unsupervised_core.convex_hull_tracker.pose_kalman_filter import (
    PoseKalmanFilter,
)
from lion.unsupervised_core.file_utils import load_predictions_parallel
from lion.unsupervised_core.outline_utils import (
    OutlineFitter,
    points_rigid_transform,
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

        self.min_semantic_threshold = 0.9
        self.min_iou_threshold = 0.1
        self.min_box_iou = 0.1

        self.icp_max_dist = 1.0
        self.icp_max_iterations = 5

        self.nms_iou_threshold = 0.3
        self.nms_stg2_iou_threshold = 0.7
        self.nms_semantic_threshold: float = 0.95
        self.nms_query_distance: float = 10.0
        self.n_points_err_thresh = 0.3

        self.min_component_points = 10

        self.n_init = 3
        self.max_age = 10  # 5 frames -> 0.5 seconds

        self.updates_per_track = {}
        self.timestamps = []

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
            if not track.is_deleted():
                track.predict(self.kf, timestamp)

        self.timestamps.append(timestamp)

    def update(
        self,
        convex_hulls: List[ConvexHullObject],
        pose: np.ndarray,
        lidar_tree: cKDTree,
        flow: np.ndarray
    ):
        # Run matching cascade.
        # matches, unmatched_tracks, unmatched_detections = \
        #     self._match(convex_hulls, pose)

        # TODO: get boxes in here and do projection etc, rather than _get_vision_clusters...
        # -> more efficienct? can match with tracks first then remaining boxes can be added?

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

        # unmatched_ious = []
        # for track_idx in unmatched_tracks:
        #     track = self.tracks[track_idx]

        #     if track.is_deleted():
        #         continue

        #     pos = track.mean[:3] # change to box[:3]
        #     radius = np.linalg.norm(track.lwh * 0.5)

        #     lidar_indices = lidar_tree.query_ball_point(pos, radius)
        #     lidar_indices = np.array(lidar_indices, int)

        #     iou = 0.0
        #     if len(lidar_indices) > self.min_component_points:
        #         track_box = track.to_box()
        #         cmp_points = lidar_tree.data[lidar_indices]

        #         # get points in the bbox
        #         n_orig_points = len(cmp_points)
        #         points_mask = ConvexHullObject.points_in_box(track_box, cmp_points.copy(), ret_mask=True)
        #         lidar_indices = lidar_indices[points_mask]
        #         cmp_points = cmp_points[points_mask]
        #         cmp_flow = flow[lidar_indices]

        #         n_in_box = len(cmp_points)

        #         flow_mean = np.mean(cmp_flow, axis=0)
        #         flow_yaw = None
        #         if np.linalg.norm(flow_mean) > 0.36: # FIXME: heading_speed_thresh * 0.1
        #             flow_yaw = np.arctan2(flow_mean[1], flow_mean[0])

        #         if n_in_box < self.min_component_points:
        #             continue

        #         cur_box = ConvexHullObject.points_to_bounding_box(
        #             cmp_points, track_box[:3], flow_yaw
        #         )

        #         iou = box_iou_3d(cur_box, track_box)

        #     unmatched_ious.append(iou)

        #     # if err > self.n_points_err_thresh:
        #     if iou < self.nms_stg2_iou_threshold:
        #         self.tracks[track_idx].mark_missed()
        #         unmatched_and_missed.add(track_idx)
        #     else:
        #         self.tracks[track_idx].update_raw_lidar(self.kf, cmp_points, cmp_flow)
        #         unmatched_and_found.add(track_idx)

        # if len(unmatched_ious) > 0:
        #     print("unmatched_ious", np.min(unmatched_ious), np.mean(unmatched_ious), np.max(unmatched_ious))

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            unmatched_and_missed.add(track_idx)

        # tracks_tree = cKDTree([x.to_box()[:3] for x in self.tracks if not x.is_deleted()])
        for detection_idx in unmatched_detections:
            # ious = np.array([box_iou_3d(convex_hulls[detection_idx].box, x.to_box()) for x in self.tracks])
            convex_hull_box = convex_hulls[detection_idx].box
            convex_hull_feature = convex_hulls[detection_idx].feature
            found_match = False
            for track in self.tracks:
                if track.is_deleted():
                    continue
                iou1 = box_iou_3d(convex_hull_box, track.to_box()) 
                # iou2 = box_iou_3d(convex_hull_box, track.history[-1].box)
                # iou = max(iou1, iou2) 

                iou = iou1

                # mean_track_features = np.stack(track.features, axis=0).mean(axis=0)
                # mean_track_features = mean_track_features / (1e-6 + np.linalg.norm(mean_track_features))
                semantic_overlap = np.dot(convex_hull_feature, track.features[-1])
                if (
                    iou > self.nms_iou_threshold
                    and semantic_overlap > self.nms_semantic_threshold
                ):
                    found_match = True
                    break
                elif iou > self.nms_stg2_iou_threshold:
                    found_match = True
                    break

            if not found_match:
                self._initiate_track(convex_hulls[detection_idx])

        new_merged_tracks = 0
        # new_merged_tracks = self.merge_tracks()

        suppressed_tracks = self.track_nms()

        print(f"Tracks updated: matches: {len(matched_tracks)}")
        print(
            f"            unmatched_and_found: {len(unmatched_and_found)} unmatched_and_missed: {(len(unmatched_and_missed))}"
        )
        print(f"            suppressed_tracks: {len(suppressed_tracks)}")
        print(f"            new_merged_tracks: {new_merged_tracks}")


    def merge_tracks(self):

        # can we find any previous timestamps where a track did not exist, then see if we can extrapolate and find if another track could merge?
        # all_timestamps = set(self.timestamps)
        timestamps = sorted(list(set(self.timestamps)))
        # track_indices = [i for i, track in enumerate(self.tracks) if track.is_confirmed()]

        new_merged_tracks = 0

        def get_priority_score(idx: int):
            track: ConvexHullTrack = self.tracks[idx]
            avg_confidence = np.mean([x.confidence for x in track.history])

            status_priority = 2
            if track.is_tentative():
                status_priority = 1
            elif track.is_deleted():
                status_priority = 0

            return (status_priority, track.hits, avg_confidence)

        track_indices = sorted(
            list(range(len(self.tracks))),
            key=get_priority_score,
            reverse=True,
        )

        if len(track_indices) > 0:
            track_features = np.stack([self.tracks[x].features[-1] for x in track_indices], axis=0)
            semantic_overlaps = track_features @ track_features.T
            tracks_boxes = np.stack([self.tracks[idx].extrapolate_box(timestamps) for idx in track_indices], axis=0)
            tracks_positions = tracks_boxes[..., :3]
            print("tracks_positions", tracks_positions.shape)

            for i, track_idx1 in enumerate(track_indices):
                track1 = self.tracks[track_idx1]
                track1_timestamps = track1.timestamps

                if track1.is_deleted():
                    break

                for j in range(i + 1, len(track_indices)):
                    track_idx2 = track_indices[j]
                    dists = np.linalg.norm(tracks_positions[i]-tracks_positions[j], axis=1)
                    mean_dist = dists.mean()

                    # these could potentially be the same object...
                    closest_idx = np.argmin(dists)
                    min_dist = dists[closest_idx]

                    box1 = tracks_boxes[i, closest_idx]
                    box2 = tracks_boxes[j, closest_idx]

                    iou = box_iou_3d(box1, box2)

                    track2 = self.tracks[track_idx2]

                    track2_timestamps = track2.timestamps

                    times1_set = set(track1_timestamps)
                    times2_set = set(track2_timestamps)

                    times_difference = times1_set.symmetric_difference(times2_set)

                    combined_timestamps = sorted(list(times1_set.union(times2_set)))

                    if min_dist < self.nms_query_distance and semantic_overlaps[i, j] > self.nms_semantic_threshold and iou > self.nms_iou_threshold and len(times_difference) > 0:
                        print("dists", dists.shape, dists.min(), dists.max())
                        print("min_dist", min_dist)
                        print("mean_dist", mean_dist)
                        print("times1, times2, times_difference", len(times1_set), len(times2_set), len(times_difference))
                        print(f"{iou=:.3f}")


                        times1 = np.array(self.tracks[track_idx1].timestamps, int)
                        times2 = np.array(self.tracks[track_idx2].timestamps, int)


                        combined_idx = None
                        for timestamp in combined_timestamps:
                            track1_indices = np.where(times1==timestamp)[0]
                            track2_indices = np.where(times2==timestamp)[0]

                            convex_hull_obj = None
                            if len(track1_indices) == 0 and len(track2_indices) == 1:
                                idx = track2_indices[0]
                                convex_hull_obj = track2.history[idx]
                            elif len(track1_indices) == 1 and len(track2_indices) == 0:
                                idx = track1_indices[0]
                                convex_hull_obj = track1.history[idx]
                            elif len(track1_indices) == 1 and len(track2_indices) == 1:
                                # merge...
                                idx1 = track1_indices[0]
                                convex_hull_obj1 = track1.history[idx1]

                                idx2 = track2_indices[0]
                                convex_hull_obj2 = track2.history[idx2]

                                points1 = convex_hull_obj1.original_points
                                points2 = convex_hull_obj2.original_points

                                # need to calculate the pose difference between...

                                print(f"{points1.shape=} {points2.shape=}")
                                points = np.concatenate([points1, points2], axis=0)

                                print(f"merged {points.shape}")


                                convex_hull_obj = ConvexHullObject(
                                    points,
                                    convex_hull_obj1.confidence,
                                    convex_hull_obj1.iou_2d,
                                    convex_hull_obj1.objectness_score,
                                    convex_hull_obj1.feature,
                                    timestamp,
                                    convex_hull_obj1.source
                                )

                                if convex_hull_obj.original_points is None:
                                    fig, ax = plt.subplots(figsize=(8,8))

                                    ax.scatter(
                                        points1[:, 0],
                                        points1[:, 1],
                                        s=3,
                                        c="blue",
                                        label="points1",
                                        alpha=0.5,
                                    )
                                
                                    ax.scatter(
                                        points2[:, 0],
                                        points2[:, 1],
                                        s=3,
                                        c="green",
                                        label="points2",
                                        alpha=0.5,
                                    )
                                    
                                    for box in [convex_hull_obj1.box, convex_hull_obj2.box]:
                                        center_xy = box[:2]
                                        length = box[3]
                                        width = box[4]
                                        yaw = box[6]

                                        # Get rotated box corners
                                        corners = get_rotated_box(center_xy, length, width, yaw)
                                        track_polygon = patches.Polygon(
                                            corners,
                                            linewidth=1,
                                            edgecolor="blue",
                                            facecolor="none",
                                            alpha=1.0,
                                            linestyle="--",
                                        )
                                        ax.add_patch(track_polygon)

                                    for box in [box1, box2]:
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
                                            alpha=1.0,
                                            linestyle="--",
                                        )
                                        ax.add_patch(track_polygon)

                                    ax.set_aspect("equal")
                                    ax.grid(True, alpha=0.3)
                                    ax.set_xlabel("X (meters)", fontsize=12)
                                    ax.set_ylabel("Y (meters)", fontsize=12)

                                    plt.tight_layout()

                                    save_path = "failed_merge.png"
                                    plt.savefig(save_path, dpi=300, bbox_inches="tight")
                                    plt.close()
                                    exit()
                            else:
                                raise ValueError(f"{len(track1_indices)=} {len(track2_indices)=}")

                            if convex_hull_obj.original_points is None:
                                continue # something wrong... e.g. too large

                            if combined_idx is None:
                                self._initiate_track(convex_hull_obj)
                                combined_idx = len(self.tracks) - 1
                            else:
                                self.tracks[combined_idx].update(self.kf, convex_hull_obj)


                        new_merged_tracks += 1
        return new_merged_tracks

    def track_nms(self) -> set:
        def get_priority_score(track: ConvexHullTrack):
            avg_confidence = np.mean([x.confidence for x in track.history])
            return (track.hits, avg_confidence)

        if len(self.tracks) == 0:
            return set()

        sorted_tracks = sorted(
            [x for x in self.tracks if not x.is_deleted()],
            key=get_priority_score,
            reverse=True,
        )

        sorted_boxes = np.stack([x.to_box() for x in sorted_tracks], axis=0)
        sorted_positions = sorted_boxes[:, :3]
        sorted_track_ids = np.array([x.track_id for x in sorted_tracks], dtype=int)

        sorted_features = np.stack([x.features[-1] for x in sorted_tracks], axis=0)
        # sorted_meshes = [x.history[-1].mesh for x in sorted_tracks]

        print(
            f"sorted_features={sorted_features.shape} {sorted_boxes.shape} {sorted_positions.shape} {sorted_track_ids.shape}"
        )

        tracks_tree = cKDTree(sorted_positions)

        keep_indices = set()
        suppressed = set()

        semantic_overlaps = sorted_features @ sorted_features.T
        np.fill_diagonal(semantic_overlaps, 0.0) # fill for min/max sake.

        ious = np.zeros_like(semantic_overlaps)

        print(
            f"{semantic_overlaps.shape=} {semantic_overlaps.min()} {semantic_overlaps.max()}"
        )

        # print("row median", np.median(semantic_overlaps, axis=1))
        # print("row max", np.max(semantic_overlaps, axis=1))

        num_tracks = len(sorted_tracks)
        for i in range(num_tracks):
            if i in suppressed:
                continue

            keep_indices.add(i)

            # Check all remaining clusters for overlap
            # for j in range(i+1, num_clusters):
            indices = tracks_tree.query_ball_point(
                sorted_positions[i], self.nms_query_distance
            )
            for j in indices:
                if j in suppressed or j in keep_indices:
                    continue

                iou = box_iou_3d(sorted_boxes[i], sorted_boxes[j])
                # iou = self._convex_hull_iou_trimesh(sorted_meshes[i], sorted_meshes[j])
                semantic_overlap = semantic_overlaps[i, j]

                ious[i, j] = iou
                ious[j, i] = iou

                # if (
                #     iou > self.nms_iou_threshold
                #     and semantic_overlap < self.nms_semantic_threshold
                # ):
                #     print(f"Wouldn't suppress {iou=:.2f} {semantic_overlap=:.2f}")
                #     suppressed.add(j)

                # Suppress the lower-priority cluster if IoU is high
                if (
                    iou > self.nms_iou_threshold
                    and semantic_overlap > self.nms_semantic_threshold
                ):
                    suppressed.add(j)

        for i, track_id in enumerate(sorted_track_ids):
            self.tracks[track_id].last_iou = np.max(ious[i, :]) # iou with itself is zero as we never check...


        # Return the kept clusters in original order (not sorted order)
        for idx in suppressed:
            track_id = sorted_track_ids[idx]

            self.tracks[track_id].state = ConvexHullTrackState.Deleted

        return suppressed

    def _match_full(self, detections: List[ConvexHullObject], pose):
        n_dets = len(detections)

        track_idxes = [
            i for i, track in enumerate(self.tracks) if not track.is_deleted()
        ]
        n_tracks = len(track_idxes)

        if n_tracks == 0 or n_dets == 0:
            tracks_unmatched = set([i for i in range(n_tracks)])
            dets_unmatched = set([i for i in range(n_dets)])

            return [], list(tracks_unmatched), list(dets_unmatched)

        # Calculate IoU matrix
        iou_matrix = np.zeros((n_tracks, n_dets))
        semantic_matrix = np.zeros((n_tracks, n_dets))
        dist_matrix = np.full((n_tracks, n_dets), 100.0)
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
            box = track.to_box()
            predicted_center = box[:3]

            predicted_boxes.append(box)

            # predicted_shape = track.to_shape_dict()

            predicted_centroids.append(predicted_center)
            # predicted_shapes.append(track.to_bev_hull())

        for i, track_idx in enumerate(track_idxes):
            track = self.tracks[track_idx]
            if track.is_deleted():
                print(
                    f"Track {track_idx} is deleted. {track.hits=} {track.age=} {track.time_since_update=}"
                )
                continue

            close_detection_indices = detections_tree.query_ball_point(
                predicted_centroids[i], self.track_query_eps
            )

            # predicted_shape = predicted_shapes[i]

            for local_det_idx in close_detection_indices:
                # Compute geometric costs

                semantic_iou = np.dot(
                    detections[local_det_idx].feature, track.features[-1]
                )

                if semantic_iou < self.min_semantic_threshold:
                    continue

                # IoU cost
                iou = 0.0
                icp_err = 0.0
                det_box = detections[local_det_idx].box
                pred_box = predicted_boxes[i]


                # hull1 = detections[local_det_idx].hull
                # hull2 = predicted_shape['hull']

                # # Calculate intersection
                # z1_min, z1_max = detections[local_det_idx].z_min, detections[local_det_idx].z_max
                # z2_min, z2_max = predicted_shape['z_min'], predicted_shape['z_max']
                # intersection_min = max(z1_min, z2_min)
                # intersection_max = min(z1_max, z2_max)
                # intersection_height = max(0, intersection_max - intersection_min)

                # # Calculate union
                # union_min = min(z1_min, z2_min)
                # union_max = max(z1_max, z2_max)
                # union_height = union_max - union_min

                # z_iou = 0.0

                # if union_height == 0:
                #     z_iou = 1.0  # Both have same z
                
                # else:
                #     z_iou = intersection_height / union_height

                # bev_iou = 0.0
                # # Handle degenerate cases (points, lines)
                # if hull1.area == 0 or hull2.area == 0:
                #     bev_iou = 0.0
                # else:

                #     # Compute intersection and union
                #     intersection = hull1.intersection(hull2).area
                #     union = hull1.union(hull2).area

                #     # Return IoU
                #     bev_iou = intersection / union if union > 0 else 0.0

                # iou = bev_iou * z_iou

                iou = box_iou_3d(pred_box, det_box)

                # iou = (iou_transformed + iou_default) / 2.0

                distance = np.linalg.norm(pred_box[:3] - det_box[:3])

                if distance < self.nms_query_distance:
                    box_transformed = np.copy(pred_box)
                    box_transformed[:3] = det_box[:3]

                    iou_transformed = box_iou_3d(pred_box, det_box)

                    iou = (iou + iou_transformed) / 2.0


                iou_matrix[i, local_det_idx] = iou
                semantic_matrix[i, local_det_idx] = semantic_iou
                icp_matrix[i, local_det_idx] = icp_err
                dist_matrix[i, local_det_idx] = distance
                icp_matrix_max = max(icp_err, icp_matrix_max)


        print("dist_matrix", dist_matrix.shape, dist_matrix.min(), dist_matrix.max())
        dist_matrix_normed = dist_matrix / (
            dist_matrix.max(axis=1, keepdims=True) + 1e-6
        )

        cost_matrix = (1.0 - iou_matrix) + (1.0 - semantic_matrix) + dist_matrix_normed


        # compute matches
        # track_matches = np.argmin(cost_matrix, axis=1)
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        # track_matches = [i for i, x in enumerate(track_matches) if (iou_matrix[i, x] > (1.0 - self.max_iou_distance)) and (semantic_matrix[i, x] > self.min_semantic_threshold)]

        track_set = set(track_idxes)
        det_set = set([i for i in range(n_dets)])

        matched_tracks = set()
        matched_dets = set()
        match_ious = []

        matches = []
        # for track_i, matched_det in enumerate(track_matches):
        for track_i, matched_det in zip(track_indices, det_indices):
            track_idx = track_idxes[track_i]
            # print(f"mach {track_idx=} {matched_det=} iou={iou_matrix[track_idx, matched_det]} semantic={semantic_matrix[track_idx, matched_det]}")
            # print(f"icp_err={icp_matrix[track_idx, matched_det]}")
            if (
                iou_matrix[track_i, matched_det] >= self.min_iou_threshold
                and semantic_matrix[track_i, matched_det] > self.min_semantic_threshold
            ):
                matches.append((track_idx, matched_det))

                matched_tracks.add(track_idx)
                matched_dets.add(matched_det)
                match_ious.append(iou_matrix[track_i, matched_det])

        tracks_unmatched = track_set.difference(matched_tracks)
        dets_unmatched = det_set.difference(matched_dets)

        print(
            f"matched_tracks: {len(matched_tracks)} tracks_unmatched {len(tracks_unmatched)}"
        )
        print(f"matched_dets: {len(matched_dets)} dets_unmatched {len(dets_unmatched)}")

        match_ious = np.array(match_ious)
        if len(match_ious) > 0:
            print("match_ious", match_ious.min(), np.median(match_ious), np.max(match_ious))

        return matches, list(tracks_unmatched), list(dets_unmatched)

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
