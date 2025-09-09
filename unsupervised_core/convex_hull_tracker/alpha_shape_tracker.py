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
from lion.unsupervised_core.convex_hull_tracker.alpha_shape_utils import AlphaShapeUtils
from lion.unsupervised_core.convex_hull_tracker.convex_hull_utils import (
    voxel_sampling_fast,
)
from lion.unsupervised_core.file_utils import load_predictions_parallel
from lion.unsupervised_core.outline_utils import (
    OutlineFitter,
    TrackSmooth,
    points_rigid_transform,
)
from lion.unsupervised_core.owlvit_frustum_tracker import OWLViTFrustumTracker
from lion.unsupervised_core.trajectory_optimizer import (
    GlobalTrajectoryOptimizer,
    optimize_with_gtsam_timed,
    simple_pairwise_icp_refinement,
)


class AlphaShapeTracker:
    """Tracker for alpha shapes using IoU matching."""

    def __init__(self, config, debug: bool = False):
        self.config = config
        self.debug = debug
        self.tracked_objects = {}  # id -> tracked object

        self.next_id = 0
        self.all_pose = None
        self.frame_alpha_shapes = {}  # timestamp -> list of (id, alpha_shape, metadata)
        self.icp_fail_max = 3
        self.ppscore_thresh = 0.7
        self.track_query_eps = 5.0  # metres
        self.min_semantic_threshold = 0.3
        self.min_box_iou = 0.1

        # Debug statistics
        if self.debug:
            self.debug_stats = {
                "total_new_tracks": 0,
                "total_updates": 0,
                "frame_stats": {},
                "iou_history": [],
            }

    def track_alpha_shapes(
        self,
        all_alpha_shapes: List[List[Dict]],
        all_pose: List[np.ndarray],
        all_timestamps: List[int],
    ):
        """Track alpha shapes across frames."""
        self.all_pose = all_pose

        if self.debug:
            print(f"\n=== ALPHA SHAPE TRACKING DEBUG ===")
            print(f"Total frames to process: {len(all_alpha_shapes)}")
            print(
                f"IoU threshold: {getattr(self.config, 'alpha_shape_iou_threshold', 0.3)}"
            )
            print("=" * 50)

        for timestamp_ns, alpha_shapes, pose in zip(
            all_timestamps, all_alpha_shapes, all_pose
        ):
            if self.debug:
                print(f"\n--- FRAME {timestamp_ns} ---")
                print(f"Input alpha shapes: {len(alpha_shapes)}")

            self._process_frame(timestamp_ns, alpha_shapes, pose)

            if self.debug:
                frame_stats = self.debug_stats["frame_stats"].get(timestamp_ns, {})
                print(f"New tracks created: {frame_stats.get('new_tracks', 0)}")
                print(f"Tracks updated: {frame_stats.get('updated_tracks', 0)}")
                print(
                    f"Tracks dropped (ICP failed): {frame_stats.get('dropped_tracks', 0)}"
                )
                print(
                    f"Unmatched detections: {frame_stats.get('unmatched_detections', 0)}"
                )
                print(f"Total tracks after frame: {len(self.tracked_objects)}")

        if self.debug:
            print(f"\n=== TRACKING SUMMARY ===")
            print(f"Total tracks created: {self.debug_stats['total_new_tracks']}")
            print(f"Total track updates: {self.debug_stats['total_updates']}")

            # Count dropped tracks
            total_dropped = sum(
                frame_stats.get("dropped_tracks", 0)
                for frame_stats in self.debug_stats["frame_stats"].values()
            )
            print(f"Total tracks dropped (ICP failures): {total_dropped}")
            print(f"Final number of tracks: {len(self.tracked_objects)}")

            # Show track length distribution
            track_lengths = np.array(
                [
                    t["last_timestamp"] - t["first_timestamp"]
                    for t in self.tracked_objects.values()
                ]
            )
            track_lengths = track_lengths / 1e9  # ns to seconds

            print(
                f"Track length stats - Min: {np.min(track_lengths):.2f} secs, Max: {np.max(track_lengths):.2f} secs, Avg: {np.mean(track_lengths):.2f} secs"
            )

            # ICP error statistics
            all_icp_errors = []
            for track in self.tracked_objects.values():
                for frame_data in track["trajectory"].values():
                    if "icp_error" in frame_data and frame_data["icp_error"] < float(
                        "inf"
                    ):
                        all_icp_errors.append(frame_data["icp_error"])

            if all_icp_errors:
                self.debug_stats["icp_errors"] = all_icp_errors
                print(
                    f"ICP error stats - Mean: {np.mean(all_icp_errors):.3f}m, Std: {np.std(all_icp_errors):.3f}m, Max: {np.max(all_icp_errors):.3f}m"
                )

        # Perform forward-backward pass for temporal consistency
        # if self.debug:
        # print(f"\nPerforming forward-backward consistency pass...")
        # self._forward_backward_consistency()
        self.enhanced_forward_backward_consistency()

    def _process_frame(
        self, timestamp_ns: int, alpha_shapes: List[Dict], pose: np.ndarray
    ):
        """Process a single frame for tracking."""
        frame_assignments = []

        # Initialize debug stats for this frame
        if self.debug:
            self.debug_stats["frame_stats"][timestamp_ns] = {
                "new_tracks": 0,
                "updated_tracks": 0,
                "dropped_tracks": 0,
                "unmatched_detections": 0,
                "input_shapes": len(alpha_shapes),
            }

        # if timestamp_ns == 0:
        #     # Initialize tracking for first frame
        #     for alpha_shape in alpha_shapes:
        #         obj_id = self._create_new_track(timestamp_ns, alpha_shape, pose)
        #         frame_assignments.append(obj_id)

        #         if self.debug:
        #             self.debug_stats["frame_stats"][timestamp_ns]["new_tracks"] += 1

        # else:
        #     # Match with existing tracks
        #     frame_assignments = self._match_frame(timestamp_ns, alpha_shapes, pose)


        # Match with existing tracks
        frame_assignments = self._match_frame(timestamp_ns, alpha_shapes, pose)


        self.frame_alpha_shapes[timestamp_ns] = frame_assignments

    def _match_frame(
        self, timestamp_ns: int, alpha_shapes: List[Dict], pose: np.ndarray
    ) -> List[Tuple]:
        """Match alpha shapes in current frame with existing tracks."""
        if not alpha_shapes:
            return []

        # Get active tracks from previous frame
        active_tracks = self._get_active_tracks(timestamp_ns)

        if self.debug:
            print(
                f"  Matching: {len(alpha_shapes)} detections vs {len(active_tracks)} active tracks"
            )

        if not active_tracks:
            # No active tracks, create new ones
            assignments = []
            for alpha_shape in alpha_shapes:
                obj_id = self._create_new_track(timestamp_ns, alpha_shape, pose)
                assignments.append(obj_id)

                if self.debug:
                    self.debug_stats["frame_stats"][timestamp_ns]["new_tracks"] += 1

            return assignments

        if self.debug:
            print(
                f"  Computing {len(active_tracks)} x {len(alpha_shapes)} IoU matrix..."
            )

        active_track_positions = []
        active_track_predicted_shapes = []
        # collect positions of active alpha shapes
        for track_id in active_tracks:
            track = self.tracked_objects[track_id]

            last_timestamp = track["last_timestamp"]

            last_data = track["trajectory"][last_timestamp]
            last_shape = last_data["alpha_shape"]
            last_pose = last_data["pose"]

            # Transform from last frame to current frame
            transform = np.linalg.inv(pose) @ last_pose

            last_centre = last_shape["centroid_3d"]

            predicted_centre = points_rigid_transform(
                last_centre.reshape(1, 3), transform
            )[0]

            tracked_points = last_shape["original_points"].copy()
            predicted_points = points_rigid_transform(tracked_points, transform)
            predicted_shape = AlphaShapeUtils.compute_alpha_shape(predicted_points)

            active_track_positions.append(predicted_centre)
            active_track_predicted_shapes.append(predicted_shape)

        tracks_tree = cKDTree(active_track_positions)

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(active_tracks), len(alpha_shapes)))
        semantic_matrix = np.zeros((len(active_tracks), len(alpha_shapes)))
        cost_matrix = np.full((len(active_tracks), len(alpha_shapes)), 100.0)
        icp_matrix = np.full((len(active_tracks), len(alpha_shapes)), 100.0)

        icp_errs = []

        for j, current_shape in enumerate(alpha_shapes):
            current_semantic_features = current_shape.get("semantic_features", None)

            indices = tracks_tree.query_ball_point(
                current_shape["centroid_3d"], self.track_query_eps
            )

            for i in indices:
                track_id = active_tracks[i]
                track = self.tracked_objects[track_id]

                last_timestamp = track["last_timestamp"]
                track_semantic_features = track["semantic_features"]

                # box_iou = bbox_iou_3d(
                #     predicted_shape["original_points"], current_shape["original_points"]
                # )

                # if box_iou < self.min_box_iou:
                #     continue

                semantic_overlap = iou = 0

                if (
                    track_semantic_features is not None
                    and current_semantic_features is not None
                ):
                    semantic_overlap = np.dot(
                        track_semantic_features, current_semantic_features
                    )
                    semantic_matrix[i, j] = semantic_overlap
                else:
                    semantic_overlap = 0.3

                # if semantic_overlap < self.min_semantic_threshold:
                #     continue

                predicted_shape = active_track_predicted_shapes[i]
                # _, _, icp_err = icp(
                #     predicted_shape["original_points"],
                #     current_shape["original_points"],
                #     ret_err=True,
                #     max_iterations=10,
                # )
                icp_err = 0.0
                # _, icp_err = self._compute_world_icp(predicted_shape['original_points'], current_shape['original_points'])
                # _, icp_err = icp_open3d_robust(predicted_shape['original_points'], current_shape['original_points'], initial_alignment='ransac', max_iterations=10)

                icp_matrix[i, j] = icp_err
                icp_errs.append(icp_err)

                if icp_err == np.inf:
                    continue

                if predicted_shape is not None:
                    iou = AlphaShapeUtils.convex_hull_iou_trimesh(predicted_shape, current_shape)
                    # iou = AlphaShapeUtils.convex_hull_iou_voxelized(predicted_shape, current_shape)
                    # iou = AlphaShapeUtils.voxel_iou_from_sets(
                    #     predicted_shape["voxel_set"], current_shape["voxel_set"]
                    # )
                    iou_matrix[i, j] = iou

                    # print(f"box_iou = {box_iou:.2f} alpha_iou = {iou:.2f}")

                # iou_matrix[i, j] = box_iou

                # Combined cost
                cost = (
                    icp_err + (1.0 - iou) + (1.0 - semantic_overlap)
                )
                cost_matrix[i, j] = cost

        print("semantic_matrix", semantic_matrix.min(), semantic_matrix.max())
        semantic_matrix = (semantic_matrix - semantic_matrix.min()) / max(
            1, semantic_matrix.max() - semantic_matrix.min()
        )
        print(
            "semantic_matrix after scaling",
            semantic_matrix.min(),
            semantic_matrix.max(),
        )
        print("iou_matrix", iou_matrix.min(), iou_matrix.max())

        if len(icp_errs) > 0:
            print("icp_errs", np.min(icp_errs), np.mean(icp_errs), np.max(icp_errs))

        # Perform assignment using IoU threshold
        iou_threshold = getattr(self.config, 'alpha_shape_iou_threshold', 0.1)
        assignments = self._hungarian_assignment(
            cost_matrix, iou_matrix, semantic_matrix, iou_threshold=iou_threshold
        )

        if self.debug:
            print(f"  Successful assignments: {len(assignments)}")

            # Log IoU statistics for this frame
            valid_ious = iou_matrix[iou_matrix > 0]
            if len(valid_ious) > 0:
                self.debug_stats["iou_history"].extend(valid_ious.tolist())
                print(
                    f"  IoU stats - Max: {np.max(valid_ious):.3f}, Mean: {np.mean(valid_ious):.3f}, Min: {np.min(valid_ious):.3f}"
                )

        frame_assignments = []
        used_shapes = set()
        dropped_tracks = set()

        # Process matched assignments
        for track_idx, shape_idx in assignments:
            if track_idx < len(active_tracks) and shape_idx < len(alpha_shapes):
                track_id = active_tracks[track_idx]
                alpha_shape = alpha_shapes[shape_idx]
                iou_score = iou_matrix[track_idx, shape_idx]
                icp_err = icp_matrix[track_idx, shape_idx]

                # Attempt to update track (may fail due to ICP)
                update_success = self._update_track(
                    track_id, timestamp_ns, alpha_shape, pose, iou_score
                )

                if update_success:
                    frame_assignments.append(track_id)
                    used_shapes.add(shape_idx)

                    if self.debug:
                        print(
                            f"    MATCHED: Track {track_id} <- Detection {shape_idx} (IoU: {iou_score:.3f}, {icp_err=:.3f})"
                        )
                else:
                    # Track was dropped due to ICP failure
                    dropped_tracks.add(track_id)
                    # Shape becomes available for new track creation

                    if self.debug:
                        print(
                            f"    DROPPED: Track {track_id} (ICP failed for Detection {shape_idx})"
                        )

        # Remove dropped tracks from tracking
        # for track_id in dropped_tracks:
        #     if track_id in self.tracked_objects:
        #         del self.tracked_objects[track_id]

        # Create new tracks for unmatched shapes
        unmatched_count = 0
        for j, alpha_shape in enumerate(alpha_shapes):
            if j not in used_shapes:
                obj_id = self._create_new_track(timestamp_ns, alpha_shape, pose)
                frame_assignments.append(obj_id)
                unmatched_count += 1

                if self.debug:
                    self.debug_stats["frame_stats"][timestamp_ns]["new_tracks"] += 1
                    # print(f"    NEW TRACK: {obj_id} <- Detection {j} (unmatched)")

        if self.debug:
            self.debug_stats["frame_stats"][timestamp_ns][
                "unmatched_detections"
            ] = unmatched_count
            print(f"  Created {unmatched_count} new tracks for unmatched detections")

        return frame_assignments

    def _hungarian_assignment(
        self,
        cost_matrix: np.ndarray,
        iou_matrix: np.ndarray,
        semantic_matrix: np.ndarray,
        iou_threshold: float,
    ) -> List[Tuple[int, int]]:
        """Hungarian assignment based on IoU threshold."""
        if iou_matrix.size == 0 or cost_matrix.size == 0:
            return []

        # Convert IoU to a cost matrix (since Hungarian solves a minimization problem)
        # cost_matrix = 1.0 - iou_matrix - semantic_matrix

        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Collect assignments that pass the threshold
        assignments = [
            (r, c)
            for r, c in zip(row_ind, col_ind)
            if iou_matrix[r, c] >= iou_threshold
        ]

        return assignments

    def _create_new_track(
        self, timestamp_ns: int, alpha_shape: Dict, pose: np.ndarray
    ) -> int:
        """Create a new tentative track."""
        track_id = self.next_id
        self.next_id += 1

        self.tracked_objects[track_id] = {
            "id": track_id,
            "state": "tentative",  # tentative → confirmed → deleted
            "hits": 1,  # number of successful matches
            "missed": 0,  # consecutive misses
            "icp_failures": 0,  # consecutive ICP failures
            "first_timestamp": timestamp_ns,
            "last_timestamp": timestamp_ns,
            "semantic_features": alpha_shape.get("semantic_features", None),
            "trajectory": {
                timestamp_ns: {
                    "alpha_shape": alpha_shape,
                    "pose": pose.copy(),
                    "ego_transform": np.eye(4),
                    "combined_transform": np.eye(4),
                    "icp_error": 0.0,
                }
            },
            "merged_alpha_shape": alpha_shape.copy(),
        }

        if self.debug:
            print(f"    NEW TRACK {track_id} (tentative)")

        return track_id

    def _update_track(
        self,
        track_id: int,
        timestamp_ns: int,
        alpha_shape: Dict,
        pose: np.ndarray,
        score: float,
    ) -> bool:
        """
        Update an existing track using ICP and lifecycle logic.
        Returns False if track should be dropped.
        """

        track = self.tracked_objects[track_id]

        icp_threshold = 1.0
        max_consecutive_misses = 100

        # Get previous frame data
        prev_timestamp = track["last_timestamp"]
        if prev_timestamp not in track["trajectory"]:
            return False

        # prev_data = track["trajectory"][prev_timestamp]
        # prev_alpha_shape = prev_data["alpha_shape"]
        # prev_pose = prev_data["pose"]

        # # Compute ego vehicle transformation
        # ego_transform = np.linalg.inv(pose) @ prev_pose

        # # Get original points for ICP
        # prev_points = prev_alpha_shape.get("original_points")
        # curr_points = alpha_shape.get("original_points")

        # # Transform both to world coordinates
        # prev_world = points_rigid_transform(prev_points, prev_pose)
        # curr_world = points_rigid_transform(curr_points, pose)

        # # Run ICP in world coordinates
        # # world_transform, icp_error, prev_inlier_idx, curr_inlier_idx = (
        # #     self._compute_world_icp(
        # #         prev_world,
        # #         curr_world,
        # #         return_inlier_indices=True,
        # #         debug=self.debug
        # #     )
        # # )
        # world_transform, icp_error = icp_open3d_robust(prev_world, curr_world, initial_alignment='ransac')

        # print("world_transform[:3, 3]", world_transform[:3, 3])

        # # confidence = (len(prev_inlier_idx) + len(curr_inlier_idx)) / (len(prev_world) + len(curr_world))
        # confidence = 0.8
        # # print(f"confidence={confidence:.2f}")

        # if world_transform is None or icp_error > icp_threshold:
        #     print("combined_transform is None or icp_error > icp_threshold")
        #     print(f"{world_transform=}")
        #     print(f"{icp_error=}")
        #     print(f"{icp_threshold=}")
        #     exit()
        #     # # Track miss
        #     # track["missed"] = track.get("missed", 0) + 1
        #     # track["icp_failures"] = track.get("icp_failures", 0) + 1

        #     # # Delete based on consecutive misses
        #     # if track["missed"] >= max_consecutive_misses:
        #     #     track["state"] = "deleted"
        #     #     if self.debug:
        #     #         print(
        #     #             f"    DELETING Track {track_id}: {track['missed']} consecutive misses"
        #     #         )
        #     #     return False

        #     # # Or delete based on ICP failures for confirmed tracks
        #     # icp_fail_max = 3 if track.get("state") == "confirmed" else 1
        #     # if track["icp_failures"] >= icp_fail_max:
        #     #     track["state"] = "deleted"
        #     #     if self.debug:
        #     #         print(
        #     #             f"    DELETING Track {track_id}: {track['icp_failures']} ICP failures"
        #     #         )
        #     #     return False

        #     # return False  # Soft failure

        # Reset failure counters on success
        track["missed"] = 0
        track["icp_failures"] = 0
        track["hits"] = track.get("hits", 0) + 1
        track["last_timestamp"] = timestamp_ns

        # Promote tentative → confirmed if enough hits
        if track.get("state", "tentative") == "tentative" and track["hits"] >= getattr(
            self, "min_hits", 3
        ):
            track["state"] = "confirmed"
            if self.debug:
                print(f"    CONFIRMED: Track {track_id}")

        # Update trajectory
        track["trajectory"][timestamp_ns] = {
            "alpha_shape": alpha_shape,
            "pose": pose.copy(),
            "score": score,
            # "ego_transform": ego_transform,
            # "world_transform": world_transform,
            # "icp_error": icp_error,
            # "icp_confidence": confidence,
        }

        if self.debug:
            self.debug_stats["total_updates"] += 1

        self.tracked_objects[track_id] = track

        return True

    def _compute_world_icp(
        self, prev_world, curr_world, return_inlier_indices=False, debug=False
    ):
        """Run ICP directly in world coordinates."""
        if len(prev_world) < 5 or len(curr_world) < 5:
            if return_inlier_indices:
                return None, float("inf"), None, None
            return None, float("inf")

        try:
            # Subsample if needed
            max_points = 200
            prev_subset = prev_world
            curr_subset = curr_world
            prev_indices = np.arange(len(prev_world))
            curr_indices = np.arange(len(curr_world))

            if len(prev_world) > max_points:
                idx = np.random.choice(len(prev_world), max_points, replace=False)
                prev_subset = prev_world[idx]
                prev_indices = prev_indices[idx]

            if len(curr_world) > max_points:
                idx = np.random.choice(len(curr_world), max_points, replace=False)
                curr_subset = curr_world[idx]
                curr_indices = curr_indices[idx]

            # Run ICP
            if return_inlier_indices:
                R, t, icp_error, prev_inliers, curr_inliers = icp(
                    prev_subset,
                    curr_subset,
                    max_iterations=50,
                    tolerance=1e-4,
                    ret_err=True,
                    return_inliers=True,
                )
            else:
                R, t, icp_error = icp(
                    prev_subset,
                    curr_subset,
                    max_iterations=50,
                    tolerance=1e-4,
                    ret_err=True,
                )

            if R is None:
                if return_inlier_indices:
                    return None, float("inf"), None, None
                return None, float("inf")

            # Build 4x4 transform
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = t

            if return_inlier_indices:
                if prev_inliers is not None and curr_inliers is not None:
                    return (
                        transform,
                        icp_error,
                        prev_indices[prev_inliers],
                        curr_indices[curr_inliers],
                    )
                return transform, icp_error, None, None

            return transform, icp_error

        except Exception as e:
            if debug:
                print(f"World ICP failed: {e}")
            if return_inlier_indices:
                return None, float("inf"), None, None
            return None, float("inf")

    def _get_active_tracks(
        self, timestamp_ns: int, timestamp_thresh: int = 1e9
    ) -> List[int]:
        """Get tracks that were active in the given frame."""
        active = []
        for track_id, track in self.tracked_objects.items():
            if abs(track["last_timestamp"] - timestamp_ns) < timestamp_thresh:
                active.append(track_id)
        return active

    def _predict_alpha_shape(
        self, track_id: int, frame_id: int, current_pose: np.ndarray
    ) -> Optional[Dict]:
        """Predict alpha shape for a track at current frame."""
        track = self.tracked_objects[track_id]

        if frame_id not in track["trajectory"]:
            return None

        last_data = track["trajectory"][frame_id]
        last_shape = last_data["alpha_shape"]
        last_pose = last_data["pose"]

        # Transform from last frame to current frame
        transform = np.linalg.inv(current_pose) @ last_pose
        predicted_shape = AlphaShapeUtils.transform_alpha_shape(last_shape, transform)

        return predicted_shape

    def align_all_frames_to_reference(self, track, reference_frame_id):
        """Transform all shapes in a track to a common coordinate system."""
        trajectory = track["trajectory"]
        aligned_shapes = {}

        # Get sorted frame IDs
        frame_ids = sorted(trajectory.keys())
        ref_idx = frame_ids.index(reference_frame_id)

        # Reference frame stays as-is
        aligned_shapes[reference_frame_id] = trajectory[reference_frame_id][
            "alpha_shape"
        ]["original_points"]

        # Forward pass: from reference frame to later frames
        accumulated_transform = np.eye(4)
        for i in range(ref_idx + 1, len(frame_ids)):
            frame_id = frame_ids[i]
            prev_frame_id = frame_ids[i - 1]

            frame_data = trajectory[frame_id]
            if "combined_transform" in frame_data:
                # combined_transform goes from prev_frame to current_frame
                # We need to accumulate: ref -> ... -> prev_frame -> current_frame
                accumulated_transform = (
                    frame_data["combined_transform"] @ accumulated_transform
                )

                # To align current frame to reference, we need the inverse
                ref_to_current_transform = np.linalg.inv(accumulated_transform)
                aligned_points = points_rigid_transform(
                    frame_data["alpha_shape"]["original_points"],
                    ref_to_current_transform,
                )
                aligned_shapes[frame_id] = aligned_points
            else:
                # No transform available, skip this frame
                if self.debug:
                    print(f"Warning: No combined_transform for frame {frame_id}")

        # Backward pass: from reference frame to earlier frames
        accumulated_transform = np.eye(4)
        for i in range(ref_idx - 1, -1, -1):
            frame_id = frame_ids[i]
            next_frame_id = frame_ids[i + 1]

            # We need the transform from current_frame to next_frame
            # which is the inverse of combined_transform (next_frame to current_frame)
            next_frame_data = trajectory[next_frame_id]
            if "combined_transform" in next_frame_data:
                # combined_transform goes from current_frame to next_frame
                # For backward accumulation, we need the inverse
                current_to_next = next_frame_data["combined_transform"]
                next_to_current = np.linalg.inv(current_to_next)

                # Accumulate: ref -> ... -> next_frame -> current_frame
                accumulated_transform = next_to_current @ accumulated_transform

                # To align current frame to reference, we need the inverse
                ref_to_current_transform = np.linalg.inv(accumulated_transform)
                aligned_points = points_rigid_transform(
                    trajectory[frame_id]["alpha_shape"]["original_points"],
                    ref_to_current_transform,
                )
                aligned_shapes[frame_id] = aligned_points
            else:
                # No transform available, skip this frame
                if self.debug:
                    print(f"Warning: No combined_transform for frame {next_frame_id}")

        return aligned_shapes

    def enhanced_forward_backward_consistency(self):
        del_ids = []
        for track_id, track in self.tracked_objects.items():
            if track.get("state") != "confirmed" or len(track["trajectory"]) < 3:
            # if len(track["trajectory"]) < 3:
                del_ids.append(track_id)
                continue

            timestamps = sorted(track["trajectory"].keys())
            trajectory = track["trajectory"]

            object_points_per_timestamp = []
            world_centers = []

            # Collect world points and centers
            for timestamp_ns in timestamps:
                traj = trajectory[timestamp_ns]
                ego_points = traj["alpha_shape"]["original_points"]
                world_points = points_rigid_transform(ego_points, traj["pose"])
                object_points_per_timestamp.append(world_points.copy())

                center = (world_points.min(axis=0) + world_points.max(axis=0)) / 2.0
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

                R, t, _ = icp(prev_points, cur_points, max_iterations=5)

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
            for i, timestamp_ns in enumerate(timestamps):
                trajectory[timestamp_ns]["optimized_pose"] = optimized_poses[i]

            # After optimization, compute object-centric representation
            object_poses, object_points = self.compute_object_centric_transforms(track)

            # Merge all points in object frame
            # merged_object_points = self.merge_object_centric_points(object_points)
            # merged_object_points = np.vstack(object_points)

            # find the best frame
            num_kept_per_frame = [0 for _ in range(len(object_points))]
            best_num_kept = -1
            best_points = None
            for i in range(len(object_points)):
                cur_points = object_points[i]
                nbr_traversals = object_points[:i] + object_points[(i+1):]

                ppscore = compute_ppscore(cur_points, nbr_traversals)
                ppscore_mask = ppscore >= 0.7

                num_kept = ppscore_mask.sum()
                num_kept_per_frame[i] = num_kept

                if num_kept > best_num_kept:
                    best_points = cur_points[ppscore_mask]
                    best_num_kept = num_kept

            print("num_kept_per_frame", num_kept_per_frame)
            if best_num_kept < 10:
                del_ids.append(track_id)
                continue

            merged_object_points = best_points

            merged_object_points = voxel_sampling_fast(merged_object_points, 0.05, 0.05, 0.05)

            # ppscore = compute_ppscore(merged_object_points, object_points)
            # print("merged_object_points ppscore", ppscore.min(), ppscore.mean(), ppscore.max())
            # ppscore_mask = ppscore >= min(np.median(ppscore), 0.7)
            # merged_object_points = merged_object_points[ppscore_mask]

            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            # # ppscore computation
            # ppscore = compute_ppscore(merged_object_points, object_points)

            # # Create the scatter plot and store the return value for the colorbar
            # scatter = ax.scatter(
            #     merged_object_points[:, 0],
            #     merged_object_points[:, 1],
            #     s=1,
            #     c=ppscore,
            #     cmap="jet",
            #     label="Object Points",
            #     alpha=1.0,
            #     vmin=0,  # Set minimum value for colormap
            #     vmax=1   # Set maximum value for colormap
            # )

            # # Add colorbar
            # cbar = plt.colorbar(scatter, ax=ax)
            # cbar.set_label('PPScore', rotation=270, labelpad=15)  # Label with padding
            # cbar.ax.tick_params(labelsize=9)  # Adjust tick label size if needed

            # ax.set_title(
            #     f"Optimized Track {track_id} - Merged Object Coordinates PPscore"
            # )
            # ax.set_aspect("equal")

            # save_folder = Path("./merged_ppscore")
            # save_folder.mkdir(exist_ok=True)
            # save_path = save_folder / f"track_{track_id}.png"
            # plt.savefig(save_path, dpi=300, bbox_inches="tight")
            # plt.close()

            # print(f"Kept {ppscore_mask.sum()}/{len(ppscore_mask)} merged object points")

            mesh = trimesh.convex.convex_hull(merged_object_points)
            merged_vertices = mesh.vertices

            for i, timestamp_ns in enumerate(timestamps):
                # update the alpha shape
                # ego_centre = track["trajectory"][timestamp_ns]["alpha_shape"][
                #     "centroid_3d"
                # ]
                ego_pose = track["trajectory"][timestamp_ns]["pose"]
                # world_centre = points_rigid_transform(
                #     ego_centre.reshape(-1, 3), ego_pose
                # )[0]

                object_cur_pose = optimized_poses[i]

                # move the merged_vertices to this object_pose
                cur_vertices = points_rigid_transform(
                    merged_vertices.copy(), object_cur_pose
                )
                # cur_vertices_centre = np.mean(cur_vertices, axis=0)

                # object_cur_pose_centre = object_cur_pose[:3, 3]

                # print("world_centre", world_centre)
                # print("object_cur_pose_centre", object_cur_pose_centre)
                # print("cur_vertices_centre", cur_vertices_centre)

                # move
                # print("norm", np.linalg.norm(object_cur_pose_centre - world_centre))

                # undo back to ego coordinates
                cur_vertices_ego = points_rigid_transform(
                    cur_vertices, np.linalg.inv(ego_pose)
                )
                # print("cur_vertices_ego mean", cur_vertices_ego.mean(axis=0))
                # print("ego_centre", ego_centre)

                track["trajectory"][timestamp_ns]["alpha_shape"] = {
                    "vertices_3d": cur_vertices_ego,
                    "centroid_3d": cur_vertices_ego.mean(axis=0),
                    "mesh": None,  # don't need anymore?
                    "original_points": cur_vertices_ego,
                }

            self._compute_oriented_boxes(track_id, timestamps, optimized_poses)

            # Store the canonical object representation
            track["object_centric_points"] = merged_object_points
            track["object_to_world_poses"] = object_poses

            self.tracked_objects[track_id] = track

        for track_id in del_ids:
            del self.tracked_objects[track_id]

    def enhanced_forward_backward_consistency_debug(self):
        for track_id, track in self.tracked_objects.items():
            if track.get("state") != "confirmed" or len(track["trajectory"]) < 3:
                continue

            timestamps = sorted(track["trajectory"].keys())
            trajectory = track["trajectory"]

            object_points_per_timestamp = []
            world_centers = []

            # Collect world points and centers
            for timestamp_ns in timestamps:
                traj = trajectory[timestamp_ns]
                ego_points = traj["alpha_shape"]["original_points"]
                world_points = points_rigid_transform(ego_points, traj["pose"])
                object_points_per_timestamp.append(world_points.copy())

                center = (world_points.min(axis=0) + world_points.max(axis=0)) / 2.0
                world_centers.append(center)

            world_centers = np.array(world_centers)

            # Compute initial heading from velocity
            if len(world_centers) > 1:
                initial_velocity = world_centers[-1] - world_centers[0]
                speed = np.linalg.norm(initial_velocity) / (
                    timestamps[-1] - timestamps[0]
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

            print("first_object_pose", first_object_pose)

            relative_poses = []
            confidences = []
            cumulative_pose = first_object_pose.copy()
            for i in range(1, len(timestamps)):
                curr_timestamp = timestamps[i]
                world_transform = trajectory[curr_timestamp].get("world_transform")

                prev_centre = world_centers[i - 1]
                undo_pose = np.linalg.inv(cumulative_pose)
                print("prev_centre", prev_centre)
                print("undo_pose[:3, 3]", undo_pose[:3, 3])

                prev_points = points_rigid_transform(
                    object_points_per_timestamp[i - 1], undo_pose
                )
                cur_points = points_rigid_transform(
                    object_points_per_timestamp[i], undo_pose
                )

                centres_transform = world_centers[i] - world_centers[i - 1]

                fig, ax = plt.subplots(1, 1, figsize=(10, 8))

                ax.scatter(
                    prev_points[:, 0],
                    prev_points[:, 1],
                    c="black",
                    s=2,
                    alpha=0.7,
                    label=f"Previous",
                )
                ax.scatter(
                    cur_points[:, 0],
                    cur_points[:, 1],
                    c="green",
                    s=2,
                    alpha=0.7,
                    label=f"Current",
                )

                # ax.set_title(f'Track {track_id} - World Coordinates')
                ax.set_aspect("equal")
                ax.grid(True)
                ax.legend()

                save_path = (
                    f"./optimized_trajectories/track_{track_id}_relative_{i}.png"
                )
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()

                R, t, icp_err = icp(prev_points, cur_points, ret_err=True)
                world_transform_t = world_transform[:3, 3]

                transform, error, result = icp_open3d_robust(
                    prev_points,
                    cur_points,
                    initial_alignment="ransac",
                    return_full_result=True,
                )

                print("icp_open3d_robust", transform, error, result)

                open3d_transform = transform[:3, 3]

                print("world_transform_t", world_transform_t)
                print("new t", t)
                print("open3d_transform", open3d_transform)
                print("centres_transform", centres_transform)

                # open3d
                world_transform = transform

                assert world_transform is not None
                relative_poses.append(world_transform)
                confidences.append(
                    trajectory[curr_timestamp].get("icp_confidence", 0.8)
                )

                cumulative_pose = cumulative_pose @ world_transform
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

            print(
                f"initial_poses: {len(initial_poses)}, constraints: {len(constraints)}, timestamps: {len(timestamps)}"
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

            print(f"Optimization quality: {quality}")

            # Extract positions
            initial_positions = np.stack(
                [pose[:3, 3] for pose in initial_poses], axis=0
            )
            optimized_positions = np.stack(
                [pose[:3, 3] for pose in optimized_poses], axis=0
            )

            print("world_centers", world_centers)
            print("initial_positions", initial_positions)
            print("optimized_positions", optimized_positions)

            # Visualization
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(object_points_per_timestamp)))

            # Transform object points to world using optimized poses
            for i, points in enumerate(object_points_per_timestamp):
                ax.scatter(
                    world_points[:, 0],
                    world_points[:, 1],
                    c=[colors[i]],
                    s=2,
                    alpha=0.7,
                    label=f"Frame {i}",
                )

            # ax.plot(initial_positions[:, 0], initial_positions[:, 1],
            #         'r-', linewidth=2, label="Initial trajectory")
            # ax.plot(optimized_positions[:, 0], optimized_positions[:, 1],
            #         'b-', linewidth=2, label="Optimized trajectory")
            # ax.plot(world_centers[:, 0], world_centers[:, 1],
            #         'g-', linewidth=2, label="World centers")

            ax.set_title(f"Track {track_id} - World Coordinates")
            ax.set_aspect("equal")
            ax.grid(True)
            ax.legend()

            save_path = f"./optimized_trajectories/track_{track_id}_fixed.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Error metrics
            optimized_err = np.linalg.norm(
                optimized_positions - world_centers, axis=1
            ).mean()
            initial_err = np.linalg.norm(
                initial_positions - world_centers, axis=1
            ).mean()

            print(
                f"Optimized error: {optimized_err:.3f}m, Initial error: {initial_err:.3f}m"
            )

            print(
                "optimized_positions - world_centers",
                optimized_positions - world_centers,
            )

            # Update tracked objects
            for i, timestamp_ns in enumerate(timestamps):
                trajectory[timestamp_ns]["optimized_pose"] = optimized_poses[i]

            # After optimization, compute object-centric representation
            object_poses, object_points = self.compute_object_centric_transforms(track)

            # Merge all points in object frame
            merged_object_points = self.merge_object_centric_points(object_points)

            ####################################################################################
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            colors = plt.cm.viridis(np.linspace(0, 1, len(object_points)))

            # Plot point clouds in world coordinates
            for i, points in enumerate(object_points):
                ax.scatter(points[:, 0], points[:, 1], c=[colors[i]], s=3, alpha=0.7)

            ax.set_title(f"Optimized Track {track_id} - Merged object coordinates")
            ax.set_aspect("equal")
            # ax.grid(True)
            # ax.legend()

            save_path = f"./optimized_trajectories/track_{track_id}_merged.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            # ppscore computation
            ppscore = compute_ppscore(merged_object_points, object_points)

            ax.scatter(
                merged_object_points[:, 0],
                merged_object_points[:, 1],
                s=1,
                c=ppscore,
                cmap="jet",
                label="Object Points",
                alpha=1.0,
            )

            ax.set_title(
                f"Optimized Track {track_id} - Merged Object Coordinates PPscore"
            )
            ax.set_aspect("equal")

            save_path = f"./optimized_trajectories/track_{track_id}_merged_ppscore.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            ####################################################################################

            # Store the canonical object representation
            track["object_centric_points"] = merged_object_points
            track["object_to_world_poses"] = object_poses

            self._visualize_optimization(
                track_id, object_points_per_timestamp, optimized_poses, timestamps
            )

    def compute_object_centric_transforms(self, track):
        """
        Compute object-to-world transforms for each timestamp.
        Returns:
            object_poses: List of 4x4 matrices that transform from object-centric to world
            object_points: List of points in object-centric coordinates
        """
        timestamps = sorted(track["trajectory"].keys())
        trajectory = track["trajectory"]

        # First, get the optimized poses if available, otherwise build them
        if "optimized_pose" in trajectory[timestamps[0]]:
            # Use optimized poses from your optimization
            object_to_world_poses = []
            for timestamp in timestamps:
                object_to_world_poses.append(trajectory[timestamp]["optimized_pose"])
        else:
            raise ValueError(
                f"must run enhanced forward backward consistency! {trajectory[timestamps[0]].keys()}"
            )

        # Now transform points to object-centric coordinates
        object_centric_points = []
        for timestamp, obj_pose in zip(timestamps, object_to_world_poses):
            # Get world points
            ego_points = trajectory[timestamp]["alpha_shape"]["original_points"]
            ego_to_world = trajectory[timestamp]["pose"]
            world_points = points_rigid_transform(ego_points, ego_to_world)

            # Transform to object-centric: multiply by inverse of object pose
            world_to_object = np.linalg.inv(obj_pose)
            object_points = points_rigid_transform(world_points, world_to_object)
            object_centric_points.append(object_points)

        return object_to_world_poses, object_centric_points

    def merge_object_centric_points(self, object_centric_points_list):
        """
        Merge multiple point clouds in object-centric coordinates.
        Could use various strategies: voxel grid, ICP alignment, etc.
        """
        # Simple concatenation
        merged_points = np.vstack(object_centric_points_list)

        # Optional: voxel grid downsampling to remove duplicates
        from scipy.spatial import cKDTree

        voxel_size = 0.1  # 10cm voxels
        voxelized = np.round(merged_points / voxel_size).astype(np.float32) * voxel_size
        unique_voxels = np.unique(voxelized, axis=0)

        return unique_voxels

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

    def _compute_oriented_boxes(self, track_id, timestamps, optimized_poses):
        """Compute oriented bounding boxes based on object motion direction"""
        trajectory = self.tracked_objects[track_id]["trajectory"]

        for i, timestamp_ns in enumerate(timestamps):
            traj = trajectory[timestamp_ns]
            vertices_3d = traj["alpha_shape"].get("vertices_3d")

            if vertices_3d is None or len(vertices_3d) < 3:
                continue

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

            # Create oriented bounding box from 3D vertices
            box = self._vertices_to_oriented_box(vertices_3d, yaw)
            trajectory[timestamp_ns]["oriented_box"] = box

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

    def _visualize_optimization(
        self, track_id, object_points_per_timestamp, optimized_poses, timestamps
    ):
        """Visualize optimization results with corrected transforms"""
        # fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # colors = plt.cm.viridis(np.linspace(0, 1, len(object_points_per_timestamp)))
        # reference_pose = optimized_poses[0]

        # # Plot aligned point clouds
        # for i, (points, pose) in enumerate(zip(object_points_per_timestamp, optimized_poses)):
        #     # Transform: object_coords -> world_coords -> reference_coords
        #     object_to_world = pose
        #     world_to_reference = np.linalg.inv(reference_pose)
        #     object_to_reference = world_to_reference @ object_to_world

        #     aligned_points = points_rigid_transform(points, object_to_reference)

        #     ax.scatter(aligned_points[:, 0], aligned_points[:, 1],
        #             c=[colors[i]], s=2, alpha=0.7, label=f'Frame {i}')

        # # Plot trajectory
        # positions = np.stack([pose[:3, 3] for pose in optimized_poses], axis=0)
        # world_to_reference = np.linalg.inv(reference_pose)
        # reference_positions = points_rigid_transform(positions, world_to_reference)

        # ax.plot(reference_positions[:, 0], reference_positions[:, 1],
        #     'r-', linewidth=2, label="Trajectory")
        # ax.scatter(reference_positions[:, 0], reference_positions[:, 1],
        #         c='red', s=30, zorder=5)

        # ax.set_title(f'Optimized Track {track_id} - Object Reference Frame')
        # ax.set_aspect('equal')
        # ax.grid(True)
        # ax.legend()

        # save_path = f"./optimized_trajectories/track_{track_id}_optimized.png"
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"Saved optimization visualization: {save_path}")

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(object_points_per_timestamp)))

        timestamps_secs = np.array(timestamps) * 1e-9
        timestamps_secs = timestamps_secs - timestamps_secs.min()

        # Plot point clouds in world coordinates
        for i, (points, pose) in enumerate(
            zip(object_points_per_timestamp, optimized_poses)
        ):
            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=[colors[i]],
                s=2,
                alpha=0.7,
                label=f"Frame {timestamps_secs[i]} original points",
            )

        # Plot trajectory (positions already in world coordinates)
        positions = np.stack([pose[:3, 3] for pose in optimized_poses], axis=0)

        ax.plot(positions[:, 0], positions[:, 1], "r-", linewidth=2, label="Trajectory")
        ax.scatter(positions[:, 0], positions[:, 1], c="red", s=30, zorder=5)

        ax.set_title(f"Optimized Track {track_id} - World Coordinates")
        ax.set_aspect("equal")
        ax.grid(True)
        # ax.legend()

        save_path = f"./optimized_trajectories/track_{track_id}_world.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _forward_backward_consistency(self):
        """Perform forward-backward pass for temporal consistency using stored ICP transforms."""
        if self.debug:
            print(f"\n--- FORWARD-BACKWARD CONSISTENCY ---")

        consistency_updates = 0

        # Forward pass: propagate merged shapes forward using stored transformations
        for track_id, track in self.tracked_objects.items():
            trajectory = track["trajectory"]
            frames = sorted(trajectory.keys())

            if len(frames) <= 1:
                continue

            if self.debug and len(frames) > 3:  # Only log for longer tracks
                print(
                    f"  Processing track {track_id}: {len(frames)} frames ({frames[0]}-{frames[-1]})"
                )

            # Collect shapes with their accumulated transformations
            shapes_with_transforms = []
            accumulated_transform = np.eye(4)

            # First frame (reference)
            first_frame_data = trajectory[frames[0]]
            shapes_with_transforms.append((first_frame_data["alpha_shape"], np.eye(4)))

            # Subsequent frames with accumulated transformations
            for i in range(1, len(frames)):
                frame_id = frames[i]
                frame_data = trajectory[frame_id]

                # Accumulate transformation from first frame to current frame
                if "combined_transform" in frame_data:
                    # Use stored combined transformation (ego + object ICP)
                    accumulated_transform = (
                        frame_data["combined_transform"] @ accumulated_transform
                    )
                else:
                    # Fallback to ego transform only
                    if "ego_transform" in frame_data:
                        accumulated_transform = (
                            frame_data["ego_transform"] @ accumulated_transform
                        )

                shapes_with_transforms.append(
                    (frame_data["alpha_shape"], accumulated_transform.copy())
                )

            # Merge all shapes using accumulated transformations
            if len(shapes_with_transforms) > 1:
                merged = AlphaShapeUtils.merge_alpha_shapes_with_icp(
                    shapes_with_transforms
                )

                if merged is not None:
                    track["final_merged_alpha_shape"] = merged
                    consistency_updates += 1
                else:
                    # Fallback to last known good shape
                    track["final_merged_alpha_shape"] = track.get(
                        "merged_alpha_shape", shapes_with_transforms[-1][0]
                    )
            else:
                track["final_merged_alpha_shape"] = shapes_with_transforms[0][0]

        if self.debug:
            print(f"  Applied {consistency_updates} consistency updates")
            print(
                f"  Tracks with final merged shapes: {len([t for t in self.tracked_objects.values() if 'final_merged_alpha_shape' in t])}"
            )

            # Print ICP error statistics
            all_icp_errors = []
            for track in self.tracked_objects.values():
                for frame_data in track["trajectory"].values():
                    if "icp_error" in frame_data and frame_data["icp_error"] < float(
                        "inf"
                    ):
                        all_icp_errors.append(frame_data["icp_error"])

            if all_icp_errors:
                print(
                    f"  ICP error stats - Mean: {np.mean(all_icp_errors):.3f}m, Max: {np.max(all_icp_errors):.3f}m, Count: {len(all_icp_errors)}"
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