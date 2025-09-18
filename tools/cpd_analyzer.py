"""
Goal
    1. run CPD to extract object proposals
    2. extract features similar to standard_feature_analyzer.py
    3. run similar analysis and av2 eval metrics
        e.g. random forest classifier on the features
        -> can we find good features to determine the quality of proposals?
        -> can we find better heuristics?

"""

import argparse
import copy
import cProfile
import io
import json
import os
import pickle as pkl
import pstats
import sys
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import trimesh
import yaml
from easydict import EasyDict
from scipy.spatial import ConvexHull, cKDTree
from torch import Tensor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_plotting_utils
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from av2.datasets.sensor.constants import RingCameras
from av2.evaluation import SensorCompetitionCategories
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.map.lane_segment import LaneSegment
from av2.map.map_api import ArgoverseStaticMap, GroundHeightLayer

# from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from av2.structures.sweep import Sweep
from av2.utils.io import read_city_SE3_ego, read_ego_SE3_sensor, read_feather
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Decision Tree Classifier
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_text,
    plot_tree,
)
from tqdm import tqdm

from lion.unsupervised_core.alpha_shape import AlphaShapeMFCF, OWLViTAlphaShapeMFCF
from lion.unsupervised_core.box_utils import (
    apply_pose_to_box,
    argo2_box_to_lidar,
    get_rotated_box,
)
from lion.unsupervised_core.c_proto_refine import C_PROTO, CSS
from lion.unsupervised_core.mfcf import MFCF
from lion.unsupervised_core.outline_utils import (
    KL_entropy_score,
    OutlineFitter,
    hierarchical_occupancy_score,
    points_rigid_transform,
    smooth_points_and_ppscore,
)
from lion.unsupervised_core.rotate_iou_cpu_eval import rotate_iou_cpu_eval

np.set_printoptions(suppress=True, precision=2)

import multiprocessing as mp
import signal
import threading
import traceback


def signal_handler(signum, frame):
    # Only print stack trace if we're in the main thread of the main process
    current_process = mp.current_process()
    if threading.current_thread() is threading.main_thread() and current_process.name == 'MainProcess':
        print(f"\n=== PROCESS KILLED (Signal {signum}) ===")
        print("Stack trace at time of termination:")
        traceback.print_stack(frame, file="cpd_analyzer_stack.log")
        print("=" * 60)
    
    # Try to gracefully shutdown multiprocessing resources
    try:
        # If you have a global pool variable, you can terminate it here
        # pool.terminate()
        # pool.join()
        pass
    except:
        pass
    
    sys.exit(1)


# Remove position-based features that shouldn't affect quality
IGNORE_COLS = [
    "proto_id_avg",
    "gt_best_iou_avg",
    "gt_best_iou",
    "object_id",
    "object_id_avg",
    "proto_id",
    "frame_idx",
    "timestamp_ns",
    "x_avg",
    "y_avg",
    "z_avg",  # Remove absolute positions
    "yaw_avg",
    "yaw",  # Remove absolute orientation
    "x",
    "y",
    "z",  # Remove any non-averaged positions if present
    "timestamp_ns_avg",
    "frame_idx_avg",
    # avoid filtering sizes
    "length",
    "width",
    "height",
    "score",
    "score_mean",
    "score_mean_avg",
    "length_avg",
    "width_avg",
    "height_avg",
    "box_volume",
    "box_volume_avg",
    "css_total_score",
    "orientation_quality",
    "css_size_score",
    # 'ground_height', 'ground_height_avg',
    # 'score_avg',
    "box_aspect_ratio_lw_avg",
    "box_aspect_ratio_wh_avg",
    "box_aspect_ratio_lw",
    "box_aspect_ratio_wh",
    "box_aspect_ratio_lh_avg",
    "box_aspect_ratio_lh",
]

LABEL_ATTR = (
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
)


@torch.jit.script
def xyz_to_quat(xyz_rad: Tensor) -> Tensor:
    """Convert euler angles (xyz - pitch, roll, yaw) to scalar first quaternions.

    Args:
        xyz_rad: (...,3) Tensor of roll, pitch, and yaw in radians.

    Returns:
        (...,4) Scalar first quaternions (wxyz).
    """
    x_rad = xyz_rad[..., 0]
    y_rad = xyz_rad[..., 1]
    z_rad = xyz_rad[..., 2]

    cy = torch.cos(z_rad * 0.5)
    sy = torch.sin(z_rad * 0.5)
    cp = torch.cos(y_rad * 0.5)
    sp = torch.sin(y_rad * 0.5)
    cr = torch.cos(x_rad * 0.5)
    sr = torch.sin(x_rad * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat_wxyz = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat_wxyz


@torch.jit.script
def yaw_to_quat(yaw_rad: Tensor) -> Tensor:
    """Convert yaw (rotation about the vertical axis) to scalar first quaternions.

    Args:
        yaw_rad: (...,1) Rotations about the z-axis.

    Returns:
        (...,4) scalar first quaternions (wxyz).
    """
    xyz_rad = torch.zeros_like(yaw_rad)[..., None].repeat_interleave(3, dim=-1)
    xyz_rad[..., -1] = yaw_rad
    quat_wxyz: Tensor = xyz_to_quat(xyz_rad)
    return quat_wxyz


def quat_to_yaw(qw, qx, qy, qz):
    """Convert quaternion to yaw angle.

    Args:
        qw, qx, qy, qz: Quaternion components

    Returns:
        float: Yaw angle in radians
    """
    # For rotation around Z-axis: yaw = 2 * atan2(qz, qw)
    return 2 * np.arctan2(qz, qw)


def log_config_to_file(cfg, pre="cfg", logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info("\n%s.%s = edict()" % (pre, key))
            log_config_to_file(cfg[key], pre=pre + "." + key, logger=logger)
            continue
        logger.info("%s.%s: %s" % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval

    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split(".")
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, "NotFoundKey: %s" % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, "NotFoundKey: %s" % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(",")
            for src in key_val_list:
                cur_key, cur_val = src.split(":")
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(",")
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(
                d[subkey]
            ), "type {} does not match original type {}".format(
                type(value), type(d[subkey])
            )
            d[subkey] = value


def merge_new_config(config, new_config):
    if "_BASE_CONFIG_" in new_config:
        with open(new_config["_BASE_CONFIG_"], "r") as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, "r") as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / "../").resolve()
cfg.LOCAL_RANK = 0


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/dataset_configs/cpd/waymo_unsupervised_cproto.yaml",
        help="specify the config for training",
    )

    parser.add_argument("--run_cpd_infos", action="store_true", default=False, help="")

    # default ones from pcdet
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        required=False,
        help="batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        required=False,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--extra_tag", type=str, default="default", help="extra tag for this experiment"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint to start from"
    )
    parser.add_argument(
        "--pretrained_model", type=str, default=None, help="pretrained_model"
    )
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm"], default="none"
    )
    parser.add_argument(
        "--tcp_port", type=int, default=23271, help="tcp port for distrbuted training"
    )
    parser.add_argument(
        "--sync_bn", action="store_true", default=False, help="whether to use sync bn"
    )
    parser.add_argument("--fix_random_seed", action="store_true", default=True, help="")
    parser.add_argument(
        "--ckpt_save_interval", type=int, default=1, help="number of training epochs"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--max_ckpt_save_num",
        type=int,
        default=30,
        help="max number of saved checkpoint",
    )
    parser.add_argument(
        "--merge_all_iters_to_one_epoch", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )

    parser.add_argument(
        "--max_waiting_mins", type=int, default=0, help="max waiting minutes"
    )
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    parser.add_argument("--save_to_file", action="store_true", default=False, help="")

    parser.add_argument("--profile", action="store_true", default=False, help="")


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(
        args.cfg_file.split("/")[1:-1]
    )  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def save_pp_score(
    log_id: str,
    save_path: Path,
    dataset_dir: Path,
    split: str = "train",
    max_win: int = 30,
    win_inte: int = 5,
    max_neighbor_dist: float = 0.3,
) -> bool:
    """
    Save point cloud persistence score for Argoverse dataset.

    Args:
        log_id: Argoverse log identifier
        save_path: Root directory where pp scores will be saved
        dataset_dir: Path to the Argoverse dataset directory
        split: Dataset split (train/val/test)
        max_win: Maximum window size for temporal context
        win_inte: Window interval for sampling frames
        max_neighbor_dist: Maximum neighbor distance for pp score computation

    Returns:
        bool: True if successful
    """
    # Create output directory
    file_path = save_path / log_id / "ppscore"
    file_path.mkdir(parents=True, exist_ok=True)

    # Construct paths for this log
    log_dir = dataset_dir / split / log_id
    sensor_dir = log_dir / "sensors"
    lidar_dir = sensor_dir / "lidar"

    # Get all lidar files and sort by timestamp
    lidar_files = list(lidar_dir.glob("*.feather"))
    lidar_files.sort(key=lambda x: int(x.stem))  # Sort by timestamp

    if not lidar_files:
        raise FileNotFoundError(f"No lidar files found in {lidar_dir}")

    # Load city SE3 ego transformations for this log
    timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_dir)

    # Create timestamp to index mapping
    timestamps = [int(f.stem) for f in lidar_files]
    timestamp_to_idx = {ts: idx for idx, ts in enumerate(timestamps)}

    lidar_output_dir = save_path / log_id

    for i, lidar_file in enumerate(tqdm(lidar_files, desc="Extracting LiDAR")):
        info_path = str(i).zfill(4) + ".npy"
        lidar_path = lidar_output_dir / info_path

        if lidar_path.exists():
            # print(f"Skipping saving LiDAR {lidar_path.name}")
            continue

        sweep = Sweep.from_feather(lidar_feather_path=lidar_file)
        # Extract xyz coordinates (equivalent to [:, 0:3])
        lidar_points = np.column_stack(
            [
                sweep.xyz[:, 0],  # x
                sweep.xyz[:, 1],  # y
                sweep.xyz[:, 2],  # z
            ]
        )

        np.save(lidar_path, lidar_points)

    win_size = 1

    for i, lidar_file in enumerate(
        tqdm(lidar_files, desc="PPScore: Iterating over lidar_output_dir files")
    ):
        current_timestamp = int(lidar_file.stem)
        # Save the result
        output_file = file_path / f"{str(i).zfill(4)}.npy"

        if output_file.exists():
            continue

        # Get current pose (inverse for transformation)
        if current_timestamp not in timestamp_city_SE3_ego_dict:
            print(f"Warning: No pose found for timestamp {current_timestamp}")
            continue

        city_SE3_ego_i = timestamp_city_SE3_ego_dict[current_timestamp]
        ego_SE3_city_i = city_SE3_ego_i.inverse()  # Equivalent to np.linalg.inv(pose)

        all_traversals = []
        cur_points = None
        max_tra = max_win

        # Sample frames within the temporal window
        for j in range(i - max_tra, i + max_tra, win_inte):
            if j < 0 or j >= len(lidar_files):
                continue

            this_tra = []

            # Process frames within the window size
            for k in range(j, j + win_size):
                if k < 0 or k >= len(lidar_files):
                    continue

                k_timestamp = timestamps[k]
                k_lidar_file = lidar_files[k]

                # Load lidar data
                if not k_lidar_file.exists():
                    continue

                try:
                    sweep = Sweep.from_feather(lidar_feather_path=k_lidar_file)
                    # Extract xyz coordinates (equivalent to [:, 0:3])
                    lidar_points = np.column_stack(
                        [
                            sweep.xyz[:, 0],  # x
                            sweep.xyz[:, 1],  # y
                            sweep.xyz[:, 2],  # z
                        ]
                    )

                    if k == i:
                        cur_points = lidar_points

                    # Get pose for frame k
                    if k_timestamp not in timestamp_city_SE3_ego_dict:
                        continue

                    city_SE3_ego_k = timestamp_city_SE3_ego_dict[k_timestamp]

                    # Transform points using SE3 (equivalent to points_rigid_transform(lidar_points, pose_k))
                    lidar_points_transformed = city_SE3_ego_k.transform_point_cloud(
                        lidar_points
                    )
                    this_tra.append(lidar_points_transformed)

                except Exception as e:
                    print(f"Warning: Failed to load {k_lidar_file}: {e}")
                    continue

            if len(this_tra) > 0:
                this_tra = np.concatenate(this_tra)
                # Transform to current frame (equivalent to points_rigid_transform(this_tra, pose_i))
                this_tra_current_frame = ego_SE3_city_i.transform_point_cloud(this_tra)
                all_traversals.append(this_tra_current_frame)

        if cur_points is None:
            print(f"Warning: No current points found for frame {i}")
            continue

        # Compute PP score
        H = compute_ppscore(
            cur_points, all_traversals, max_neighbor_dist=max_neighbor_dist
        )

        np.save(output_file, np.array(H).astype(np.float16))

    return True


def count_neighbors(ptc, trees, max_neighbor_dist=0.3):
    neighbor_count = {}
    for seq in trees.keys():
        neighbor_count[seq] = trees[seq].query_ball_point(
            ptc[:, :3], r=max_neighbor_dist, return_length=True
        )
    return np.stack(list(neighbor_count.values())).T


def compute_ephe_score(count):
    N = count.shape[1]
    P = count / (np.expand_dims(count.sum(axis=1), -1) + 1e-8)
    H = (-P * np.log(P + 1e-8)).sum(axis=1) / np.log(N)

    return H


def compute_ppscore(cur_frame, neighbor_traversals=None, max_neighbor_dist=0.3):

    trees = {}

    for seq_id, points in enumerate(neighbor_traversals):
        trees[seq_id] = cKDTree(points)

    count = count_neighbors(cur_frame, trees, max_neighbor_dist)

    H = compute_ephe_score(count)

    return H


def points_rigid_transform(cloud, pose):
    if cloud.shape[0] == 0:
        return cloud
    mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
    pose_mat = np.mat(pose)
    mat[:, 0:3] = cloud[:, 0:3]
    mat = np.mat(mat)
    transformed_mat = pose_mat * mat.T
    T = np.array(transformed_mat.T, dtype=np.float32)
    return T[:, 0:3]


def save_infos(
    log_id: str, save_path: Path, dataset_dir: Path, split: str = "train"
) -> bool:
    """
    Save pose information for Argoverse dataset in the format expected by the pipeline.

    Args:
        log_id: Argoverse log identifier
        save_path: Root directory where infos will be saved
        dataset_dir: Path to the Argoverse dataset directory
        split: Dataset split (train/val/test)

    Returns:
        bool: True if successful
    """
    # Create output directory
    output_dir = save_path / log_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output pickle file path
    pkl_path = output_dir / f"{log_id}.pkl"

    # Construct paths for this log
    log_dir = dataset_dir / split / log_id
    sensor_dir = log_dir / "sensors"
    lidar_dir = sensor_dir / "lidar"

    # Get all lidar files and sort by timestamp
    lidar_files = list(lidar_dir.glob("*.feather"))
    lidar_files.sort(key=lambda x: int(x.stem))  # Sort by timestamp

    if not lidar_files:
        raise FileNotFoundError(f"No lidar files found in {lidar_dir}")

    # Load city SE3 ego transformations for this log
    timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_dir)

    # Create infos list
    infos = []

    for i, lidar_file in enumerate(lidar_files):
        timestamp_ns = int(lidar_file.stem)

        # Get pose for this timestamp
        if timestamp_ns not in timestamp_city_SE3_ego_dict:
            print(f"Warning: No pose found for timestamp {timestamp_ns}")
            continue

        city_SE3_ego = timestamp_city_SE3_ego_dict[timestamp_ns]

        # Convert SE3 to 4x4 transformation matrix (numpy array)
        # This is equivalent to the "pose" field in the original code
        pose_matrix = city_SE3_ego.transform_matrix

        # Create info dictionary for this frame
        info = {
            "pose": pose_matrix,
            "timestamp_ns": timestamp_ns,
            "frame_idx": i,
            "log_id": log_id,
            # Add other fields as needed by your pipeline
        }

        infos.append(info)

    # Save infos to pickle file
    with open(pkl_path, "wb") as f:
        pkl.dump(infos, f)

    print(f"Saved {len(infos)} frame infos to {pkl_path}")
    return True


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


def plot_lane_segments(
    ax: matplotlib.axes.Axes,
    lane_segments: list[LaneSegment],
    lane_color: np.ndarray = np.array([0.2, 0.2, 0.2]),
) -> None:
    """
    Args:
        ax:
        lane_segments:
    """
    for ls in lane_segments:
        pts_city = ls.polygon_boundary
        ALPHA = 1.0  # 0.1
        vector_plotting_utils.plot_polygon_patch_mpl(
            polygon_pts=pts_city, ax=ax, color=lane_color, alpha=ALPHA, zorder=1
        )

        for bound_type, bound_city in zip(
            [ls.left_mark_type, ls.right_mark_type],
            [ls.left_lane_boundary, ls.right_lane_boundary],
        ):
            if "YELLOW" in bound_type:
                mark_color = "y"
            elif "WHITE" in bound_type:
                mark_color = "w"
            else:
                mark_color = "grey"  # "b" lane_color #

            LOOSELY_DASHED = (0, (5, 10))

            if "DASHED" in bound_type:
                linestyle = LOOSELY_DASHED
            else:
                linestyle = "solid"

            if "DOUBLE" in bound_type:
                left, right = polyline_utils.get_double_polylines(
                    polyline=bound_city.xyz[:, :2], width_scaling_factor=0.1
                )
                ax.plot(
                    left[:, 0],
                    left[:, 1],
                    mark_color,
                    alpha=ALPHA,
                    linestyle=linestyle,
                    zorder=2,
                )
                ax.plot(
                    right[:, 0],
                    right[:, 1],
                    mark_color,
                    alpha=ALPHA,
                    linestyle=linestyle,
                    zorder=2,
                )
            else:
                ax.plot(
                    bound_city.xyz[:, 0],
                    bound_city.xyz[:, 1],
                    mark_color,
                    alpha=ALPHA,
                    linestyle=linestyle,
                    zorder=2,
                )


def load_and_plot_objects(
    output_info_path,
    use_first_frame=True,
    plot_trajectories=False,
    figsize=(12, 10),
    save_path=None,
    gts_dataframe=None,
    ref_timestamp_ns=None,
    log_id: str = "",
    score_thresh=-1,
):
    """
    Load object detection data and plot BEV with bounding boxes and trajectories.
    Can also overlay ground truth data from a dataframe.

    Args:
        output_info_path (str): Path to the pickle file containing object info
        use_first_frame (bool): If True, use first frame as reference. If False, use frame with most objects
        plot_trajectories (bool): Whether to plot object trajectories
        figsize (tuple): Figure size for the plot
        save_path (str): Optional path to save the plot
        gts_dataframe (pd.DataFrame, optional): Ground truth dataframe in Argoverse 2 format
        ref_timestamp_ns (int, optional): Specific timestamp to use for GT filtering

    Returns:
        dict: Loaded object information
    """

    # Load the pickle file
    if not os.path.exists(output_info_path):
        raise FileNotFoundError(f"Output info path not found: {output_info_path}")

    with open(output_info_path, "rb") as f:
        outline_infos = pkl.load(f)

    print(f"Loaded {len(outline_infos)} frames of data")

    # Extract object trajectories across all frames
    object_trajectories = defaultdict(list)
    frame_object_counts = []

    for frame_idx, frame_info in enumerate(outline_infos):
        outline_boxes = frame_info.get("outline_box", [])
        outline_ids = frame_info.get("outline_ids", [])
        outline_cls = frame_info.get("outline_cls", [])
        outline_scores = frame_info.get("outline_score", [])
        pose = frame_info.get("pose", np.eye(4))

        frame_object_counts.append(len([x for x in outline_scores if x > 0]))

        for box_idx, (box, obj_id, cls, score) in enumerate(
            zip(outline_boxes, outline_ids, outline_cls, outline_scores)
        ):
            if (
                len(box) >= 7 and score > score_thresh
            ):  # Ensure we have [x, y, z, length, width, height, yaw]
                # Transform to global coordinates if needed
                center_global = box[:3]  # Assuming already in global coordinates

                object_trajectories[obj_id].append(
                    {
                        "frame_idx": frame_idx,
                        "center": center_global[:2],  # x, y only for BEV
                        "box": box,
                        "class": cls,
                        "score": score,
                        "pose": pose,
                    }
                )

    # Choose reference frame
    if use_first_frame or len(frame_object_counts) == 0:
        ref_frame_idx = 0
    else:
        ref_frame_idx = np.argmax(frame_object_counts)

    print(
        f"Using frame {ref_frame_idx} as reference (contains {frame_object_counts[ref_frame_idx]} objects)"
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Load and plot lidar points if available
    try:
        lidar_path = (
            Path(output_info_path).parent / f"{str(ref_frame_idx).zfill(4)}.npy"
        )
        if lidar_path.exists():
            lidar_points = np.load(lidar_path)[:, 0:3]
            ax.scatter(
                lidar_points[:, 0],
                lidar_points[:, 1],
                s=1,
                c="blue",
                label="Lidar Points",
                alpha=0.5,
            )
            print(f"Loaded lidar points from {lidar_path}")
        else:
            print(f"Lidar file not found: {lidar_path}")
    except Exception as e:
        print(f"Could not load lidar points: {e}")

    # Color map for different classes
    class_colors = plt.cm.Set3(np.linspace(0, 1, 12))
    class_to_color = {}

    ref_frame_info = outline_infos[ref_frame_idx]
    # pprint(ref_frame_info)
    outline_boxes = ref_frame_info.get("outline_box", [])
    outline_ids = ref_frame_info.get("outline_ids", [])
    outline_cls = ref_frame_info.get("outline_cls", [])
    outline_scores = ref_frame_info.get(
        "outline_score", [score_thresh + 0.1 for x in outline_boxes]
    )

    print("outline_boxes", len(outline_boxes))
    print("outline_ids", len(outline_ids))
    print("outline_cls", len(outline_cls))
    print("outline_scores", len(outline_scores))
    print("outline_scores", outline_scores)

    print("outline_boxes", outline_boxes)

    outline_boxes_filtered = [
        x for x, y in zip(outline_boxes, outline_scores) if y > score_thresh
    ]

    # Plot ground truth boxes if provided
    if gts_dataframe is not None:
        print("Plotting ground truth boxes...")

        # Get reference frame info for timestamp
        ref_timestamp = ref_frame_info.get(
            "timestamp", ref_frame_info.get("timestamp_ns", ref_timestamp_ns)
        )

        ref_frame_info["gt_ious"] = [0.0 for x in outline_boxes_filtered]

        gts_timestamps = gts_dataframe["timestamp_ns"].values

        print("gts_timestamps", gts_timestamps.min(), gts_timestamps.max())
        print(
            "ref_timestamp",
            ref_timestamp,
            gts_timestamps.min() <= ref_timestamp <= gts_timestamps.max(),
        )
        print("type", type(ref_timestamp))

        if ref_timestamp is not None:
            gt_frame = gts_dataframe.loc[[(log_id, int(ref_timestamp))]]

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

            pred_lidar_boxes = torch.tensor(outline_boxes_filtered, dtype=torch.float32)

            print("gt_lidar_boxes", gt_lidar_boxes.shape)
            print("pred_lidar_boxes", pred_lidar_boxes.shape)

            if len(gt_lidar_boxes) > 0:
                ious = rotate_iou_cpu_eval(gt_lidar_boxes, pred_lidar_boxes).reshape(
                    gt_lidar_boxes.shape[0], pred_lidar_boxes.shape[0], 2
                )
                ious = ious[:, :, 0]
            else:
                ious = torch.zeros((0, len(pred_lidar_boxes)), dtype=torch.float32)

            # dists = cdist(gt_lidar_boxes[:, :3].to(dtype=torch.float32), pred_lidar_boxes[:, :3]).numpy()
            print("ious", ious.shape)
            # print("dists", dists.shape)

            # pred_assignments = ious.argmax(axis=1)
            # assignment_ious = ious[pred_assignments, torch.arange(len(pred_assignments))]

            # assert ious.max(axis=0) == assignment_ious, f"{ious.max(axis=0)} {assignment_ious=}"

            pred_ious = ious.max(axis=0)

            print(
                f"TPS/FPS {pred_ious.shape}",
                (pred_ious > 0.3).sum(),
                (pred_ious <= 0.3).sum(),
            )

            print("ious", ious.shape)
            if np.prod(ious.shape) > 0: 
                best_ious = ious.max(axis=1)
                best_ious_idx = ious.argmax(axis=1)
            else:
                best_ious = None

            # best_dists = dists[np.arange(len(dists)), best_ious_idx]

            print("best_ious", best_ious)
            # print("best_dists", best_dists)

            # if 'timestamp_ns' in gts_dataframe.columns:
            #     gt_frame = gts_dataframe[gts_dataframe['timestamp_ns'] == int(ref_timestamp)]
            # else:
            #     # If dataframe has index with timestamp_ns
            #     if isinstance(gts_dataframe.index, pd.MultiIndex) and 'timestamp_ns' in gts_dataframe.index.names:
            #         gt_frame = gts_dataframe.xs(int(ref_timestamp), level='timestamp_ns', drop_level=False)
            #     else:
            #         gt_frame = gts_dataframe  # Use all if no timestamp filtering possible

            print(
                f"Found {len(gt_frame)} ground truth objects for timestamp {ref_timestamp}"
            )

            # Plot GT boxes
            for idx, (_, gt_row) in enumerate(gt_frame.iterrows()):
                try:
                    
                    iou = best_ious[idx] if best_ious is not None else 0.0
                    # Extract box parameters from Argoverse format
                    center_xy = [gt_row["tx_m"], gt_row["ty_m"]]
                    length = gt_row["length_m"]
                    width = gt_row["width_m"]

                    # Convert quaternion to yaw
                    qw, qx, qy, qz = (
                        gt_row["qw"],
                        gt_row["qx"],
                        gt_row["qy"],
                        gt_row["qz"],
                    )
                    yaw = quat_to_yaw(qw, qx, qy, qz)

                    # Get category for coloring
                    category = gt_row.get("category", "UNKNOWN")

                    # Get rotated box corners
                    corners = get_rotated_box(center_xy, length, width, yaw)

                    # Create polygon patch for GT (different style)
                    gt_polygon = patches.Polygon(
                        corners,
                        linewidth=3,
                        edgecolor="red",
                        facecolor="none",
                        alpha=0.8,
                        linestyle="-",
                        label="Ground Truth" if _ == 0 else "",
                    )
                    ax.add_patch(gt_polygon)

                    # Add GT label
                    # ax.text(center_xy[0], center_xy[1] + length/2 + 1, f'GT: {category}',
                    #        ha='center', va='bottom', fontsize=7, fontweight='bold',
                    #        bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7, edgecolor='white'),
                    #        color='white')

                    ax.text(
                        center_xy[0],
                        center_xy[1] + length / 2 + 1,
                        f"{iou:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="red",
                            alpha=0.7,
                            edgecolor="white",
                        ),
                        color="white",
                    )

                except Exception as e:
                    print(f"Error plotting GT box: {e}")
                    continue
        else:
            print("Warning: Could not determine timestamp for GT filtering")

    # Plot trajectories first (so they appear behind boxes)
    if plot_trajectories:
        for obj_id, trajectory in object_trajectories.items():
            if len(trajectory) > 1:
                # Filter trajectory points with score > 0
                valid_points = [
                    point for point in trajectory if point["score"] > score_thresh
                ]
                if len(valid_points) > 1:
                    trajectory_points = np.array(
                        [point["center"] for point in valid_points]
                    )
                    ax.plot(
                        trajectory_points[:, 0],
                        trajectory_points[:, 1],
                        alpha=0.6,
                        linewidth=1,
                        color="gray",
                        linestyle="--",
                    )

                    # Add arrows to show direction
                    for i in range(
                        0,
                        len(trajectory_points) - 1,
                        max(1, len(trajectory_points) // 5),
                    ):
                        if i + 1 < len(trajectory_points):
                            dx = trajectory_points[i + 1, 0] - trajectory_points[i, 0]
                            dy = trajectory_points[i + 1, 1] - trajectory_points[i, 1]
                            ax.arrow(
                                trajectory_points[i, 0],
                                trajectory_points[i, 1],
                                dx * 0.5,
                                dy * 0.5,
                                head_width=0.5,
                                head_length=0.3,
                                fc="gray",
                                ec="gray",
                                alpha=0.5,
                            )

    # Plot objects from reference frame

    # print(
    #     "outline_scores",
    #     np.array(outline_scores).shape,
    #     np.min(outline_scores),
    #     np.median(outline_scores),
    #     np.max(outline_scores),
    # )
    # print("scores > 0:", np.array(outline_scores)[np.array(outline_scores) > 0])

    for box, obj_id, cls, score in zip(
        outline_boxes, outline_ids, outline_cls, outline_scores
    ):
        if len(box) >= 7 and score > score_thresh:
            center_xy = box[:2]
            length = box[3]
            width = box[4]
            yaw = box[6]

            # Get color for this class
            if cls not in class_to_color:
                class_to_color[cls] = class_colors[
                    len(class_to_color) % len(class_colors)
                ]

            # Get rotated box corners
            corners = get_rotated_box(center_xy, length, width, yaw)

            # Create polygon patch for predictions
            polygon = patches.Polygon(
                corners,
                linewidth=1.5,
                edgecolor="black",
                facecolor=class_to_color[cls],
                alpha=1.0,
                label=(
                    f"Pred: {cls}"
                    if cls
                    not in [p.get_label().replace("Pred: ", "") for p in ax.patches]
                    else ""
                ),
            )
            ax.add_patch(polygon)

            # Commented out text labels as per user's modification
            # ax.text(center_xy[0], center_xy[1], f'{cls}\nID:{obj_id}\n{score:.2f}',
            #        ha='center', va='center', fontsize=8, fontweight='bold',
            #        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Set equal aspect ratio and labels
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)

    # Count valid objects (score > 0)
    valid_objects = len([s for s in outline_scores if s > score_thresh])
    title = f"Bird's Eye View - Log {log_id} Frame {ref_frame_idx}\nPredictions: {valid_objects}, Total Frames: {len(outline_infos)}"
    if gts_dataframe is not None:
        gt_count = len(gt_frame) if "gt_frame" in locals() else 0
        title += f", GT Objects: {gt_count}"

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Create legend for classes and GT
    legend_elements = []
    if class_to_color:
        legend_elements.extend(
            [
                patches.Patch(facecolor=color, edgecolor="black", label=f"Pred: {cls}")
                for cls, color in class_to_color.items()
            ]
        )

    if gts_dataframe is not None:
        legend_elements.append(
            patches.Patch(
                facecolor="none",
                edgecolor="red",
                linewidth=3,
                linestyle="-",
                label="Ground Truth (Best Pred IoU)",
            )
        )

    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    # Save wide version
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Now zoom in
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    # Remove artists outside the new view
    for artist in plt.gca().texts:
        if not (
            -50 <= artist.get_position()[0] <= 50
            and -50 <= artist.get_position()[1] <= 50
        ):
            artist.remove()

    plt.savefig(save_path.replace(".png", "_tight.png"), dpi=300, bbox_inches="tight")

    # Print summary statistics
    valid_objects = len([s for s in outline_scores if s > score_thresh])
    print("\nSummary Statistics:")
    print(f"Total frames: {len(outline_infos)}")
    print(f"Total unique objects: {len(object_trajectories)}")
    print(f"Valid objects in reference frame (score > 0): {valid_objects}")
    print(f"Total objects in reference frame: {len(outline_boxes)}")
    print(f"Classes present: {list(class_to_color.keys())}")

    if gts_dataframe is not None:
        gt_count = len(gt_frame) if "gt_frame" in locals() else 0
        print(f"Ground truth objects in reference frame: {gt_count}")
        if gt_count > 0:
            gt_categories = (
                gt_frame["category"].unique() if "gt_frame" in locals() else []
            )
            print(f"GT categories: {list(gt_categories)}")

    return outline_infos


def load_and_plot_alpha_shapes(
    output_info_path: str,
    use_first_frame: bool = True,
    ref_frame_idx: int = None,
    plot_trajectories: bool = True,
    figsize: Tuple = (12, 10),
    save_path: Optional[str] = None,
    gts_dataframe=None,
    ref_timestamp_ns: Optional[int] = None,
    log_id: str = "",
    score_thresh: float = -1,
) -> Dict:
    """
    Load and plot alpha shapes in BEV.

    Args:
        output_info_path: Path to the pickle file containing alpha shape info
        use_first_frame: If True, use first frame as reference
        plot_trajectories: Whether to plot object trajectories
        figsize: Figure size for the plot
        save_path: Optional path to save the plot
        gts_dataframe: Ground truth dataframe in Argoverse 2 format
        ref_timestamp_ns: Specific timestamp for GT filtering
        log_id: Log ID for filtering
        score_thresh: Score threshold for filtering

    Returns:
        Loaded alpha shape information
    """
    # Load the pickle file
    if not os.path.exists(output_info_path):
        raise FileNotFoundError(f"Output info path not found: {output_info_path}")

    with open(output_info_path, "rb") as f:
        alpha_shape_infos = pkl.load(f)

    print(f"Loaded {len(alpha_shape_infos)} frames of alpha shape data")

    # Extract trajectories
    object_trajectories = defaultdict(list)
    object_frames = defaultdict(list)
    frame_object_counts = []

    log_dir = Path("/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val") / log_id
    raster_ground_height_layer = GroundHeightLayer.from_file(log_dir / "map")

    for frame_idx, frame_info in enumerate(alpha_shape_infos):
        alpha_shapes = frame_info.get("alpha_shapes", [])
        alpha_ids = frame_info.get("outline_ids", [])
        alpha_poses = frame_info.get("outline_poses", [])
        timestamp = frame_info['timestamp_ns']
        # tracked_objects = frame_info.get("tracked_objects")
        # print("frame_info", frame_info.keys())
        # print("frame_info", {k: len(v) for k, v in frame_info.items() if isinstance(v, list)})

        frame_object_counts.append(len(alpha_shapes))

        for alpha_id, alpha_pose in zip(alpha_ids, alpha_poses):
            # track = tracked_objects[alpha_id]

            # assert track.track_id == alpha_id, f"{track.track_id=} {alpha_id=}"

            # print(track.keys())
            # pprint(track)

            object_frames[alpha_id].append(timestamp)
            object_trajectories[alpha_id].append(alpha_pose[:3, 3])


            # if alpha_id not in object_trajectories:
            #     object_trajectories[alpha_id] = np.array([x[:3, 3] for x in track.optimized_poses])
            #     object_frames[alpha_id] = set(i for i in track.timestamps)

            # for object_pose in track['object_to_world_poses']:
            #     translation = object_pose[:3, 3]
            #     print("object_pose", translation)
            #     object_trajectories[alpha_id].append(
            #         {
            #             "frame_idx": frame_idx,
            #             "world_position": translation,
            #             # "alpha_shape": alpha_shape,
            #         }
            #     )                
        # for alpha_shape, obj_id in zip(alpha_shapes, alpha_ids):
        #     if alpha_shape is not None:
        #         # centroid = alpha_shape.get('centroid_2d', np.array([0, 0]))

        #         pprint(alpha_shape)

        #         centroid = alpha_shape.get("centroid_3d", None)
        #         object_trajectories[obj_id].append(
        #             {
        #                 "frame_idx": frame_idx,
        #                 "center": centroid,
        #                 "alpha_shape": alpha_shape,
        #             }
        #         )
            
    for k in object_trajectories.keys():
        object_trajectories[k] = np.array(object_trajectories[k])

    # Choose reference frame
    if ref_frame_idx is None or (use_first_frame or len(frame_object_counts) == 0):
        ref_frame_idx = 0
    elif ref_frame_idx is None:
        ref_frame_idx = np.argmax(frame_object_counts)

    # Plot alpha shapes from reference frame
    ref_frame_info = alpha_shape_infos[ref_frame_idx]
    alpha_shapes = ref_frame_info.get("alpha_shapes", [])
    alpha_ids = ref_frame_info.get("outline_ids", [])

    ref_frame_timestamp = ref_frame_info['timestamp_ns']

    # keep only objects that have the ref_frame_idx
    # valid_obj_ids = {
    #     obj_id
    #     for obj_id, frames in object_frames.items()
    #     if ref_frame_timestamp in frames
    # }

    valid_obj_ids = list(object_frames.keys())

    print(f"valid_obj_ids={len(valid_obj_ids)} {len(object_trajectories)=}")
    print(f"{len(alpha_ids)=}")

    # filter trajectories in place
    object_trajectories = {
        obj_id: object_trajectories[obj_id] for obj_id in valid_obj_ids
    }

    print(
        f"Using frame {ref_frame_idx} as reference (contains {frame_object_counts[ref_frame_idx]} alpha shapes)"
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)



    # pprint(alpha_shapes)

    pose = ref_frame_info.get("pose", np.eye(4))

    log_dir = Path(
        f"/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val/{log_id}/"
    )
    avm = ArgoverseStaticMap.from_map_dir(log_dir / "map", build_raster=True)

    # scaled to [0,1] for matplotlib.
    PURPLE_RGB = [201, 71, 245]
    PURPLE_RGB_MPL = np.array(PURPLE_RGB) / 255

    crosswalk_color = PURPLE_RGB_MPL
    CROSSWALK_ALPHA = 0.6
    for pc in avm.get_scenario_ped_crossings():
        vector_plotting_utils.plot_polygon_patch_mpl(
            polygon_pts=pc.polygon[:, :2],
            ax=ax,
            color=crosswalk_color,
            alpha=CROSSWALK_ALPHA,
            zorder=3,
        )

    plot_lane_segments(ax=ax, lane_segments=avm.get_scenario_lane_segments())

    # Load and plot lidar points if available
    try:
        lidar_path = (
            Path(output_info_path).parent / f"{str(ref_frame_idx).zfill(4)}.npy"
        )
        if lidar_path.exists():
            lidar_points = np.load(lidar_path)[:, 0:3]
            lidar_points = points_rigid_transform(lidar_points, pose)

            is_ground = raster_ground_height_layer.get_ground_points_boolean(
                lidar_points
            ).astype(bool)
            is_not_ground = ~is_ground

            lidar_points = lidar_points[is_not_ground]

            ax.scatter(
                lidar_points[:, 0],
                lidar_points[:, 1],
                s=1,
                c="blue",
                label="Lidar Points",
                alpha=0.5,
            )
            print(f"Loaded lidar points from {lidar_path}")
        else:
            print(f"Lidar file not found: {lidar_path}")
    except Exception as e:
        print(f"Could not load lidar points: {e}")

    # Color map for alpha shapes
    # colors = plt.cm.Set3(np.linspace(0, 1, 12))
    colors = plt.cm.get_cmap("tab20c").colors

    # Plot trajectories if requested
    if plot_trajectories:
        for obj_id, trajectory_points in object_trajectories.items():
            if len(trajectory_points) > 1:
                color = colors[obj_id % len(colors)]

                # print("trajectory_points", trajectory_points.min(axis=0), trajectory_points.max(axis=0))

                # trajectory_points = points_rigid_transform(
                #     trajectory_points.reshape(-1, 3), pose
                # )

                ax.plot(
                    trajectory_points[:, 0],
                    trajectory_points[:, 1],
                    alpha=0.6,
                    linewidth=1,
                    color=color,
                    linestyle="--",
                    zorder=100
                )

                # ax.plot(trajectory_points[:, 0], trajectory_points[:, 1],
                #        alpha=0.6, linewidth=1, color='gray', linestyle='--')

    print(f"Plotting {len(alpha_shapes)} alpha shapes {len(alpha_ids)} alpha ids")

    for alpha_shape, obj_id in zip(alpha_shapes, alpha_ids):
        if alpha_shape is not None:
            color = colors[obj_id % len(colors)]

            if "vertices_2d" in alpha_shape:
                vertices_2d = alpha_shape["vertices_2d"]
                z_center = (alpha_shape["z_min"] + alpha_shape["z_max"]) / 2
                points_3d = np.column_stack(
                    [vertices_2d, np.full(len(vertices_2d), z_center)]
                )

                points_3d = points_rigid_transform(points_3d.reshape(-1, 3), pose)

                vertices_2d = points_3d[:, :2]
            else:
                vertices_3d = alpha_shape["vertices_3d"]

                vertices_3d = points_rigid_transform(vertices_3d.reshape(-1, 3), pose)

                vertices_2d = vertices_3d[:, :2]

                hull = ConvexHull(vertices_2d)
                vertices_2d = vertices_2d[hull.vertices]

            # Create polygon patch for alpha shape
            polygon = patches.Polygon(
                vertices_2d,
                linewidth=2,
                edgecolor="black",
                facecolor=color,
                alpha=0.7,
                label=f"Alpha Shape {obj_id}",
            )
            ax.add_patch(polygon)

            # Add centroid marker
            # centroid = alpha_shape['centroid_2d']
            centroid = np.mean(vertices_2d, axis=0)
            ax.plot(centroid[0], centroid[1], "ko", markersize=4)

            # Add ID label
            ax.text(
                centroid[0],
                centroid[1],
                f"{obj_id}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    # Plot ground truth boxes if provided
    if gts_dataframe is not None:
        print("Plotting ground truth boxes...")

        # Get reference frame info for timestamp
        ref_timestamp = ref_frame_info.get(
            "timestamp", ref_frame_info.get("timestamp_ns", ref_timestamp_ns)
        )

        if ref_timestamp is not None:
            gt_frame = gts_dataframe.loc[[(log_id, int(ref_timestamp))]]

            print(
                f"Found {len(gt_frame)} ground truth objects for timestamp {ref_timestamp}"
            )

            # Plot GT boxes
            for idx, (_, gt_row) in enumerate(gt_frame.iterrows()):
                try:
                    # Extract box parameters from Argoverse format
                    # center_xyz = gt_row[['tx_m', 'ty_m', 'tz_m']].values

                    center_xy, yaw, length, width = apply_pose_to_box(gt_row, pose)

                    # Get category for coloring
                    category = gt_row.get("category", "UNKNOWN")

                    # Get rotated box corners
                    corners = get_rotated_box(center_xy, length, width, yaw)

                    # Create polygon patch for GT (different style)
                    gt_polygon = patches.Polygon(
                        corners,
                        linewidth=3,
                        edgecolor="red",
                        facecolor="none",
                        alpha=0.8,
                        linestyle="-",
                        label="Ground Truth" if idx == 0 else "",
                    )
                    ax.add_patch(gt_polygon)

                except Exception as e:
                    print(f"Error plotting GT box: {e}")
                    continue
        else:
            print("Warning: Could not determine timestamp for GT filtering")

    # Set equal aspect ratio and labels
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)

    title = f"Alpha Shapes BEV - Log {log_id} Frame {ref_frame_idx}\n{len(alpha_shapes)} Alpha Shapes, Total Frames: {len(alpha_shape_infos)}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Extract the center from pose translation
    center = pose[:2, 3]  # Get x, y translation from pose matrix

    # Now zoom in centered on the pose
    plt.xlim(center[0] - 50, center[0] + 50)
    plt.ylim(center[1] - 50, center[1] + 50)

    # Remove artists outside the new view
    for artist in plt.gca().texts:
        pos = artist.get_position()
        if not (
            center[0] - 50 <= pos[0] <= center[0] + 50
            and center[1] - 50 <= pos[1] <= center[1] + 50
        ):
            artist.remove()

    plt.savefig(
        str(save_path).replace(".png", "_tight.png"), dpi=300, bbox_inches="tight"
    )

    # Print summary
    print("\nAlpha Shape Summary:")
    print(f"Total frames: {len(alpha_shape_infos)}")
    print(f"Total unique objects: {len(object_trajectories)}")
    print(f"Alpha shapes in reference frame: {len(alpha_shapes)}")

    return alpha_shape_infos


def load_and_plot_alpha_shapes_camera(
    output_info_path: str,
    use_first_frame: bool = True,
    ref_frame_idx: int = None,
    plot_trajectories: bool = True,
    figsize: Tuple = (12, 10),
    save_path: Optional[str] = None,
    gts_dataframe=None,
    ref_timestamp_ns: Optional[int] = None,
    log_id: str = "",
    score_thresh: float = -1,
) -> Dict:
    """
    Load and plot alpha shapes in BEV.

    Args:
        output_info_path: Path to the pickle file containing alpha shape info
        use_first_frame: If True, use first frame as reference
        plot_trajectories: Whether to plot object trajectories
        figsize: Figure size for the plot
        save_path: Optional path to save the plot
        gts_dataframe: Ground truth dataframe in Argoverse 2 format
        ref_timestamp_ns: Specific timestamp for GT filtering
        log_id: Log ID for filtering
        score_thresh: Score threshold for filtering

    Returns:
        Loaded alpha shape information
    """
    # Load the pickle file
    if not os.path.exists(output_info_path):
        raise FileNotFoundError(f"Output info path not found: {output_info_path}")

    with open(output_info_path, "rb") as f:
        alpha_shape_infos = pkl.load(f)

    print(f"Loaded {len(alpha_shape_infos)} frames of alpha shape data")

    # Extract trajectories
    object_trajectories = defaultdict(list)
    object_frames = defaultdict(list)
    frame_object_counts = []

    log_dir = Path("/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val") / log_id
    raster_ground_height_layer = GroundHeightLayer.from_file(log_dir / "map")

    for frame_idx, frame_info in enumerate(alpha_shape_infos):
        alpha_shapes = frame_info.get("alpha_shapes", [])
        alpha_ids = frame_info.get("outline_ids", [])
        alpha_poses = frame_info.get("outline_poses", [])
        timestamp = frame_info['timestamp_ns']
        frame_object_counts.append(len(alpha_shapes))

        for alpha_id, alpha_pose in zip(alpha_ids, alpha_poses):

            object_frames[alpha_id].append(timestamp)
            object_trajectories[alpha_id].append(alpha_pose[:3, 3])

    for k in object_trajectories.keys():
        object_trajectories[k] = np.array(object_trajectories[k])

    # Choose reference frame
    if ref_frame_idx is None or (use_first_frame or len(frame_object_counts) == 0):
        ref_frame_idx = 0
    elif ref_frame_idx is None:
        ref_frame_idx = np.argmax(frame_object_counts)

    # Plot alpha shapes from reference frame
    ref_frame_info = alpha_shape_infos[ref_frame_idx]
    alpha_shapes = ref_frame_info.get("alpha_shapes", [])
    alpha_ids = ref_frame_info.get("outline_ids", [])
    pose = ref_frame_info.get("pose", np.eye(4))

    sweep_timestamp_ns = ref_frame_info['timestamp_ns']

    valid_obj_ids = list(object_frames.keys())

    # filter trajectories in place
    object_trajectories = {
        obj_id: object_trajectories[obj_id] for obj_id in valid_obj_ids
    }

    print(
        f"Using frame {ref_frame_idx} as reference (contains {frame_object_counts[ref_frame_idx]} alpha shapes)"
    )

    print(f"Plotting {len(alpha_shapes)} alpha shapes {len(alpha_ids)} alpha ids")

    ring_cameras = [x.value for x in list(RingCameras)]
    # avm = ArgoverseStaticMap.from_map_dir(log_dir / "map", build_raster=True)

    camera_models = {
        cam_name: PinholeCamera.from_feather(log_dir=log_dir, cam_name=cam_name)
        for cam_name in ring_cameras
    }
    timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_dir)


    # Ggather all cameras and timestamps
    sensor_dir = log_dir / "sensors"
    cameras_dir = sensor_dir / "cameras"

    # TODO
    src_sensor_name = "lidar"
    synchronization_cache_path = (
        Path.home() / ".cache" / "av2" / "synchronization_cache.feather"
    )
    synchronization_cache = read_feather(synchronization_cache_path)
    # Finally, create a MultiIndex set the sync records index and sort it.
    synchronization_cache.set_index(
        keys=["split", "log_id", "sensor_name"], inplace=True
    )
    synchronization_cache.sort_index(inplace=True)

    src_timedelta_ns = pd.Timedelta(sweep_timestamp_ns)

    for camera_name in ring_cameras:
        camera_dir = cameras_dir / camera_name

        src_to_target_records = synchronization_cache.loc[
            ("val", log_id, src_sensor_name)
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
        cam_timestamp_ns_str = str(target_timestamp_ns.asm8.item())

        city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[target_timestamp_ns.asm8.item()]
        city_SE3_ego_lidar_t = timestamp_city_SE3_ego_dict[sweep_timestamp_ns]
        camera_model = camera_models[camera_name]


        image_path = camera_dir / f"{cam_timestamp_ns_str}.jpg"

        img_vis = cv2.imread(str(image_path))

        height, width = img_vis.shape[:2]
        instance_mask = np.zeros((height, width), dtype=np.int32)
        depth_buffer = np.full((height, width), np.inf)  # Z-buffer for occlusion handling
        rendered_image = np.zeros_like(img_vis)
        rendered_mask = np.zeros((height, width), dtype=bool)
        
        
        cmap = plt.get_cmap("tab20", len(alpha_shapes))
        
        for shape_id, alpha_shape in enumerate(alpha_shapes):
            vertices = alpha_shape['vertices_3d']
            
            mesh = trimesh.convex.convex_hull(vertices)

            (
                uv_points,
                points_cam,
                is_valid_points,
            ) = camera_model.project_ego_to_img_motion_compensated(
                mesh.vertices,
                city_SE3_ego_cam_t=city_SE3_ego_cam_t,
                city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
            )
            
            color = cmap(shape_id)
            color = tuple((np.array(color)[:3]*255).astype(int).tolist())
            
            uvs = uv_points[:, :2]
            depths = points_cam[:, 2]
            cam_mask = is_valid_points

            # Render each triangular face
            for face in mesh.faces:
                # Get the vertices of the triangle
                tri_verts_2d = uvs[face]  # (3, 2): Triangle in 2D space
                tri_depths = depths[face]  # (3,): Depth of the triangle vertices
                tri_mask = cam_mask[face]
                
                # Compute the average depth of the triangle
                avg_depth = np.mean(tri_depths)
                
                # if avg_depth < 0:
                if np.any(tri_depths < 0) or not np.all(tri_mask):
                    # print('tri_depths', tri_depths)
                    continue
                
                # Create a mask for the triangle
                tri_mask = np.zeros((height, width), dtype=np.uint8)
                tri_verts_2d = tri_verts_2d.astype(int)
                cv2.fillConvexPoly(tri_mask, tri_verts_2d, 1)
                
                # Find pixels where this triangle is visible (closer than current depth buffer)
                visible_pixels = (tri_mask > 0) & (avg_depth < depth_buffer)
                
                # Update depth buffer, instance mask, and rendered image for visible pixels
                depth_buffer[visible_pixels] = avg_depth
                instance_mask[visible_pixels] = shape_id
                rendered_image[visible_pixels] = color  # Example color for this alpha shape
                rendered_mask[visible_pixels] = True
                
        opacity = 0.7
        blended_image = cv2.addWeighted(rendered_image, opacity, img_vis, 1 - opacity, 0)
        result = img_vis.copy()
        result[rendered_mask] = blended_image[rendered_mask]

        cv2.imwrite(save_path.replace(".png", f"_{camera_name}.png"), result)
        cv2.imwrite(save_path.replace(".png", f"_{camera_name}_orig.png"), img_vis)


def analyze_object_data(output_info_path):
    """
    Analyze the object detection data and provide detailed statistics.

    Args:
        output_info_path (str): Path to the pickle file containing object info

    Returns:
        dict: Analysis results
    """

    with open(output_info_path, "rb") as f:
        outline_infos = pkl.load(f)

    analysis = {
        "total_frames": len(outline_infos),
        "objects_per_frame": [],
        "classes_distribution": defaultdict(int),
        "object_lifecycles": defaultdict(list),
        "average_scores": defaultdict(list),
    }

    for frame_idx, frame_info in enumerate(outline_infos):
        outline_boxes = frame_info.get("outline_box", [])
        outline_ids = frame_info.get("outline_ids", [])
        outline_cls = frame_info.get("outline_cls", [])
        outline_scores = frame_info.get("outline_score", [])

        analysis["objects_per_frame"].append(len(outline_boxes))

        for obj_id, cls, score in zip(outline_ids, outline_cls, outline_scores):
            analysis["classes_distribution"][cls] += 1
            analysis["object_lifecycles"][obj_id].append(frame_idx)
            analysis["average_scores"][cls].append(score)

    # Calculate averages
    for cls in analysis["average_scores"]:
        analysis["average_scores"][cls] = np.mean(analysis["average_scores"][cls])

    # Print analysis
    print("=== Object Detection Data Analysis ===")
    print(f"Total frames: {analysis['total_frames']}")
    print(f"Average objects per frame: {np.mean(analysis['objects_per_frame']):.1f}")
    print(f"Max objects in single frame: {max(analysis['objects_per_frame'])}")
    print(f"Frame with most objects: {np.argmax(analysis['objects_per_frame'])}")
    print("\nClass distribution:")
    for cls, count in analysis["classes_distribution"].items():
        avg_score = analysis["average_scores"][cls]
        print(f"  {cls}: {count} detections (avg score: {avg_score:.3f})")

    print("\nObject lifecycles:")
    for obj_id, frames in analysis["object_lifecycles"].items():
        print(
            f"  Object {obj_id}: appears in {len(frames)} frames "
            f"(frames {min(frames)}-{max(frames)})"
        )

    return analysis


def lidar_box_to_argo2(boxes):
    """Convert boxes from [x,y,z,length,width,height,yaw] to Argoverse format.

    Args:
        boxes (np.ndarray or torch.Tensor): Boxes in format [x,y,z,l,w,h,yaw]

    Returns:
        torch.Tensor: Boxes in Argoverse format [x,y,z,l,w,h,qw,qx,qy,qz]
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    elif not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)

    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)

    cnt_xyz = boxes[:, :3]  # x, y, z centers
    lwh = boxes[:, 3:6]  # length, width, height
    yaw = boxes[:, 6]  # yaw angle

    # Convert yaw to quaternion
    quat = yaw_to_quat(yaw)  # [qw, qx, qy, qz]

    # Combine: [x, y, z, length, width, height, qw, qx, qy, qz]
    argo_cuboid = torch.cat([cnt_xyz, lwh, quat], dim=1)
    return argo_cuboid


def convert_to_argoverse2(
    output_info_path: str,
    log_id: str,
    output_path: Optional[str] = None,
    category_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Convert object detection results to Argoverse 2 format.

    Args:
        output_info_path (str): Path to pickle file containing detection results
        log_id (str): Argoverse log ID for this sequence
        output_path (str, optional): Path to save the .feather file
        category_mapping (Dict[str, str], optional): Custom category mapping

    Returns:
        pd.DataFrame: Formatted results in Argoverse 2 format
    """

    # Load the detection results
    with open(output_info_path, "rb") as f:
        outline_infos = pkl.load(f)

    print(f"Loaded {len(outline_infos)} frames from {output_info_path}")

    # Use default category mapping if none provided
    if category_mapping is None:
        raise ValueError("category_mapping is None")

    serialized_dts_list = []

    print("Converting predictions to Argoverse 2 format...")

    for frame_idx, frame_info in enumerate(outline_infos):
        outline_boxes = frame_info.get("outline_box", [])
        outline_ids = frame_info.get("outline_ids", [])
        outline_cls = frame_info.get("outline_cls", [])
        outline_scores = frame_info.get("outline_score", [])

        # Get timestamp - adjust this based on your data structure
        timestamp_ns = frame_info.get(
            "timestamp", frame_info.get("timestamp_ns", frame_idx * 100000000)
        )  # fallback to frame-based timestamp

        if len(outline_boxes) == 0:
            continue

        # Convert boxes to numpy array if needed
        if isinstance(outline_boxes, list):
            boxes_array = np.array(outline_boxes)
        else:
            boxes_array = outline_boxes

        # Ensure we have the right format [x,y,z,l,w,h,yaw]
        if boxes_array.shape[1] < 7:
            print(
                f"Warning: Frame {frame_idx} boxes have {boxes_array.shape[1]} dimensions, expected 7"
            )
            continue

        # Convert boxes to Argoverse format
        argo_boxes = lidar_box_to_argo2(boxes_array)

        # Map categories
        mapped_categories = []
        for cls in outline_cls:
            cls_str = str(cls)
            if cls_str in category_mapping:
                mapped_categories.append(category_mapping[cls_str])
            else:
                print(f"Warning: Unknown category '{cls}', mapping to REGULAR_VEHICLE")
                mapped_categories.append(
                    SensorCompetitionCategories.REGULAR_VEHICLE.value
                )

        # Create DataFrame for this frame
        serialized_dts = pd.DataFrame(argo_boxes.numpy(), columns=list(LABEL_ATTR))

        serialized_dts["score"] = outline_scores
        serialized_dts["log_id"] = log_id
        serialized_dts["timestamp_ns"] = int(timestamp_ns)
        serialized_dts["category"] = mapped_categories

        # Add object tracking ID if available
        # if outline_ids:
        serialized_dts["track_uuid"] = [f"{log_id}_{obj_id}" for obj_id in outline_ids]

        serialized_dts_list.append(serialized_dts)

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{len(outline_infos)}")

    if not serialized_dts_list:
        print("Warning: No valid detections found!")
        return pd.DataFrame()

    # Concatenate all frames
    dts = pd.concat(serialized_dts_list, ignore_index=True)

    # Set index and sort
    # dts = (
    #     dts.set_index(["log_id", "timestamp_ns"], drop=False)
    #     .sort_index()
    # )

    # Sort by score (highest first) and reset index
    dts = (
        dts[dts["score"] > 0]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    print(f"Converted {len(dts)} total detections across {len(outline_infos)} frames")
    print(f"Categories found: {sorted(dts['category'].unique())}")

    # Save to feather file if path provided
    if output_path is not None:
        if not output_path.suffix == ".feather":
            output_path = output_path.with_suffix(".feather")

        # Set index back for saving
        dts.to_feather(output_path)
        print(f"Result saved to {output_path}")

    # dts = dts.set_index(["log_id", "timestamp_ns"]).sort_index()
    return dts


def validate_argoverse2_format(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame matches Argoverse 2 format requirements.

    Args:
        df (pd.DataFrame): DataFrame to validate

    Returns:
        bool: True if format is valid
    """
    df = df.copy().reset_index(drop=True)
    required_columns = set(
        list(LABEL_ATTR) + ["score", "log_id", "timestamp_ns", "category"]
    )
    actual_columns = set(df.columns)

    print("actual_columns", actual_columns)

    missing_columns = required_columns - actual_columns
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False

    # Check data types
    if not pd.api.types.is_integer_dtype(df["timestamp_ns"]):
        print("timestamp_ns should be integer type")
        return False

    # Check categories are valid
    valid_categories = set(cat.value for cat in SensorCompetitionCategories)
    invalid_categories = set(df["category"].unique()) - valid_categories
    if invalid_categories:
        print(f"Invalid categories found: {invalid_categories}")
        return False

    print("Argoverse 2 format validation passed!")
    return True


class FeatureExtractor:
    """Extract accurate heuristic features by re-running CSS and other computations."""

    def __init__(self, config):
        """Initialize with the same config used in C_PROTO."""
        self.config = config
        self.css = CSS(config.RefinerConfig.CSSConfig)

        # Initialize OutlineFitter for point processing
        self.outline_estimator = OutlineFitter(
            sensor_height=config.GeneratorConfig.sensor_height,
            ground_min_threshold=config.RefinerConfig.GroundMin,
            ground_min_distance=config.GeneratorConfig.ground_min_distance,
            cluster_dis=config.GeneratorConfig.cluster_dis,
            cluster_min_points=config.GeneratorConfig.cluster_min_points,
            discard_max_height=config.GeneratorConfig.discard_max_height,
            min_box_volume=config.GeneratorConfig.min_box_volume,
            min_box_height=config.GeneratorConfig.min_box_height,
            max_box_volume=config.GeneratorConfig.max_box_volume,
            max_box_len=config.GeneratorConfig.max_box_len,
        )

    def compute_css_components(self, points, box, name):
        """
        Compute individual CSS components.
        Returns: (distance_score, mlo_score, size_score, total_score)
        """
        predefined_size = self.css.predifined_size[name]

        # Distance score
        dis_dis = np.linalg.norm(box[0:3])
        if dis_dis > self.css.max_dis:
            dis_dis = self.css.max_dis
        dis_score = 1 - dis_dis / self.css.max_dis

        # MLO score
        mlo_score = hierarchical_occupancy_score(points, box, self.css.mlo_parts)

        # Size score
        new_box = copy.deepcopy(box)
        this_size_norm = new_box[3:6] / new_box[3:6].sum()
        this_temp_norm = np.array(predefined_size)
        this_temp_norm = this_temp_norm / this_temp_norm.sum()
        size_score = KL_entropy_score(this_size_norm, this_temp_norm)

        # Total score
        weights = np.array(self.css.weights) / np.sum(self.css.weights)
        final_score = (
            dis_score * weights[0] + mlo_score * weights[1] + size_score * weights[2]
        )

        return dis_score, mlo_score, size_score, final_score

    def extract_advanced_objectness_features(
        self,
        points: np.ndarray,
        ppscore: np.ndarray,
        box: np.ndarray,
        prev_boxes=None,
        timestamps=None,
    ):
        """Extract advanced objectness features for unsupervised 3D object detection."""
        x, y, z, l, w, h, yaw = box[:7]

        # Get points near box (larger radius for context)
        dist_to_box = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
        mask_near = dist_to_box < max(l, w) * 2.0  # Larger context
        points_near = points[mask_near]
        ppscore_near = ppscore[mask_near]

        features = {}

        if len(points_near) == 0:
            return self._get_empty_features()

        # Smooth points
        points_near, ppscore_near = smooth_points_and_ppscore(points_near, ppscore_near)

        # Define box regions
        z_min, z_max = z - h / 2, z + h / 2
        mask_height = (points_near[:, 2] > z_min + 0.2) & (points_near[:, 2] < z_max)
        points_in_box = points_near[mask_height]
        ppscore_in_box = ppscore_near[mask_height]

        # Remove ground points
        non_ground_points = (
            self.outline_estimator.remove_ground(points_in_box)
            if len(points_in_box) > 0
            else np.array([])
        )

        # ============ POINT CLOUD STRUCTURE FEATURES ============

        # 1. Point Density Gradients
        features.update(
            self._compute_density_gradients(
                points_near, points_in_box, x, y, z, l, w, h
            )
        )

        # 2. Local Surface Normal Consistency
        features.update(self._compute_surface_normals(non_ground_points))

        # 3. Planarity/Curvature Measures
        features.update(self._compute_planarity_curvature(non_ground_points))

        # 4. Point Distribution Uniformity
        features.update(
            self._compute_distribution_uniformity(non_ground_points, l, w, h)
        )

        # 5. Edge/Corner Detection
        features.update(self._compute_edge_features(non_ground_points, l, w, h))

        # ============ CONTEXTUAL OBJECTNESS FEATURES (Top 3) ============

        # 1. Ground Plane Interaction (how well object "sits" on ground)
        features.update(
            self._compute_ground_interaction(points_near, non_ground_points, z_min)
        )

        # 2. Isolation Score (separation from surrounding points)
        features.update(
            self._compute_isolation_score(points_near, points_in_box, x, y, z, l, w, h)
        )

        # 3. Multi-scale Clustering Consistency
        features.update(self._compute_multiscale_consistency(non_ground_points))

        # ============ TEMPORAL/MOTION FEATURES (Top 3) ============
        if prev_boxes is not None and timestamps is not None:
            # 1. Velocity Consistency
            features.update(
                self._compute_velocity_consistency(box, prev_boxes, timestamps)
            )

            # 2. Motion Coherence (rigid body assumption)
            features.update(self._compute_motion_coherence(box, prev_boxes, timestamps))

            # 3. Track Stability
            features.update(self._compute_track_stability(prev_boxes, timestamps))
        else:
            # Default values when no temporal data
            features.update(
                {
                    "velocity_consistency": 0.0,
                    "velocity_std": 0.0,
                    "motion_coherence": 0.0,
                    "track_stability": 0.0,
                    "track_length_frames": 1,
                }
            )

        # Add existing features
        features.update(
            self._compute_basic_features(
                points_in_box, ppscore_in_box, non_ground_points, l, w, h
            )
        )

        return features

    def _compute_density_gradients(self, points_near, points_in_box, x, y, z, l, w, h):
        """Compute point density gradients - objects should have higher internal density."""
        features = {}

        if len(points_in_box) == 0:
            return {
                "density_gradient": 0.0,
                "density_ratio": 0.0,
                "density_contrast": 0.0,
            }

        # Internal density
        box_volume = l * w * h
        internal_density = len(points_in_box) / box_volume

        # External density (ring around box)
        dist_to_center = np.sqrt(
            (points_near[:, 0] - x) ** 2 + (points_near[:, 1] - y) ** 2
        )
        box_radius = max(l, w) / 2
        mask_external = (dist_to_center > box_radius) & (
            dist_to_center < box_radius * 2
        )
        points_external = points_near[mask_external]

        if len(points_external) > 0:
            external_volume = np.pi * ((box_radius * 2) ** 2 - box_radius**2) * h
            external_density = len(points_external) / external_volume

            features["density_ratio"] = internal_density / (external_density + 1e-6)
            features["density_contrast"] = internal_density - external_density
        else:
            features["density_ratio"] = internal_density
            features["density_contrast"] = internal_density

        # Local density gradient within box
        if len(points_in_box) > 10:
            center_mask = (
                np.sqrt((points_in_box[:, 0] - x) ** 2 + (points_in_box[:, 1] - y) ** 2)
                < box_radius * 0.5
            )
            center_density = np.sum(center_mask) / (np.pi * (box_radius * 0.5) ** 2 * h)
            features["density_gradient"] = center_density / (internal_density + 1e-6)
        else:
            features["density_gradient"] = 1.0

        return features

    def _compute_surface_normals(self, points):
        """Compute surface normal consistency - objects have coherent surfaces."""
        features = {}

        if len(points) < 10:
            return {
                "normal_consistency": 0.0,
                "normal_variance": 1.0,
                "surface_coherence": 0.0,
            }

        # Use PCA on local neighborhoods to estimate normals
        nbrs = NearestNeighbors(n_neighbors=min(8, len(points))).fit(points)
        normals = []

        for i in range(len(points)):
            distances, indices = nbrs.kneighbors([points[i]])
            local_points = points[indices[0]]

            if len(local_points) >= 3:
                # PCA to find normal
                local_centered = local_points - np.mean(local_points, axis=0)
                _, _, V = np.linalg.svd(local_centered)
                normal = V[-1]  # Smallest eigenvector
                normals.append(normal)

        if len(normals) > 0:
            normals = np.array(normals)

            # Consistency: how aligned are the normals
            dot_products = np.abs(np.dot(normals, normals.T))
            features["normal_consistency"] = np.mean(dot_products)
            features["normal_variance"] = np.var(np.linalg.norm(normals, axis=1))

            # Surface coherence: variation in normal directions
            mean_normal = np.mean(normals, axis=0)
            mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-6)
            coherence = np.mean([np.abs(np.dot(n, mean_normal)) for n in normals])
            features["surface_coherence"] = coherence
        else:
            features["normal_consistency"] = 0.0
            features["normal_variance"] = 1.0
            features["surface_coherence"] = 0.0

        return features

    def _compute_planarity_curvature(self, points):
        """Compute planarity and curvature measures."""
        features = {}

        if len(points) < 4:
            return {"planarity": 0.0, "curvature": 0.0, "surface_variation": 1.0}

        # PCA on all points
        centered = points - np.mean(points, axis=0)
        _, s, _ = np.linalg.svd(centered)

        # Planarity: ratio of eigenvalues
        if len(s) >= 3:
            features["planarity"] = (s[1] - s[2]) / (s[0] + 1e-6)
            features["surface_variation"] = s[2] / (
                s[0] + 1e-6
            )  # How much variation in smallest direction
        else:
            features["planarity"] = 0.0
            features["surface_variation"] = 1.0

        # Local curvature estimation
        if len(points) > 10:
            nbrs = NearestNeighbors(n_neighbors=min(8, len(points))).fit(points)
            curvatures = []

            for i in range(min(50, len(points))):  # Sample for efficiency
                distances, indices = nbrs.kneighbors([points[i]])
                local_points = points[indices[0]]

                if len(local_points) >= 4:
                    local_centered = local_points - np.mean(local_points, axis=0)
                    _, local_s, _ = np.linalg.svd(local_centered)
                    if len(local_s) >= 3:
                        # Curvature as ratio of smallest to largest eigenvalue
                        curvatures.append(local_s[2] / (local_s[0] + 1e-6))

            features["curvature"] = np.mean(curvatures) if curvatures else 0.0
        else:
            features["curvature"] = 0.0

        return features

    def _compute_distribution_uniformity(self, points, l, w, h):
        """Compute how uniformly points are distributed (objects vs noise)."""
        features = {}

        if len(points) < 5:
            return {
                "distribution_uniformity": 0.0,
                "spatial_entropy": 0.0,
                "clustering_metric": 0.0,
            }

        # Spatial entropy - divide space into grid and compute entropy
        x_bins = max(3, int(l))
        y_bins = max(3, int(w))
        z_bins = max(2, int(h))

        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

        # Create 3D histogram
        hist, _ = np.histogramdd(
            points,
            bins=[x_bins, y_bins, z_bins],
            range=[[x_min, x_max], [y_min, y_max], [z_min, z_max]],
        )

        # Normalize and compute entropy
        hist_norm = hist.flatten()
        hist_norm = hist_norm[hist_norm > 0]  # Remove empty bins
        hist_norm = hist_norm / np.sum(hist_norm)

        features["spatial_entropy"] = entropy(hist_norm)

        # Distribution uniformity using nearest neighbor distances
        if len(points) > 1:
            nbrs = NearestNeighbors(n_neighbors=min(5, len(points))).fit(points)
            distances, _ = nbrs.kneighbors(points)
            mean_nn_dist = np.mean(distances[:, 1:])  # Exclude self (distance 0)
            std_nn_dist = np.std(distances[:, 1:])
            features["distribution_uniformity"] = 1.0 / (
                1.0 + std_nn_dist / (mean_nn_dist + 1e-6)
            )
        else:
            features["distribution_uniformity"] = 0.0

        # Clustering metric - how clustered vs spread out
        if len(points) > 2:
            distances = cdist(points, points)
            mean_dist = np.mean(distances[distances > 0])
            max_dist = np.max(distances)
            features["clustering_metric"] = mean_dist / (max_dist + 1e-6)
        else:
            features["clustering_metric"] = 0.0

        return features

    def _compute_edge_features(self, points, l, w, h):
        """Detect edges and corners - objects have distinct boundaries."""
        features = {}

        if len(points) < 10:
            return {"edge_density": 0.0, "corner_score": 0.0, "boundary_coherence": 0.0}

        # Simple edge detection using local variance
        nbrs = NearestNeighbors(n_neighbors=min(8, len(points))).fit(points)
        edge_scores = []

        for i in range(len(points)):
            distances, indices = nbrs.kneighbors([points[i]])
            local_points = points[indices[0]]

            # Variance in local neighborhood indicates edges
            local_var = np.mean(np.var(local_points, axis=0))
            edge_scores.append(local_var)

        features["edge_density"] = np.mean(edge_scores)

        # Corner detection - points with high local curvature
        corner_scores = []
        for i in range(min(len(points), 50)):  # Sample for efficiency
            distances, indices = nbrs.kneighbors([points[i]])
            local_points = points[indices[0]]

            if len(local_points) >= 4:
                # Fit plane and measure deviation
                centered = local_points - np.mean(local_points, axis=0)
                _, s, _ = np.linalg.svd(centered)
                if len(s) >= 3:
                    corner_scores.append(
                        s[2]
                    )  # Smallest eigenvalue indicates corner-ness

        features["corner_score"] = np.mean(corner_scores) if corner_scores else 0.0

        # Boundary coherence - how well-defined are the boundaries
        box_volume = l * w * h
        expected_boundary_points = (
            2 * (l * w + l * h + w * h) / (l * w * h) * len(points)
        )  # Rough estimate
        actual_high_variance = np.sum(
            np.array(edge_scores) > np.percentile(edge_scores, 75)
        )
        features["boundary_coherence"] = min(
            1.0, actual_high_variance / (expected_boundary_points + 1e-6)
        )

        return features

    def _compute_ground_interaction(self, points_near, non_ground_points, z_min):
        """Compute how well object interacts with ground plane."""
        features = {}

        if len(points_near) == 0:
            return {
                "ground_support": 0.0,
                "ground_contact_ratio": 0.0,
                "elevation_consistency": 0.0,
            }

        # Ground height estimation
        ground_height = np.percentile(points_near[:, 2], 5)

        # How much of object is supported by ground
        ground_tolerance = 0.3
        near_ground = points_near[
            np.abs(points_near[:, 2] - ground_height) < ground_tolerance
        ]
        features["ground_support"] = len(near_ground) / (len(points_near) + 1e-6)

        # Contact ratio - points at bottom of box near ground
        bottom_points = points_near[points_near[:, 2] < z_min + 0.5]
        ground_contact = bottom_points[
            np.abs(bottom_points[:, 2] - ground_height) < ground_tolerance
        ]
        features["ground_contact_ratio"] = len(ground_contact) / (
            len(bottom_points) + 1e-6
        )

        # Elevation consistency - how consistent is the object height above ground
        if len(non_ground_points) > 0:
            elevations = non_ground_points[:, 2] - ground_height
            features["elevation_consistency"] = 1.0 / (1.0 + np.std(elevations))
        else:
            features["elevation_consistency"] = 0.0

        return features

    def _compute_isolation_score(self, points_near, points_in_box, x, y, z, l, w, h):
        """Compute how isolated/separated the object is from surroundings."""
        features = {}

        if len(points_in_box) == 0:
            return {
                "isolation_score": 0.0,
                "separation_quality": 0.0,
                "background_contrast": 0.0,
            }

        # Points around the box (but not in it)
        dist_to_center = np.sqrt(
            (points_near[:, 0] - x) ** 2 + (points_near[:, 1] - y) ** 2
        )
        box_radius = max(l, w) / 2

        mask_around = (dist_to_center > box_radius) & (dist_to_center < box_radius * 2)
        points_around = points_near[mask_around]

        if len(points_around) > 0:
            # Minimum distance from box points to surrounding points
            distances = cdist(points_in_box, points_around)
            min_distances = np.min(distances, axis=1) if distances.size > 0 else [0]
            features["isolation_score"] = np.mean(min_distances)

            # Separation quality - how cleanly separated
            separation_threshold = max(l, w) * 0.2
            well_separated = np.sum(np.array(min_distances) > separation_threshold)
            features["separation_quality"] = well_separated / len(points_in_box)

            # Background contrast in density
            box_density = len(points_in_box) / (l * w * h)
            around_volume = np.pi * ((box_radius * 2) ** 2 - box_radius**2) * h
            around_density = len(points_around) / around_volume
            features["background_contrast"] = box_density / (around_density + 1e-6)
        else:
            features["isolation_score"] = max(
                l, w
            )  # High isolation if no surrounding points
            features["separation_quality"] = 1.0
            features["background_contrast"] = 100.0  # Very high contrast

        return features

    def _compute_multiscale_consistency(self, points):
        """Compute clustering consistency at multiple scales."""
        features = {}

        if len(points) < 10:
            return {"multiscale_consistency": 0.0, "scale_stability": 0.0}

        from sklearn.cluster import DBSCAN

        # Try different eps values (scales)
        scales = [0.1, 0.2, 0.5, 1.0, 2.0]
        n_clusters_at_scale = []

        for eps in scales:
            if len(points) > 0:
                clustering = DBSCAN(eps=eps, min_samples=3).fit(points)
                n_clusters = len(set(clustering.labels_)) - (
                    1 if -1 in clustering.labels_ else 0
                )
                n_clusters_at_scale.append(n_clusters)
            else:
                n_clusters_at_scale.append(0)

        # Consistency: should have similar number of clusters across scales for coherent objects
        if len(n_clusters_at_scale) > 1:
            features["multiscale_consistency"] = 1.0 / (
                1.0 + np.std(n_clusters_at_scale)
            )
            features["scale_stability"] = (
                1.0
                if np.max(n_clusters_at_scale) == np.min(n_clusters_at_scale)
                else 0.5
            )
        else:
            features["multiscale_consistency"] = 1.0
            features["scale_stability"] = 1.0

        return features

    def _compute_velocity_consistency(self, current_box, prev_boxes, timestamps):
        """Compute velocity consistency over time."""
        features = {}

        if len(prev_boxes) < 2:
            return {"velocity_consistency": 0.0, "velocity_std": 0.0}

        # Calculate velocities between consecutive frames
        velocities = []
        for i in range(1, len(prev_boxes)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt > 0:
                dx = prev_boxes[i][0] - prev_boxes[i - 1][0]
                dy = prev_boxes[i][1] - prev_boxes[i - 1][1]
                v = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(v)

        if len(velocities) > 1:
            features["velocity_consistency"] = 1.0 / (1.0 + np.std(velocities))
            features["velocity_std"] = np.std(velocities)
        else:
            features["velocity_consistency"] = 1.0
            features["velocity_std"] = 0.0

        return features

    def _compute_motion_coherence(self, current_box, prev_boxes, timestamps):
        """Compute motion coherence (rigid body assumption)."""
        features = {}

        if len(prev_boxes) < 3:
            return {"motion_coherence": 0.0}

        # Check if motion follows smooth trajectory
        positions = np.array([[box[0], box[1]] for box in prev_boxes])

        # Fit polynomial to trajectory
        if len(positions) >= 3:
            t = np.array(timestamps)
            try:
                # Fit 2nd order polynomial
                coeffs_x = np.polyfit(t, positions[:, 0], min(2, len(positions) - 1))
                coeffs_y = np.polyfit(t, positions[:, 1], min(2, len(positions) - 1))

                # Compute R-squared for fit quality
                x_pred = np.polyval(coeffs_x, t)
                y_pred = np.polyval(coeffs_y, t)

                ss_res_x = np.sum((positions[:, 0] - x_pred) ** 2)
                ss_tot_x = np.sum((positions[:, 0] - np.mean(positions[:, 0])) ** 2)
                r2_x = 1 - (ss_res_x / (ss_tot_x + 1e-6))

                ss_res_y = np.sum((positions[:, 1] - y_pred) ** 2)
                ss_tot_y = np.sum((positions[:, 1] - np.mean(positions[:, 1])) ** 2)
                r2_y = 1 - (ss_res_y / (ss_tot_y + 1e-6))

                features["motion_coherence"] = (r2_x + r2_y) / 2
            except:
                features["motion_coherence"] = 0.0
        else:
            features["motion_coherence"] = 0.0

        return features

    def _compute_track_stability(self, prev_boxes, timestamps):
        """Compute track stability metrics."""
        features = {}

        features["track_length_frames"] = len(prev_boxes)

        if len(prev_boxes) < 2:
            return {"track_stability": 1.0, "track_length_frames": len(prev_boxes)}

        # Size consistency over time
        sizes = np.array([[box[3], box[4], box[5]] for box in prev_boxes])  # l, w, h
        size_stds = np.std(sizes, axis=0)
        size_stability = 1.0 / (1.0 + np.mean(size_stds))

        features["track_stability"] = size_stability

        return features

    def _compute_basic_features(
        self, points_in_box, ppscore_in_box, non_ground_points, l, w, h
    ):
        """Compute basic features to maintain compatibility."""
        features = {}

        # PPS Score features
        features["ppscore_in_box_mean"] = (
            np.mean(ppscore_in_box) if len(ppscore_in_box) > 0 else 0
        )
        features["ppscore_in_box_std"] = (
            np.std(ppscore_in_box) if len(ppscore_in_box) > 0 else 0
        )
        features["ppscore_in_box_min"] = (
            np.min(ppscore_in_box) if len(ppscore_in_box) > 0 else 0
        )
        features["ppscore_in_box_max"] = (
            np.max(ppscore_in_box) if len(ppscore_in_box) > 0 else 0
        )

        # Point counts
        features["num_points_in_cluster"] = len(non_ground_points)
        features["points_above_ground"] = len(non_ground_points)

        # Densities
        box_volume = l * w * h
        features["point_density"] = (
            len(points_in_box) / box_volume if box_volume > 0 else 0
        )
        features["cluster_density"] = (
            len(non_ground_points) / box_volume if box_volume > 0 else 0
        )

        return features

    def _get_empty_features(self):
        """Return empty features when no points are available."""
        return {
            # Structure features
            "density_gradient": 0.0,
            "density_ratio": 0.0,
            "density_contrast": 0.0,
            "normal_consistency": 0.0,
            "normal_variance": 1.0,
            "surface_coherence": 0.0,
            "planarity": 0.0,
            "curvature": 0.0,
            "surface_variation": 1.0,
            "distribution_uniformity": 0.0,
            "spatial_entropy": 0.0,
            "clustering_metric": 0.0,
            "edge_density": 0.0,
            "corner_score": 0.0,
            "boundary_coherence": 0.0,
            # Contextual features
            "ground_support": 0.0,
            "ground_contact_ratio": 0.0,
            "elevation_consistency": 0.0,
            "isolation_score": 0.0,
            "separation_quality": 0.0,
            "background_contrast": 0.0,
            "multiscale_consistency": 0.0,
            "scale_stability": 0.0,
            # Temporal features
            "velocity_consistency": 0.0,
            "velocity_std": 0.0,
            "motion_coherence": 0.0,
            "track_stability": 0.0,
            "track_length_frames": 1,
            # Basic features
            "ppscore_in_box_mean": 0,
            "ppscore_in_box_std": 0,
            "ppscore_in_box_min": 0,
            "ppscore_in_box_max": 0,
            "num_points_in_cluster": 0,
            "points_above_ground": 0,
            "point_density": 0,
            "cluster_density": 0,
        }

    def extract_point_features(
        self, points: np.ndarray, ppscore: np.ndarray, box: np.ndarray
    ):
        """Extract point cloud features for a box."""
        x, y, z, l, w, h, yaw = box[:7]

        # Get points near box
        dist_to_box = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
        mask_near = dist_to_box < max(l, w)
        points_near = points[mask_near]
        ppscore_near = ppscore[mask_near]

        # print("points_near", mask_near.shape, mask_near.sum())

        # print("points centre", (points.max(axis=0) + points.min(axis=0))/2)
        # print("points centre", (box.max(axis=0) + box.min(axis=0))/2)

        features = {}

        if len(points_near) > 0:
            # Smooth points (from original code)
            # points_near = smooth_points(points_near)
            points_near, ppscore_near = smooth_points_and_ppscore(
                points_near, ppscore_near
            )

            # Height range
            z_min = z - h / 2
            z_max = z + h / 2

            # Points in box
            mask_height = (points_near[:, 2] > z_min + 0.2) & (
                points_near[:, 2] < z_max
            )
            points_in_box = points_near[mask_height]
            ppscore_in_box = ppscore_near[mask_height]

            # Remove ground points
            non_ground_points = (
                self.outline_estimator.remove_ground(points_in_box)
                if len(points_in_box) > 0
                else np.array([])
            )

            features["ppscore_in_box_mean"] = (
                np.mean(ppscore_in_box) if len(ppscore_in_box) > 0 else 0
            )
            features["ppscore_in_box_std"] = (
                np.std(ppscore_in_box) if len(ppscore_in_box) > 0 else 0
            )
            features["ppscore_in_box_min"] = (
                np.min(ppscore_in_box) if len(ppscore_in_box) > 0 else 0
            )
            features["ppscore_in_box_max"] = (
                np.max(ppscore_in_box) if len(ppscore_in_box) > 0 else 0
            )

            # Clustering to get main cluster
            if len(non_ground_points) > 10:
                clusters, labels = self.outline_estimator.clustering(non_ground_points)
                if len(clusters) > 0:
                    # Get largest cluster
                    max_cluster = max(clusters, key=len)
                    features["cluster_points"] = max_cluster
                    features["num_points_in_cluster"] = len(max_cluster)
                    features["num_clusters"] = len(clusters)
                else:
                    features["cluster_points"] = np.array([])
                    features["num_points_in_cluster"] = 0
                    features["num_clusters"] = 0
            else:
                features["cluster_points"] = non_ground_points
                features["num_points_in_cluster"] = len(non_ground_points)
                features["num_clusters"] = 1 if len(non_ground_points) > 0 else 0

            # Ground height estimation
            if len(points_near) > 0:
                ground_height = np.percentile(points_near[:, 2], 5)
            else:
                ground_height = z_min

            features["ground_height"] = ground_height
            features["height_above_ground"] = z_min - ground_height
            features["points_above_ground"] = len(non_ground_points)

            # Point density
            box_volume = l * w * h
            features["point_density"] = (
                len(points_in_box) / box_volume if box_volume > 0 else 0
            )
            features["cluster_density"] = (
                features["num_points_in_cluster"] / box_volume if box_volume > 0 else 0
            )

            # # Point distribution metrics
            # if len(points_in_box) > 0:
            #     features['points_std_x'] = np.std(points_in_box[:, 0])
            #     features['points_std_y'] = np.std(points_in_box[:, 1])
            #     features['points_std_z'] = np.std(points_in_box[:, 2])
            # else:
            #     features['points_std_x'] = 0
            #     features['points_std_y'] = 0
            #     features['points_std_z'] = 0

            # Point distribution metrics - percentiles in local box coordinates
            percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            dimensions = ["x", "y", "z"]

            if len(points_in_box) > 0:
                x, y, z, l, w, h, yaw = box[:7]

                # Transform points to local box coordinates
                # 1. Translate to box center
                points_centered = points_in_box[:, :3] - np.array([x, y, z])

                # 2. Rotate by -yaw to align with box axes
                cos_yaw = np.cos(-yaw)
                sin_yaw = np.sin(-yaw)
                rotation_matrix = np.array(
                    [[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]]
                )
                points_local = points_centered @ rotation_matrix.T

                # 3. Normalize by box dimensions to get relative positions [-0.5, 0.5]
                box_dims = np.array([l, w, h])
                points_normalized = points_local / box_dims

                # Compute percentiles for each dimension
                for dim_idx, dim_name in enumerate(dimensions):
                    for p in percentiles:
                        feature_name = f"points_percentile_{dim_name}_{p:.2f}"
                        features[feature_name] = np.quantile(
                            points_normalized[:, dim_idx], p
                        )

                # Additional distribution stats in local coordinates
                for dim_idx, dim_name in enumerate(dimensions):
                    features[f"points_std_{dim_name}"] = np.std(
                        points_normalized[:, dim_idx]
                    )
                    features[f"points_range_{dim_name}"] = np.ptp(
                        points_normalized[:, dim_idx]
                    )
                    features[f"points_span_{dim_name}"] = np.max(
                        points_normalized[:, dim_idx]
                    ) - np.min(points_normalized[:, dim_idx])
            else:
                # Handle empty case
                for dim_idx, dim_name in enumerate(dimensions):
                    for p in percentiles:
                        feature_name = f"points_percentile_{dim_name}_{p:.2f}"
                        features[feature_name] = 0
                    features[f"points_std_{dim_name}"] = 0
                    features[f"points_range_{dim_name}"] = 0
                    features[f"points_span_{dim_name}"] = 0
        else:
            features = {
                "cluster_points": np.array([]),
                "num_points_in_cluster": 0,
                "num_clusters": 0,
                "ground_height": z - h / 2,
                "height_above_ground": 0,
                "points_above_ground": 0,
                "point_density": 0,
                "cluster_density": 0,
                "points_std_x": 0,
                "points_std_y": 0,
                "points_std_z": 0,
                "ppscore_in_box_mean": 0,
                "ppscore_in_box_std": 0,
                "ppscore_in_box_min": 0,
                "ppscore_in_box_max": 0,
            }

        return features

    def compute_motion_features(self, outline_infos, obj_id, current_frame):
        """Compute motion features for an object."""
        # Collect all positions for this object
        positions = []
        frames = []
        scores = []

        for frame_idx, frame_info in enumerate(outline_infos):
            if "outline_ids" in frame_info and obj_id in frame_info["outline_ids"]:
                idx = list(frame_info["outline_ids"]).index(obj_id)
                box = frame_info["outline_box"][idx]
                pose = frame_info.get("pose", np.eye(4))

                # Transform to global coordinates
                global_pos = points_rigid_transform(
                    np.array([[box[0], box[1], box[2]]]), pose
                )
                positions.append(global_pos[0])
                frames.append(frame_idx)

                if "outline_score" in frame_info and idx < len(
                    frame_info["outline_score"]
                ):
                    scores.append(frame_info["outline_score"][idx])
                else:
                    scores.append(0)

        features = {}

        if len(positions) > 1:
            positions = np.array(positions)

            # Position variance
            mean_pos = np.mean(positions[:, :2], axis=0)
            distances = np.linalg.norm(positions[:, :2] - mean_pos, axis=1)
            features["position_std"] = np.std(distances)
            features["position_range"] = np.max(distances) - np.min(distances)

            # Velocity
            if len(frames) > 1:
                time_diffs = np.diff(frames)
                pos_diffs = np.diff(positions[:, :2], axis=0)
                velocities = np.linalg.norm(pos_diffs, axis=1) / (time_diffs + 1e-6)
                features["velocity_mean"] = np.mean(velocities)
                features["velocity_max"] = np.max(velocities)
                features["velocity_std"] = np.std(velocities)
            else:
                features["velocity_mean"] = 0
                features["velocity_max"] = 0
                features["velocity_std"] = 0

            # Static classification (using config threshold)
            features["is_static"] = (
                1
                if features["position_std"] < self.config.RefinerConfig.StaticThresh
                else 0
            )

            # Temporal consistency
            features["track_length"] = len(frames)
            features["score_mean"] = np.mean(scores)
            features["score_std"] = np.std(scores)
            features["temporal_consistency"] = 1.0 / (1.0 + features["score_std"])

        else:
            features = {
                "position_std": 0,
                "position_range": 0,
                "velocity_mean": 0,
                "velocity_max": 0,
                "velocity_std": 0,
                "is_static": 1,
                "track_length": 1,
                "score_mean": scores[0] if scores else 0,
                "score_std": 0,
                "temporal_consistency": 1.0,
            }

        return features

    # def compute_motion_featuresX(self, outline_infos, obj_id, current_frame, root_path, seq_name):
    #     """Compute motion features for an object."""
    #     # Collect all positions for this object
    #     positions = []
    #     frames = []
    #     scores = []

    #     css_totals = defaultdict(int)
    #     css_totals_num = 0

    #     for frame_idx, frame_info in enumerate(outline_infos):
    #         outline_cls = frame_info.get('outline_cls', [])
    #         if 'outline_ids' in frame_info and obj_id in frame_info['outline_ids']:
    #             idx = list(frame_info['outline_ids']).index(obj_id)
    #             box = frame_info['outline_box'][idx]
    #             pose = frame_info.get('pose', np.eye(4))

    #             # Transform to global coordinates
    #             global_pos = points_rigid_transform(np.array([[box[0], box[1], box[2]]]), pose)
    #             positions.append(global_pos[0])
    #             frames.append(frame_idx)

    #             if 'outline_score' in frame_info and idx < len(frame_info['outline_score']):
    #                 scores.append(frame_info['outline_score'][idx])
    #             else:
    #                 scores.append(0)

    #             # individual css scoring
    #             # Load lidar points for this frame
    #             info_path = str(frame_idx).zfill(4) + '.npy'
    #             lidar_path = Path(root_path) / seq_name / info_path

    #             lidar_points = np.load(lidar_path)[:, 0:3]

    #             # Extract point features
    #             point_features = self.extract_point_features(lidar_points, box)

    #             # Map category
    #             cls_name = outline_cls[obj_id] if obj_id < len(outline_cls) else 'Unknown'

    #             # Compute CSS components if we have points
    #             if len(point_features['cluster_points']) > 0 and cls_name in self.css.predifined_size:
    #                 dis_score, mlo_score, size_score, total_css = self.compute_css_components(
    #                     point_features['cluster_points'], box, cls_name
    #                 )
    #                 css_totals['css_distance_score'] += dis_score
    #                 css_totals['css_mlo_score'] += mlo_score
    #                 css_totals['css_size_score'] += size_score
    #                 css_totals_num += 1
    #             else:
    #                 # Use stored score or zero
    #                 css_totals['css_distance_score'] = 0
    #                 css_totals['css_mlo_score'] = 0
    #                 css_totals['css_size_score'] = 0

    #     features = {}

    #     for k, v in css_totals.items():
    #         features[f'{k}_avg'] = v / max(css_totals_num, 1)

    #     if len(positions) > 1:
    #         positions = np.array(positions)

    #         # Position variance
    #         mean_pos = np.mean(positions[:, :2], axis=0)
    #         distances = np.linalg.norm(positions[:, :2] - mean_pos, axis=1)
    #         features['position_std'] = np.std(distances)
    #         features['position_range'] = np.max(distances) - np.min(distances)

    #         # Velocity
    #         if len(frames) > 1:
    #             time_diffs = np.diff(frames)
    #             pos_diffs = np.diff(positions[:, :2], axis=0)
    #             velocities = np.linalg.norm(pos_diffs, axis=1) / (time_diffs + 1e-6)
    #             features['velocity_mean'] = np.mean(velocities)
    #             features['velocity_max'] = np.max(velocities)
    #             features['velocity_std'] = np.std(velocities)
    #         else:
    #             features['velocity_mean'] = 0
    #             features['velocity_max'] = 0
    #             features['velocity_std'] = 0

    #         # Static classification (using config threshold)
    #         features['is_static'] = 1 if features['position_std'] < self.config.RefinerConfig.StaticThresh else 0

    #         # Temporal consistency
    #         features['track_length'] = len(frames)
    #         features['score_mean'] = np.mean(scores)
    #         features['score_std'] = np.std(scores)
    #         features['temporal_consistency'] = 1.0 / (1.0 + features['score_std'])

    #     else:
    #         features = {
    #             'position_std': 0,
    #             'position_range': 0,
    #             'velocity_mean': 0,
    #             'velocity_max': 0,
    #             'velocity_std': 0,
    #             'is_static': 1,
    #             'track_length': 1,
    #             'score_mean': scores[0] if scores else 0,
    #             'score_std': 0,
    #             'temporal_consistency': 1.0
    #         }

    #     return features


def analyze_lidar_shadow(
    detection, lidar_points, ego_pos, debug=False, debug_path="shadow_debug.png"
):
    """Estimate LiDAR points within 3D shadow frustum behind object"""
    det_center = detection[:3]
    ego_pos = np.asarray(ego_pos, dtype=np.float64).reshape(3)
    lidar_points = np.asarray(lidar_points, dtype=np.float64)
    lidar_xyz = lidar_points[:, :3]
    lidar_xy = lidar_xyz[:, :2]

    # Object dimensions and orientation
    length = detection[3]
    width = detection[4]
    height = detection[5]
    yaw = detection[6]

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

    # rescale to min norm
    # min_norm = np.linalg.norm(box_corners, axis=1).min()
    # corner_left = corner_left / (np.linalg.norm(corner_left) + 1e-6)
    # corner_right = corner_right / (np.linalg.norm(corner_left) + 1e-6)

    # corner_left = corner_left * min_norm
    # corner_right = corner_right * min_norm

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
        ax.set_title("LiDAR Shadow Frustum (BEV)")

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


def estimate_occlusion_raycasting(
    detection: np.ndarray,
    # ego_pose: np.ndarray,
    lidar_points: np.ndarray,
    voxel_map,
    num_rays: int = 16,
    voxel_size: float = 0.30,
    min_distance_threshold: float = 2.0,
) -> Dict[str, float]:
    """
    Enhanced batch occlusion estimation using raycasting for multiple objects.

    Args:
        detection: Detection array with [x, y, z, l, w, h, ...] (cuboid parameters)
        ego_pose: Ego vehicle position (4,4)
        lidar_points: LiDAR point cloud (N, 3)
        voxel_map: Voxel occupancy map for raycasting
        num_rays: Number of rays to cast around each object center
        voxel_size: Voxel size for occupancy grid
        min_distance_threshold: Minimum distance for objects to be considered occluders

    Returns:
        Dictionary with occlusion metrics
    """
    ego_pos = np.zeros((3,), dtype=np.float32)

    target_center = detection[:3]
    target_size = detection[3:6]
    target_distance = np.linalg.norm(target_center - ego_pos)

    # Early exit if object is too close
    if target_distance < min_distance_threshold:
        return _get_default_metrics(target_distance)

    # Generate ray endpoints vectorized
    ray_endpoints = _generate_ray_endpoints_vectorized(
        target_center, target_size, num_rays
    )

    # Cast all rays at once and analyze occlusion
    occlusion_results = _batch_raycast_analysis(ego_pos, ray_endpoints, voxel_map)

    # Calculate ground-based metrics efficiently
    ground_metrics = _calculate_ground_metrics(lidar_points, target_center, detection)

    # Calculate path obstruction efficiently
    path_obstruction = _calculate_path_obstruction(ego_pos, target_center, lidar_points)

    # Calculate viewing angle
    ego_to_det = target_center - ego_pos
    viewing_angle = np.arccos(
        np.clip(
            np.dot(ego_to_det, np.array([1, 0, 0])) / np.linalg.norm(ego_to_det), -1, 1
        )
    )

    return {
        "overall_occlusion": occlusion_results["occlusion_ratio"],
        "lidar_occlusion": occlusion_results["lidar_occlusion_ratio"],
        "avg_intersections_per_ray": occlusion_results["avg_intersections"],
        "distance_to_ego": target_distance,
        "confidence": min(1.0, len(lidar_points) / 1000.0),
        "points_near_ground": ground_metrics["points_near_ground"],
        "height_above_nearby_ground": ground_metrics["height_above_ground"],
        "path_obstruction_ratio": path_obstruction,
        "viewing_angle": viewing_angle,
    }


def _get_default_metrics(distance: float) -> Dict[str, float]:
    """Return default metrics for edge cases."""
    return {
        "overall_occlusion": 0.0,
        "lidar_occlusion": 0.0,
        "avg_intersections_per_ray": 0.0,
        "distance_to_ego": distance,
        "confidence": 0.0,
        "points_near_ground": 0,
        "height_above_nearby_ground": 999.0,
        "path_obstruction_ratio": 0.0,
        "viewing_angle": 0.0,
    }


def _generate_ray_endpoints_vectorized(
    target_center: np.ndarray, target_size: np.ndarray, num_rays: int
) -> np.ndarray:
    """Generate ray endpoints around target object center efficiently."""
    ray_radius = target_size.max() * 0.3

    # Primary ray to object center
    endpoints = [target_center]

    if num_rays > 1:
        # Vectorized generation of additional rays
        indices = np.arange(num_rays - 1)
        theta = 2 * np.pi * indices / (num_rays - 1)
        phi = np.pi * (indices % 4) / 8

        # Vectorized spherical to cartesian conversion
        cos_phi = np.cos(phi)
        offsets = ray_radius * np.column_stack(
            [np.cos(theta) * cos_phi, np.sin(theta) * cos_phi, np.sin(phi)]
        )

        additional_endpoints = target_center[None, :] + offsets
        endpoints.extend(additional_endpoints)

    return np.array(endpoints)


def _batch_raycast_analysis(
    ego_pos: np.ndarray, ray_endpoints: np.ndarray, voxel_map
) -> Dict[str, float]:
    """Perform batch raycasting analysis."""
    total_rays = len(ray_endpoints)
    occluded_rays = 0
    lidar_occlusions = 0
    total_intersections = 0

    # Process rays in batch if possible, otherwise fall back to loop
    for endpoint in ray_endpoints:
        lidar_occluded, lidar_intersections = raycast_through_voxels(
            ego_pos, endpoint, voxel_map
        )

        if lidar_occluded:
            occluded_rays += 1
            lidar_occlusions += 1

        total_intersections += lidar_intersections

    return {
        "occlusion_ratio": occluded_rays / total_rays,
        "lidar_occlusion_ratio": lidar_occlusions / total_rays,
        "avg_intersections": total_intersections / total_rays,
    }


def _calculate_ground_metrics(
    lidar_points: np.ndarray, target_center: np.ndarray, detection: np.ndarray
) -> Dict[str, float]:
    """Calculate ground-related metrics efficiently."""
    # Filter ground points in one operation
    ground_z_threshold = detection[2] - detection[5] / 2
    ground_mask = lidar_points[:, 2] < ground_z_threshold

    if not np.any(ground_mask):
        return {"points_near_ground": 0, "height_above_ground": 999.0}

    ground_points = lidar_points[ground_mask]

    # Calculate distances to target center efficiently
    distances_2d = np.linalg.norm(ground_points[:, :2] - target_center[:2], axis=1)
    nearby_mask = distances_2d < 1.0
    nearby_ground = ground_points[nearby_mask]

    height_above_ground = 999.0
    if len(nearby_ground) > 0:
        height_above_ground = ground_z_threshold - np.max(nearby_ground[:, 2])

    return {
        "points_near_ground": len(nearby_ground),
        "height_above_ground": height_above_ground,
    }


def _calculate_path_obstruction(
    ego_pos: np.ndarray, target_center: np.ndarray, lidar_points: np.ndarray
) -> float:
    """Calculate path obstruction ratio efficiently."""
    ego_to_det = target_center - ego_pos
    ego_to_det_norm = np.linalg.norm(ego_to_det)

    if ego_to_det_norm == 0:
        return 0.0

    # Vectorized cross product for distance from line calculation
    points_rel = lidar_points - ego_pos
    cross_products = np.cross(points_rel, ego_to_det)

    # Handle both 2D and 3D cases
    if cross_products.ndim > 1:
        line_distances = np.linalg.norm(cross_products, axis=1) / ego_to_det_norm
    else:
        line_distances = np.abs(cross_products) / ego_to_det_norm

    # Points close to the line
    path_mask = line_distances < 0.5
    path_points = lidar_points[path_mask]

    if len(path_points) == 0:
        return 0.0

    # Points before the detection along the ray
    projections = np.dot(path_points - ego_pos, ego_to_det)
    before_det_mask = (projections > 0) & (projections < ego_to_det_norm**2)

    return np.sum(before_det_mask) / len(path_points)


def estimate_occlusion_raycastingX(
    detection: np.ndarray,
    ego_pos: np.ndarray,
    lidar_points: np.ndarray,
    voxel_map,
    camera_positions: Optional[Dict[str, np.ndarray]] = None,
    num_rays: int = 16,
    voxel_size: float = 0.30,
    min_distance_threshold: float = 2.0,
    do_shadow_vis: bool = False,
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
    # Create voxel occupancy map from LiDAR

    target_center = detection[:3]
    # Find potential occluding objects (closer to ego and within reasonable distance)
    target_distance = np.linalg.norm(target_center - ego_pos)

    # Generate ray endpoints around target object
    ray_endpoints = []

    # Primary ray to object center
    ray_endpoints.append(target_center)

    # Additional rays around object center (small sphere sampling)
    target_size = detection[3:6].max()

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

        if ray_occluded:
            occluded_rays += 1

        total_intersections += intersections

    # Calculate occlusion metrics
    occlusion_ratio = occluded_rays / total_rays
    lidar_occlusion_ratio = lidar_occlusions / total_rays

    ground_points = lidar_points[lidar_points[:, 2] < detection[2] - detection[5] / 2]
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

    return {
        "overall_occlusion": occlusion_ratio,
        "lidar_occlusion": lidar_occlusion_ratio,
        # "camera_occlusion": camera_occlusion,
        "avg_intersections_per_ray": total_intersections / total_rays,
        "distance_to_ego": target_distance,
        "confidence": min(1.0, len(lidar_points) / 1000.0),
        # "shadow_point_ratio": analyze_lidar_shadow(
        #     detection, lidar_points, ego_pos, debug=do_shadow_vis, debug_path="cpd_shadow.png"
        # ),
        "points_near_ground": len(nearby_ground),
        "height_above_nearby_ground": (
            detection[2] - detection[5] / 2 - np.max(nearby_ground[:, 2])
            if len(nearby_ground) > 0
            else 999
        ),
        "path_obstruction_ratio": len(path_points_before_det)
        / max(1, len(path_points)),
        "viewing_angle": np.arccos(
            np.dot(ego_to_det, np.array([1, 0, 0])) / np.linalg.norm(ego_to_det)
        ),
    }


def extract_all_features_accurate(
    output_info_path: str,
    log_id: str,
    config,
    root_path: str = None,
    output_path: Optional[str] = None,
    category_mapping: Optional[Dict[str, str]] = None,
    gts_dataframe: pd.DataFrame = None,
    use_first_frame=False,
    do_single_frame=False,
    score_thresh: float = 0,
) -> pd.DataFrame:
    """
    Extract all heuristic features accurately by re-running CSS and other computations.

    Args:
        output_info_path: Path to C_PROTO output pickle
        log_id: Argoverse log ID
        config: Same config used in C_PROTO
        root_path: Root directory containing lidar data
        output_path: Where to save the output feather file
        category_mapping: Category name mapping

    Returns:
        DataFrame with all features
    """

    # Initialize feature extractor with config
    extractor = FeatureExtractor(config)

    # Load detection results
    with open(output_info_path, "rb") as f:
        outline_infos = pkl.load(f)

    if root_path is None:
        root_path = Path(output_info_path).parent.parent

    print(f"Loaded {len(outline_infos)} frames from {output_info_path}")

    if category_mapping is None:
        raise ValueError("category_mapping is None")

    if root_path is None:
        root_path = Path(output_info_path).parent.parent

    seq_name = Path(output_info_path).parent.name

    all_detections = []

    print("Extracting accurate features for all detections...")
    num_matched = 0

    # Extract object trajectories across all frames
    object_trajectories = defaultdict(list)
    frame_object_counts = []

    for frame_idx, frame_info in enumerate(outline_infos):
        timestamp_ns = frame_info.get(
            "timestamp", frame_info.get("timestamp_ns", frame_idx * 100000000)
        )
        gt_count = len(
            gts_dataframe[
                (gts_dataframe["timestamp_ns"] == int(timestamp_ns))
                & (gts_dataframe["log_id"] == log_id)
            ]
        )

        frame_object_counts.append(len(frame_info.get("outline_box", [])))
        continue

        # assert gt_count > 0
        # frame_object_counts.append(gt_count)
        # continue

        outline_boxes = frame_info.get("outline_box", [])
        outline_ids = frame_info.get("outline_ids", [])
        outline_cls = frame_info.get("outline_cls", [])
        outline_scores = frame_info.get("outline_score", [])
        pose = frame_info.get("pose", np.eye(4))

        frame_object_counts.append(len([x for x in outline_scores if x > 0]))

        for box_idx, (box, obj_id, cls, score) in enumerate(
            zip(outline_boxes, outline_ids, outline_cls, outline_scores)
        ):
            if (
                len(box) >= 7 and score > 0
            ):  # Ensure we have [x, y, z, length, width, height, yaw]
                # Transform to global coordinates if needed
                center_global = box[:3]  # Assuming already in global coordinates

                object_trajectories[obj_id].append(
                    {
                        "frame_idx": frame_idx,
                        "center": center_global[:2],  # x, y only for BEV
                        "box": box,
                        "class": cls,
                        "score": score,
                        "pose": pose,
                    }
                )

    # Choose reference frame
    if use_first_frame or len(frame_object_counts) == 0:
        ref_frame_idx = 0
    else:
        ref_frame_idx = np.argmax(frame_object_counts)

    print(
        f"Using frame {ref_frame_idx} as reference (contains {frame_object_counts[ref_frame_idx]} objects)"
    )

    for frame_idx, frame_info in enumerate(outline_infos):
        if do_single_frame and frame_idx != ref_frame_idx:
            continue
        print("outline_box", len(frame_info.get("outline_box", [])))
        # Get and filter in one step
        filtered = [
            (box, id_, cls, score, proto_id)
            for box, id_, cls, score, proto_id in zip(
                frame_info.get("outline_box", []),
                frame_info.get("outline_ids", []),
                frame_info.get("outline_cls", []),
                frame_info.get(
                    "outline_score",
                    [score_thresh + 0.1 for _ in frame_info.get("outline_box", [])],
                ),
                frame_info.get(
                    "outline_proto_id",
                    [i for i in range(len(frame_info.get("outline_box", [])))],
                ),
            )
            if score > score_thresh
        ]

        outline_boxes, outline_ids, outline_cls, outline_scores, outline_proto_ids = (
            list(zip(*filtered)) if filtered else ([], [], [], [], [])
        )
        pose = frame_info.get("pose", np.eye(4))

        assert (
            len(outline_boxes)
            == len(outline_ids)
            == len(outline_cls)
            == len(outline_scores)
            == len(outline_proto_ids)
        )

        if len(outline_boxes) == 0:
            print("len(outline_boxes) == 0")
            continue

        timestamp_ns = frame_info.get(
            "timestamp", frame_info.get("timestamp_ns", frame_idx * 100000000)
        )

        # gt_frame = gts_dataframe[
        #     (gts_dataframe["timestamp_ns"] == int(timestamp_ns))
        #     & (gts_dataframe["log_id"] == log_id)
        # ]
        gt_frame = gts_dataframe.loc[[(log_id, int(timestamp_ns))]]

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

        pred_lidar_boxes = torch.tensor(outline_boxes, dtype=torch.float32)

        if len(gt_lidar_boxes) > 0 and len(pred_lidar_boxes) > 0:
            ious = rotate_iou_cpu_eval(gt_lidar_boxes, pred_lidar_boxes).reshape(
                gt_lidar_boxes.shape[0], pred_lidar_boxes.shape[0], 2
            )
            ious = ious[:, :, 0]
        else:
            ious = torch.zeros(
                (len(gt_lidar_boxes), len(pred_lidar_boxes)), dtype=torch.float32
            )
            print("gt_frame", gt_frame)
            print(f"{timestamp_ns=}")
            print(f"{log_id=}")

            # raise Exception(f"no gt or something? {gt_lidar_boxes.shape} {pred_lidar_boxes.shape}")

        # Load lidar points for this frame
        info_path = str(frame_idx).zfill(4) + ".npy"
        lidar_path = Path(root_path) / seq_name / info_path

        H_path = Path(root_path) / seq_name / "ppscore" / info_path

        if not lidar_path.exists():
            print(f"Warning: {lidar_path=} not found, skipping frame {frame_idx}")
            continue

        if not H_path.exists():
            print(f"Warning: {H_path=} not found, skipping frame {frame_idx}")
            continue

        lidar_points = np.load(lidar_path)[:, 0:3]
        H = np.load(H_path)

        voxel_map = create_voxel_occupancy_map(lidar_points, 0.3)

        vis_box_id = np.random.randint(0, len(outline_boxes))

        # Process each detection
        for box_idx, box in enumerate(outline_boxes):
            detection = {}

            if outline_scores[box_idx] < score_thresh:
                continue

            gt_ious_ = ious[:, box_idx]

            if len(gt_ious_) > 0:
                best_iou = gt_ious_.max().item()
                best_iou_cat = gt_categories[gt_ious_.argmax()]
                num_matched += 1
            else:
                best_iou = 0
                best_iou_cat = "NONE"

                # raise ValueError(f"No iou? ious.shape={ious.shape} {box_idx=} outline_boxes={len(outline_boxes)}")

            # Basic info
            x, y, z, l, w, h, yaw = box[:7]
            detection["x"] = x
            detection["y"] = y
            detection["z"] = z
            detection["length"] = l
            detection["width"] = w
            detection["height"] = h
            detection["yaw"] = yaw
            detection["frame_idx"] = frame_idx
            detection["timestamp_ns"] = int(timestamp_ns)
            detection["log_id"] = log_id

            # GT -> for classification later
            detection["gt_best_iou"] = best_iou  # allows multiple to match
            detection["gt_best_iou_cat"] = best_iou_cat

            # Object info
            obj_id = outline_ids[box_idx] if box_idx < len(outline_ids) else -1
            cls_name = outline_cls[box_idx] if box_idx < len(outline_cls) else "Unknown"
            proto_id = (
                outline_proto_ids[box_idx] if box_idx < len(outline_proto_ids) else -1
            )

            detection["object_id"] = obj_id
            detection["class_name"] = cls_name
            detection["proto_id"] = proto_id
            detection["track_uuid"] = f"{log_id}_{obj_id}"

            # Map category
            if str(cls_name) in category_mapping:
                detection["category"] = category_mapping[str(cls_name)]
            else:
                detection["category"] = list(category_mapping.values())[0]

            # Extract point features
            point_features = extractor.extract_point_features(lidar_points, H, box)
            detection.update(
                {k: v for k, v in point_features.items() if k != "cluster_points"}
            )

            detection.update(
                estimate_occlusion_raycasting(box, lidar_points, voxel_map)
            )

            # Compute CSS components if we have points
            if (
                len(point_features["cluster_points"]) > 0
                and cls_name in extractor.css.predifined_size
            ):
                dis_score, mlo_score, size_score, total_css = (
                    extractor.compute_css_components(
                        point_features["cluster_points"], box, cls_name
                    )
                )
                detection["css_distance_score"] = dis_score
                detection["css_mlo_score"] = mlo_score
                detection["css_size_score"] = size_score
                detection["css_total_score"] = total_css
            else:
                # Use stored score or zero
                stored_score = (
                    outline_scores[box_idx] if box_idx < len(outline_scores) else 0
                )
                detection["css_distance_score"] = 0
                detection["css_mlo_score"] = 0
                detection["css_size_score"] = 0
                detection["css_total_score"] = stored_score

            detection["score"] = detection["css_total_score"]  # Use CSS as main score

            # Box geometry features
            detection["box_volume"] = l * w * h
            detection["box_aspect_ratio_lw"] = l / w if w > 0 else 0
            detection["box_aspect_ratio_lh"] = l / h if h > 0 else 0
            detection["box_aspect_ratio_wh"] = w / h if h > 0 else 0
            detection["distance_from_sensor"] = np.sqrt(x**2 + y**2)

            # Motion features (computed across all frames)
            motion_features = extractor.compute_motion_features(
                outline_infos, obj_id, frame_idx
            )
            detection.update(motion_features)

            prev_boxes = []
            prev_timestamps = []

            for frame_idx, frame_info in enumerate(outline_infos):
                if "outline_ids" in frame_info and obj_id in frame_info["outline_ids"]:
                    idx = list(frame_info["outline_ids"]).index(obj_id)
                    box = frame_info["outline_box"][idx]
                    # pose = frame_info.get('pose', np.eye(4))

                    prev_boxes.append(box)

                    # Transform to global coordinates
                    # global_pos = points_rigid_transform(np.array([[box[0], box[1], box[2]]]), pose)
                    # prev_boxes.append(global_pos[0])
                    prev_timestamps.append(
                        frame_info.get(
                            "timestamp",
                            frame_info.get("timestamp_ns", frame_idx * 100000000),
                        )
                    )

                    # box = np.array(box)
                    # print('box', box)
                    # box[:3] = global_pos
                    # print('box w/global_pos', box)

            if len(prev_boxes) > 0:
                prev_boxes = np.array(prev_boxes)
                prev_timestamps = np.array(prev_timestamps)
            else:
                prev_boxes = None
                prev_timestamps = None

            # advanced_features = extractor.extract_advanced_objectness_features(
            #     lidar_points, H, box, prev_boxes, prev_timestamps
            # )
            # detection.update(advanced_features)

            # Additional heuristics from config
            detection["max_distance_threshold"] = extractor.css.max_dis
            detection["passes_min_volume"] = (
                1
                if detection["box_volume"] >= config.GeneratorConfig.min_box_volume
                else 0
            )
            detection["passes_max_volume"] = (
                1
                if detection["box_volume"] <= config.GeneratorConfig.max_box_volume
                else 0
            )
            detection["passes_min_height"] = (
                1 if h >= config.GeneratorConfig.min_box_height else 0
            )
            detection["passes_max_length"] = (
                1 if l <= config.GeneratorConfig.max_box_len else 0
            )

            # Orientation quality (from C_PROTO logic)
            detection["orientation_quality"] = (
                detection["css_total_score"] > config.RefinerConfig.OrienThresh
                if "OrienThresh" in dir(config.RefinerConfig)
                else 0
            )

            all_detections.append(detection)

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{len(outline_infos)}")

    # Create DataFrame
    df = pd.DataFrame(all_detections)

    print(
        f"{num_matched=} total={len(all_detections)} {(num_matched/max(1, len(all_detections)))*100.0:.2f}% matched with GT"
    )

    feature_cols = [
        "css_distance_score",
        "css_mlo_score",
        "css_size_score",
        "css_total_score",
        "num_points_in_cluster",
        "point_density",
        "cluster_density",
        "height_above_ground",
        "distance_from_sensor",
        "position_std",
        "velocity_mean",
        "temporal_consistency",
        "overall_occlusion",
        "box_volume",
        "length",
        "width",
        "height",
    ]

    # Calculate averages for each track_uuid and add as new columns
    # for col in feature_cols:
    #     if col in df.columns:
    #         df[f'{col}_avg'] = df.groupby('track_uuid')[col].transform('mean')
    #     else:
    #         print(f"{col=} not available")

    # for col in feature_cols:
    #     if pd.api.types.is_numeric_dtype(df[col]):
    #         df[f'{col}_avg'] = df.groupby('track_uuid')[col].transform('mean')

    if len(df) == 0:
        print("Warning: No valid detections found!")
        return df

    # Sort by score
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    print(f"\nExtracted {len(df)} detections with {len(df.columns)} features")
    print(f"Categories: {sorted(df['category'].unique())}")

    # Print feature statistics
    print("\nFeature Statistics:")

    for col in df.columns:
        # if col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(
                f"  {col:25s}: mean={df[col].mean():7.3f}, std={df[col].std():7.3f}, "
                f"min={df[col].min():7.3f}, max={df[col].max():7.3f}"
            )

    # Save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        if not output_path.suffix == ".feather":
            output_path = output_path.with_suffix(".feather")
        df.to_feather(output_path)
        print(f"\nSaved to {output_path}")

    return df


def train_rf_model(features_df: pd.DataFrame, target_col="gt_best_iou"):
    # Select numeric features more carefully
    feature_cols = []
    for col in features_df.columns:
        if (
            col != target_col
            and col not in IGNORE_COLS
            and pd.api.types.is_numeric_dtype(features_df[col])
        ):
            # Check for sufficient variance
            if features_df[col].var() > 1e-10:
                feature_cols.append(col)

    print(
        f"Selected {len(feature_cols)} features with sufficient variance: {feature_cols}"
    )

    X_processed = features_df[feature_cols].copy()

    # Better handling of infinite/missing values
    X_processed = X_processed.replace([np.inf, -np.inf], np.nan)

    # Use different imputation strategies for different feature types
    for col in X_processed.columns:
        if X_processed[col].isna().sum() > 0:
            X_processed[col] = X_processed[col].fillna(X_processed[col].mean())

    # Robust scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    # Extract target variable
    y = features_df[target_col].copy()

    # Handle missing values in target if any
    if y.isna().sum() > 0:
        print(
            f"Warning: {y.isna().sum()} missing values in target column. Dropping these rows."
        )
        mask = ~y.isna()
        X_scaled = X_scaled[mask]
        y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"R² score: {r2:.4f}")

    # Return the trained model, scaler, and feature columns for future predictions
    return {
        "model": rf_reg,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "r2_score": r2,
        "test_predictions": y_pred,
        "test_actual": y_test,
        "feature_importance": pd.DataFrame(
            {"feature": feature_cols, "importance": rf_reg.feature_importances_}
        ).sort_values("importance", ascending=False),
    }


def train_interpretable_tree(
    features_df: pd.DataFrame,
    target_col="gt_best_iou",
    max_depth=10,
    min_samples_leaf=5,
):
    """
    Train an interpretable decision tree and extract decision rules
    """

    # Remove subsequent versions of the same object to debias it from certain objects that appear for a long time
    features_df = features_df.drop_duplicates(subset=["track_uuid"], keep="first")

    feature_cols = []
    for col in features_df.columns:
        if (
            col != target_col
            and col not in IGNORE_COLS
            and pd.api.types.is_numeric_dtype(features_df[col])
        ):
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
        print(
            f"Warning: {y.isna().sum()} missing values in target. Dropping these rows."
        )
        mask = ~y.isna()
        X_scaled = X_scaled[mask]
        y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    print(
        "X_train, X_test, y_train, y_test",
        [x.shape for x in [X_train, X_test, y_train, y_test]],
    )

    # Decision Tree with interpretability constraints
    dt_reg = DecisionTreeRegressor(
        max_depth=max_depth,  # Limit depth for interpretability
        min_samples_leaf=min_samples_leaf,  # Avoid overfitting
        min_samples_split=20,  # Require meaningful splits
        max_features=None,  # Use all features for full interpretability
        random_state=42,
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
        "model": dt_reg,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "test_r2": r2,
        "test_rmse": rmse,
        # 'X_test': X_test,
        # 'y_test': y_test,
        # 'y_pred': y_pred,
        "feature_importance": pd.DataFrame(
            {"feature": feature_cols, "importance": dt_reg.feature_importances_}
        ).sort_values("importance", ascending=False),
    }


def train_interpretable_classifier(
    features_df: pd.DataFrame, iou_threshold=0.3, max_depth=10, min_samples_leaf=5
):
    """
    Train an interpretable decision tree classifier for TP/FP detection
    Based on IoU threshold with ground truth
    """

    # Remove subsequent versions of the same object to debias it from certain objects that appear for a long time
    features_df = features_df.drop_duplicates(subset=["track_uuid"], keep="first")

    print("after deduplication", len(features_df))

    # Create classification labels
    # TP: IoU >= threshold, FP: IoU < threshold (including 0)
    # Note: FN would be in ground truth but not in predictions (not in this dataframe)
    y = (features_df["gt_best_iou"] >= iou_threshold).astype(int)

    gt_best_iou = features_df["gt_best_iou"].values

    print("y", y.shape, (y == 0).sum(), (y > 0).sum())
    print(
        "gt_best_iou",
        gt_best_iou.shape,
        (gt_best_iou == 0).sum(),
        (gt_best_iou > 0).sum(),
    )

    feature_cols = []
    for col in features_df.columns:
        if col not in IGNORE_COLS and pd.api.types.is_numeric_dtype(features_df[col]):
            if features_df[col].var() > 1e-10:
                feature_cols.append(col)

    print(f"Selected {len(feature_cols)} features for classification")
    print(
        f"Class distribution: TP={y.sum()} ({y.mean():.1%}), FP={(~y).sum()} ({(~y).mean():.1%})"
    )

    X_processed = features_df[feature_cols].copy()
    X_processed = X_processed.replace([np.inf, -np.inf], np.nan)

    for col in X_processed.columns:
        if X_processed[col].isna().sum() > 0:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    X_scaled = X_processed.values

    # Handle missing labels
    if y.isna().sum() > 0:
        mask = ~y.isna()
        X_scaled = X_scaled[mask]
        y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    dt_clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=20,
        class_weight="balanced",  # Handle class imbalance
        random_state=42,
    )

    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)
    y_pred_proba = dt_clf.predict_proba(X_test)[:, 1]

    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.5

    print(f"\nClassification Results (IoU threshold = {iou_threshold}):")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    print("\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": dt_clf.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())

    # Method 1: Text representation of the tree
    tree_rules = export_text(
        dt_clf,
        feature_names=feature_cols,
        max_depth=100,
        spacing=2,
        class_names=["FP", "TP"],
    )
    print("=" * 50)
    print("DECISION TREE RULES w/class names:")
    print("=" * 50)
    print(tree_rules)

    return {
        "model": dt_clf,
        "feature_cols": feature_cols,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "feature_importance": feature_importance,
        "iou_threshold": iou_threshold,
    }


def train_interpretable_tree_category_classifier(
    features_df: pd.DataFrame,
    target_col="gt_best_iou_cat",
    max_depth=10,
    min_samples_leaf=5,
):
    """
    Train an interpretable decision tree classifier and extract decision rules for categorical targets
    """

    # Remove subsequent versions of the same object to debias it from certain objects that appear for a long time
    # features_df = features_df.drop_duplicates(subset=['track_uuid'], keep='first')

    feature_cols = []
    for col in features_df.columns:
        if (
            col != target_col
            and col not in IGNORE_COLS
            and pd.api.types.is_numeric_dtype(features_df[col])
        ):
            # Check for sufficient variance
            if features_df[col].var() > 1e-10:
                feature_cols.append(col)

    print(f"Selected {len(feature_cols)} features with sufficient variance")

    X_processed = features_df[feature_cols].copy()
    X_processed = X_processed.replace([np.inf, -np.inf], np.nan)

    for col in X_processed.columns:
        if X_processed[col].isna().sum() > 0:
            X_processed[col] = X_processed[col].fillna(X_processed[col].mean())

    # Using unscaled features for better interpretability
    X_scaled = X_processed.values
    scaler = None

    y = features_df[target_col].copy()

    # REPLACE WITH NONE IF IOU IS LOW
    y_low_mask = features_df["gt_best_iou"] < 0.3
    y[y_low_mask] = "NONE"

    # Handle missing values in target
    if y.isna().sum() > 0:
        print(
            f"Warning: {y.isna().sum()} missing values in target. Dropping these rows."
        )
        mask = ~y.isna()
        X_scaled = X_scaled[mask]
        y = y[mask]

    # Create class mapping for categorical target
    unique_classes = sorted(y.unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    print(f"Found {len(unique_classes)} unique classes: {unique_classes}")

    # Remove classes with insufficient samples for training
    min_samples_per_class = 10
    class_counts = y.value_counts()
    classes_to_keep = class_counts[class_counts > min_samples_per_class].index.tolist()

    print("Class counts before filtering:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} samples")

    if len(classes_to_keep) < len(unique_classes):
        # Filter out rows with insufficient class samples
        mask = y.isin(classes_to_keep)
        X_scaled = X_scaled[mask]
        y = y[mask]

        print(
            f"\nRemoved {len(unique_classes) - len(classes_to_keep)} classes with ≤{min_samples_per_class} samples"
        )
        print(f"Remaining classes: {classes_to_keep}")
        print(f"Total samples after filtering: {len(y)}")

        # Update unique classes and create new class mapping
        unique_classes = sorted(classes_to_keep)

    # Create class mapping for categorical target (after filtering)
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    print("\nFinal class mapping:")
    for cls, idx in class_to_idx.items():
        print(f"  {cls} -> {idx}")

    # Convert string labels to numerical indices
    y_encoded = y.map(class_to_idx)

    # Remove classes with insufficient samples for training
    min_samples_per_class = 10
    class_counts = y_encoded.value_counts()
    classes_to_keep = class_counts[class_counts > min_samples_per_class].index

    print("Class counts before filtering:")
    for idx, count in class_counts.items():
        class_name = idx_to_class[idx]
        print(f"  {class_name}: {count} samples")

    if len(classes_to_keep) < len(unique_classes):
        # Filter out rows with insufficient class samples
        mask = y_encoded.isin(classes_to_keep)
        X_scaled = X_scaled[mask]
        y_encoded = y_encoded[mask]

        # Update class mappings to only include kept classes
        kept_class_names = [idx_to_class[idx] for idx in classes_to_keep]
        class_to_idx = {cls: idx for idx, cls in enumerate(kept_class_names)}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        unique_classes = kept_class_names

        # Re-encode y with new class indices
        y_string = y_encoded.map(
            lambda x: (
                [k for k, v in class_to_idx.items() if v == x][0]
                if x in classes_to_keep
                else None
            )
        )
        y_encoded = y_string.map(class_to_idx)

        print(
            f"\nRemoved {len(class_counts) - len(classes_to_keep)} classes with ≤{min_samples_per_class} samples"
        )
        print(f"Remaining classes: {kept_class_names}")
        print(f"Total samples after filtering: {len(y_encoded)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    print(
        "X_train, X_test, y_train, y_test",
        [x.shape for x in [X_train, X_test, y_train, y_test]],
    )

    # Decision Tree Classifier with interpretability constraints
    dt_clf = DecisionTreeClassifier(
        max_depth=max_depth,  # Limit depth for interpretability
        min_samples_leaf=min_samples_leaf,  # Avoid overfitting
        min_samples_split=20,  # Require meaningful splits
        max_features=None,  # Use all features for full interpretability
        random_state=42,
    )

    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)

    y_test = np.array(y_test.values)

    print("y_test y_pred", y_test.shape, y_pred.shape)

    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)

    # precision = average_precision_score(y_test.reshape(-1, 1), y_pred.reshape(-1, 1), average='weighted')
    precision = -1
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    print(f"Decision Tree Precision (weighted): {precision:.4f}")
    print(f"Decision Tree Recall (weighted): {recall:.4f}")
    print(f"Decision Tree F1 Score (weighted): {f1:.4f}")
    print(f"Tree depth: {dt_clf.get_depth()}")
    print(f"Number of leaves: {dt_clf.get_n_leaves()}")

    # Class distribution in test set
    print("\nClass distribution in test set:")
    test_classes = pd.Series(y_test).map(idx_to_class).value_counts().sort_index()
    for cls, count in test_classes.items():
        print(f"  {cls}: {count} samples")

    # Method 1: Text representation of the tree
    tree_rules = export_text(
        dt_clf,
        feature_names=feature_cols,
        max_depth=100,
        spacing=2,
        class_names=unique_classes,
    )
    print("=" * 50)
    print("DECISION TREE RULES w/class names:")
    print("=" * 50)
    print(tree_rules)

    return {
        "model": dt_clf,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "unique_classes": unique_classes,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "feature_importance": pd.DataFrame(
            {"feature": feature_cols, "importance": dt_clf.feature_importances_}
        ).sort_values("importance", ascending=False),
    }


def extract_decision_rules(model_result):
    """
    Extract human-readable decision rules from the trained tree
    """
    dt_model = model_result["model"]
    feature_cols = model_result["feature_cols"]

    # Method 1: Text representation of the tree
    tree_rules = export_text(
        dt_model, feature_names=feature_cols, max_depth=100, spacing=2
    )
    print("=" * 50)
    print("DECISION TREE RULES:")
    print("=" * 50)
    print(tree_rules)

    return tree_rules


def get_cproto_output_path(output_dir: Path, log_id: str) -> Path:
    return output_dir / log_id / f"{log_id}_outline_C_PROTO.pkl"


def get_mfcf_output_path(output_dir: Path, log_id: str) -> Path:
    return output_dir / log_id / f"{log_id}_outline_MFCF.pkl"


def get_alpha_shape_output_path(output_dir: Path, log_id: str) -> Path:
    # return output_dir / log_id / f"{log_id}_alpha_shapes.pkl"
    method_name = "MFCF"
    return output_dir / log_id / (log_id + "_alpha_shapes_" + str(method_name) + ".pkl")


def get_owl_alpha_shape_output_path(output_dir: Path, log_id: str) -> Path:
    # return output_dir / log_id / f"{log_id}_alpha_shapes.pkl"
    method_name = "MFCF_owlvit_hybrid"
    return output_dir / log_id / (log_id + "_alpha_shapes_" + str(method_name) + ".pkl")


def analyze_mfcf(args, cfg, dataset_dir: Path, output_dir: Path):
    val_anno_path = (
        "/home/uqdetche/lidar_longtail_mining/lion/data/argo2/val_anno.feather"
    )

    val_data_path = Path(
        "/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val"
    )

    log_ids = [entry.name for entry in val_data_path.iterdir() if entry.is_dir()]
    print(log_ids[:10])

    cproto_log_ids = []

    for log_id in log_ids:
        cproto_output_path = get_mfcf_output_path(output_dir, log_id)

        if cproto_output_path.exists():
            print(f"{cproto_output_path=}")
            cproto_log_ids.append(log_id)

    # cproto_log_ids = ['25e5c600-36fe-3245-9cc0-40ef91620c22']
    # cproto_log_ids = cproto_log_ids[:5]

    print(f"Found {len(cproto_log_ids)} cproto_log_ids")

    vis_log_id = np.random.choice(cproto_log_ids)
    print(f"{vis_log_id=}")

    vis_cproto_output_path = get_mfcf_output_path(output_dir, vis_log_id)

    vis_annotations_feather = Path(
        f"/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val/{vis_log_id}/annotations.feather"
    )
    assert vis_annotations_feather.exists()

    gts = pd.read_feather(vis_annotations_feather)

    # add log_id
    gts = gts.assign(
        log_id=pd.Series([vis_log_id for _ in range(len(gts))], dtype="string").values
    )

    gts_timestamps = gts["timestamp_ns"].values

    print("gts_timestamps", gts_timestamps.min(), gts_timestamps.max())

    gts = gts.set_index(["log_id", "timestamp_ns"], drop=False).sort_values("category")

    # Create BEV plot with trajectories
    outline_infos = load_and_plot_objects(
        vis_cproto_output_path,
        use_first_frame=False,  # Use frame with most objects
        plot_trajectories=True,
        gts_dataframe=gts.copy(),  # Add GT overlay
        save_path="mfcf_bev_visualization.png",
        log_id=vis_log_id,
    )

    gts = pd.read_feather(val_anno_path)
    gts = gts.set_index(["log_id", "timestamp_ns"], drop=False).sort_values("category")

    category_mapping = {
        "Vehicle": SensorCompetitionCategories.REGULAR_VEHICLE.value,
        "Cyclist": SensorCompetitionCategories.BICYCLE.value,
        "Pedestrian": SensorCompetitionCategories.PEDESTRIAN.value,
    }

    features_output_path = output_dir / "mfcf_features.feather"
    # Extract features with accurate CSS re-computation
    if not features_output_path.exists():
        serialized_dts_list = []
        for log_id in tqdm(cproto_log_ids, desc="Extracting all features"):
            cproto_output_path = get_mfcf_output_path(output_dir, log_id)

            cur_features_output_path = output_dir / log_id / "mfcf_features.feather"

            # TODO: COMMENT OUT LATER
            if cur_features_output_path.exists():
                os.remove(cur_features_output_path)

            if not cur_features_output_path.exists():
                
                if args.profile:
                    pr = cProfile.Profile()
                    pr.enable()

                df_features = extract_all_features_accurate(
                    output_info_path=str(cproto_output_path),
                    log_id=log_id,
                    config=cfg,  # Pass the same config used in C_PROTO
                    root_path=str(output_dir),
                    output_path=str(cur_features_output_path),
                    category_mapping=category_mapping,
                    gts_dataframe=gts.copy(),
                    score_thresh=-1,  # no score thresh
                )

                if args.profile:
                    pr.disable()

                    s = io.StringIO()
                    sortby = "cumtime"
                    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    ps.print_stats()
                    with open(
                        "mfcf_analyzer_extract_all_features_accurate_profile_stats.txt", "w"
                    ) as f:
                        f.write(s.getvalue())
            else:
                print(f"read from {cur_features_output_path}")
                df_features = pd.read_feather(cur_features_output_path)

            serialized_dts_list.append(df_features)

        df_features = pd.concat(serialized_dts_list, ignore_index=True)

        # Save if path provided
        if features_output_path is not None:
            if not features_output_path.suffix == ".feather":
                features_output_path = features_output_path.with_suffix(".feather")
            df_features.to_feather(features_output_path)
            print(f"\nSaved to {features_output_path}")
    else:
        df_features = pd.read_feather(features_output_path)

    print("unique log_ids", df_features["log_id"].unique())
    print("unique log_ids", df_features["log_id"].unique().__len__())
    print("unique timestamps", df_features["timestamp_ns"].unique().__len__())
    print("unique object ids", df_features["object_id"].unique().__len__())
    print("unique proto ids", df_features["proto_id"].unique().__len__())
    print("unique track uuids", df_features["track_uuid"].unique().__len__())

    track_uuids = df_features["track_uuid"].unique()
    print(
        "df_features['track_uuid'].value_counts()",
        df_features["track_uuid"].value_counts(),
    )
    print("track_uuids", track_uuids.shape, track_uuids.min(), track_uuids.max())

    vc = df_features["track_uuid"].value_counts().value_counts()

    plt.figure(figsize=(6, 4))
    ax = vc.plot(kind="bar", color="skyblue")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Distribution of Value Counts")

    # Stagger x-axis ticks: only show ~10 evenly spaced ticks
    n_ticks = 10
    all_ticks = np.arange(len(vc))
    tick_positions = all_ticks[:: max(1, len(vc) // n_ticks)]
    tick_labels = vc.index[tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.tight_layout()
    plt.savefig("mfcf_value_counts_barplot.png")
    plt.close()

    dist_normalized = (
        df_features["track_uuid"].value_counts().value_counts(normalize=True)
    )
    print(dist_normalized)
    # exit()

    print(f"\nTotal features extracted: {len(df_features.columns)}")
    print("Feature columns:", df_features.columns.tolist())

    # model_results = train_rf_model(df_features)

    # pprint(model_results)

    max_depth = 5

    dt_result = train_interpretable_tree(df_features, max_depth=max_depth)
    # pprint(dt_result)

    extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("mfcf_decision_tree.png")

    dt_result = train_interpretable_classifier(
        df_features, max_depth=max_depth, iou_threshold=0.1
    )
    # pprint(dt_result)

    # extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("mfcf_tp_decision_tree.png")

    dt_result = train_interpretable_tree_category_classifier(
        df_features, max_depth=max_depth
    )
    # pprint(dt_result)

    # extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
        class_names=dt_result["unique_classes"],
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("mfcf_tp_decision_tree_category.png")


def create_infos(args, cfg, dataset_dir: Path, output_dir: Path):
    # Load av2...
    # TODO : load more than 1 log_id
    # log_id = "0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2"
    # timestamp_ns = 315971843759817000

    val_anno_path = (
        "/home/uqdetche/lidar_longtail_mining/lion/data/argo2/val_anno.feather"
    )
    # dataset_dir="/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val"
    # TODO: move elswhere

    # # Recreate dataloader in this process
    # dataloader = ModifiedSensorDataloader(dataset_dir=dataset_dir.parent)
    # split = dataset_dir.name

    # # Load sensor data and compute occlusions
    # target_datum = dataloader.get_sensor_data(log_id, split, timestamp_ns, cam_names=[])

    val_data_path = Path(
        "/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val"
    )

    log_ids = [entry.name for entry in val_data_path.iterdir() if entry.is_dir()]
    print(log_ids[:10])

    if args.run_cpd_infos:
        for log_id in tqdm(log_ids, desc="Running CPD on log ids..."):
            print("Saving pose information...")
            save_infos(
                log_id,
                output_dir,
                dataset_dir,
                split="val",  # TODO: auto
            )

            # generate pp score
            print("Saving ppscores...")

            save_pp_score(
                log_id,
                output_dir,
                dataset_dir,
                split="val",  # TODO: auto
                # TODO: load from config
            )

            # create outline box
            print("Creating outline boxes using mfcf...")

            # os.path.join(
            #         root_path, seq_name, seq_name + "_outline_" + str(method_name) + ".pkl"
            #     )
            mfcf_output_path = output_dir / log_id / f"{log_id}_outline_MFCF.pkl"

            if not mfcf_output_path.exists():
                outliner = MFCF(log_id, root_path=str(output_dir), config=cfg)
                outliner()
            else:
                print(f"{mfcf_output_path.name} already exists!")

            print("Refining boxes using C_PROTO...")
            # create refiner
            # pr = cProfile.Profile()
            # pr.enable()
            refiner = C_PROTO(log_id, root_path=str(output_dir), config=cfg)
            refiner()


def analyze_cproto(args, cfg, dataset_dir: Path, output_dir: Path):
    val_anno_path = (
        "/home/uqdetche/lidar_longtail_mining/lion/data/argo2/val_anno.feather"
    )

    val_data_path = Path(
        "/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val"
    )

    log_ids = [entry.name for entry in val_data_path.iterdir() if entry.is_dir()]

    cproto_log_ids = []

    for log_id in log_ids:
        cproto_output_path = get_cproto_output_path(output_dir, log_id)

        if cproto_output_path.exists():
            print(f"{cproto_output_path=}")
            cproto_log_ids.append(log_id)

    # cproto_log_ids = ['25e5c600-36fe-3245-9cc0-40ef91620c22']
    # cproto_log_ids = cproto_log_ids[:5]

    print(f"Found {len(cproto_log_ids)} cproto_log_ids")

    vis_log_id = np.random.choice(cproto_log_ids)
    print(f"{vis_log_id=}")

    vis_cproto_output_path = get_cproto_output_path(output_dir, vis_log_id)

    with open(vis_cproto_output_path, "rb") as f:
        cpd_infos = pkl.load(f)

    # pprint(cpd_infos[0])

    vis_annotations_feather = Path(
        f"/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val/{vis_log_id}/annotations.feather"
    )
    assert vis_annotations_feather.exists()

    gts = pd.read_feather(vis_annotations_feather)

    # add log_id
    gts = gts.assign(
        log_id=pd.Series([vis_log_id for _ in range(len(gts))], dtype="string").values
    )

    gts_timestamps = gts["timestamp_ns"].values

    print("gts_timestamps", gts_timestamps.min(), gts_timestamps.max())

    # analysis = analyze_object_data(vis_cproto_output_path)
    # gts = pd.read_feather(Path(val_anno_path))
    gts = gts.set_index(["log_id", "timestamp_ns"], drop=False).sort_values("category")

    # Create BEV plot with trajectories
    outline_infos = load_and_plot_objects(
        vis_cproto_output_path,
        use_first_frame=False,  # Use frame with most objects
        plot_trajectories=False,
        gts_dataframe=gts.copy(),  # Add GT overlay
        save_path="cpd_bev_visualization.png",
        log_id=vis_log_id,
    )

    gts = pd.read_feather(val_anno_path)
    gts = gts.set_index(["log_id", "timestamp_ns"], drop=False).sort_values("category")

    category_mapping = {
        "Vehicle": SensorCompetitionCategories.REGULAR_VEHICLE.value,
        "Cyclist": SensorCompetitionCategories.BICYCLE.value,
        "Pedestrian": SensorCompetitionCategories.PEDESTRIAN.value,
    }

    features_output_path = output_dir / "cpd_features.feather"
    # features_output_path = output_dir / f"cpd_features_aug14.feather"

    # Extract features with accurate CSS re-computation
    if not features_output_path.exists():
        serialized_dts_list = []
        for log_id in tqdm(cproto_log_ids, desc="Extracting all features"):
            cproto_output_path = get_cproto_output_path(output_dir, log_id)

            cur_features_output_path = output_dir / log_id / "cpd_features.feather"

            # TODO: COMMENT OUT LATER
            # if not cur_features_output_path.exists():
            #     continue

            # TODO: COMMENT OUT LATER
            if cur_features_output_path.exists():
                os.remove(cur_features_output_path)

            if not cur_features_output_path.exists():
                pr = cProfile.Profile()
                pr.enable()

                df_features = extract_all_features_accurate(
                    output_info_path=str(cproto_output_path),
                    log_id=log_id,
                    config=cfg,  # Pass the same config used in C_PROTO
                    root_path=str(output_dir),
                    output_path=str(cur_features_output_path),
                    category_mapping=category_mapping,
                    gts_dataframe=gts.copy(),
                    score_thresh=-1,  # no score thresh
                )
                pr.disable()

                import io
                import pstats

                s = io.StringIO()
                sortby = "cumtime"
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                ps.print_stats()
                with open(
                    "cpd_analyzer_extract_all_features_accurate_profile_stats.txt", "w"
                ) as f:
                    f.write(s.getvalue())
            else:
                df_features = pd.read_feather(cur_features_output_path)

            serialized_dts_list.append(df_features)

        df_features = pd.concat(serialized_dts_list, ignore_index=True)

        # Save if path provided
        if features_output_path is not None:
            if not features_output_path.suffix == ".feather":
                features_output_path = features_output_path.with_suffix(".feather")
            df_features.to_feather(features_output_path)
            print(f"\nSaved to {features_output_path}")
    else:
        df_features = pd.read_feather(features_output_path)

    print("unique log_ids", df_features["log_id"].unique())
    print("unique log_ids", df_features["log_id"].unique().__len__())
    print("unique timestamps", df_features["timestamp_ns"].unique().__len__())
    print("unique object ids", df_features["object_id"].unique().__len__())
    print("unique proto ids", df_features["proto_id"].unique().__len__())
    print("unique track uuids", df_features["track_uuid"].unique().__len__())

    track_uuids = df_features["track_uuid"].unique()
    print(
        "df_features['track_uuid'].value_counts()",
        df_features["track_uuid"].value_counts(),
    )
    print("track_uuids", track_uuids.shape, track_uuids.min(), track_uuids.max())

    vc = df_features["track_uuid"].value_counts().value_counts()

    plt.figure(figsize=(6, 4))
    ax = vc.plot(kind="bar", color="skyblue")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Distribution of Value Counts")

    # Stagger x-axis ticks: only show ~10 evenly spaced ticks
    n_ticks = 10
    all_ticks = np.arange(len(vc))
    tick_positions = all_ticks[:: max(1, len(vc) // n_ticks)]
    tick_labels = vc.index[tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.tight_layout()
    plt.savefig("value_counts_barplot.png")
    plt.close()

    dist_normalized = (
        df_features["track_uuid"].value_counts().value_counts(normalize=True)
    )
    print(dist_normalized)
    # exit()

    print(f"\nTotal features extracted: {len(df_features.columns)}")
    print("Feature columns:", df_features.columns.tolist())

    # model_results = train_rf_model(df_features)

    # pprint(model_results)

    max_depth = 5

    dt_result = train_interpretable_tree(df_features, max_depth=max_depth)
    pprint(dt_result)

    extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("cpd_decision_tree.png")

    dt_result = train_interpretable_classifier(
        df_features, max_depth=max_depth, iou_threshold=0.1
    )
    pprint(dt_result)

    # extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("cpd_tp_decision_tree.png")

    dt_result = train_interpretable_tree_category_classifier(
        df_features, max_depth=max_depth
    )
    pprint(dt_result)

    # extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
        class_names=dt_result["unique_classes"],
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("cpd_tp_decision_tree_category.png")
    # # Convert your data to Argoverse 2 format
    # df = convert_to_argoverse2(
    #     output_info_path=cproto_output_path,
    #     log_id=log_id,
    #     output_path=cproto_argo_path,
    #     category_mapping=category_mapping
    # )

    # # Validate the format
    # validate_argoverse2_format(df)

    # gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")
    # df = df.set_index(["log_id", "timestamp_ns"]).sort_values("category")

    # valid_uuids_gts = gts.index.tolist()
    # valid_uuids_dts = df.index.tolist()
    # valid_uuids = set(valid_uuids_gts) & set(valid_uuids_dts)
    # print("valid_uuids", valid_uuids)
    # gts = gts.loc[list(valid_uuids)].sort_index()

    # categories = set(x.value for x in SensorCompetitionCategories)
    # categories &= set(gts["category"].unique().tolist())

    # from av2.evaluation.detection.utils import DetectionCfg

    # cfg = DetectionCfg(
    #     dataset_dir=Path(dataset_dir/'val'),
    #     categories=tuple(sorted(categories)),
    #     max_range_m=150,
    #     eval_only_roi_instances=True,
    # )

    # from av2.evaluation.detection.eval import evaluate, evaluate_hierarchy

    # # Evaluate using Argoverse detection API.
    # eval_dts, eval_gts, metrics = evaluate(
    #     df.reset_index(), gts.reset_index(), cfg
    # )

    # valid_categories = sorted(categories) + ["AVERAGE_METRICS"]
    # ap_dict = {}
    # for index, row in metrics.iterrows():
    #     ap_dict[index] = row.to_json()

    # print('ap_dict', ap_dict)

    # with open("cpd_infos_0.json", 'w') as f:
    #     json.dump(cpd_infos[0], f, cls=NumpyEncoder, indent=2)


def analyze_owl_alpha_shapes(args, cfg, dataset_dir: Path, output_dir: Path):
    val_anno_path = (
        "/home/uqdetche/lidar_longtail_mining/lion/data/argo2/val_anno.feather"
    )
    val_data_path = Path(
        "/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val"
    )

    log_ids = [entry.name for entry in val_data_path.iterdir() if entry.is_dir()]

    owlvit_alpha_shape_log_ids = []

    for log_id in log_ids:
        output_path = get_owl_alpha_shape_output_path(output_dir, log_id)

        # if output_path.exists():
        #     os.remove(output_path)

        if output_path.exists():
            print(f"{output_path=}")

            owlvit_alpha_shape_log_ids.append(log_id)

    print(f"existing {len(owlvit_alpha_shape_log_ids)=}")

    owlvit_predictions_dir = Path("/media/local-data/uqdetche/argo2_val_owlv2preds/")
    owlvit_log_ids = [x.name for x in owlvit_predictions_dir.iterdir()]
    print(f"{len(owlvit_log_ids)=}")
    # exit()


    not_done_log_ids = list(set(owlvit_log_ids).difference(owlvit_alpha_shape_log_ids))
    not_done_log_ids = sorted(not_done_log_ids)
    print(f"not_done_log_ids {len(not_done_log_ids)}")

    vis_log_id = '42f92807-0c5e-3397-bd45-9d5303b4db2a'
    # vis_log_id = 'c2d44a70-9fd4-3298-ad0a-c4c9712e6f1e'
    vis_log_id = '27c03d98-6ac3-38a3-ba5e-102b184d01ef'
    vis_log_id = 'e72ef05c-8b94-3885-a34f-fff3b2b954b4'
    vis_log_id = np.random.choice(not_done_log_ids)

    if len(owlvit_alpha_shape_log_ids) == 0 or True:
        # for log_id in [np.random.choice(not_done_log_ids)]:  # TODO: do all
        for log_id in [vis_log_id]:
            # pr = cProfile.Profile()
            # pr.enable()
            print(f"running on {log_id=}")
            alpha_shape = OWLViTAlphaShapeMFCF(
                log_id,
                root_path=str(output_dir),
                owlvit_predictions_dir=owlvit_predictions_dir,
                config=cfg,
                debug=True,
            )
            alpha_shape()

            vis_log_id = log_id

            # pr.disable()

            # import pstats, io

            # s = io.StringIO()
            # sortby = "cumtime"
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # with open("OWLViTAlphaShapeMFCF_cprofile.txt", "w") as f:
            #     f.write(s.getvalue())

            owlvit_alpha_shape_log_ids.append(log_id)

    print(f"Found {len(owlvit_alpha_shape_log_ids)} owlvit_alpha_shape_log_ids")

    vis_log_id = np.random.choice(owlvit_alpha_shape_log_ids) if vis_log_id is None else vis_log_id
    print(f"{vis_log_id=}")

    vis_path = get_owl_alpha_shape_output_path(output_dir, vis_log_id)

    vis_annotations_feather = Path(
        f"/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val/{vis_log_id}/annotations.feather"
    )
    assert vis_annotations_feather.exists()

    gts = pd.read_feather(vis_annotations_feather)
    gts = gts.assign(
        log_id=pd.Series([vis_log_id for _ in range(len(gts))], dtype="string").values
    )
    gts = gts.set_index(["log_id", "timestamp_ns"], drop=False).sort_values("category")


    outline_infos = load_and_plot_objects(
        vis_path,
        use_first_frame=False,  # Use frame with most objects
        plot_trajectories=True,
        gts_dataframe=gts.copy(),  # Add GT overlay
        save_path="owlvit_alpha_shape_box_bev_visualization.png",
        log_id=vis_log_id,
    )

    # Create BEV plot with trajectories
    outline_infos = load_and_plot_alpha_shapes(
        vis_path,
        use_first_frame=False,  # Use frame with most objects
        plot_trajectories=True,
        gts_dataframe=gts.copy(),  # Add GT overlay
        save_path="owlvit_alpha_shape_bev_visualization.png",
        log_id=vis_log_id,
    )

    outline_infos = load_and_plot_alpha_shapes_camera(
        vis_path,
        use_first_frame=False,  # Use frame with most objects
        plot_trajectories=True,
        gts_dataframe=gts.copy(),  # Add GT overlay
        save_path="owlvit_alpha_shape_bev_visualization.png",
        log_id=vis_log_id,
    )

    alpha_shape_vis_path = Path("./owlvit_alpha_shapes_vis")
    alpha_shape_vis_path.mkdir(exist_ok=True)

    for frame in range(20):
        save_path = alpha_shape_vis_path / f"frame_{frame}.png"
        load_and_plot_alpha_shapes(
            vis_path,
            use_first_frame=False,  # Use frame with most objects
            ref_frame_idx=frame,
            plot_trajectories=True,
            gts_dataframe=gts.copy(),  # Add GT overlay
            save_path=save_path,
            log_id=vis_log_id,
        )

    gts = pd.read_feather(val_anno_path)
    gts = gts.set_index(["log_id", "timestamp_ns"], drop=False).sort_values("category")

    category_mapping = {
        "Vehicle": SensorCompetitionCategories.REGULAR_VEHICLE.value,
        "Cyclist": SensorCompetitionCategories.BICYCLE.value,
        "Pedestrian": SensorCompetitionCategories.PEDESTRIAN.value,
    }

    features_output_path = output_dir / "owlvit_alpha_shape_features.feather"
    # Extract features with accurate CSS re-computation
    if not features_output_path.exists():
        serialized_dts_list = []
        for log_id in tqdm(owlvit_alpha_shape_log_ids, desc="Extracting all features"):
            cproto_output_path = get_owl_alpha_shape_output_path(output_dir, log_id)

            cur_features_output_path = (
                output_dir / log_id / "owlvit_alpha_shape_features.feather"
            )

            # TODO: COMMENT OUT LATER
            if cur_features_output_path.exists():
                os.remove(cur_features_output_path)

            if not cur_features_output_path.exists():
                df_features = extract_all_features_accurate(
                    output_info_path=str(cproto_output_path),
                    log_id=log_id,
                    config=cfg,  # Pass the same config used in C_PROTO
                    root_path=str(output_dir),
                    output_path=str(cur_features_output_path),
                    category_mapping=category_mapping,
                    gts_dataframe=gts.copy(),
                    score_thresh=-1,  # no score thresh
                    # use_first_frame=True
                )
            else:
                df_features = pd.read_feather(cur_features_output_path)

            serialized_dts_list.append(df_features)

        df_features = pd.concat(serialized_dts_list, ignore_index=True)

        print("Actual dtypes in DataFrame:")
        print(df_features.dtypes.value_counts())
        print("\nUnique dtypes:")
        print(df_features.dtypes.unique())

        # # Correct way to exclude string-like columns
        # non_string_cols = df_features.select_dtypes(exclude=['object']).columns.tolist()
        # print(f"{non_string_cols=}")

        # non_string_cols = df_features.select_dtypes(exclude=['unicode', 'string']).columns.tolist()
        # print(f"{non_string_cols=}")

        # Save if path provided
        # if features_output_path is not None:
        #     if not features_output_path.suffix == ".feather":
        #         features_output_path = features_output_path.with_suffix(".feather")
        #     df_features.to_feather(features_output_path)
        #     print(f"\nSaved to {features_output_path}")
    else:
        df_features = pd.read_feather(features_output_path)

    print("unique log_ids", df_features["log_id"].unique())
    print("unique log_ids", df_features["log_id"].unique().__len__())
    print("unique timestamps", df_features["timestamp_ns"].unique().__len__())
    print("unique object ids", df_features["object_id"].unique().__len__())
    print("unique proto ids", df_features["proto_id"].unique().__len__())
    print("unique track uuids", df_features["track_uuid"].unique().__len__())

    track_uuids = df_features["track_uuid"].unique()
    print(
        "df_features['track_uuid'].value_counts()",
        df_features["track_uuid"].value_counts(),
    )
    print("track_uuids", track_uuids.shape, track_uuids.min(), track_uuids.max())

    vc = df_features["track_uuid"].value_counts().value_counts()

    plt.figure(figsize=(6, 4))
    ax = vc.plot(kind="bar", color="skyblue")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Distribution of Value Counts")

    # Stagger x-axis ticks: only show ~10 evenly spaced ticks
    n_ticks = 10
    all_ticks = np.arange(len(vc))
    tick_positions = all_ticks[:: max(1, len(vc) // n_ticks)]
    tick_labels = vc.index[tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.tight_layout()
    plt.savefig("owl_alpha_shape_value_counts_barplot.png")
    plt.close()

    dist_normalized = (
        df_features["track_uuid"].value_counts().value_counts(normalize=True)
    )
    print(dist_normalized)
    # exit()

    print(f"\nTotal features extracted: {len(df_features.columns)}")
    print("Feature columns:", df_features.columns.tolist())

    # model_results = train_rf_model(df_features)

    # pprint(model_results)

    max_depth = 5

    dt_result = train_interpretable_tree(df_features, max_depth=max_depth)
    # pprint(dt_result)

    extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("owl_alpha_shape_decision_tree.png")

    dt_result = train_interpretable_classifier(
        df_features, max_depth=max_depth, iou_threshold=0.1
    )
    # pprint(dt_result)

    # extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("owl_alpha_shape_tp_decision_tree.png")

    dt_result = train_interpretable_tree_category_classifier(
        df_features, max_depth=max_depth
    )
    # pprint(dt_result)

    # extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
        class_names=dt_result["unique_classes"],
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("owl_alpha_shape_tp_decision_tree_category.png")

def analyze_alpha_shape(args, cfg, dataset_dir: Path, output_dir: Path):
    val_anno_path = (
        "/home/uqdetche/lidar_longtail_mining/lion/data/argo2/val_anno.feather"
    )

    val_data_path = Path(
        "/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val"
    )

    log_ids = [entry.name for entry in val_data_path.iterdir() if entry.is_dir()]
    print(log_ids[:10])

    alpha_shape_log_ids = []

    for log_id in log_ids:
        output_path = get_alpha_shape_output_path(output_dir, log_id)

        if output_path.exists():
            print(f"{output_path=}")

            alpha_shape_log_ids.append(log_id)

    print("existing alpha_shape_log_ids", alpha_shape_log_ids)
    # exit()

    if len(alpha_shape_log_ids) == 0:
        for log_id in np.random.choice(log_ids, 10):  # TODO: do all
            pr = cProfile.Profile()
            pr.enable()
            alpha_shape = AlphaShapeMFCF(
                log_id, root_path=str(output_dir), config=cfg, debug=True
            )
            alpha_shape()

            pr.disable()

            import io
            import pstats

            s = io.StringIO()
            sortby = "cumtime"
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            with open("AlphaShapeMFCF_cprofile.txt", "w") as f:
                f.write(s.getvalue())

    print(f"Found {len(alpha_shape_log_ids)} alpha_shape_log_ids")

    vis_log_id = np.random.choice(alpha_shape_log_ids)
    print(f"{vis_log_id=}")

    vis_path = get_alpha_shape_output_path(output_dir, vis_log_id)

    vis_annotations_feather = Path(
        f"/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor/val/{vis_log_id}/annotations.feather"
    )
    assert vis_annotations_feather.exists()

    gts = pd.read_feather(vis_annotations_feather)
    gts = gts.assign(
        log_id=pd.Series([vis_log_id for _ in range(len(gts))], dtype="string").values
    )
    gts = gts.set_index(["log_id", "timestamp_ns"], drop=False).sort_values("category")

    # Create BEV plot with trajectories
    outline_infos = load_and_plot_alpha_shapes(
        vis_path,
        use_first_frame=True,  # Use frame with most objects
        plot_trajectories=True,
        gts_dataframe=gts.copy(),  # Add GT overlay
        save_path="alpha_shape_bev_visualization.png",
        log_id=vis_log_id,
    )

    alpha_shape_vis_path = Path("./alpha_shapes_vis")
    alpha_shape_vis_path.mkdir(exist_ok=True)

    for frame in range(20):
        save_path = alpha_shape_vis_path / f"frame_{frame}.png"
        load_and_plot_alpha_shapes(
            vis_path,
            use_first_frame=False,  # Use frame with most objects
            ref_frame_idx=frame,
            plot_trajectories=True,
            gts_dataframe=gts.copy(),  # Add GT overlay
            save_path=save_path,
            log_id=vis_log_id,
        )

    outline_infos = load_and_plot_objects(
        vis_path,
        use_first_frame=True,  # Use frame with most objects
        plot_trajectories=True,
        gts_dataframe=gts.copy(),  # Add GT overlay
        save_path="alpha_shape_box_bev_visualization.png",
        log_id=vis_log_id,
    )

    return

    gts = pd.read_feather(val_anno_path)
    gts = gts.set_index(["log_id", "timestamp_ns"], drop=False).sort_values("category")

    category_mapping = {
        "Vehicle": SensorCompetitionCategories.REGULAR_VEHICLE.value,
        "Cyclist": SensorCompetitionCategories.BICYCLE.value,
        "Pedestrian": SensorCompetitionCategories.PEDESTRIAN.value,
    }

    features_output_path = output_dir / "alpha_shape_features.feather"
    # Extract features with accurate CSS re-computation
    if not features_output_path.exists():
        serialized_dts_list = []
        for log_id in tqdm(alpha_shape_log_ids, desc="Extracting all features"):
            cproto_output_path = get_alpha_shape_output_path(output_dir, log_id)

            cur_features_output_path = (
                output_dir / log_id / "alpha_shape_features.feather"
            )

            # TODO: COMMENT OUT LATER
            if cur_features_output_path.exists():
                os.remove(cur_features_output_path)

            if not cur_features_output_path.exists():
                df_features = extract_all_features_accurate(
                    output_info_path=str(cproto_output_path),
                    log_id=log_id,
                    config=cfg,  # Pass the same config used in C_PROTO
                    root_path=str(output_dir),
                    output_path=str(cur_features_output_path),
                    category_mapping=category_mapping,
                    gts_dataframe=gts.copy(),
                    score_thresh=-1,  # no score thresh
                )
            else:
                df_features = pd.read_feather(cur_features_output_path)

            serialized_dts_list.append(df_features)

        df_features = pd.concat(serialized_dts_list, ignore_index=True)

        # Save if path provided
        if features_output_path is not None:
            if not features_output_path.suffix == ".feather":
                features_output_path = features_output_path.with_suffix(".feather")
            df_features.to_feather(features_output_path)
            print(f"\nSaved to {features_output_path}")
    else:
        df_features = pd.read_feather(features_output_path)

    print("unique log_ids", df_features["log_id"].unique())
    print("unique log_ids", df_features["log_id"].unique().__len__())
    print("unique timestamps", df_features["timestamp_ns"].unique().__len__())
    print("unique object ids", df_features["object_id"].unique().__len__())
    print("unique proto ids", df_features["proto_id"].unique().__len__())
    print("unique track uuids", df_features["track_uuid"].unique().__len__())

    track_uuids = df_features["track_uuid"].unique()
    print(
        "df_features['track_uuid'].value_counts()",
        df_features["track_uuid"].value_counts(),
    )
    print("track_uuids", track_uuids.shape, track_uuids.min(), track_uuids.max())

    vc = df_features["track_uuid"].value_counts().value_counts()

    plt.figure(figsize=(6, 4))
    ax = vc.plot(kind="bar", color="skyblue")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Distribution of Value Counts")

    # Stagger x-axis ticks: only show ~10 evenly spaced ticks
    n_ticks = 10
    all_ticks = np.arange(len(vc))
    tick_positions = all_ticks[:: max(1, len(vc) // n_ticks)]
    tick_labels = vc.index[tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.tight_layout()
    plt.savefig("alpha_shape_value_counts_barplot.png")
    plt.close()

    dist_normalized = (
        df_features["track_uuid"].value_counts().value_counts(normalize=True)
    )
    print(dist_normalized)
    # exit()

    print(f"\nTotal features extracted: {len(df_features.columns)}")
    print("Feature columns:", df_features.columns.tolist())

    # model_results = train_rf_model(df_features)

    # pprint(model_results)

    max_depth = 5

    dt_result = train_interpretable_tree(df_features, max_depth=max_depth)
    # pprint(dt_result)

    extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("alpha_shape_decision_tree.png")

    dt_result = train_interpretable_classifier(
        df_features, max_depth=max_depth, iou_threshold=0.1
    )
    # pprint(dt_result)

    # extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("alpha_shape_tp_decision_tree.png")

    dt_result = train_interpretable_tree_category_classifier(
        df_features, max_depth=max_depth
    )
    # pprint(dt_result)

    # extract_decision_rules(dt_result)

    dt_model = dt_result["model"]
    feature_cols = dt_result["feature_cols"]

    plt.figure(figsize=(25, 12))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
        class_names=dt_result["unique_classes"],
    )  # Limit depth for readability
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig("alpha_shape_tp_decision_tree_category.png")


def main():
    # Register the signal handler
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    args, cfg = parse_config()

    output_dir = cfg.ROOT_DIR / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # print("output_dir", str(output_dir))

    dataset_dir = Path("/home/uqdetche/lidar_longtail_mining/lion/data/argo2/sensor")

    analyze_owl_alpha_shapes(args, cfg, dataset_dir, output_dir)
    # analyze_alpha_shape(args, cfg, dataset_dir, output_dir)
    # analyze_cproto(args, cfg, dataset_dir, output_dir)
    # analyze_mfcf(args, cfg, dataset_dir, output_dir)
    # create_infos(args, cfg, dataset_dir, output_dir)


if __name__ == "__main__":
    main()
