from typing import Tuple
import numpy as np
import open3d as o3d

from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import DBSCAN
from lion.unsupervised_core.outline_utils import points_rigid_transform

def quat_to_yaw(qw, qx, qy, qz):
    """Convert quaternion to yaw angle.
    
    Args:
        qw, qx, qy, qz: Quaternion components
        
    Returns:
        float: Yaw angle in radians
    """
    # For rotation around Z-axis: yaw = 2 * atan2(qz, qw)
    return 2 * np.arctan2(qz, qw)


def get_rotated_box(center_xy, length, width, yaw):
    """Return 4 corners of rotated rectangle (BEV)"""
    dx = length / 2
    dy = width / 2
    corners = np.array(
        [
            [dx, dy],
            [dx, -dy],
            [-dx, -dy],
            [-dx, dy],
        ]
    )
    rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rotated = (rot_mat @ corners.T).T
    return rotated + center_xy

def get_rotated_3d_box_corners(bbox):
    """Return 4 corners of rotated rectangle (BEV)"""
    center_xy = bbox[:2]
    center_z = bbox[2]
    length = bbox[3]
    width = bbox[4]
    height = bbox[5]
    yaw = bbox[6]

    dx = length / 2
    dy = width / 2
    corners = np.array(
        [
            [dx, dy],
            [dx, -dy],
            [-dx, -dy],
            [-dx, dy],
        ]
    )
    rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rotated = (rot_mat @ corners.T).T
    xy_corners = rotated + center_xy

    z1, z2 = center_z - height / 2, center_z + height / 2

    xyz_corners_lower = np.concatenate([xy_corners, np.full((4, 1), fill_value=z1)], axis=1)
    xyz_corners_upper = np.concatenate([xy_corners, np.full((4, 1), fill_value=z2)], axis=1)

    xyz_corners = np.concatenate([xyz_corners_lower, xyz_corners_upper], axis=0)

    return xyz_corners


def apply_pose_to_box(gt_row, pose):
    """
    Apply 3D pose transformation to box parameters for BEV plotting

    Args:
        gt_row: Row containing box parameters (tx_m, ty_m, tz_m, qw, qx, qy, qz, etc.)
        pose: 4x4 transformation matrix

    Returns:
        transformed_center_xy: [x, y] center position after transformation
        transformed_yaw: yaw angle after transformation
        length, width: unchanged dimensions
    """

    # Extract original box parameters
    center_xyz = np.array([gt_row["tx_m"], gt_row["ty_m"], gt_row["tz_m"]])
    length = gt_row["length_m"]
    width = gt_row["width_m"]

    # Convert quaternion to yaw
    qw, qx, qy, qz = gt_row["qw"], gt_row["qx"], gt_row["qy"], gt_row["qz"]
    original_yaw = quat_to_yaw(qw, qx, qy, qz)

    # Method 1: Transform center position using your existing function
    transformed_center_xyz = points_rigid_transform(
        center_xyz.reshape(1, 3), pose
    ).reshape(3)
    transformed_center_xy = [transformed_center_xyz[0], transformed_center_xyz[1]]

    # Method 2: Transform yaw using rotation matrix from pose
    # Extract rotation matrix from pose (3x3 upper-left)
    rotation_matrix = pose[:3, :3]

    # Create a unit vector in the original yaw direction
    original_direction = np.array([np.cos(original_yaw), np.sin(original_yaw), 0])

    # Transform the direction vector
    transformed_direction = rotation_matrix @ original_direction

    # Calculate new yaw from transformed direction
    transformed_yaw = np.arctan2(transformed_direction[1], transformed_direction[0])

    return transformed_center_xy, transformed_yaw, length, width

def compute_plane_equation(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute plane equation (normal, d) from 3 points.
    Plane equation: normal · x + d = 0
    """
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # normalize
    d = -np.dot(normal, p1)
    return normal, d

def points_in_frustum(lidar_points: np.ndarray, frustum_corners: np.ndarray) -> np.ndarray:
    """
    Find which lidar points are inside the frustum defined by 8 corners.
    
    Args:
        lidar_points: (N, 3) array of lidar points in ego coordinates
        frustum_corners: (8, 3) array of frustum corners from _get_box_frustum_corners_in_ego
                        First 4 corners are near plane, last 4 are far plane
                        Corners should be ordered: [bottom-left, bottom-right, top-right, top-left]
    
    Returns:
        Boolean mask (N,) indicating which points are inside the frustum
    """
    near_corners = frustum_corners[:4]
    far_corners = frustum_corners[4:]
    frustum_center = np.mean(frustum_corners, axis=0)
    
    # Face definitions with points ordered for outward-pointing normals
    faces_info = [
        ("near", [near_corners[0], near_corners[1], near_corners[2]]),
        ("far", [far_corners[0], far_corners[3], far_corners[2]]),  
        ("left", [near_corners[0], near_corners[3], far_corners[3]]),
        ("right", [near_corners[1], far_corners[1], far_corners[2]]),
        ("bottom", [near_corners[0], far_corners[0], far_corners[1]]),
        ("top", [near_corners[3], near_corners[2], far_corners[2]])
    ]
    
    n_points = lidar_points.shape[0]
    inside_mask = np.ones(n_points, dtype=bool)
    
    for face_name, face_points in faces_info:
        normal, d = compute_plane_equation(face_points[0], face_points[1], face_points[2])
        
        # Ensure normal points inward by checking against center
        face_center = np.mean(face_points, axis=0)
        to_frustum_center = frustum_center - face_center
        
        # If normal points away from frustum center, flip it
        if np.dot(normal, to_frustum_center) < 0:
            normal = -normal
            d = -d
        
        # Points inside should be on positive side of inward-pointing normal
        distances = np.dot(lidar_points, normal) + d
        inside_mask &= (distances > 0)
        
        # Debug
        # print(f"{face_name}: normal={normal}, center_check={np.dot(normal, to_frustum_center)}")
    
    return inside_mask


def find_connected_components_lidar(points, threshold=0.5):
    """
    Find connected components in a LiDAR point cloud using distance threshold.
    
    Parameters:
    -----------
    points : np.ndarray
        Nx3 array of 3D coordinates (x, y, z)
    threshold : float
        Distance threshold for connectivity (e.g., 0.5 meters)
        
    Returns:
    --------
    labels : np.ndarray
        Component label for each point (same component = same label)
    n_components : int
        Number of connected components found
    """
    # Use KDTree-based neighbor search for efficiency
    nbrs = NearestNeighbors(radius=threshold, algorithm='kd_tree')
    nbrs.fit(points)
    
    # Get sparse adjacency matrix (points connected if within threshold)
    adjacency = nbrs.radius_neighbors_graph(points, mode='connectivity')
    
    # Find connected components using efficient scipy algorithm
    n_components, labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    
    return labels, n_components

def fast_connected_components(points, eps=0.5):
    """Faster connected components using DBSCAN or KDTree."""
    # Use DBSCAN for larger point clouds (optimized C implementation)
    clustering = DBSCAN(eps=eps, min_samples=1, n_jobs=1, algorithm='kd_tree').fit(points)
    labels = clustering.labels_
    # n_components = len(set(labels)) - (1 if -1 in labels else 0)
    n_components = labels.max() + 1
    
    return labels, n_components


def box_iou(boxes1, boxes2):
    """
    Compute IoU (Intersection over Union) between two sets of 2D bounding boxes.
    
    Args:
        boxes1: np.ndarray of shape (N, 4) containing N bounding boxes in format [x1, y1, x2, y2]
        boxes2: np.ndarray of shape (M, 4) containing M bounding boxes in format [x1, y1, x2, y2]
    
    Returns:
        np.ndarray of shape (N, M) where element (i,j) is IoU between boxes1[i] and boxes2[j]
    """
    # Ensure inputs are numpy arrays
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)
    
    # Validate input shapes
    if boxes1.ndim != 2 or boxes1.shape[1] != 4:
        raise ValueError(f"boxes1 must have shape (N, 4), got {boxes1.shape}")
    if boxes2.ndim != 2 or boxes2.shape[1] != 4:
        raise ValueError(f"boxes2 must have shape (M, 4), got {boxes2.shape}")
    
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # Expand dimensions for broadcasting: boxes1 -> (N, 1, 4), boxes2 -> (1, M, 4)
    boxes1_expanded = boxes1[:, np.newaxis, :]  # (N, 1, 4)
    boxes2_expanded = boxes2[np.newaxis, :, :]  # (1, M, 4)
    
    # Calculate intersection coordinates
    # Top-left corner of intersection: max of top-left corners
    inter_x1 = np.maximum(boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0])
    inter_y1 = np.maximum(boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1])
    
    # Bottom-right corner of intersection: min of bottom-right corners  
    inter_x2 = np.minimum(boxes1_expanded[:, :, 2], boxes2_expanded[:, :, 2])
    inter_y2 = np.minimum(boxes1_expanded[:, :, 3], boxes2_expanded[:, :, 3])
    
    # Calculate intersection area
    inter_width = np.maximum(0, inter_x2 - inter_x1)
    inter_height = np.maximum(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    
    # Calculate areas of both boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    
    # Broadcast areas for union calculation
    area1_expanded = area1[:, np.newaxis]  # (N, 1)
    area2_expanded = area2[np.newaxis, :]  # (1, M)
    
    # Calculate union area
    union = area1_expanded + area2_expanded - intersection
    
    # Calculate IoU, avoiding division by zero
    iou = np.where(union > 0, intersection / union, 0)
    
    return iou


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B.
    Inputs:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      R: mxm rotation matrix
      t: mx1 translation vector
    """
    assert A.shape == B.shape

    # Get number of dimensions
    m = A.shape[1]

    # Translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Rotation matrix
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = centroid_B.T - R @ centroid_A.T

    return R, t


def icp(
    A,
    B,
    max_iterations=100,
    tolerance=1e-6,
    ret_err=False,
    fixed_indices=False,
    return_inliers=False,
):
    """
    Iterative Closest Point (ICP) algorithm: aligns point cloud A to point cloud B.
    Returns the final transformation from the original A to B.

    Args:
        A: Source point cloud (Nx3)
        B: Target point cloud (Mx3)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        ret_err: Whether to return the final error
        fixed_indices: Whether to use fixed correspondences
        return_inliers: Whether to return inlier indices

    Returns:
        Various combinations based on flags:
        - (R_final, t_final) if neither ret_err nor return_inliers
        - (R_final, t_final, A_transformed) if not ret_err and not return_inliers
        - (R_final, t_final, error) if ret_err and not return_inliers
        - (R_final, t_final, error, A_inliers, B_inliers) if ret_err and return_inliers
    """
    A = np.copy(A)
    B = np.copy(B)
    A_original = np.copy(A)  # Keep original for inlier calculation

    prev_error = float("inf")

    # Initialize final transformation
    R_final = np.eye(A.shape[1])
    t_final = np.zeros(A.shape[1])

    tree = cKDTree(B) if not fixed_indices else None
    final_indices = None
    final_distances = None

    for i in range(max_iterations):
        # Find the nearest neighbors in B for each point in A
        if not fixed_indices:
            distances, indices = tree.query(A)
        else:
            indices = np.arange(len(B))
            distances = np.linalg.norm(A - B, axis=1)

        # Store final correspondences
        final_indices = indices
        final_distances = distances

        # Compute the transformation
        R, t = best_fit_transform(A, B[indices])

        # Update the final transformation
        R_final = R @ R_final
        t_final = R @ t_final + t

        # Apply the transformation
        A = (R @ A.T).T + t

        # Check for convergence
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    # Handle return values based on flags
    if return_inliers and final_indices is not None and final_distances is not None:
        # Define inliers based on distance threshold
        # Use a reasonable threshold - you can adjust this based on your needs
        distance_threshold = np.percentile(
            final_distances, 75
        )  # Top 75% closest points
        # Alternatively, use a fixed threshold: distance_threshold = 0.1  # 10cm threshold

        inlier_mask = final_distances <= distance_threshold
        A_inlier_indices = np.where(inlier_mask)[0]
        B_inlier_indices = final_indices[inlier_mask]

        if ret_err:
            return R_final, t_final, prev_error, A_inlier_indices, B_inlier_indices
        else:
            return R_final, t_final, A_inlier_indices, B_inlier_indices

    if ret_err:
        return R_final, t_final, prev_error

    return R_final, t_final, A

def icp_open3d_robust(source_points, target_points, 
                     voxel_size=None,
                     max_correspondence_dist=None,
                     max_iterations=50,
                     use_point_to_plane=False,
                     initial_alignment='none',  # 'none', 'ransac', or 4x4 matrix
                     return_full_result=False):
    """
    Robust ICP alignment using Open3D with optional initial alignment.
    
    Args:
        source_points: Source point cloud (Nx3 numpy array)
        target_points: Target point cloud (Mx3 numpy array)  
        voxel_size: Voxel size for downsampling (None = auto-compute from point cloud)
        max_correspondence_dist: Max correspondence distance (None = auto-compute)
        max_iterations: Maximum number of ICP iterations
        use_point_to_plane: Use point-to-plane ICP (requires normals)
        initial_alignment: 'none', 'ransac', or a 4x4 transformation matrix
        return_full_result: Return full result object with more metrics
    
    Returns:
        transform: 4x4 transformation matrix (numpy array)
        error: Final RMSE error
        (optionally) result: Full Open3D result object if return_full_result=True
    """
    # Convert to Open3D point clouds
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    
    # Auto-compute voxel size if not provided
    if voxel_size is None:
        source_bb = source.get_axis_aligned_bounding_box()
        target_bb = target.get_axis_aligned_bounding_box()
        voxel_size = max(source_bb.get_max_extent(), 
                         target_bb.get_max_extent()) * 0.01
    
    # Auto-compute max correspondence distance if not provided
    if max_correspondence_dist is None:
        max_correspondence_dist = voxel_size * 3.0
    
    # Compute normals if using point-to-plane
    if use_point_to_plane:
        search_radius = voxel_size * 2.0
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30))
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30))
    
    # Handle initial alignment
    if isinstance(initial_alignment, np.ndarray):
        init_transform = initial_alignment
    elif initial_alignment == 'ransac':
        init_transform = _ransac_initial_alignment(
            source, target, voxel_size)
    else:  # 'none'
        init_transform = np.eye(4)
    
    # Choose ICP method
    if use_point_to_plane:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    
    # Run ICP
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_dist,
        init=init_transform,
        estimation_method=estimation,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=max_iterations
        )
    )
    
    if return_full_result:
        return result.transformation, result.inlier_rmse, result
    else:
        return result.transformation, result.inlier_rmse


def _ransac_initial_alignment(source, target, voxel_size):
    """
    Helper function for RANSAC-based initial alignment.
    """
    # Downsample
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # Estimate normals
    radius_normal = voxel_size * 2
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Compute FPFH features
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    # RANSAC
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return result.transformation


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

def bbox_iou_3d(cluster1, cluster2):
    """
    Calculate 3D IoU between axis-aligned bounding boxes of two point clusters.
    
    Args:
        cluster1: Array of 3D points, shape (n_points, 3) or (n_points * 3,) flattened
        cluster2: Array of 3D points, shape (m_points, 3) or (m_points * 3,) flattened
    
    Returns:
        float: IoU value between 0 and 1
    """
    # Convert inputs to proper shape if they're flattened
    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)
    
    # Reshape if flattened (assuming 3D points)
    if cluster1.ndim == 1:
        if len(cluster1) % 3 != 0:
            raise ValueError(f"cluster1 length {len(cluster1)} is not divisible by 3")
        cluster1 = cluster1.reshape(-1, 3)
    
    if cluster2.ndim == 1:
        if len(cluster2) % 3 != 0:
            raise ValueError(f"cluster2 length {len(cluster2)} is not divisible by 3")
        cluster2 = cluster2.reshape(-1, 3)
    
    # Get bounding box coordinates for each cluster
    box1_min = np.min(cluster1, axis=0)  # [x_min, y_min, z_min]
    box1_max = np.max(cluster1, axis=0)  # [x_max, y_max, z_max]
    
    box2_min = np.min(cluster2, axis=0)  # [x_min, y_min, z_min] 
    box2_max = np.max(cluster2, axis=0)  # [x_max, y_max, z_max]
    
    # Calculate intersection bounds
    intersection_min = np.maximum(box1_min, box2_min)
    intersection_max = np.minimum(box1_max, box2_max)
    
    # Calculate intersection dimensions (0 if no overlap)
    intersection_dims = np.maximum(0, intersection_max - intersection_min)
    intersection_vol = np.prod(intersection_dims)
    
    # Calculate volumes of both bounding boxes
    box1_vol = np.prod(box1_max - box1_min)
    box2_vol = np.prod(box2_max - box2_min)
    
    # Calculate union volume
    union_vol = box1_vol + box2_vol - intersection_vol
    
    # Avoid division by zero
    if union_vol == 0:
        return 0.0
    
    return intersection_vol / union_vol