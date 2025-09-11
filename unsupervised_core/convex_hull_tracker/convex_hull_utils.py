from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon


def voxel_sampling_fast(points, res_x=0.1, res_y=0.1, res_z=0.1):
    """Ultra-fast vectorized voxel sampling"""
    if len(points) == 0:
        return points
    
    # Vectorized voxel coordinate computation
    mins = points.min(axis=0)
    voxel_coords = ((points - mins) / [res_x, res_y, res_z]).astype(np.int32)
    
    # Create unique voxel indices using numpy
    _, unique_indices, inverse_indices = np.unique(
        voxel_coords, axis=0, return_index=True, return_inverse=True
    )
    
    # Option 1: Take first point in each voxel (fastest)
    return points[unique_indices]

def analytical_y_rotation(
    A: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Analytical least squares solution for y-axis rotation only.
    """
    # Center the data
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Extract x and z coordinates for y-axis rotation
    AAx, AAz = AA[:, 0], AA[:, 2]
    BBx, BBz = BB[:, 0], BB[:, 2]

    # Analytical solution for optimal theta
    numerator = np.sum(BBx * AAz - BBz * AAx)
    denominator = np.sum(BBx * AAx + BBz * AAz)
    theta = np.arctan2(numerator, denominator)

    # Construct rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array([[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]])

    # Translation
    t = centroid_B - R @ centroid_A

    return R, t, theta


def rigid_icp(
    points1: np.ndarray,
    points2: np.ndarray,
    tolerance: float = 0.1,
    max_iterations=10,
    debug: bool = False,
    relative: bool = True,
):
    # Center both point clouds by the same reference point (centroid of points1)
    centroid_A = np.mean(points1, axis=0)

    A = np.copy(points1) - centroid_A
    B = np.copy(points2) - centroid_A  # Note: both centered by same point

    prev_error = float("inf")

    # Initialize final transformation
    R_final = np.eye(A.shape[1])
    t_final = np.zeros((3,))

    tree = cKDTree(B)

    init_distances, B_indices = tree.query(A)

    # if init_distances.mean() < tolerance:
    #     if debug:
    #         print("init is good!")
    #         R, t, theta = analytical_y_rotation(A, B[B_indices])

    #         print("init R, t", np.round(R, 2), np.round(t, 2))

    #     A_indices = np.arange(len(A))

    #     # Convert back: t_final_orig = t_final + centroid_A - R_final @ centroid_A
    #     # t_final_orig = t_final + centroid_A - R_final @ centroid_A
    #     return R_final, t_final, A_indices, B_indices, init_distances.mean()

    for i in range(max_iterations):
        distances, indices = tree.query(A)
        R, t, theta = analytical_y_rotation(A, B[indices])

        # Update transformations
        R_final = R @ R_final
        t_final = R @ t_final + t

        # Apply transformation
        A = (R @ A.T).T + t

        mean_error = np.mean(distances)
        if debug:
            print(f"rigid_icp {i=} mean_error={mean_error:.2f}")

        if (
            np.abs(prev_error - mean_error) < tolerance
            or np.abs(mean_error) < tolerance
        ):
            break
        prev_error = mean_error

    # Convert final transformation back to original coordinate system
    # Since both were centered by centroid_A, we need:
    # points2 = R_final @ points1 + t_final_orig
    # where: points2 - centroid_A = R_final @ (points1 - centroid_A) + t_final
    # So: t_final_orig = t_final + centroid_A - R_final @ centroid_A
    t_original = t_final + centroid_A - R_final @ centroid_A

    if debug:
        print(f"t_final before", np.round(t_final, 2))
    if debug:
        print(f"t_original", np.round(t_original, 2))
    # Verify on original coordinates
    A = np.copy(points1)

    # t_original = t_final + centroid_A - R_final @ centroid_A
    A_final = (R_final @ points1.T).T + t_original

    # # Centered coordinates approach, converted back to original
    # A_centered = np.copy(points1) - centroid_A
    # A_final_centered = (R_final @ A_centered.T).T + t_final
    # A_final_from_centered = A_final_centered + centroid_A  # Convert back to original coords

    # # These should be the same!
    # mean_error = np.linalg.norm(A_final - A_final_from_centered, axis=1).mean()
    # print(f"Comparison of both approaches: mean_error={mean_error:.6f}")
    # assert np.allclose(A_final, A_final_from_centered), f"Both approaches should give same result!"

    # Final error computation
    tree_orig = cKDTree(points2)
    distances, B_indices = tree_orig.query(A_final)
    A_indices = np.arange(len(A_final))

    if debug:
        print(
            f"final distances {distances.min():.6f} {distances.mean():.6f} {distances.max():.6f}"
        )
        print(f"original translation magnitude: {np.linalg.norm(centroid_A):.6f}")
        print(f"centroid_A: {np.round(centroid_A, 2)}")
        print(f"centroid_b {np.round(np.mean(points2, axis=0), 2)}")

    distances_mask = distances <= tolerance
    A_indices = A_indices[distances_mask]
    B_indices = B_indices[distances_mask]

    final_err = distances.mean()

    if relative:
        return R_final, t_final, A_indices, B_indices, final_err
    else:
        return R_final, t_original, A_indices, B_indices, final_err


def hungarian_matching(points1, points2):
    """One-to-one matching using Hungarian algorithm"""
    # Compute distance matrix
    distances = np.linalg.norm(points1[:, np.newaxis] - points2[np.newaxis, :], axis=2)

    # Hungarian assignment
    row_indices, col_indices = linear_sum_assignment(distances)

    distances = distances[row_indices, col_indices]

    return row_indices, col_indices, distances


def bidirectional_matching(points1, points2):
    """Mutual nearest neighbor matching"""
    if len(points1) == 0 or len(points2) == 0:
        return np.array([]), np.array([]), np.array([])

    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    # Forward matching: points1 -> points2
    distances_1to2, indices_1to2 = tree2.query(points1)

    # Backward matching: points2 -> points1
    distances_2to1, indices_2to1 = tree1.query(points2)

    # Find mutual matches
    mutual_matches = []
    mutual_distances = []

    for i, (j, dist) in enumerate(zip(indices_1to2, distances_1to2)):
        if indices_2to1[j] == i:  # Mutual nearest neighbors
            mutual_matches.append((i, j))
            mutual_distances.append(dist)

    if mutual_matches:
        row_indices, col_indices = zip(*mutual_matches)
        return np.array(row_indices), np.array(col_indices), np.array(mutual_distances)
    else:
        return np.array([]), np.array([]), np.array([])

def relative_object_pose_multiresolution(
    points1: np.ndarray,
    points2: np.ndarray,
    resolutions=[0.2, 0.1, 0.05],  # Coarse to fine
    iterations_per_level=[5, 5, 10],
    tolerance=0.1,
    debug=False
):
    T_final = np.eye(4)
    
    for res, max_iter in zip(resolutions, iterations_per_level):
        # Downsample at current resolution
        A = voxel_sampling_fast(points1, res, res, res)
        B = voxel_sampling_fast(points2, res, res, res)
        
        # Apply current transformation estimate
        A = (T_final[:3, :3] @ A.T).T + T_final[:3, 3]
        
        # Run ICP at this resolution
        R, t, _, _, icp_cost = relative_object_pose(
            A, B, 
            tolerance=tolerance*res/resolutions[-1],  # Scale tolerance
            max_iterations=max_iter,
            debug=debug
        )
        print(f"{res=:.2f} {icp_cost=:.3f}")
        
        # Update cumulative transformation
        T_iter = np.eye(4)
        T_iter[:3, :3] = R
        T_iter[:3, 3] = t
        T_final = T_iter @ T_final
    
    # Final matching on full resolution for inliers
    A_final = (T_final[:3, :3] @ points1.T).T + T_final[:3, 3]
    A_indices, B_indices, distances = hungarian_matching(A_final, points2)
    
    # Robust outlier rejection
    distance_threshold = np.percentile(distances, 90)
    inlier_mask = distances <= distance_threshold
    
    return T_final[:3, :3], T_final[:3, 3], A_indices[inlier_mask], B_indices[inlier_mask], distances[inlier_mask].mean()

def relative_object_pose(
    points1: np.ndarray,
    points2: np.ndarray,
    tolerance: float = 0.1,
    max_iterations=10,
    debug: bool = False,
    hungarian: bool =True
):
    """
    ICP for pre-centered points (assumes both clouds centered at origin)
    """
    A = np.copy(points1)
    B = np.copy(points2)

    # simplify
    A = voxel_sampling_fast(A, 0.05, 0.05, 0.05)
    B = voxel_sampling_fast(B, 0.05, 0.05, 0.05)

    # Check if points are actually centered
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    if np.linalg.norm(centroid_A) > 0.1 or np.linalg.norm(centroid_B) > 0.1:
        if debug:
            print(f"Warning: Points not centered! A: {centroid_A}, B: {centroid_B}")

    prev_error = float("inf")

    # Track transformation as 4x4 matrix for clarity
    T_final = np.eye(4)

    matching_func = hungarian_matching if hungarian else bidirectional_matching

    for i in range(max_iterations):
        # distances, indices = tree.query(A)
        A_indices, B_indices, distances = matching_func(A, B)

        A_, B_ = A[A_indices], B[B_indices]

        # Modified analytical_y_rotation for centered points
        R, t = analytical_z_rotation_centered(A_, B_)

        # Update transformation matrix
        T_iter = np.eye(4)
        T_iter[:3, :3] = R
        T_iter[:3, 3] = t
        T_final = T_iter @ T_final

        # Apply transformation
        A = (R @ A.T).T + t

        mean_error = np.mean(distances)
        if debug:
            print(f"rigid_icp {i=} mean_error={mean_error:.2f}")

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Extract final R and t
    R_final = T_final[:3, :3]
    t_final = T_final[:3, 3]

    # Verify
    A_final = (R_final @ points1.T).T + t_final
    # tree_orig = cKDTree(points2)
    # distances, B_indices = tree_orig.query(A_final)
    # A_indices = np.arange(len(A_final))
    A_indices, B_indices, distances = matching_func(A_final, points2)

    distance_threshold = np.percentile(distances, 75)  # Top 75% closest points

    distance_threshold = max(distance_threshold, 0.1)

    inlier_mask = distances <= distance_threshold
    A_indices = np.where(inlier_mask)[0]
    B_indices = B_indices[inlier_mask]

    if debug:
        print(
            f"final distances {distances.min():.6f} {distances.mean():.6f} {distances.max():.6f}"
        )

    final_err = distances.mean()
    # final_err = np.median(distances)

    return R_final, t_final, A_indices, B_indices, final_err


def analytical_y_rotation_centered(
    A: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Y-axis rotation for already-centered points.
    """
    # DON'T re-center if points are already centered!
    # Just use them directly

    # Extract x and z coordinates for y-axis rotation
    Ax, Az = A[:, 0], A[:, 2]
    Bx, Bz = B[:, 0], B[:, 2]

    # Analytical solution for optimal theta
    numerator = np.sum(Bx * Az - Bz * Ax)
    denominator = np.sum(Bx * Ax + Bz * Az)
    theta = np.arctan2(numerator, denominator)

    # Construct rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array([[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]])

    # For centered points, translation is just the difference after rotation
    A_rotated = (R @ A.T).T
    t = np.mean(B - A_rotated, axis=0)

    return R, t

def analytical_z_rotation_centered(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-axis rotation for already-centered points (yaw only)."""
    
    t = np.mean(B - A, axis=0)

    A_translated = A + t

    # Extract x and y coordinates for z-axis rotation
    Ax, Ay = A_translated[:, 0], A_translated[:, 1]  # X,Y → Z-axis rotation (yaw)
    # Ax, Ay = A[:, 0], A[:, 1]  # X,Y → Z-axis rotation (yaw)
    Bx, By = B[:, 0], B[:, 1]

    # Analytical solution for optimal theta
    numerator = np.sum(Bx * Ay - By * Ax)
    denominator = np.sum(Bx * Ax + By * Ay)
    theta = np.arctan2(numerator, denominator)

    # Z-axis rotation matrix
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    R = np.array([[cos_theta, -sin_theta, 0], 
                  [sin_theta,  cos_theta, 0], 
                  [0,          0,         1]])

    # Translation after rotation
    # A_rotated = (R @ A.T).T
    # t = np.mean(B - A_rotated, axis=0)

    return R, t


def relative_object_rotation(
    points1: np.ndarray,
    points2: np.ndarray,
    tolerance: float = 0.1,
    max_iterations=10,
    debug: bool = False,
):
    """
    ICP for pre-centered points (assumes both clouds centered at origin)
    """
    A = np.copy(points1)
    B = np.copy(points2)

    # Check if points are actually centered
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    if np.linalg.norm(centroid_A) > 0.1 or np.linalg.norm(centroid_B) > 0.1:
        if debug:
            print(f"Warning: Points not centered! A: {centroid_A}, B: {centroid_B}")

    prev_error = float("inf")

    # Track transformation as 4x4 matrix for clarity
    R_final = np.eye(3)

    tree = cKDTree(B)

    for i in range(max_iterations):
        distances, indices = tree.query(A)

        # Modified analytical_y_rotation for centered points
        R, _ = analytical_z_rotation_centered(A, B[indices])

        # Update transformation matrix
        R_final = R @ R_final

        # Apply transformation
        A = (R @ A.T).T

        mean_error = np.mean(distances)
        if debug:
            print(f"rigid_icp {i=} mean_error={mean_error:.2f}")

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Verify
    A_final = (R_final @ points1.T).T
    tree_orig = cKDTree(points2)
    distances, B_indices = tree_orig.query(A_final)

    if debug:
        print(
            f"final distances {distances.min():.6f} {distances.mean():.6f} {distances.max():.6f}"
        )

    return R_final


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

def icp_hungarian(
    points1: np.ndarray,
    points2: np.ndarray,
    tolerance: float = 0.1,
    max_iterations=10,
    debug: bool = False,
    hungarian: bool =True
):
    """
    ICP for pre-centered points (assumes both clouds centered at origin)
    """
    A = np.copy(points1)
    B = np.copy(points2)

    # Check if points are actually centered
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    if np.linalg.norm(centroid_A) > 0.1 or np.linalg.norm(centroid_B) > 0.1:
        if debug:
            print(f"Warning: Points not centered! A: {centroid_A}, B: {centroid_B}")

    prev_error = float("inf")

    # Track transformation as 4x4 matrix for clarity
    T_final = np.eye(4)

    matching_func = hungarian_matching if hungarian else bidirectional_matching

    for i in range(max_iterations):
        # distances, indices = tree.query(A)
        A_indices, B_indices, distances = matching_func(A, B)

        A_, B_ = A[A_indices], B[B_indices]

        # Modified analytical_y_rotation for centered points
        R, t = best_fit_transform(A_, B_)

        # Update transformation matrix
        T_iter = np.eye(4)
        T_iter[:3, :3] = R
        T_iter[:3, 3] = t
        T_final = T_iter @ T_final

        # Apply transformation
        A = (R @ A.T).T + t

        mean_error = np.mean(distances)
        if debug:
            print(f"rigid_icp {i=} mean_error={mean_error:.2f}")

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Extract final R and t
    R_final = T_final[:3, :3]
    t_final = T_final[:3, 3]

    # Verify
    A_final = (R_final @ points1.T).T + t_final
    # tree_orig = cKDTree(points2)
    # distances, B_indices = tree_orig.query(A_final)
    # A_indices = np.arange(len(A_final))
    A_indices, B_indices, distances = matching_func(A_final, points2)

    distance_threshold = np.percentile(distances, 75)  # Top 75% closest points

    distance_threshold = max(distance_threshold, 0.1)

    inlier_mask = distances <= distance_threshold
    A_indices = np.where(inlier_mask)[0]
    B_indices = B_indices[inlier_mask]

    if debug:
        print(
            f"final distances {distances.min():.6f} {distances.mean():.6f} {distances.max():.6f}"
        )

    final_err = distances.mean()
    # final_err = np.median(distances)

    return R_final, t_final, A_indices, B_indices, final_err

def render_triangle_with_fillpoly_barycentric(tri_verts_2d, tri_depths, depth_buffer, render_image, color):
    """Use fillPoly to mask triangle, then barycentric interpolation for depth"""
    
    height, width = depth_buffer.shape
    
    # Create triangle mask using fillPoly
    triangle_mask = np.zeros((height, width), dtype=np.uint8)
    tri_verts_int = tri_verts_2d.astype(np.int32).reshape(1, -1, 2)
    cv2.fillPoly(triangle_mask, tri_verts_int, 1)
    
    # Get pixel coordinates where mask is 1
    y_coords, x_coords = np.where(triangle_mask == 1)
    
    if len(y_coords) == 0:
        return
    
    # Stack coordinates for barycentric calculation
    pixel_coords = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
    
    # Compute barycentric coordinates for masked pixels
    barycentric_coords = compute_barycentric_vectorized(pixel_coords, tri_verts_2d[0], tri_verts_2d[1], tri_verts_2d[2])
    
    # Interpolate depth using barycentric coordinates
    interpolated_depths = (barycentric_coords[:, 0] * tri_depths[0] + 
                          barycentric_coords[:, 1] * tri_depths[1] + 
                          barycentric_coords[:, 2] * tri_depths[2])
    
    # Update depth buffer where this triangle is closer
    closer_mask = interpolated_depths < depth_buffer[y_coords, x_coords]
    
    if np.any(closer_mask):
        update_y = y_coords[closer_mask]
        update_x = x_coords[closer_mask]
        update_depths = interpolated_depths[closer_mask]
        
        depth_buffer[update_y, update_x] = update_depths
        render_image[update_y, update_x] = color

def compute_barycentric_vectorized(points, v0, v1, v2):
    """Vectorized barycentric coordinate computation"""
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = points - v0[None, :]
    
    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.sum(v0p * v0v2[None, :], axis=1)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.sum(v0p * v0v1[None, :], axis=1)
    
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-10)  # Add small epsilon
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1 - u - v
    
    return np.stack([w, v, u], axis=1)

def save_depth_buffer_colorized(depth_buffer, save_path, cmap=cv2.COLORMAP_VIRIDIS):
    """Save depth buffer as colorized image using OpenCV colormap"""
    
    # Create a copy to avoid modifying original
    depth_vis = depth_buffer.copy()
    
    # Handle infinite values (background)
    valid_mask = np.isfinite(depth_vis)
    
    if not np.any(valid_mask):
        print("Warning: No valid depth values found")
        return
    
    # Get min/max of valid depths for normalization
    min_depth = np.min(depth_vis[valid_mask])
    max_depth = np.max(depth_vis[valid_mask])
    
    # Set invalid depths to max_depth (will appear as furthest color)
    depth_vis[~valid_mask] = max_depth
    
    # Normalize to 0-255 range
    if max_depth > min_depth:
        depth_normalized = ((depth_vis - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth_vis, dtype=np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, cmap)
    
    # Optionally set background to black
    background_mask = ~valid_mask
    depth_colored[background_mask] = [0, 0, 0]  # Black background
    
    # Save the image
    cv2.imwrite(save_path, depth_colored)
    
    print(f"Depth range: {min_depth:.2f} - {max_depth:.2f}m")
    print(f"Saved colorized depth to: {save_path}")
    
    return depth_colored

def point_in_triangle_vectorized(points, v0, v1, v2):
    """Vectorized point-in-triangle test using cross products"""
    # Compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = points - v0[None, :]
    
    # Compute dot products
    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.sum(v0p * v0v2[None, :], axis=1)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.sum(v0p * v0v1[None, :], axis=1)
    
    # Compute barycentric coordinates
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    # Check if point is inside triangle
    return (u >= 0) & (v >= 0) & (u + v <= 1)


def box_iou_3d(box1: np.ndarray, box2: np.ndarray):
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

def yaw_circular_mean(yaw1: float, yaw2: float, weight1: float = 0.5, weight2: float = 0.5):
    mean_x = weight1 * np.cos(yaw1) + weight2 * np.cos(yaw2)
    mean_y = weight1 * np.sin(yaw1) + weight2 * np.sin(yaw2)
    return np.arctan2(mean_y, mean_x) 