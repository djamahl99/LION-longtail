from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


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

        # print("distances", distances.shape, distances.min(), distances.max(), np.median(distances))

        # distance_threshold = np.median(distances)
        # distance_mask = distances <= distance_threshold
        # A_indices = A_indices[distance_mask]
        # B_indices = B_indices[distance_mask]
        # distances = distances[distance_mask]

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
    
    # Extract x and y coordinates for z-axis rotation
    Ax, Ay = A[:, 0], A[:, 1]  # X,Y → Z-axis rotation (yaw)
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
    A_rotated = (R @ A.T).T
    t = np.mean(B - A_rotated, axis=0)

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
