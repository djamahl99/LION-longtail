from pathlib import Path

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, splev, splprep
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

from lion.unsupervised_core.convex_hull_tracker.pose_kalman_filter import wrap_angle

from .box_utils import compute_ppscore, icp


def smooth_trajectory_with_vehicle_model(timestamps_ns, optimized_poses, 
                                       heading_speed_thresh=1.0, 
                                       smoothing_factor=None,
                                       apply_dynamics_model=True):
    """
    Vehicle motion model approach:
    1. Smooth path geometry with B-splines
    2. Derive yaw from path tangents (when moving fast enough)
    3. Apply vehicle dynamics for consistency
    """
    times_secs = np.array(timestamps_ns) * 1e-9
    times_secs = times_secs - times_secs[0]  # Start from 0

    if len(optimized_poses) < 4:
        return optimized_poses
    
    # Extract positions and yaws
    positions = np.array([pose[:3, 3] for pose in optimized_poses])
    original_yaws = np.array([Rotation.from_matrix(pose[:3, :3]).as_rotvec()[2] 
                             for pose in optimized_poses])
    
    n_points = len(positions)
    
    # Step 1: Smooth the path geometry using parametric B-spline
    if smoothing_factor is None:
        # Base smoothing on path length and noise level
        path_length = np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1))
        smoothing_factor = max(n_points * 0.1, path_length * 0.01)
    
    # smoothing_factor = None

    # print(f"Smoothing 2D path with factor: {smoothing_factor:.3f}")
    
    # Parametric spline for XY path
    tck_xy, u_xy = splprep([positions[:, 0], positions[:, 1]], 
                           u=times_secs, s=smoothing_factor, k=min(3, n_points-1))
    
    # Smooth Z separately
    z_spline = UnivariateSpline(times_secs, positions[:, 2], 
                               s=smoothing_factor * 0.5, k=min(3, n_points-1))
    
    # Evaluate smoothed path
    smoothed_xy = np.array(splev(times_secs, tck_xy)).T
    smoothed_z = z_spline(times_secs)
    smoothed_positions = np.column_stack([smoothed_xy, smoothed_z])
    
    # Step 2: Compute velocities and speeds from smoothed path
    xy_derivatives = np.array(splev(times_secs, tck_xy, der=1)).T
    z_derivative = z_spline.derivative()(times_secs)
    
    velocities_3d = np.column_stack([xy_derivatives, z_derivative])
    speeds_2d = np.linalg.norm(xy_derivatives, axis=1)
    
    # Step 3: Derive yaw from path tangents (when moving fast enough)
    path_yaws = np.arctan2(xy_derivatives[:, 1], xy_derivatives[:, 0])
    
    # Determine which frames to use path-based yaw vs original yaw
    use_path_yaw = speeds_2d > heading_speed_thresh
    n_fast_frames = np.sum(use_path_yaw)
    
    print(f"Using path-based yaw for {n_fast_frames}/{n_points} frames (speed > {heading_speed_thresh:.1f} m/s)")
    
    # Initialize output yaws
    smoothed_yaws = original_yaws.copy()
    
    if n_fast_frames > 0:
        # For fast frames, use path tangent yaw
        # Handle angle wrapping carefully
        fast_indices = np.where(use_path_yaw)[0]

        fps = path_yaws[fast_indices]
        xps = times_secs[fast_indices]
        smoothed_yaws = np.interp(times_secs, xps, fps)
    else:
        median_yaw = np.median(original_yaws)
        smoothed_yaws[:] = median_yaw

    
    # Step 5: Create smoothed pose matrices
    smoothed_poses = []
    for i in range(n_points):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(np.array([0, 0, smoothed_yaws[i]])).as_matrix()
        pose[:3, 3] = smoothed_positions[i]
        smoothed_poses.append(pose)
    
    # Compute quality metrics
    position_rmse = np.sqrt(np.mean(np.sum((positions - smoothed_positions)**2, axis=1)))
    yaw_rmse = np.sqrt(np.mean((original_yaws - smoothed_yaws)**2))
    
    print(f"Vehicle Motion Model Results:")
    print(f"  Position RMSE: {position_rmse:.3f}m")
    print(f"  Yaw RMSE: {np.degrees(yaw_rmse):.2f}°")
    # print(f"  Max curvature: {np.max(np.abs(curvatures)):.4f} rad/m")
    print(f"  Speed range: {np.min(speeds_2d):.1f} - {np.max(speeds_2d):.1f} m/s")
    
    return smoothed_poses


def smooth_with_vehicle_model(times_secs, optimized_poses, knot_interval=0.5, smoothing_factor=None):
    """
    Stable vehicle model: smooth position and yaw separately but with coupling constraints
    This avoids the integration drift problem while maintaining vehicle dynamics
    """
    positions = np.array([pose[:3, 3] for pose in optimized_poses])
    yaws = np.array([Rotation.from_matrix(pose[:3, :3]).as_rotvec()[2] for pose in optimized_poses])
    yaws_unwrapped = np.unwrap(yaws)
    
    if smoothing_factor is None:
        smoothing_factor = len(positions) * 0.1
    
    # STABLE APPROACH: Smooth position and yaw separately first
    # Then apply vehicle coupling constraints
    
    # 1. Smooth XY positions with parametric spline (maintains path shape)
    tck_xy, u = splprep([positions[:, 0], positions[:, 1]], 
                        u=times_secs, s=smoothing_factor, k=3)
    smoothed_xy = np.array(splev(times_secs, tck_xy)).T
    
    # 2. Smooth Z separately 
    z_spline = UnivariateSpline(times_secs, positions[:, 2], s=smoothing_factor * 0.5, k=3)
    smoothed_z = z_spline(times_secs)
    
    # 3. Smooth yaw with continuity constraint
    yaw_spline = UnivariateSpline(times_secs, yaws_unwrapped, s=smoothing_factor * 0.1, k=3)
    smoothed_yaws_unwrapped = yaw_spline(times_secs)
    
    # 4. Apply vehicle dynamics coupling constraint
    # Adjust yaw to be more consistent with motion direction, but gently
    coupling_strength = 0.3  # How much to couple (0=independent, 1=fully coupled)
    
    for i in range(1, len(times_secs) - 1):
        # Compute motion direction from smoothed positions
        dx = smoothed_xy[i+1, 0] - smoothed_xy[i-1, 0]
        dy = smoothed_xy[i+1, 1] - smoothed_xy[i-1, 1]
        
        if np.sqrt(dx*dx + dy*dy) > 1e-6:  # Only if moving significantly
            motion_yaw = np.arctan2(dy, dx)
            current_yaw = smoothed_yaws_unwrapped[i]
            
            # Find the closest equivalent angle (handle wrapping)
            yaw_options = [motion_yaw, motion_yaw + 2*np.pi, motion_yaw - 2*np.pi]
            yaw_errors = [abs(wrap_angle(opt - current_yaw)) for opt in yaw_options]
            best_motion_yaw = yaw_options[np.argmin(yaw_errors)]
            
            # Gently pull yaw toward motion direction
            yaw_correction = coupling_strength * (best_motion_yaw - current_yaw)
            smoothed_yaws_unwrapped[i] += yaw_correction
    
    # 5. Compute vehicle dynamics metrics
    distances = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(smoothed_xy, axis=0), axis=1))])
    curvatures = np.zeros(len(positions))
    speeds = np.zeros(len(positions))
    
    for i in range(1, len(times_secs)):
        dt = times_secs[i] - times_secs[i-1]
        ds = distances[i] - distances[i-1]
        
        if ds > 1e-6 and dt > 1e-6:
            speeds[i] = ds / dt
            dyaw = smoothed_yaws_unwrapped[i] - smoothed_yaws_unwrapped[i-1]
            curvatures[i] = dyaw / ds
        else:
            speeds[i] = speeds[i-1] if i > 0 else 0
            curvatures[i] = curvatures[i-1] if i > 0 else 0
    
    # Wrap yaws back to [-π, π]
    smoothed_yaws_wrapped = np.array([wrap_angle(yaw) for yaw in smoothed_yaws_unwrapped])
    
    # Combine smoothed positions
    smoothed_positions = np.column_stack([smoothed_xy, smoothed_z])
    
    # Create smoothed pose matrices
    smoothed_poses = []
    for i in range(len(times_secs)):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(np.array([0, 0, smoothed_yaws_wrapped[i]])).as_matrix()
        pose[:3, 3] = smoothed_positions[i]
        smoothed_poses.append(pose)
    
    # Quality metrics
    position_rmse = np.sqrt(np.mean(np.sum((positions - smoothed_positions)**2, axis=1)))
    yaw_rmse = np.sqrt(np.mean((yaws - smoothed_yaws_wrapped)**2))
    
    # Check for reasonable results
    if position_rmse > 2.0:
        print(f"WARNING: Large position RMSE ({position_rmse:.3f}m), smoothing may be too aggressive")
        print("Consider increasing smoothing_factor or reducing coupling_strength")
    
    # Compute derivatives for velocities
    xy_derivatives = np.array(splev(times_secs, tck_xy, der=1)).T
    z_derivative = z_spline.derivative()(times_secs)
    yaw_derivative = yaw_spline.derivative()(times_secs)
    
    velocities_2d = xy_derivatives
    velocities_3d = np.column_stack([velocities_2d, z_derivative])
    angular_velocities = yaw_derivative
    
    spline_info = {
        'method': 'vehicle_model_stable',
        'splines': {
            'xy': tck_xy,
            'z': z_spline,
            'yaw': yaw_spline
        },
        'coupling_strength': coupling_strength,
        'smoothing_factor': smoothing_factor,
        'quality': {
            'position_rmse': position_rmse,
            'yaw_rmse_deg': np.degrees(yaw_rmse),
            'max_curvature': np.max(np.abs(curvatures)),
            'max_speed': np.max(speeds)
        },
        'dynamics': {
            'velocities': velocities_3d,
            'angular_velocities': angular_velocities,
            'curvatures': curvatures,
            'speeds': speeds
        },
        'times_secs': times_secs
    }
    
    print(f"Stable vehicle model smoothing:")
    print(f"  Position RMSE: {position_rmse:.3f}m")
    print(f"  Yaw RMSE: {np.degrees(yaw_rmse):.2f}°")
    print(f"  Max curvature: {np.max(np.abs(curvatures)):.4f} rad/m")
    print(f"  Coupling strength: {coupling_strength}")
    
    return smoothed_poses, spline_info

def refine_with_gtsam(optimized_poses):
    """Use GTSAM to refine the already-good poses"""
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # Very tight prior on first pose
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    )
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(optimized_poses[0]), prior_noise))
    
    # Add poses as initial estimates
    for i, pose in enumerate(optimized_poses):
        initial_estimate.insert(i, gtsam.Pose3(pose))
    
    # Add sequential constraints with reasonable noise
    for i in range(len(optimized_poses) - 1):
        relative_pose = np.linalg.inv(optimized_poses[i]) @ optimized_poses[i+1]
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.05,  # Tight yaw constraint
                     0.2, 0.2, 0.2])
        )
        graph.add(gtsam.BetweenFactorPose3(i, i+1, gtsam.Pose3(relative_pose), noise))
    
    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(50)  # Few iterations since starting point is good
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    
    return [result.atPose3(i).matrix() for i in range(len(optimized_poses))]

def optimize_with_gtsam_timed(initial_poses, constraints, timestamps_ns):
    n_poses = len(initial_poses)
    timestamps = np.array(timestamps_ns) * 1e-9
    
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # 1. Better prior - fix only first pose with small uncertainty
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.0, 0.0, 0.0,  # reasonable rotation uncertainty (rad)
                  1.0, 1.0, 1.0])  # Small translation uncertainty (m)
    )
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(initial_poses[0]), prior_noise))
    
    # 2. Add initial estimates
    for i, pose_matrix in enumerate(initial_poses):
        initial_estimate.insert(i, gtsam.Pose3(pose_matrix))
    
    # 3. Add constraints with confidence-based noise
    for constraint in constraints:
        i, j = constraint['frame_i'], constraint['frame_j']
        relative_pose = gtsam.Pose3(constraint['relative_pose'])
        confidence = constraint['confidence']
        is_loop = constraint.get('is_loop', False)
        
        # Time-based uncertainty
        dt = abs(timestamps[j] - timestamps[i])
        
        # if is_loop:
        #     # Loop closures: tighter constraints if confident
        #     base_rot = 0.05 / (confidence + 0.1)
        #     base_trans = 0.1 / (confidence + 0.1)
        # else:
        #     # Sequential: scale with time gap
        #     time_factor = 1.0 + dt * 0.01  # Gradual increase
        #     base_rot = 0.1 * time_factor / (confidence + 0.1)
        #     base_trans = 0.2 * time_factor / (confidence + 0.1)
        
        # Z-rotation only constraint (vehicles don't pitch/roll)
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([10.0, 10.0, 10.0,  # Large X,Y rotation uncertainty
                     3.0, 3.0, 3.0])
        )
        
        # Add robust loss for outlier rejection
        robust_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            noise
        )
        
        graph.add(gtsam.BetweenFactorPose3(i, j, relative_pose, robust_noise))
    
    # 4. Add velocity smoothness for consecutive poses
    if n_poses > 2:
        for i in range(1, n_poses - 1):
            dt_prev = timestamps[i] - timestamps[i-1]
            dt_next = timestamps[i+1] - timestamps[i]
            
            # Constant velocity prior
            vel_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1.0, 1.0, 0.01,  # Tight on Z rotation
                         0.5, 0.5, 0.5])    # Moderate on translation
            )
            
            # Expected pose based on constant velocity
            T_prev = initial_poses[i-1]
            T_curr = initial_poses[i]
            T_next = initial_poses[i+1]
            
            # Velocity from i-1 to i
            vel_pose = np.linalg.inv(T_prev) @ T_curr
            
            # Predicted next pose
            dt_ratio = dt_next / dt_prev
            predicted_delta = np.eye(4)
            predicted_delta[:3, 3] = vel_pose[:3, 3] * dt_ratio
            # Handle rotation interpolation for Z-axis only
            
            predicted_pose = gtsam.Pose3(T_curr @ predicted_delta)
            
            graph.add(gtsam.PriorFactorPose3(i+1, predicted_pose, vel_noise))
    
    # 5. Optimize with better parameters
    params = gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(1e-6)
    params.setAbsoluteErrorTol(1e-6)
    params.setMaxIterations(200)
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    
    # Extract results
    optimized_poses = [result.atPose3(i).matrix() for i in range(n_poses)]
    marginals = gtsam.Marginals(graph, result)
    
    quality = {
        'initial_error': graph.error(initial_estimate),
        'final_error': graph.error(result),
        'iterations': optimizer.iterations(),
        'error_reduction': 1 - graph.error(result) / graph.error(initial_estimate)
    }
    
    return optimized_poses, marginals, quality

def optimize_with_gtsam_timed_(initial_poses, constraints, timestamps_ns):
    """
    Optimize 3D pose graph with temporal information using GTSAM
    
    Args:
        initial_poses: List of 4x4 transformation matrices
        constraints: List of dicts with 'frame_i', 'frame_j', 'relative_pose', 'confidence'
        timestamps_ns: List of timestamps in nanoseconds for each pose
    """
    n_poses = len(initial_poses)
    # print(f"Number of poses: {n_poses} (indices 0 to {n_poses-1})")
    
    # Check all constraints for invalid indices
    for i, constraint in enumerate(constraints):
        frame_i = constraint['frame_i']
        frame_j = constraint['frame_j']
        if frame_i >= n_poses or frame_j >= n_poses or frame_i < 0 or frame_j < 0:
            print(f"ERROR: Constraint {i} has invalid indices: {frame_i} -> {frame_j}")
            print(f"Valid range is 0 to {n_poses-1}")
            raise ValueError(f"Constraint references invalid frame index")
            
    # Convert timestamps to seconds
    timestamps = np.array(timestamps_ns) * 1e-9
    
    # 1. Create factor graph container
    graph = gtsam.NonlinearFactorGraph()
    
    # 2. Create initial estimates
    initial_estimate = gtsam.Values()

    trans_sigma = 1.0
    rot_sigma = 1.0
    
    # 3. Add prior on first pose (fix it as reference frame)
    # For Pose3: 6 DOF - 3 rotation, 3 translation
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.0, 0.0, rot_sigma, trans_sigma, trans_sigma, trans_sigma])
    )
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(initial_poses[0]), prior_noise))
    
    # 4. Add all poses to initial estimate
    for i, pose_matrix in enumerate(initial_poses):
        initial_estimate.insert(i, gtsam.Pose3(pose_matrix))
    
    # 5. Add between factors from constraints
    for constraint in constraints:
        i, j = constraint['frame_i'], constraint['frame_j']
        relative_pose = gtsam.Pose3(constraint['relative_pose'])
        confidence = constraint['confidence']
        
        # Scale uncertainty based on time difference
        dt = abs(timestamps[j] - timestamps[i])
        time_factor = 1.0 + dt * 0.1  # Increase uncertainty with time gap
        
        # # Create noise model based on confidence and time
        # base_sigma = 1.0 / (confidence + 1e-6)
        
        # # Different uncertainty for rotation (first 3) vs translation (last 3)
        # rot_sigma = base_sigma * time_factor * 0.1  # radians
        # trans_sigma = base_sigma * time_factor * 0.01  # meters
        
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.0, 0.0, rot_sigma,
                     trans_sigma, trans_sigma, trans_sigma])
        )
        
        graph.add(gtsam.BetweenFactorPose3(i, j, relative_pose, noise))
    
    # 6. Add smoothness constraints for consecutive poses (optional)
    if len(initial_poses) > 2:
        add_temporal_smoothness(graph, initial_poses, timestamps)
    
    # 7. Optimize using Levenberg-Marquardt
    params = gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(1e-5)
    params.setAbsoluteErrorTol(1e-5)
    params.setMaxIterations(100)
    params.setVerbosityLM("SILENT")  # or "ERROR", "VALUES", "DELTA", "LINEAR"
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    
    # 8. Extract optimized poses
    optimized_poses = []
    for i in range(len(initial_poses)):
        optimized_pose = result.atPose3(i)
        optimized_poses.append(optimized_pose.matrix())
    
    # 9. Calculate marginals (covariances) if needed
    marginals = gtsam.Marginals(graph, result)
    
    # 10. Compute quality metrics
    quality = {
        'initial_error': graph.error(initial_estimate),
        'final_error': graph.error(result),
        'iterations': optimizer.iterations()
    }
    
    return optimized_poses, marginals, quality

def optimize_with_gtsam_timed_positions(initial_positions, constraints, timestamps_ns):
    """
    Optimize poses with constraints: only yaw rotation allowed (rx=0, ry=0)
    
    Args:
        initial_positions: List of 3D positions [x, y, z]
        constraints: List of dicts with 'frame_i', 'frame_j', 'relative_pose', 'confidence'
        timestamps_ns: List of timestamps in nanoseconds
    """
    n_poses = len(initial_positions)
    timestamps = np.array(timestamps_ns) * 1e-9
    
    # 1. Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # 2. Create initial poses with zero pitch/roll, initial yaw=0
    for i, position in enumerate(initial_positions):
        pose = np.eye(4)
        pose[:3, 3] = position
        # Start with yaw=0, let GTSAM optimize it
        initial_estimate.insert(i, gtsam.Pose3(pose))
    
    # 3. Prior on first pose position and zero pitch/roll, loose yaw
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.001, 0.001, 0.5, 0.001, 0.001, 0.001])  # Loose on rz (yaw)
    )
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(initial_estimate.atPose3(0).matrix()), prior_noise))
    
    # 4. Add relative pose constraints from ICP
    for constraint in constraints:
        i, j = constraint['frame_i'], constraint['frame_j']
        relative_pose = gtsam.Pose3(constraint['relative_pose'])
        confidence = constraint['confidence']
        
        dt = abs(timestamps[j] - timestamps[i])
        time_factor = 1.0 + dt * 0.1
        base_sigma = 1.0 / (confidence + 1e-6)
        
        rot_sigma = base_sigma * time_factor * 0.1
        trans_sigma = base_sigma * time_factor * 0.01
        
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([rot_sigma, rot_sigma, rot_sigma,
                     trans_sigma, trans_sigma, trans_sigma])
        )
        
        graph.add(gtsam.BetweenFactorPose3(i, j, relative_pose, noise))
    
    # 5. ENFORCE: Add strong priors to keep rx=0, ry=0 for ALL poses
    zero_pitch_roll_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.0, 0.0, 10.0, 0.1, 0.1, 0.1])  # Very tight on rx,ry; loose on rz,x,y,z
    )
    
    for i in range(1, n_poses):  # Skip first pose (already has prior)
        # Create pose with zero pitch/roll at current position
        zero_pitch_roll_pose = np.eye(4)
        zero_pitch_roll_pose[:3, 3] = initial_positions[i]
        
        graph.add(gtsam.PriorFactorPose3(i, gtsam.Pose3(zero_pitch_roll_pose), zero_pitch_roll_noise))
    
    # 6. Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(1e-5)
    params.setAbsoluteErrorTol(1e-5)
    params.setMaxIterations(100)
    params.setVerbosityLM("SILENT")
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    
    # 7. Extract optimized poses and enforce rx=ry=0
    optimized_poses = []
    for i in range(len(initial_positions)):
        optimized_pose = result.atPose3(i)
        pose_matrix = optimized_pose.matrix()
        
        optimized_poses.append(pose_matrix)

        # # FORCE: Extract only yaw and reconstruct pose
        # rotation_matrix = pose_matrix[:3, :3]
        # yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        # # Reconstruct with only yaw rotation
        # cos_yaw = np.cos(yaw)
        # sin_yaw = np.sin(yaw)
        # clean_pose = np.eye(4)
        # clean_pose[:3, :3] = np.array([
        #     [cos_yaw, -sin_yaw, 0],
        #     [sin_yaw,  cos_yaw, 0],
        #     [0,        0,       1]
        # ])
        # clean_pose[:3, 3] = initial_positions[i]  # Use exact geometric center
        
        # optimized_poses.append(clean_pose)
    
    # 8. Calculate quality metrics
    marginals = gtsam.Marginals(graph, result)
    quality = {
        'initial_error': graph.error(initial_estimate),
        'final_error': graph.error(result),
        'iterations': optimizer.iterations()
    }
    
    return optimized_poses, marginals, quality

def add_temporal_smoothness(graph, initial_poses, timestamps):
    """
    Add smoothness constraints based on constant velocity assumption
    """
    smoothness_weight = 0.5  # Tune this based on your application
    
    for i in range(len(initial_poses) - 2):
        dt1 = timestamps[i+1] - timestamps[i]
        dt2 = timestamps[i+2] - timestamps[i+1]
        
        if dt1 <= 0 or dt2 <= 0:
            continue
        
        # For consecutive poses, we expect similar relative transformations
        pose_i = gtsam.Pose3(initial_poses[i])
        pose_i1 = gtsam.Pose3(initial_poses[i+1])
        pose_i2 = gtsam.Pose3(initial_poses[i+2])
        
        # Get relative transformation from i to i+2
        relative_i_to_i2 = pose_i.between(pose_i2)
        
        # Add soft constraint between non-consecutive poses
        smoothness_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([
                smoothness_weight * 0.1, smoothness_weight * 0.1, smoothness_weight * 0.1,
                smoothness_weight * 0.05, smoothness_weight * 0.05, smoothness_weight * 0.05
            ])
        )
        
        graph.add(gtsam.BetweenFactorPose3(i, i+2, relative_i_to_i2, smoothness_noise))

def compute_trajectory_quality(result, graph):
    """
    Compute quality metrics for the optimized trajectory
    """
    # Calculate final error
    error = graph.error(result)
    
    # Chi-squared test for consistency
    chi2 = 2.0 * error
    dof = graph.size() - result.size() * 6  # degrees of freedom
    
    return {
        'total_error': error,
        'chi2': chi2,
        'dof': dof,
        'avg_error': error / graph.size() if graph.size() > 0 else 0
    }

def simple_pairwise_icp_refinement(city_points_per_timestamp, initial_poses):
    """
    Much simpler approach: just refine each consecutive pair with ICP
    """
    print("\n=== SIMPLE PAIRWISE ICP REFINEMENT ===")
    
    refined_poses = [initial_poses[0].copy()]
    
    for i in range(len(city_points_per_timestamp) - 1):
        curr_points = city_points_per_timestamp[i]
        next_points = city_points_per_timestamp[i + 1]
        
        print(f"\nRefining frames {i} -> {i+1}")
        
        # Current approach: just do ICP directly on world coordinates
        # This might work better than the complex object-frame approach
        try:
            R, t, A_inliers, B_inliers = icp(
                curr_points, next_points,
                max_iterations=50,
                return_inliers=True,
                ret_err=False
            )
            
            print(f"  ICP inliers: {len(A_inliers)}/{len(curr_points)}")
            
            # Create world-space transformation
            world_relative = np.eye(4)
            world_relative[:3, :3] = R
            world_relative[:3, 3] = t
            
            # Apply transformation to get next pose
            next_refined_pose = refined_poses[i] @ world_relative
            refined_poses.append(next_refined_pose)
            
            # Check the refinement quality
            original_dist = np.linalg.norm(initial_poses[i+1][:3, 3] - refined_poses[i][:3, 3])
            refined_dist = np.linalg.norm(next_refined_pose[:3, 3] - refined_poses[i][:3, 3]) 
            print(f"  Original distance: {original_dist:.2f}m")
            print(f"  Refined distance: {refined_dist:.2f}m")
            
        except Exception as e:
            print(f"  ICP failed: {e}")
            refined_poses.append(initial_poses[i + 1].copy())
    
    return refined_poses

class GlobalTrajectoryOptimizer:
    """
    Multi-stage trajectory optimization:
    1. Pairwise ICP initialization
    2. Staggered pair constraints 
    3. Global pose graph optimization
    4. Inlier detection and refinement
    """
    
    def __init__(self, max_icp_iterations=50, stagger_step=2, max_stagger_gap=5):
        self.icp_max_iterations = max_icp_iterations
        self.stagger_step = stagger_step  # For staggered pairs (0-2, 1-3, etc.)
        self.max_stagger_gap = max_stagger_gap
        
    def optimize_trajectory(self, point_clouds, initial_poses=None):
        """
        Main trajectory optimization pipeline
        
        Args:
            point_clouds: List of numpy arrays, each (N_i, 3) representing point clouds
            initial_poses: Optional list of initial 4x4 pose matrices
            
        Returns:
            optimized_poses: List of optimized 4x4 pose matrices
            inlier_masks: List of boolean masks for inlier points in each frame
            trajectory_info: Dict with optimization metadata
        """
        n_frames = len(point_clouds)
        
        # Stage 1: Pairwise ICP for initialization
        print("Stage 1: Pairwise ICP initialization...")
        pairwise_poses, pairwise_constraints = self._pairwise_icp_stage(point_clouds, initial_poses)
        
        # Stage 2: Add staggered pair constraints  
        # print("Stage 2: Adding staggered pair constraints...")
        # staggered_constraints = self._staggered_pairs_stage(point_clouds, pairwise_poses)
        # staggered_constraints = []

        # Stage 3: Global pose graph optimization
        print("Stage 3: Global pose graph optimization...")
        all_constraints = pairwise_constraints
        optimized_poses = self._global_optimization_stage(pairwise_poses, all_constraints)

        return optimized_poses
        
        # Stage 4: Inlier detection and refinement
        # print("Stage 4: Inlier detection...")
        # inlier_masks, refined_poses = self._inlier_detection_stage(
            # point_clouds, optimized_poses
        # )
        
        trajectory_info = {
            'n_pairwise_constraints': len(pairwise_constraints),
            'n_staggered_constraints': len(staggered_constraints),
            'n_inliers_per_frame': [mask.sum() for mask in inlier_masks]
        }
        
        return refined_poses, inlier_masks, trajectory_info
    
    def _pairwise_icp_stage(self, point_clouds, initial_poses):
        """Stage 1: Sequential pairwise ICP"""
        n_frames = len(point_clouds)
        poses = []
        constraints = []
        
        # Initialize first pose
        if initial_poses is not None:
            poses.append(initial_poses[0])
        else:
            initial_pose = np.eye(4)
            first_points = point_clouds[0]
            centre = (first_points.min(axis=0) + first_points.max(axis=0)) / 2.0
            initial_pose[:3, 3] = centre

            poses.append(initial_pose)
            
        # Sequential pairwise ICP
        for i in range(n_frames - 1):
            cur_points = point_clouds[i]
            next_points = point_clouds[i + 1]
            
            # Your existing ICP function
            R, t, A_inliers, B_inliers = icp(
                cur_points, next_points, 
                max_iterations=self.icp_max_iterations,
                return_inliers=True,
                ret_err=False
            )
            
            # Convert to 4x4 pose matrix
            relative_pose = np.eye(4)
            relative_pose[:3, :3] = R
            relative_pose[:3, 3] = t
            
            # Accumulate pose
            next_pose = poses[i] @ relative_pose
            poses.append(next_pose)
            
            # Store constraint for global optimization
            constraint = {
                'frame_i': i,
                'frame_j': i + 1,
                'relative_pose': relative_pose,
                'inliers_i': A_inliers,
                'inliers_j': B_inliers,
                'confidence': len(A_inliers) / len(cur_points)  # Inlier ratio as confidence
            }
            constraints.append(constraint)
            
        return poses, constraints
    
    def _staggered_pairs_stage(self, point_clouds, initial_poses):
        """Stage 2: Add staggered pair constraints"""
        n_frames = len(point_clouds)
        staggered_constraints = []
        
        # Try different stagger gaps: 2, 3, 4, etc.
        for gap in range(self.stagger_step, min(self.max_stagger_gap + 1, n_frames)):
            for i in range(n_frames - gap):
                j = i + gap
                
                cur_points = point_clouds[i] 
                target_points = point_clouds[j]
                
                # Initialize with accumulated pose from pairwise stage
                initial_relative = np.linalg.inv(initial_poses[i]) @ initial_poses[j]
                
                # Apply initial transform to get better ICP starting point
                transformed_cur = self._transform_points(cur_points, initial_relative)
                
                # Run ICP
                R, t, A_inliers, B_inliers = icp(
                    transformed_cur, target_points,
                    max_iterations=self.icp_max_iterations,
                    return_inliers=True, 
                    ret_err=False
                )
                
                # Create relative pose
                icp_pose = np.eye(4)
                icp_pose[:3, :3] = R
                icp_pose[:3, 3] = t
                
                # Combine with initial estimate
                relative_pose = initial_relative @ icp_pose
                
                constraint = {
                    'frame_i': i,
                    'frame_j': j, 
                    'relative_pose': relative_pose,
                    'inliers_i': A_inliers,
                    'inliers_j': B_inliers,
                    'confidence': len(A_inliers) / len(cur_points),
                    'gap': gap
                }
                staggered_constraints.append(constraint)
                
        return staggered_constraints
        
    def _global_optimization_stage(self, initial_poses, constraints):
        """Stage 3: Global pose graph optimization using least squares"""
        n_frames = len(initial_poses)
        
        # Parameterize poses as [x, y, z, qx, qy, qz, qw] for each frame
        # Fix first frame as reference
        def poses_to_params(poses):
            params = []
            for i in range(1, len(poses)):  # Skip first frame (reference)
                pose = poses[i]
                t = pose[:3, 3]
                q = self._rotation_to_quaternion(pose[:3, :3])
                params.extend([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])
            return np.array(params)
        
        def params_to_poses(params):
            poses = [initial_poses[0]]  # First pose fixed
            for i in range(n_frames - 1):
                start_idx = i * 7
                t = params[start_idx:start_idx+3]
                q = params[start_idx+3:start_idx+7] 
                q = q / np.linalg.norm(q)  # Normalize quaternion
                
                pose = np.eye(4)
                pose[:3, :3] = self._quaternion_to_rotation(q)
                pose[:3, 3] = t
                poses.append(pose)
            return poses
        
        def residual_function(params):
            poses = params_to_poses(params)
            residuals = []
            
            for constraint in constraints:
                i, j = constraint['frame_i'], constraint['frame_j']
                expected_relative = constraint['relative_pose']
                confidence = constraint['confidence']
                
                # Compute actual relative pose
                actual_relative = np.linalg.inv(poses[i]) @ poses[j]
                
                # Compute pose difference
                diff_pose = np.linalg.inv(expected_relative) @ actual_relative
                
                # Extract translation and rotation errors
                t_error = diff_pose[:3, 3]
                R_error = diff_pose[:3, :3]
                
                # Convert rotation error to angle-axis
                angle_error = self._rotation_matrix_to_angle_axis(R_error)
                
                # Weight by confidence (inlier ratio)
                weight = np.sqrt(confidence)
                
                # Add weighted residuals
                residuals.extend(weight * t_error)
                residuals.extend(weight * angle_error)

            # regularization_weight = 0.01
            # for i in range(len(params)):
            #     residuals.append(regularization_weight * (params[i] - initial_params[i]))
                
            return np.array(residuals)
        
        # Run optimization
        initial_params = poses_to_params(initial_poses)
        result = least_squares(residual_function, initial_params, method='trf')
        
        optimized_poses = params_to_poses(result.x)
        return optimized_poses
    
    def _inlier_detection_stage(self, point_clouds, optimized_poses):
        """Stage 4: Find inliers using globally optimized trajectory"""
        n_frames = len(point_clouds)
        
        # Transform all points to object coordinate frames
        # Since points are in world coords, use inv(pose) to get object coords
        aligned_point_clouds = []
        for i, (points, pose) in enumerate(zip(point_clouds, optimized_poses)):
            world_to_object = np.linalg.inv(pose)
            object_frame_points = self._transform_points(points, world_to_object)
            aligned_point_clouds.append(object_frame_points)
        
        # Apply ppscore to find temporally consistent points
        inlier_masks = []
        for i, object_points in enumerate(aligned_point_clouds):
            # Use all other frames as neighbors for ppscore
            neighbor_frames = aligned_point_clouds[:i] + aligned_point_clouds[i+1:]
            
            ppscore = compute_ppscore(object_points, neighbor_frames)
            
            # Threshold ppscore to get inliers (adjust threshold as needed)
            ppscore_threshold = np.percentile(ppscore, 70)  # Top 30% most consistent
            spatial_inliers = ppscore >= ppscore_threshold
            
            # Additional spatial clustering to remove outliers
            if spatial_inliers.sum() > 10:  # Need minimum points for clustering
                inlier_points = object_points[spatial_inliers]
                clustering = DBSCAN(eps=0.5, min_samples=3).fit(inlier_points)
                
                # Keep largest cluster
                labels = clustering.labels_
                if len(np.unique(labels[labels >= 0])) > 0:
                    largest_cluster = np.bincount(labels[labels >= 0]).argmax()
                    cluster_mask = labels == largest_cluster
                    
                    # Map back to original indices
                    final_inliers = np.zeros(len(object_points), dtype=bool)
                    spatial_inlier_indices = np.where(spatial_inliers)[0]
                    final_inliers[spatial_inlier_indices[cluster_mask]] = True
                    
                    inlier_masks.append(final_inliers)
                else:
                    inlier_masks.append(spatial_inliers)
            else:
                inlier_masks.append(spatial_inliers)
        
        # Optional: Refine poses using only inliers
        refined_poses = self._refine_poses_with_inliers(
            point_clouds, optimized_poses, inlier_masks
        )
        
        return inlier_masks, refined_poses
    
    def _refine_poses_with_inliers(self, point_clouds, poses, inlier_masks):
        """Optional refinement step using only inlier points"""
        refined_poses = poses.copy()
        
        # Re-run pairwise ICP with inlier points only
        for i in range(len(point_clouds) - 1):
            cur_inliers = point_clouds[i][inlier_masks[i]]
            next_inliers = point_clouds[i + 1][inlier_masks[i + 1]]
            
            if len(cur_inliers) > 10 and len(next_inliers) > 10:  # Minimum points needed
                R, t, _, _ = icp(
                    cur_inliers, next_inliers,
                    max_iterations=self.icp_max_iterations,
                    return_inliers=True,
                    ret_err=False
                )
                
                # Update relative transformation
                relative_pose = np.eye(4)
                relative_pose[:3, :3] = R  
                relative_pose[:3, 3] = t
                
                # Apply small correction to pose
                correction_weight = 0.3  # Blend with original estimate
                current_relative = np.linalg.inv(refined_poses[i]) @ refined_poses[i + 1]
                blended_relative = self._blend_poses(current_relative, relative_pose, correction_weight)
                
                refined_poses[i + 1] = refined_poses[i] @ blended_relative
                
        return refined_poses
    
    def analyze_trajectory_quality(self, point_clouds, poses, inlier_masks):
        """Analyze trajectory quality metrics"""
        metrics = {}
        
        # 1. Trajectory smoothness (acceleration changes)
        positions = np.array([pose[:3, 3] for pose in poses])
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0) 
        
        metrics['trajectory_smoothness'] = np.mean(np.linalg.norm(accelerations, axis=1))
        
        # 2. Inlier consistency across frames
        inlier_ratios = [mask.mean() for mask in inlier_masks]
        metrics['mean_inlier_ratio'] = np.mean(inlier_ratios)
        metrics['inlier_std'] = np.std(inlier_ratios)
        
        # 3. Point cloud overlap between consecutive frames
        overlaps = []
        for i in range(len(point_clouds) - 1):
            overlap = self._compute_overlap(
                point_clouds[i], point_clouds[i + 1], 
                poses[i], poses[i + 1]
            )
            overlaps.append(overlap)
        metrics['mean_overlap'] = np.mean(overlaps)
        
        return metrics
    
    def visualize_optimization_results(self, point_clouds, initial_poses, final_poses, 
                                     inlier_masks, save_path=None):
        """Create comprehensive visualization of optimization results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Trajectory comparison
        initial_positions = np.array([pose[:3, 3] for pose in initial_poses])
        final_positions = np.array([pose[:3, 3] for pose in final_poses])
        
        ax1.plot(initial_positions[:, 0], initial_positions[:, 1], 'r--', 
                label='Initial (Pairwise ICP)', linewidth=2)
        ax1.plot(final_positions[:, 0], final_positions[:, 1], 'b-', 
                label='Optimized (Global)', linewidth=2)
        ax1.scatter(initial_positions[0, 0], initial_positions[0, 1], 
                   c='green', s=100, marker='o', label='Start')
        ax1.scatter(final_positions[-1, 0], final_positions[-1, 1], 
                   c='red', s=100, marker='x', label='End')
        ax1.set_title('Trajectory Comparison')
        ax1.legend()
        ax1.grid(True)
        ax1.set_aspect('equal')
        
        # 2. Inlier ratios over time
        inlier_ratios = [mask.mean() for mask in inlier_masks]
        ax2.plot(range(len(inlier_ratios)), inlier_ratios, 'g-', linewidth=2)
        ax2.set_title('Inlier Ratio Over Time')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Inlier Ratio')
        ax2.grid(True)
        
        # 3. Point cloud with inliers highlighted (middle frame)
        mid_frame = len(point_clouds) // 2
        mid_points = point_clouds[mid_frame]
        mid_inliers = inlier_masks[mid_frame]
        
        ax3.scatter(mid_points[~mid_inliers, 0], mid_points[~mid_inliers, 1], 
                   c='red', s=1, alpha=0.3, label='Outliers')
        ax3.scatter(mid_points[mid_inliers, 0], mid_points[mid_inliers, 1],
                   c='blue', s=1, alpha=0.7, label='Inliers')
        ax3.set_title(f'Inliers vs Outliers (Frame {mid_frame})')
        ax3.legend()
        ax3.set_aspect('equal')
        ax3.grid(True)
        
        # 4. Aligned point clouds overlay - FIXED TRANSFORMATION  
        # Since point_clouds are in world coordinates, transform each to object coordinate frame
        # If optimization worked correctly, they should all overlap when viewed in object space
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(point_clouds)))
        
        for i, (points, pose) in enumerate(zip(point_clouds, final_poses)):
            # Transform world coordinates to object coordinate frame  
            # pose = object-to-world transformation
            # So world-to-object = inv(pose)
            world_to_object = np.linalg.inv(pose)
            object_frame_points = self._transform_points(points, world_to_object)
            inliers = object_frame_points[inlier_masks[i]]
            
            ax4.scatter(inliers[:, 0], inliers[:, 1], 
                       c=[colors[i]], s=1, alpha=0.6, label=f'Frame {i}')
        
        ax4.set_title('Object-Frame Aligned Point Clouds (Inliers Only)')
        ax4.set_aspect('equal') 
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def extract_consistent_object_points(self, point_clouds, optimized_poses, inlier_masks):
        """
        Extract points that are consistent across the trajectory.
        This integrates with your ppscore approach.
        """
        # Transform all point clouds to object coordinate frame
        # Since point clouds are in world coords, use inv(pose) to get object coords
        aligned_clouds = []
        
        for i, (points, pose, inliers) in enumerate(zip(point_clouds, optimized_poses, inlier_masks)):
            # Use only inlier points
            inlier_points = points[inliers]
            
            # Transform world coordinates to object coordinate frame
            world_to_object = np.linalg.inv(pose)
            object_frame_points = self._transform_points(inlier_points, world_to_object)
            aligned_clouds.append(object_frame_points)
        
        # Apply your ppscore function to find most consistent points
        if len(aligned_clouds) > 1:
            # Use last frame as query, others as neighbors
            query_points = aligned_clouds[-1]
            neighbor_clouds = aligned_clouds[:-1]
            
            ppscore = compute_ppscore(query_points, neighbor_clouds)
            
            # Extract highly consistent points (top percentile)
            consistency_threshold = np.percentile(ppscore, 80)
            consistent_mask = ppscore >= consistency_threshold
            
            consistent_points = query_points[consistent_mask]
            
            return {
                'aligned_clouds': aligned_clouds,
                'ppscore': ppscore,
                'consistent_points': consistent_points,
                'consistent_mask': consistent_mask
            }
        else:
            return None
    
    # Utility functions
    def _transform_points(self, points, pose):
        """Transform points using 4x4 pose matrix"""
        points_h = np.hstack([points, np.ones((len(points), 1))])
        transformed = (pose @ points_h.T).T
        return transformed[:, :3]
    
    def _rotation_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [x,y,z,w]"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s  
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([x, y, z, w])
    
    def _quaternion_to_rotation(self, q):
        """Convert quaternion [x,y,z,w] to rotation matrix"""
        x, y, z, w = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)], 
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]
        ])
    
    def _rotation_matrix_to_angle_axis(self, R):
        """Convert rotation matrix to angle-axis representation"""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle < 1e-6:
            return np.zeros(3)
        
        axis = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / (2 * np.sin(angle))
        return angle * axis
    
    def _blend_poses(self, pose1, pose2, weight):
        """Blend two poses with given weight"""
        # Simple linear blending for translation
        t_blend = (1 - weight) * pose1[:3, 3] + weight * pose2[:3, 3]
        
        # Quaternion SLERP for rotation  
        q1 = self._rotation_to_quaternion(pose1[:3, :3])
        q2 = self._rotation_to_quaternion(pose2[:3, :3])
        q_blend = self._slerp_quaternions(q1, q2, weight)
        
        result = np.eye(4)
        result[:3, :3] = self._quaternion_to_rotation(q_blend)
        result[:3, 3] = t_blend
        return result
    
    def _slerp_quaternions(self, q1, q2, t):
        """Spherical linear interpolation between quaternions"""
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
            
        if dot > 0.9995:
            # Linear interpolation for nearly identical quaternions
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        theta_0 = np.arccos(np.abs(dot))
        theta = theta_0 * t
        
        q2_perp = q2 - q1 * dot
        q2_perp = q2_perp / np.linalg.norm(q2_perp)
        
        return q1 * np.cos(theta) + q2_perp * np.sin(theta)
    
    def _compute_overlap(self, points1, points2, pose1, pose2, threshold=0.5):
        """Compute overlap ratio between two point clouds"""
        # Transform both to object coordinate frame for comparison
        # Since points are in world coords, use inv(pose) to get object coords
        object_points1 = self._transform_points(points1, np.linalg.inv(pose1))
        object_points2 = self._transform_points(points2, np.linalg.inv(pose2))
        
        # Build KDTree for points2 in object space
        tree = cKDTree(object_points2)
        
        # Count points1 that have close neighbors in points2  
        distances, _ = tree.query(object_points1, k=1)
        overlapping = np.sum(distances < threshold)
        
        return overlapping / len(points1)