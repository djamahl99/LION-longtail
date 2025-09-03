import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pathlib import Path

from .box_utils import icp, compute_ppscore

import gtsam

def optimize_with_gtsam_timed(initial_poses, constraints, timestamps_ns):
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
    
    # 3. Add prior on first pose (fix it as reference frame)
    # For Pose3: 6 DOF - 3 rotation, 3 translation
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
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
        
        # Create noise model based on confidence and time
        base_sigma = 1.0 / (confidence + 1e-6)
        
        # Different uncertainty for rotation (first 3) vs translation (last 3)
        rot_sigma = base_sigma * time_factor * 0.1  # radians
        trans_sigma = base_sigma * time_factor * 0.01  # meters
        
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([rot_sigma, rot_sigma, rot_sigma,
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
        self.max_icp_iterations = max_icp_iterations
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
                max_iterations=self.max_icp_iterations,
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
                    max_iterations=self.max_icp_iterations,
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
                    max_iterations=self.max_icp_iterations,
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