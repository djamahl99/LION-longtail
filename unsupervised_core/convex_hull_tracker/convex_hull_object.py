import numpy as np
import trimesh
from lion.unsupervised_core.box_utils import get_rotated_3d_box_corners
from lion.unsupervised_core.convex_hull_tracker.pose_kalman_filter import PoseKalmanFilter
from lion.unsupervised_core.outline_utils import points_rigid_transform


MAX_POINTS = 1024

class ConvexHullObject(object):
    original_points: np.ndarray = None
    confidence: float
    feature: np.ndarray
    timestamp: int
    source: str
    box: np.ndarray
    def __init__(
        self,
        original_points: np.ndarray,
        confidence: float,
        feature: np.ndarray,
        timestamp: int,
        source: str = "vision_guided",
    ):
        dims_mins = original_points.min(axis=0)
        dims_maxes = original_points.max(axis=0)

        lwh = dims_maxes - dims_mins

        if np.any(lwh > 15) or np.any(lwh < 0.1):
            return None

        if len(original_points) > MAX_POINTS:
            orig_len = len(original_points)
            indices = np.random.randint(0, orig_len, size=MAX_POINTS)
            original_points = original_points[indices]
            print(f"Resampled original_points {orig_len} -> {len(original_points)}")

        self.original_points = original_points.copy()
        self.confidence = confidence
        self.feature = feature.copy()
        self.timestamp = timestamp
        self.source = source

        self.mesh = trimesh.convex.convex_hull(self.original_points)
        self.box = ConvexHullObject.points_to_bounding_box(self.original_points, centre=self.mesh.centroid)
        self.centroid_3d = self.box[:3]
        self.pose_vector = PoseKalmanFilter.box_to_pose_vector(self.box)

        obj_pose = PoseKalmanFilter.pose_vector_to_transform(self.pose_vector)          
        
        world_to_object = np.linalg.inv(obj_pose)
        self.object_points = points_rigid_transform(original_points, world_to_object)

    @staticmethod
    def points_to_bounding_box(points3d: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """Convert single alpha shape to bounding box [x, y, z, length, width, height, yaw]."""
        try:
            points_2d = points3d[:, :2]
            points_z = points3d[:, 2]

            z_min = points_z.min()
            z_max = points_z.max()

            # Center
            # center_2d = (points_2d.min(axis=0) + points_2d.max(axis=0)) / 2.0
            # center_x, center_y = center_2d
            # center_z = (z_min + z_max) / 2
            centre_x, centre_y, centre_z = centre
            height = z_max - z_min

            # Estimate orientation using PCA
            try:
                centered_vertices = points_2d - centre[:3]
                cov_matrix = np.cov(centered_vertices.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # Sort by eigenvalue (largest first)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Primary direction (largest eigenvalue) and secondary direction
                primary_direction = eigenvectors[:, 0]  # Length direction
                secondary_direction = eigenvectors[:, 1]  # Width direction
                
                # Calculate yaw from primary direction
                yaw = np.arctan2(primary_direction[1], primary_direction[0])
                
                # Project vertices onto principal directions to get length and width
                projections_length = np.dot(centered_vertices, primary_direction)
                projections_width = np.dot(centered_vertices, secondary_direction)
                
                # Length and width are ranges of projections
                length = np.max(projections_length) - np.min(projections_length)
                width = np.max(projections_width) - np.min(projections_width)

            except:
                # Fallback to axis-aligned approach
                min_x, min_y = np.min(points_2d, axis=0)
                max_x, max_y = np.max(points_2d, axis=0)
                length = max_x - min_x
                width = max_y - min_y
                yaw = 0.0

            return np.array([centre_x, centre_y, centre_z, length, width, height, yaw])

        except Exception as e:
            print(f"Error converting alpha shape to box: {e}")
            
            return np.zeros((7,), dtype=np.float32)
