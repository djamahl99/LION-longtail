from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh

from lion.unsupervised_core.box_utils import get_rotated_3d_box_corners
from lion.unsupervised_core.convex_hull_tracker.convex_hull_utils import (
    voxel_sampling_fast,
)
from lion.unsupervised_core.convex_hull_tracker.pose_kalman_filter import (
    PoseKalmanFilter,
)
from lion.unsupervised_core.outline_utils import points_rigid_transform

MAX_POINTS = 1024


class LShapeCriterion(Enum):
    """Criteria for L-shape fitting optimization."""
    AREA = "area"
    NEAREST = "nearest"
    VARIANCE = "variance"

class LShapedFIT:
    """
    L-shape fitting implementation based on "Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners".
    
    This implementation matches the C++ version from the paper as closely as possible.
    """
    
    def __init__(self, 
                 min_dist_of_nearest_crit: float = 0.01,
                 dtheta_deg_for_search: float = 1.0,
                 criterion: LShapeCriterion = LShapeCriterion.VARIANCE):
        """
        Initialize L-shape fitting parameters.
        
        Args:
            min_dist_of_nearest_crit: Minimum distance threshold for nearest criterion
            dtheta_deg_for_search: Angular resolution in degrees for search
            criterion: Optimization criterion to use
        """
        self.min_dist_of_nearest_crit = min_dist_of_nearest_crit
        self.dtheta_deg_for_search = dtheta_deg_for_search
        self.criterion = criterion
        self.vertex_pts = []
        self.a = []  # Line equation coefficients ax + by + c = 0
        self.b = []
        self.c = []
    
    def fit_box(self, points_2d: np.ndarray) -> Optional[Dict]:
        """
        Fit L-shaped bounding box to 2D points.
        
        Args:
            points_2d: Nx2 array of 2D points
            
        Returns:
            Dict with keys: center, size, angle_rad, vertices
            Returns None if fitting fails
        """
        if len(points_2d) < 3:
            return None
        
        # Convert to matrix format (N x 2)
        matrix_pts = points_2d.astype(np.float64)
        
        dtheta = np.deg2rad(self.dtheta_deg_for_search)
        
        minimal_cost = -np.inf
        best_theta = np.inf
        
        # Search for best direction (0 to π/2 - dtheta)
        loop_number = int(np.ceil((np.pi/2.0 - dtheta) / dtheta))
        
        for k in range(loop_number):
            theta = k * dtheta
            
            # Ensure theta is in valid range
            if theta < (np.pi/2.0 - dtheta):
                # Create orthogonal unit vectors
                e1 = np.array([np.cos(theta), np.sin(theta)])
                e2 = np.array([-np.sin(theta), np.cos(theta)])
                
                # Project points onto the two directions
                c1 = matrix_pts @ e1  # Shape: (N,)
                c2 = matrix_pts @ e2  # Shape: (N,)
                
                # Calculate cost based on selected criterion
                if self.criterion == LShapeCriterion.AREA:
                    cost = self._calc_area_criterion(c1, c2)
                elif self.criterion == LShapeCriterion.NEAREST:
                    cost = self._calc_nearest_criterion(c1, c2)
                elif self.criterion == LShapeCriterion.VARIANCE:
                    cost = self._calc_variances_criterion(c1, c2)
                else:
                    print("L-Shaped Algorithm Criterion Is Not Supported.")
                    return None
                
                if minimal_cost < cost:
                    minimal_cost = cost
                    best_theta = theta
            else:
                break
        
        # Check if valid solution found
        if minimal_cost <= -np.inf or best_theta >= np.inf:
            print("RotatedRect Fit Failed.")
            return None
        
        # Calculate final bounding box with best theta
        sin_s = np.sin(best_theta)
        cos_s = np.cos(best_theta)
        
        e1_s = np.array([cos_s, sin_s])
        e2_s = np.array([-sin_s, cos_s])
        
        c1_s = matrix_pts @ e1_s
        c2_s = matrix_pts @ e2_s
        
        min_c1_s = np.min(c1_s)
        max_c1_s = np.max(c1_s)
        min_c2_s = np.min(c2_s)
        max_c2_s = np.max(c2_s)
        
        # Store line equation coefficients for rectangle edges
        self.a = [cos_s, -sin_s, cos_s, -sin_s]
        self.b = [sin_s, cos_s, sin_s, cos_s]
        self.c = [min_c1_s, min_c2_s, max_c1_s, max_c2_s]
        
        # Calculate rectangle vertices and properties
        return self._calc_rect_contour(best_theta)
    
    def _calc_area_criterion(self, c1: np.ndarray, c2: np.ndarray) -> float:
        """Calculate area criterion (negative bounding box area)."""
        c1_min = np.min(c1)
        c1_max = np.max(c1)
        c2_min = np.min(c2)
        c2_max = np.max(c2)
        
        alpha = -(c1_max - c1_min) * (c2_max - c2_min)
        return alpha
    
    def _calc_nearest_criterion(self, c1: np.ndarray, c2: np.ndarray) -> float:
        """Calculate nearest criterion (sum of inverse distances to edges)."""
        c1_min = np.min(c1)
        c1_max = np.max(c1)
        c2_min = np.min(c2)
        c2_max = np.max(c2)
        
        # Distance to nearest edge for each projection
        d1 = np.minimum(np.abs(c1_max - c1), np.abs(c1 - c1_min))
        d2 = np.minimum(np.abs(c2_max - c2), np.abs(c2 - c2_min))
        
        beta = 0.0
        for i in range(len(d1)):
            d = max(min(d1[i], d2[i]), self.min_dist_of_nearest_crit)
            beta += (1.0 / d)
        
        return beta
    
    def _calc_variances_criterion(self, c1: np.ndarray, c2: np.ndarray) -> float:
        """Calculate variance criterion (negative variance of edge distances)."""
        c1_min = np.min(c1)
        c1_max = np.max(c1)
        c2_min = np.min(c2)
        c2_max = np.max(c2)
        
        # Distance to nearest edge for each projection
        d1 = np.minimum(np.abs(c1_max - c1), np.abs(c1 - c1_min))
        d2 = np.minimum(np.abs(c2_max - c2), np.abs(c2 - c2_min))
        
        # Split points based on which direction has smaller distance to edge
        e1 = []
        e2 = []
        
        for i in range(len(d1)):
            if d1[i] < d2[i]:
                e1.append(d1[i])
            else:
                e2.append(d2[i])
        
        v1 = 0.0
        if len(e1) > 0:
            v1 = -self._calc_var(e1)
        
        v2 = 0.0
        if len(e2) > 0:
            v2 = -self._calc_var(e2)
        
        gamma = v1 + v2
        return gamma
    
    def _calc_var(self, v: list) -> float:
        """Calculate standard deviation of a list of values."""
        if len(v) <= 1:
            return 0.0
        
        mean_val = np.mean(v)
        variance = np.sum((np.array(v) - mean_val) ** 2) / (len(v) - 1)
        return np.sqrt(variance)
    
    def _calc_cross_point(self, a0: float, a1: float, b0: float, b1: float, 
                         c0: float, c1: float) -> Tuple[float, float]:
        """Calculate intersection point of two lines."""
        # Lines: a0*x + b0*y + c0 = 0 and a1*x + b1*y + c1 = 0
        x = (b0 * (-c1) - b1 * (-c0)) / (a0 * b1 - a1 * b0)
        y = (a1 * (-c0) - a0 * (-c1)) / (a0 * b1 - a1 * b0)
        return x, y
    
    def _calc_rect_contour(self, angle_rad: float) -> Dict:
        """Calculate rectangle vertices and return bounding box parameters."""
        self.vertex_pts = []
        
        # Calculate four corner points by intersecting adjacent edges
        top_left_x, top_left_y = self._calc_cross_point(
            self.a[0], self.a[1], self.b[0], self.b[1], self.c[0], self.c[1])
        self.vertex_pts.append([top_left_x, top_left_y])
        
        top_right_x, top_right_y = self._calc_cross_point(
            self.a[1], self.a[2], self.b[1], self.b[2], self.c[1], self.c[2])
        self.vertex_pts.append([top_right_x, top_right_y])
        
        bottom_right_x, bottom_right_y = self._calc_cross_point(
            self.a[2], self.a[3], self.b[2], self.b[3], self.c[2], self.c[3])
        self.vertex_pts.append([bottom_right_x, bottom_right_y])
        
        bottom_left_x, bottom_left_y = self._calc_cross_point(
            self.a[3], self.a[0], self.b[3], self.b[0], self.c[3], self.c[0])
        self.vertex_pts.append([bottom_left_x, bottom_left_y])
        
        # Calculate center, size, and return formatted result
        vertices = np.array(self.vertex_pts)
        center = np.mean(vertices, axis=0)
        
        # Calculate width and height
        edge1 = vertices[1] - vertices[0]
        edge2 = vertices[3] - vertices[0]
        width = np.linalg.norm(edge1)
        height = np.linalg.norm(edge2)
        
        return {
            'center': center,
            'size': np.array([width, height]),
            'angle_rad': angle_rad,
            'vertices': vertices
        }
    
    def get_rect_vertices(self) -> np.ndarray:
        """Get the four corner points of the fitted rectangle."""
        return np.array(self.vertex_pts)
        