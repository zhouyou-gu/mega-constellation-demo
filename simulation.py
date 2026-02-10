#!/usr/bin/env python3
"""Mega-constellation digital twin and visualization.

This module loads Starlink TLE data, filters invalid satellites, computes Earth’s
rotation, and visualizes Earth plus satellites in a 3D scene. It is structured to
separate concerns:

1) Numerical kernels (Numba-accelerated) for geometry and filtering
2) Visualization setup (Vispy scene construction)
3) Digital twin orchestration (state update, rendering, and profiling)
Author: Zhouyou Gu (SUTD) – zhouyou_gu@sutd.edu.sg
"""

import cProfile
import io
import logging
import math
import os
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
import psutil
from numba import njit, prange
from PIL import Image
from scipy.spatial import cKDTree

from skyfield.api import load
from skyfield.sgp4lib import TEME
from sgp4.api import SatrecArray
from vispy import app, gloo, scene
import vispy.io as vispy_io
from vispy.geometry import MeshData, create_sphere
from vispy.gloo.util import _screenshot
from vispy.visuals.filters import TextureFilter
from vispy.visuals.transforms import MatrixTransform, STTransform

# ---------------------------
# Logging and profiling tools
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def cprofile_context():
    """Context manager to log the most expensive profiler entries."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # shows top 10 lines
        logger.debug("++%s", s.getvalue())


# ---------------------------
# Configuration and constants
# ---------------------------
@dataclass(frozen=True)
class DigitalTwinConfig:
    """Configuration parameters for digital twin behavior and visualization."""

    for_theta_deg: float = 15.0
    lisl_max_distance_km: float = 3000.0
    time_scale: float = 10.0
    earth_radius_km: float = 6371.0
    plot_potential_lisl: bool = False
    texture_path: str = "population_density_texture.png"
    arrow_length_scale: float = 0.01


@dataclass(frozen=True)
class CaptureConfig:
    """Configuration for screenshots and GIF recording."""

    screenshot_dir: str = "images"
    gif_fps: int = 15
    gif_max_frames: int = 300
    gif_output_dir: str = "images/gifs"
    gif_scale: float = 0.3
    gif_colors: int = 128


DEFAULT_CONFIG = DigitalTwinConfig()
DEFAULT_CAPTURE_CONFIG = CaptureConfig()


# ---------------------------
# Helper utilities (non-Numba)
# ---------------------------
def expand_edges_with_original(edges: np.ndarray, repeat_per_node: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand edges by duplicating each edge in a grid fashion.

    Parameters:
        edges (np.ndarray): Array of edges.
        repeat_per_node (int): Number of repetitions per node.

    Returns:
        tuple: (repeated_original, expanded_edges)
    """
    grid = np.stack(np.meshgrid(np.arange(repeat_per_node), np.arange(repeat_per_node),
                                indexing='ij'), axis=-1).reshape(-1, 2)
    expanded_edges = (edges[:, None, :] * repeat_per_node + grid[None, :, :]).reshape(-1, 2)
    repeated_original = np.repeat(edges, repeat_per_node * repeat_per_node, axis=0)
    return repeated_original, expanded_edges

# ---------------------------
# Numba-accelerated kernels
# ---------------------------
@njit(parallel=True,cache=True)
def update_arrows(velocities, positions):
    """
    Compute the direction arrows in parallel.

    Parameters:
    -----------
    velocities : np.ndarray
        Array of shape (n, 3) containing velocity vectors.
    positions : np.ndarray
        Array of shape (n, 3) containing position vectors.

    Returns:
    --------
    front : np.ndarray
        Normalized velocity vectors.
    back : np.ndarray
        Negated front vectors.
    down : np.ndarray
        Normalized position vectors.
    right : np.ndarray
        Cross product of down and front vectors.
    left : np.ndarray
        Negated right vectors.
    """
    n = velocities.shape[0]
    front = np.empty_like(velocities)
    back = np.empty_like(velocities)
    down = np.empty_like(positions)
    right = np.empty_like(positions)
    left = np.empty_like(positions)

    for i in prange(n):
        # Normalize the velocity vector for the "front" arrow
        v0 = velocities[i, 0]
        v1 = velocities[i, 1]
        v2 = velocities[i, 2]
        norm_v = math.sqrt(v0*v0 + v1*v1 + v2*v2)
        if norm_v == 0:
            # Degenerate velocity; keep direction as zero-vector
            front[i, 0] = 0.0
            front[i, 1] = 0.0
            front[i, 2] = 0.0
        else:
            front[i, 0] = v0 / norm_v
            front[i, 1] = v1 / norm_v
            front[i, 2] = v2 / norm_v

        # "Back" is simply the negative of "front"
        back[i, 0] = -front[i, 0]
        back[i, 1] = -front[i, 1]
        back[i, 2] = -front[i, 2]

        # Normalize the position vector for the "down" arrow
        p0 = positions[i, 0]
        p1 = positions[i, 1]
        p2 = positions[i, 2]
        norm_p = math.sqrt(p0*p0 + p1*p1 + p2*p2)
        if norm_p == 0:
            # Degenerate position; keep direction as zero-vector
            down[i, 0] = 0.0
            down[i, 1] = 0.0
            down[i, 2] = 0.0
        else:
            down[i, 0] = p0 / norm_p
            down[i, 1] = p1 / norm_p
            down[i, 2] = p2 / norm_p

        # Compute the cross product for "right": cross(down, front)
        right[i, 0] = down[i, 1] * front[i, 2] - down[i, 2] * front[i, 1]
        right[i, 1] = down[i, 2] * front[i, 0] - down[i, 0] * front[i, 2]
        right[i, 2] = down[i, 0] * front[i, 1] - down[i, 1] * front[i, 0]

        # "Left" is simply the negative of "right"
        left[i, 0] = -right[i, 0]
        left[i, 1] = -right[i, 1]
        left[i, 2] = -right[i, 2]

    return front, back, down, right, left

@njit(parallel=True,cache=True)
def filter_and_compute_pair(edges, view_from_stack, view_to_stack, cos_threshold):
    """
    This function replicates the following operations:
    
      1. Create boolean arrays (i_j_indicator and j_i_indicator) by comparing each element 
         of view_from_stack and view_to_stack to cos_threshold.
      2. For each row, use a manual "any" to compute binary indicators.
      3. Filter rows (and corresponding edges) where both binary indicators are True.
      4. For the filtered rows, compute the broadcasted pairwise boolean AND between 
         view_from_stack and view_to_stack indicators and flatten the result.
    
    Parameters
    ----------
    edges : np.ndarray
        Array of shape (n, m_edges) representing edge data (e.g., indices).
    view_from_stack : np.ndarray
        Array of shape (n, m1) with float values.
    view_to_stack : np.ndarray
        Array of shape (n, m2) with float values.
    cos_threshold : float
        Threshold for comparison.
        
    Returns
    -------
    filtered_edges : np.ndarray
        Filtered rows of edges.
    filtered_view_from_stack : np.ndarray
        Filtered rows of view_from_stack.
    filtered_view_to_stack : np.ndarray
        Filtered rows of view_to_stack.
    p_lisl_LT_pair : np.ndarray
        Flattened boolean array from the broadcasted pairwise AND between 
        the filtered view_from_stack and view_to_stack boolean indicators.
    """
    n = view_from_stack.shape[0]
    m1 = view_from_stack.shape[1]
    m2 = view_to_stack.shape[1]
    m_edges = edges.shape[1]
    
    # Step 1: Compute boolean indicator arrays.
    i_j_indicator = np.empty((n, m1), dtype=np.bool_)
    j_i_indicator = np.empty((n, m2), dtype=np.bool_)
    for i in prange(n):
        for j in range(m1):
            i_j_indicator[i, j] = view_from_stack[i, j] > cos_threshold
        for j in range(m2):
            j_i_indicator[i, j] = view_to_stack[i, j] > cos_threshold
    
    # Step 2: Compute per-row "any" (binary indicators).
    i_j_binary = np.empty(n, dtype=np.bool_)
    j_i_binary = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        flag_from = False
        for j in range(m1):
            if i_j_indicator[i, j]:
                flag_from = True
                break
        i_j_binary[i] = flag_from

        flag_to = False
        for j in range(m2):
            if j_i_indicator[i, j]:
                flag_to = True
                break
        j_i_binary[i] = flag_to
    
    # Step 3: Final indicator: only rows where both are True.
    final_indicator = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        final_indicator[i] = i_j_binary[i] and j_i_binary[i]
    
    # Count rows passing the final condition.
    count = 0
    for i in range(n):
        if final_indicator[i]:
            count += 1
            
    # Allocate filtered arrays.
    filtered_edges = np.empty((count, m_edges), dtype=edges.dtype)
    filtered_view_from_stack = np.empty((count, m1), dtype=view_from_stack.dtype)
    filtered_view_to_stack = np.empty((count, m2), dtype=view_to_stack.dtype)
    filtered_i_j_indicator = np.empty((count, m1), dtype=np.bool_)
    filtered_j_i_indicator = np.empty((count, m2), dtype=np.bool_)
    
    # Copy over the rows that pass the threshold.
    idx = 0
    for i in range(n):
        if final_indicator[i]:
            for j in range(m_edges):
                filtered_edges[idx, j] = edges[i, j]
            for j in range(m1):
                filtered_view_from_stack[idx, j] = view_from_stack[i, j]
                filtered_i_j_indicator[idx, j] = i_j_indicator[i, j]
            for j in range(m2):
                filtered_view_to_stack[idx, j] = view_to_stack[i, j]
                filtered_j_i_indicator[idx, j] = j_i_indicator[i, j]
            idx += 1
    
    # Step 4: Compute the broadcasted pairwise AND.
    # For each filtered row, we want to compute a boolean matrix of shape (m1, m2)
    total_elements = count * m1 * m2
    p_lisl_LT_pair = np.empty(total_elements, dtype=np.bool_)
    for i in prange(count):
        for j in range(m1):
            for k in range(m2):
                flat_index = i * (m1 * m2) + j * m2 + k
                p_lisl_LT_pair[flat_index] = filtered_i_j_indicator[i, j] and filtered_j_i_indicator[i, k]
                
    return filtered_edges, filtered_view_from_stack, filtered_view_to_stack, p_lisl_LT_pair


@njit(parallel=True,cache=True)
def compute_directions(positions, edges):
    # positions: array of shape (n, d) with n points in d dimensions
    # edges: array of shape (m, 2) where each row defines an edge by indices into positions
    m = edges.shape[0]
    d = positions.shape[1]
    directions = np.empty((m, d))
    
    # Parallel loop over edges
    for i in prange(m):
        idx0 = edges[i, 0]
        idx1 = edges[i, 1]
        
        # Compute the difference vector
        diff = np.empty(d)
        for j in range(d):
            diff[j] = positions[idx1, j] - positions[idx0, j]
        
        # Compute the Euclidean norm
        norm = 0.0
        for j in range(d):
            norm += diff[j] * diff[j]
        norm = np.sqrt(norm)

        # Normalize the difference vector (guard against zero length)
        if norm == 0:
            for j in range(d):
                directions[i, j] = 0.0
        else:
            for j in range(d):
                directions[i, j] = diff[j] / norm

    return directions

@njit(parallel=True, cache=True)
def compute_view_LT_pair_min_cos(filtered_view_from_stack, filtered_view_to_stack, p_lisl_LT_pair):
    # Assume shapes:
    # filtered_view_from_stack: (A, B)
    # filtered_view_to_stack: (A, C)
    A, B = filtered_view_from_stack.shape
    A2, C = filtered_view_to_stack.shape
    # Allocate an output array for the broadcasted minimum with shape (A, B, C)
    min_vals = np.empty((A, B, C), dtype=filtered_view_from_stack.dtype)
    
    # Manually compute the elementwise minimum for the broadcasted arrays.
    for i in prange(A):
        for j in range(B):
            for k in range(C):
                a_val = filtered_view_from_stack[i, j]
                b_val = filtered_view_to_stack[i, k]
                if a_val < b_val:
                    min_vals[i, j, k] = a_val
                else:
                    min_vals[i, j, k] = b_val
                    
    # Flatten the 3D array to 1D for mask-based selection.
    flat_min_vals = min_vals.reshape(-1)
    N = flat_min_vals.shape[0]
    
    # First, count the number of True (or nonzero) entries in the binary indicator.
    count = 0
    for i in range(N):
        if p_lisl_LT_pair[i] != 0:  # Works for both booleans and 0/1 integers.
            count += 1
            
    # Allocate final result with shape (count, 1)
    final_result = np.empty((count, 1), dtype=filtered_view_from_stack.dtype)
    k = 0
    for i in range(N):
        if p_lisl_LT_pair[i] != 0:
            final_result[k, 0] = flat_min_vals[i]
            k += 1
    
    return final_result

@njit(parallel=True, cache=True)
def expand_and_filter_edges(edges, p_lisl_LT_pair, repeat_per_node=4):
    """
    Expand each edge in a grid fashion and filter the expanded arrays based on a boolean mask.
    
    Parameters
    ----------
    edges : np.ndarray
        2D array of shape (num_edges, 2) where each row represents an edge.
    p_lisl_LT_pair : np.ndarray
        Boolean 1D array of length (num_edges * repeat_per_node^2) indicating which expanded entries to keep.
    repeat_per_node : int, optional
        Number of repetitions per node (default is 4). The expansion grid will have shape (repeat_per_node, repeat_per_node).
    
    Returns
    -------
    filtered_repeated : np.ndarray
        Filtered array of repeated original edges (shape (num_selected, 2)).
    filtered_expanded : np.ndarray
        Filtered array of expanded edges with grid offsets (shape (num_selected, 2)).
    """
    num_edges = edges.shape[0]
    num_grid = repeat_per_node * repeat_per_node
    total = num_edges * num_grid

    # Step 1: Compute prefix sum over the boolean mask (serial loop).
    cumsum = np.empty(total, dtype=np.int64)
    count = 0
    for i in range(total):
        if p_lisl_LT_pair[i]:
            count += 1
        cumsum[i] = count
    total_true = count

    # Allocate output arrays.
    filtered_repeated = np.empty((total_true, 2), dtype=edges.dtype)
    filtered_expanded = np.empty((total_true, 2), dtype=edges.dtype)

    # Step 2: Loop over all expansion entries in parallel.
    # For each flattened index, if the mask is True, compute the corresponding edge expansion and copy it.
    for i in prange(total):
        if p_lisl_LT_pair[i]:
            # Determine output position from prefix sum.
            pos = cumsum[i] - 1  # Adjust for 0-indexing.
            # Map flattened index to original edge index and grid index.
            e = i // num_grid
            g = i % num_grid
            # Original (repeated) edge.
            filtered_repeated[pos, 0] = edges[e, 0]
            filtered_repeated[pos, 1] = edges[e, 1]
            # Compute grid offsets.
            offset0 = g // repeat_per_node  # row offset
            offset1 = g % repeat_per_node   # column offset
            # Expanded edge: multiply the original edge by repeat_per_node and add the grid offset.
            filtered_expanded[pos, 0] = edges[e, 0] * repeat_per_node + offset0
            filtered_expanded[pos, 1] = edges[e, 1] * repeat_per_node + offset1

    return filtered_repeated, filtered_expanded

@njit(parallel=True,cache=True)
def compute_weighted_edges(velocities, positions, filtered_repeated, 
                           filtered_expanded, view_LT_pair_min_cos, FOR_THETA, MIN_TIME=100):
    n = filtered_repeated.shape[0]
    
    # Preallocate arrays for intermediate computations.
    relative_speed = np.empty((n, 3), dtype=velocities.dtype)
    relative_direction = np.empty((n, 3), dtype=positions.dtype)
    cross = np.empty((n, 3), dtype=velocities.dtype)
    angular_speed = np.empty(n, dtype=velocities.dtype)
    view_time_approx = np.empty(n, dtype=velocities.dtype)
    
    theta_rad = math.radians(FOR_THETA)
    
    # Compute relative values in parallel.
    for i in prange(n):
        idx0 = filtered_repeated[i, 0]
        idx1 = filtered_repeated[i, 1]
        
        # Calculate relative speed and direction for each axis.
        for j in range(3):
            relative_speed[i, j] = velocities[idx1, j] - velocities[idx0, j]
            relative_direction[i, j] = positions[idx1, j] - positions[idx0, j]
        
        # Manually compute the cross product.
        cross[i, 0] = relative_speed[i, 1]*relative_direction[i, 2] - relative_speed[i, 2]*relative_direction[i, 1]
        cross[i, 1] = relative_speed[i, 2]*relative_direction[i, 0] - relative_speed[i, 0]*relative_direction[i, 2]
        cross[i, 2] = relative_speed[i, 0]*relative_direction[i, 1] - relative_speed[i, 1]*relative_direction[i, 0]
        
        # Compute norms.
        cross_norm = math.sqrt(cross[i, 0]**2 + cross[i, 1]**2 + cross[i, 2]**2)
        dir_norm = math.sqrt(relative_direction[i, 0]**2 + relative_direction[i, 1]**2 + relative_direction[i, 2]**2)
        
        # Avoid division by zero.
        if dir_norm == 0:
            angular_speed[i] = 0.0
        else:
            angular_speed[i] = cross_norm / dir_norm
        
        # Compute view time approximation.
        # Note: view_LT_pair_min_cos[i] should be in the domain of acos, i.e. between -1 and 1.
        acos_val = math.acos(view_LT_pair_min_cos[i])
        # Protect against angular_speed being zero.
        if angular_speed[i] == 0:
            view_time_approx[i] = 1e10  # Use a large number to represent near-infinite view time.
        else:
            view_time_approx[i] = (theta_rad - acos_val) / math.fabs(angular_speed[i])
    
    # Count how many edges meet the criterion.
    count = 0
    for i in range(n):
        if view_time_approx[i] > MIN_TIME:
            count += 1

    # Determine number of columns in filtered_expanded.
    num_cols = filtered_expanded.shape[1]
    # Allocate result array: each row is filtered_expanded row concatenated with one view_time value.
    res = np.empty((count, num_cols + 1), dtype=filtered_expanded.dtype)
    
    k = 0
    for i in range(n):
        if view_time_approx[i] > MIN_TIME:
            # Copy the corresponding row from filtered_expanded.
            for j in range(num_cols):
                res[k, j] = filtered_expanded[i, j]
            # Append the view time value.
            res[k, num_cols] = view_time_approx[i]
            k += 1
    
    return res

@njit(parallel=True,cache=True)
def compute_view_stacks(front, back, right, left, edges, direction):
    """
    Compute dot products for source and destination sides in parallel.

    Parameters:
        front, back, right, left (np.ndarray): Arrays of shape (N, 3) representing direction vectors.
        edges (np.ndarray): Array of shape (num_edges, 2) containing indices.
        direction (np.ndarray): Array of shape (num_edges, 3) representing normalized directions.

    Returns:
        tuple: Two arrays of shape (num_edges, 4) for view_from_stack and view_to_stack.
    """
    num_edges = edges.shape[0]
    # Pre-allocate output arrays.
    view_from_stack = np.empty((num_edges, 4), dtype=direction.dtype)
    view_to_stack = np.empty((num_edges, 4), dtype=direction.dtype)
    
    for i in prange(num_edges):
        # Get the source and destination indices for this edge.
        src = edges[i, 0]
        dst = edges[i, 1]
        d0 = direction[i, 0]
        d1 = direction[i, 1]
        d2 = direction[i, 2]
        
        # Compute dot products for source side.
        view_from_stack[i, 0] = front[src, 0]*d0 + front[src, 1]*d1 + front[src, 2]*d2
        view_from_stack[i, 1] = back[src, 0]*d0 + back[src, 1]*d1 + back[src, 2]*d2
        view_from_stack[i, 2] = right[src, 0]*d0 + right[src, 1]*d1 + right[src, 2]*d2
        view_from_stack[i, 3] = left[src, 0]*d0 + left[src, 1]*d1 + left[src, 2]*d2

        # Compute dot products for destination side with -direction.
        view_to_stack[i, 0] = front[dst, 0]*(-d0) + front[dst, 1]*(-d1) + front[dst, 2]*(-d2)
        view_to_stack[i, 1] = back[dst, 0]*(-d0) + back[dst, 1]*(-d1) + back[dst, 2]*(-d2)
        view_to_stack[i, 2] = right[dst, 0]*(-d0) + right[dst, 1]*(-d1) + right[dst, 2]*(-d2)
        view_to_stack[i, 3] = left[dst, 0]*(-d0) + left[dst, 1]*(-d1) + left[dst, 2]*(-d2)
        
    return view_from_stack, view_to_stack


@njit(parallel=True,cache=True)
def optimize_edge_and_color_data(edges_color, connected_edges, connected_sat, positions, lift=False):
    M = connected_edges.shape[0]
    M_sat = connected_sat.shape[0]
    
    # Build edges_color_from and edges_color_to.
    # Each is computed by taking the row from edges_color indexed by connected_edges mod 4,
    # duplicating it, then reshaping to yield an array of shape (2*M, 4).
    edges_color_from = np.empty((2 * M, 4), dtype=edges_color.dtype)
    edges_color_to   = np.empty((2 * M, 4), dtype=edges_color.dtype)
    
    for i in prange(M):
        idx_from = connected_edges[i, 0] % 4
        idx_to   = connected_edges[i, 1] % 4
        # Duplicate the row for "from" and "to".
        for j in range(4):
            edges_color_from[2 * i, j]     = edges_color[idx_from, j]
            edges_color_from[2 * i + 1, j] = edges_color[idx_from, j]
            edges_color_to[2 * i, j]       = edges_color[idx_to, j]
            edges_color_to[2 * i + 1, j]   = edges_color[idx_to, j]
    
    # Concatenate edges_color_from and edges_color_to along axis 0.
    edges_color_data = np.empty((4 * M, 4), dtype=edges_color.dtype)
    for i in prange(2 * M):
        for j in range(4):
            edges_color_data[i, j] = edges_color_from[i, j]
            edges_color_data[i + 2 * M, j] = edges_color_to[i, j]

    # Compute positions for connected_sat.
    p_from = np.empty((M_sat, 3), dtype=positions.dtype)
    p_to   = np.empty((M_sat, 3), dtype=positions.dtype)
    p_mid  = np.empty((M_sat, 3), dtype=positions.dtype)
    
    for i in prange(M_sat):
        idx_from = connected_sat[i, 0]
        idx_to   = connected_sat[i, 1]
        for j in range(3):
            # Multiply by a slight factor (1.0001)
            if lift:
                p_from[i, j] = positions[idx_from, j] * 1.0001
                p_to[i, j]   = positions[idx_to, j] * 1.0001
            else:
                p_from[i, j] = positions[idx_from, j]
                p_to[i, j]   = positions[idx_to, j]
            # p_mid is computed as the average.
            p_mid[i, j]  = (p_from[i, j] + p_to[i, j]) * 0.5

    # Build c_lisl_data_from = reshape(concatenate(p_from, p_mid, axis=1)) => shape (2*M_sat, 3)
    c_lisl_data_from = np.empty((2 * M_sat, 3), dtype=positions.dtype)
    for i in prange(M_sat):
        for j in range(3):
            c_lisl_data_from[2 * i, j]     = p_from[i, j]
            c_lisl_data_from[2 * i + 1, j] = p_mid[i, j]
    
    # Build c_lisl_data_to = reshape(concatenate(p_mid, p_to, axis=1)) => shape (2*M_sat, 3)
    c_lisl_data_to = np.empty((2 * M_sat, 3), dtype=positions.dtype)
    for i in prange(M_sat):
        for j in range(3):
            c_lisl_data_to[2 * i, j]     = p_mid[i, j]
            c_lisl_data_to[2 * i + 1, j] = p_to[i, j]
    
    # Concatenate c_lisl_data_from and c_lisl_data_to along axis 0.
    c_lisl_data = np.empty((4 * M_sat, 3), dtype=positions.dtype)
    for i in prange(2 * M_sat):
        for j in range(3):
            c_lisl_data[i, j] = c_lisl_data_from[i, j]
            c_lisl_data[i + 2 * M_sat, j] = c_lisl_data_to[i, j]
    
    return edges_color_data, c_lisl_data

@njit(cache=True)
def greedy_max_weight_matching(E: np.ndarray) -> list:
    """
    Compute a greedy heuristic maximum weight matching for a NumPy array of edges.
    
    Parameters:
        E (np.ndarray): Array with shape (num_edges, 3) where each row is [u, v, weight].
    
    Returns:
        list: List of tuples (u, v) representing the selected edges.
    """
    sorted_indices = np.argsort(-E[:, 2])
    E_sorted = E[sorted_indices]
    
    matching = []
    matched_nodes = set()
    
    for edge in E_sorted:
        u, v, weight = edge
        u, v = int(u), int(v)
        if u not in matched_nodes and v not in matched_nodes:
            matching.append((u, v))
            matched_nodes.add(u)
            matched_nodes.add(v)
    
    return matching


def compute_face_texcoords(vertices: np.ndarray) -> list:
    """
    Compute texture coordinates for a single face of vertices.

    Parameters:
        vertices (np.ndarray): Array of vertex coordinates for one face.

    Returns:
        list: List of [u, v] texture coordinates for the face.
    """
    has_negative = any(
        p[0] < 0 and q[0] < 0 and p[1] * q[1] < 0 for p, q in combinations(vertices, 2)
    )
    face_texcoords = []
    for v in vertices:
        x, y, z = v
        theta = np.arctan2(y, x)
        phi = np.arccos(z / np.linalg.norm(v))
        if has_negative and theta < 0:
            theta += 2 * np.pi
        u = (theta + np.pi) / (2 * np.pi) / 2
        v_coord = phi / np.pi
        face_texcoords.append([u, v_coord])
    return face_texcoords


def compute_texcoords(faces: np.ndarray) -> np.ndarray:
    """
    Compute texture coordinates for all faces.

    Parameters:
        faces (np.ndarray): Array where each element represents a face's vertices.

    Returns:
        np.ndarray: Array of texture coordinates.
    """
    texcoords = []
    for face in faces:
        texcoords.extend(compute_face_texcoords(face))
    return np.array(texcoords)


def load_starlink_data(url: str, reload: bool = True) -> Tuple[object, list, SatrecArray]:
    """
    Load Starlink TLE data from the provided URL.

    Parameters:
        url (str): URL to the TLE file.
        reload (bool): Whether to reload the TLE data.

    Returns:
        tuple: (timescale, valid_satellites, sat_array)
    """
    # Initialize timescale and load TLEs from remote source.
    ts = load.timescale()
    satellites = load.tle_file(url, reload=reload)
    if not satellites:
        raise Exception("No Starlink satellites were loaded; check the TLE URL.")
    logger.debug("Loaded %d Starlink satellites from %s", len(satellites), url)

    # Filter out satellites with invalid positions to avoid propagator errors.
    valid_satellites = []
    for sat in satellites:
        pos = sat.at(ts.now())
        if np.isnan(pos.position.km).any():
            message = pos.message if pos.message else "position is invalid"
            logger.warning("Skipping %s due to error: %s", sat.name, message)
            continue
        valid_satellites.append(sat)

    models = [sat.model for sat in valid_satellites]
    sat_array = SatrecArray(models)
    return ts, valid_satellites, sat_array


# ---------------------------
# Visualization utilities
# ---------------------------
def _load_earth_texture(texture_path: str) -> np.ndarray:
    """Load and duplicate the Earth texture for spherical mapping."""
    try:
        texture_image = Image.open(texture_path)
    except Exception as exc:
        logger.error("Error loading texture image: %s", exc)
        raise

    texture = np.array(texture_image)
    # Duplicate texture horizontally to reduce seam artifacts.
    return np.hstack([texture, texture])


def _create_text_overlays(canvas: scene.SceneCanvas) -> Dict[str, scene.visuals.Text]:
    """Create and position text overlays for status and profiling output."""
    w, h = canvas.size
    font_size = 10

    text_top = scene.visuals.Text(
        text="Waiting...",
        color='black',
        font_size=font_size,
        bold=False,
        pos=(0, 0),
        anchor_x='left',
        anchor_y='bottom',
        parent=canvas.central_widget,
    )

    text_bot = scene.visuals.Text(
        text="Waiting...",
        color='black',
        font_size=font_size,
        bold=False,
        pos=(0, h),
        anchor_x='left',
        anchor_y='top',
        parent=canvas.central_widget,
    )

    text_top_right = scene.visuals.Text(
        text="Waiting...",
        color='black',
        font_size=font_size,
        bold=False,
        pos=(w, 0),
        anchor_x='right',
        anchor_y='bottom',
        parent=canvas.central_widget,
    )
    
    text_bot_right = scene.visuals.Text(
        text="Waiting...",
        color='black',
        font_size=font_size,
        bold=False,
        pos=(w, h),
        anchor_x='right',
        anchor_y='top',
        parent=canvas.central_widget,
    )

    return {
        "text_top": text_top,
        "text_bot": text_bot,
        "text_top_right": text_top_right,
        "text_bot_right": text_bot_right,
    }


def setup_visualization(config: DigitalTwinConfig = DEFAULT_CONFIG) -> Dict[str, object]:
    """
    Set up the Vispy visualization environment including canvas, view, sphere, and markers.

    Returns:
        dict: Dictionary containing references to visualization components.
    """
    canvas = scene.SceneCanvas(
        title='Mega-Constellation Digital Twin',
        size=(1200, 700),
        position=(0, 0),
        keys='interactive',
        show=True,
        bgcolor=(1.0, 1.0, 1.0, 1.0),
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=0, elevation=45, distance=2.5)

    # # Set up the OpenGL state for non-transparent objects.
    # gloo.set_state(depth_test=True, depth_mask=True, blend=False, cull_face=False)

    # Global arrow for reference
    global_arrow = scene.visuals.Arrow(
        pos=np.array([[0, 0, 0], [1, 1, 1]]), color='black', width=3, arrow_size=20,
        arrow_type='stealth', parent=view.scene
    )
    view.add(global_arrow)

    # Axes with scaling.
    axes = scene.visuals.XYZAxis(parent=view.scene)
    view.add(axes)
    axes.transform = STTransform(scale=(2, 2, 2))

    # Load texture image for the Earth sphere.
    texture = _load_earth_texture(config.texture_path)
    # Create sphere mesh and compute texture coordinates.
    sphere = create_sphere(rows=20, cols=40, radius=1, method='latitude', offset=False)
    vertices = sphere.get_vertices(indexed='faces')
    texcoords = compute_texcoords(vertices)

    mesh_data = MeshData(vertices=vertices)
    texture_filter = TextureFilter(texture, texcoords)

    sphere_visual = scene.visuals.Mesh(
        meshdata=mesh_data, color=(1.0, 1.0, 1.0, 1.0), shading=None
    )
    sphere_visual.attach(texture_filter)
    sphere_visual.set_gl_state('opaque')
    view.add(sphere_visual)

    # Apply transformation to the sphere.
    sphere_visual.transform = MatrixTransform()

    # Enable depth testing for transparent objects.
    gloo.set_state(depth_test=True, depth_mask=True, blend=True,
               blend_func=('src_alpha', 'one_minus_src_alpha'))
    
    # Create markers for satellite positions.
    scatter = scene.visuals.Markers()
    view.add(scatter)

    # Satellite arrows for LT directions.
    satellite_arrow = scene.visuals.Arrow()
    view.add(satellite_arrow)

    # Lines for satellite LISL.
    p_lisl = scene.visuals.Arrow()
    view.add(p_lisl)

    # Lines for connected LISL.
    c_lisl = scene.visuals.Arrow()
    view.add(c_lisl)
    text_overlays = _create_text_overlays(canvas)

    return {
        "canvas": canvas,
        "view": view,
        "sphere_visual": sphere_visual,
        "scatter": scatter,
        "arrow": satellite_arrow,
        "p_lisl": p_lisl,
        "c_lisl": c_lisl,
        **text_overlays,
    }



# ---------------------------
# Digital Twin orchestration
# ---------------------------
class DigitalTwin:
    """Main digital twin controller for satellite state and visualization updates."""

    # Standardized color palette for LT directions.
    FRONT_COLOR = np.array([0, 0, 0.85, 1])
    BACK_COLOR = np.array([0.05, 0.75, 0.05, 1])
    RIGHT_COLOR = np.array([1, 0, 0, 1])
    LEFT_COLOR = np.array([0.75, 0.75, 0, 1])

    def __init__(
        self,
        ts,
        sat_array,
        viz: Dict[str, object],
        config: DigitalTwinConfig = DEFAULT_CONFIG,
        capture_config: CaptureConfig = DEFAULT_CAPTURE_CONFIG,
    ):
        """
        Initialize the digital twin.

        Parameters:
            ts: Skyfield timescale.
            sat_array: Vectorized satellite propagation array.
            sphere_visual: Vispy visual for the Earth sphere.
            scatter: Vispy visual for satellite markers.
            satellite_arrow: Vispy visual for satellite LT arrows.
            p_lisl: Vispy visual for primary LISL lines.
            c_lisl: Vispy visual for connected LISL lines.
            canvas: Vispy SceneCanvas.
        """
        self.ts = ts
        self.sat_array = sat_array
        self.viz = viz
        self.config = config
        self.capture_config = capture_config
        
        self.digital_twin_start_time = self.ts.now()
        self.real_start_time = time.perf_counter()
        self.update_count = 0
        self.accumulated_update_time = 0
        self.average_update_time = 1
        self.cpu_usage_sum = 0.0
        self.average_cpu_usage = 0.0

        # Initialize satellite positions and velocities.
        self.positions = None
        self.velocities = None
        # Initialize directional vectors.
        self.front = None
        self.back = None
        self.down = None
        self.right = None
        self.left = None
        
        self.profiled_time = {}
        self.screenshot_count = 0
        self.gif_count = 0
        self.gif_frames = []
        self.gif_recording = False
        self.gif_start_time = None
        self.gif_timer = app.Timer(
            interval=1.0 / self.capture_config.gif_fps,
            connect=self._record_gif_frame,
            start=False,
        )
        
        # Connect double-click event handler
        self.viz['canvas'].events.mouse_double_click.connect(self.handle_double_click)
        self.viz['canvas'].events.mouse_press.connect(self.handle_mouse_press)
        self.viz['canvas'].events.mouse_release.connect(self.handle_mouse_release)

        self._update_earth_rotation()

    def get_simulation_time(self):
        """
        Compute the current digital twin time based on the time scaling factor.

        Returns:
            Skyfield Time: The current digital twin time.
        """
        elapsed_real = time.perf_counter() - self.real_start_time
        elapsed_scaled = elapsed_real * self.config.time_scale
        delta_days = elapsed_scaled / 86400  # Convert seconds to days.
        new_tt_jd = self.digital_twin_start_time.tt + delta_days
        return self.ts.tt(jd=new_tt_jd)

    def compute_rotation(self) -> float:
        """
        Compute the initial rotation angle from the current GMST.

        Returns:
            float: Rotation angle in degrees.
        """
        t_now = self.get_simulation_time()
        gmst_hours = t_now.gmst
        rotation_angle_deg = gmst_hours * 15  # 15° per hour.
        logger.debug("GMST: %.2f hours, Rotation angle: %.2f degrees", gmst_hours, rotation_angle_deg)
        return rotation_angle_deg

    def _update_earth_rotation(self):
        """Update the Earth's rotation transformation."""
        logger.debug("Updating Earth rotation... %d", self.update_count)
        self.viz['sphere_visual'].transform.reset()
        self.viz['sphere_visual'].transform.rotate(self.compute_rotation(), (0, 0, 1))

    def _update_satellite_positions(self):
        """Update satellite positions and velocities."""
        logger.debug("Updating satellite positions and velocities... %d", self.update_count)
        current_time = self.get_simulation_time()
        error_upd, pos_upd, vel_upd = self.sat_array.sgp4(
            np.array([current_time.whole]),
            np.array([current_time.ut1_fraction])
        )
        positions = np.array(pos_upd).reshape(-1, 3) / self.config.earth_radius_km
        velocities = np.array(vel_upd).reshape(-1, 3) / self.config.earth_radius_km

        R_icrs_to_teme = TEME.rotation_at(current_time)
        R_teme_to_icrs = R_icrs_to_teme.T
        self.positions = positions @ R_teme_to_icrs
        self.velocities = velocities @ R_teme_to_icrs

        self.viz['scatter'].set_data(self.positions, face_color=[0, 0, 0, 0.5], size=10, edge_width=0)

    def _update_satellite_arrows(self):
        """Update satellite LT direction arrows."""
        logger.debug("Updating satellite arrows... %d", self.update_count)
        self.front, self.back, self.down, self.right, self.left = update_arrows(self.velocities, self.positions)

        a_from = np.tile(self.positions, (4, 1))
        a_to = np.concatenate((self.front, self.back, self.right, self.left), axis=0)
        a_to = a_to * self.config.arrow_length_scale + a_from
        a_data = np.concatenate((a_from, a_to), axis=1).reshape(-1, 3)

        num_arrows = self.positions.shape[0] * 8
        arrow_color = np.zeros((num_arrows, 4))
        arrow_color[: num_arrows // 4, :] = self.FRONT_COLOR
        arrow_color[num_arrows // 4: num_arrows // 2, :] = self.BACK_COLOR
        arrow_color[num_arrows // 2: 3 * num_arrows // 4, :] = self.RIGHT_COLOR
        arrow_color[3 * num_arrows // 4:, :] = self.LEFT_COLOR

        self.viz['arrow'].set_data(pos=a_data, color=arrow_color, width=10, connect='segments')

    def _update_links(self):
        """Update satellite link visualizations using KDTree and matching."""
        logger.debug("Updating satellite links... %d", self.update_count)
        
        # 1) Build KDTree for efficient neighbor queries.
        tic = time.perf_counter()
        tree = cKDTree(self.positions)
        distance_threshold = self.config.lisl_max_distance_km / self.config.earth_radius_km
        edges = tree.query_pairs(r=distance_threshold, output_type='ndarray')
        toc = time.perf_counter()
        self.profiled_time['kdtree'] = toc - tic
        
        
        # 2) Compute candidate edges and view constraints.
        # Precompute cosine threshold once.
        cos_threshold = math.cos(math.radians(self.config.for_theta_deg))

        # Compute normalized direction for each edge.
        tic = time.perf_counter()
        direction = compute_directions(self.positions, edges)
        toc = time.perf_counter()
        self.profiled_time['direction'] = toc - tic
        
        # Compute view stacks for each edge.
        tic = time.perf_counter()
        view_from_stack, view_to_stack = compute_view_stacks(
            self.front, self.back, self.right, self.left, edges, direction
        )
        toc = time.perf_counter()
        self.profiled_time['view_stacks'] = toc - tic

        # Generate boolean indicators using the precomputed threshold.
        tic = time.perf_counter()
        filtered_edges, filtered_view_from_stack, filtered_view_to_stack, p_lisl_LT_pair = filter_and_compute_pair(
            edges, view_from_stack, view_to_stack, cos_threshold
        )
        toc = time.perf_counter()
        self.profiled_time['p_lisl_LT_pair'] = toc - tic
        
        
        # 3) Filter view stacks for valid edges and compute pairwise minimum.
        tic = time.perf_counter()
        view_LT_pair_min_cos = compute_view_LT_pair_min_cos(filtered_view_from_stack, filtered_view_to_stack, p_lisl_LT_pair.reshape(-1))
        toc = time.perf_counter()
        self.profiled_time['view_LT_pair'] = toc - tic

        # 4) Expand edges and filter using the computed pair indicator.
        tic = time.perf_counter()
        filtered_repeated, filtered_expanded = expand_and_filter_edges(filtered_edges, p_lisl_LT_pair)
        toc = time.perf_counter()
        self.profiled_time['expand_edges'] = toc - tic
        
        tic = time.perf_counter()
        # Set colors for all potential LISL edges.
        edges_color = np.array([self.FRONT_COLOR, self.BACK_COLOR, self.RIGHT_COLOR, self.LEFT_COLOR])
        # Draw the potential LISL edges.
        if self.config.plot_potential_lisl:
            # Build the color array using np.array for clarity.
            edges_color_data, p_lisl_data = optimize_edge_and_color_data(edges_color, filtered_expanded, filtered_repeated, self.positions)
            edges_color_data[:, 3] = 0.25
            self.viz['p_lisl'].set_data(pos=p_lisl_data, color=edges_color_data, width=0.0001, connect='segments')
            
        toc = time.perf_counter()
        self.profiled_time['draw_p_lisl'] = toc - tic
    
        # 5) Compute weighted edges and greedy matching.
        tic = time.perf_counter()
        weighted_edges = compute_weighted_edges(
            self.velocities,
            self.positions,
            filtered_repeated,
            filtered_expanded,
            view_LT_pair_min_cos.reshape(-1),
            self.config.for_theta_deg,
        )
        toc = time.perf_counter()
        self.profiled_time['weight_edges'] = toc - tic
        
        tic = time.perf_counter()
        matching = greedy_max_weight_matching(weighted_edges)
        toc = time.perf_counter()
        self.profiled_time['greedy_matching'] = toc - tic    
    
        # 6) Filter edges based on the matching result.
        tic = time.perf_counter()
        m = len(matching)
        flat_array = np.fromiter((x for pair in ((min(e), max(e)) for e in matching)
                                  for x in pair), dtype=int, count=2*m)
        connected_lts = flat_array.reshape(-1, 2)
        connected_sat = connected_lts // 4
        toc = time.perf_counter()
        self.profiled_time['filter_matching'] = toc - tic
        
        # 7) Draw the connected edges.
        tic = time.perf_counter()
        edges_color_data, c_lisl_data = optimize_edge_and_color_data(edges_color, connected_lts, connected_sat, self.positions, lift=True)
        self.viz['c_lisl'].set_data(pos=c_lisl_data, color=edges_color_data, width=2, connect='segments')
        toc = time.perf_counter()
        self.profiled_time['draw_matching'] = toc - tic
        
        # Update the top-left overlay with link statistics.
        self.viz['text_top'].text = self._build_link_summary_text(
            edges,
            filtered_edges,
            filtered_expanded,
            connected_lts,
        )

        
    def update(self, event):
        """
        Update function called on each timer tick to update the digital twin.
        """
        start_time = time.perf_counter()
        cpu_usage = psutil.cpu_percent()
        self.cpu_usage_sum += cpu_usage
        self.average_cpu_usage = self.cpu_usage_sum / (self.update_count + 1)
        mem_usage = psutil.Process().memory_info().rss / 1e6
        logger.info(f"CPU Usage: {cpu_usage}%, Memory Usage: {mem_usage:.2f} MB")

        try:
            # Update Earth rotation for the current digital twin time.
            tic = time.perf_counter()
            self._update_earth_rotation()
            toc = time.perf_counter()
            self.profiled_time['earth_rotation'] = toc - tic
            
            # Update satellite state vectors (position and velocity).
            tic = time.perf_counter()
            self._update_satellite_positions()
            toc = time.perf_counter()
            self.profiled_time['satellite_positions'] = toc - tic
            
            # Update LT direction arrows for each satellite.
            tic = time.perf_counter()
            self._update_satellite_arrows()
            toc = time.perf_counter()
            self.profiled_time['satellite_arrows'] = toc - tic
            
            # Update LISL candidate edges and matching visualization.
            tic = time.perf_counter()
            self._update_links()
            toc = time.perf_counter()
            self.profiled_time['update_links'] = toc - tic
            
            
            self.viz['text_bot'].text = self._build_status_text(cpu_usage, mem_usage)
            self.viz['text_top_right'].text = self._build_profile_text()
            self.viz['text_bot_right'].text = self._build_author_text()
        except Exception as e:
            logger.error("Unexpected error during update: %s", e)


        elapsed = time.perf_counter() - start_time
        self.update_count += 1
        self.accumulated_update_time += elapsed
        self.average_update_time = 0.9 * self.average_update_time + 0.1 * elapsed

    def handle_double_click(self, event):
        """
        Handle double-click events to save a screenshot of the current visualization.
        Screenshots are saved to the /images directory with a timestamp counter.
        """
        try:
            timestamp = self.get_simulation_time().utc_strftime('%Y%m%d_%H%M%S')
            image = self._capture_canvas_image()
            filepath = self._save_png_screenshot(image, timestamp)
            self.screenshot_count += 1
            logger.info(f"Screenshot saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")

    def handle_mouse_press(self, event):
        """
        Start GIF recording when the left mouse button is pressed and held.
        """
        if event.button != 1:
            return
        if self.gif_recording:
            return
        self.gif_frames = []
        self.gif_recording = True
        self.gif_start_time = self.get_simulation_time()
        self.gif_timer.start()

    def handle_mouse_release(self, event):
        """
        Stop GIF recording and save when the left mouse button is released.
        """
        if event.button != 1:
            return
        if not self.gif_recording:
            return
        self.gif_recording = False
        self.gif_timer.stop()
        self._save_gif()

    def _record_gif_frame(self, event):
        """Capture a single frame for GIF recording."""
        if not self.gif_recording:
            return
        if len(self.gif_frames) >= self.capture_config.gif_max_frames:
            self.gif_recording = False
            self.gif_timer.stop()
            self._save_gif()
            return

        frame = Image.fromarray(self._capture_canvas_image())
        self.gif_frames.append(frame)

    # ---------------------------
    # Screenshot and GIF utilities
    # ---------------------------
    def _capture_canvas_image(self) -> np.ndarray:
        """Capture the current canvas image as a NumPy array."""
        self.viz['canvas'].update()
        self.viz['canvas'].show()
        return _screenshot(
            viewport=(0, 0, self.viz['canvas'].physical_size[0], self.viz['canvas'].physical_size[1]),
            alpha=True,
        )

    def _ensure_dir(self, *parts: str) -> str:
        """Ensure a directory exists under the project root and return its path."""
        path = os.path.join(os.path.dirname(__file__), *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def _save_png_screenshot(self, image: np.ndarray, timestamp: str) -> str:
        """Save a PNG screenshot and return the output path."""
        images_dir = self._ensure_dir(self.capture_config.screenshot_dir)
        filename = f"screenshot_{timestamp}_{self.screenshot_count:05d}.png"
        filepath = os.path.join(images_dir, filename)
        vispy_io.write_png(filepath, image)
        return filepath

    def _save_gif(self):
        """Save the recorded frames as a GIF."""
        if not self.gif_frames:
            return

        output_dir = self._ensure_dir(self.capture_config.gif_output_dir)

        if self.gif_start_time is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
        else:
            timestamp = self.gif_start_time.utc_strftime('%Y%m%d_%H%M%S')

        filename = f"recording_{timestamp}_{self.gif_count:05d}.gif"
        filepath = os.path.join(output_dir, filename)

        duration_ms = int(1000 / max(1, self.capture_config.gif_fps))
        scale = self.capture_config.gif_scale
        if scale <= 0:
            scale = 1.0

        if scale != 1.0:
            w, h = self.gif_frames[0].size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            frames = [frame.resize(new_size, resample=Image.LANCZOS) for frame in self.gif_frames]
        else:
            frames = self.gif_frames

        colors = max(2, min(256, int(self.capture_config.gif_colors)))
        first = frames[0].convert('P', palette=Image.ADAPTIVE, colors=colors)
        rest = [frame.convert('P', palette=Image.ADAPTIVE, colors=colors) for frame in frames[1:]]
        first.save(
            filepath,
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            optimize=True,
        )

        self.gif_frames = []
        self.gif_count += 1
        logger.info(f"GIF saved to {filepath}")

    def _build_link_summary_text(
        self,
        edges: np.ndarray,
        filtered_edges: np.ndarray,
        filtered_expanded: np.ndarray,
        connected_lts: np.ndarray,
    ) -> str:
        """Format link statistics shown in the top-left overlay."""
        text = "CONSTELLATION CONFIG:\n"
        text += f"#n_sats: {self.positions.shape[0]}\n"
        text += f"#n_q_es: {edges.shape[0]}\n"
        text += f"#n_p_sp: {filtered_edges.shape[0]}\n"
        text += f"#n_p_lp: {filtered_expanded.shape[0]}\n"
        text += f"#n_c_lp: {connected_lts.shape[0]}\n"
        text += f"FOR_THETA: +/-{self.config.for_theta_deg:.0f}°\n"
        text += f"MAX_DIST: {self.config.lisl_max_distance_km:.0f} km\n"
        return text

    def _build_status_text(self, cpu_usage: float, mem_usage: float) -> str:
        """Format runtime performance and time scaling stats."""
        text = "STATUS:\n"
        text += f"AVG: {self.average_update_time:.4f} s\n"
        text += f"UPD: {self.update_count}\n"
        text += f"TOT: {time.perf_counter() - self.real_start_time:.2f} s\n"
        text += f"FPS: {1./self.average_update_time:.2f}\n"
        text += f"TSc: {self.config.time_scale:.2f}\n"
        text += f"SWT: {self.accumulated_update_time*self.config.time_scale:.2f} s\n"
        text += f"DAT: {self.get_simulation_time().utc_strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += f"CPU: {cpu_usage}%\n"
        text += f"CPU_AVG: {self.average_cpu_usage:.2f}%\n"
        text += f"MEM: {mem_usage:.2f} MB\n"
        return text

    def _build_profile_text(self) -> str:
        """Format profiling breakdown for the top-right overlay."""
        text = "TIME PROFILE:\n"
        for key, value in self.profiled_time.items():
            text += f"{key}: {value:.4f} s\n"
        return text

    def _build_author_text(self) -> str:
        """Format the window title with digital twin stats."""
        author = f"Mega-Constellation Digital Twin\n" 
        author += f"Auth.: Z. Gu, Supr.: J. Park, Aff.: SUTD\n"
        return author

    def update_dummy(self, event):
        """
        Dummy update function to keep the timer running.
        """
        self.viz['canvas'].update()

def main():
    # Load Starlink data.
    satellite_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
    # satellite_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle'
    # satellite_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
    # satellite_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=eutelsat&FORMAT=tle'
    # satellite_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium&FORMAT=tle'
    # satellite_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-NEXT&FORMAT=tle'
    # satellite_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
    
    ts, valid_satellites, sat_array = load_starlink_data(satellite_url, reload=True)

    # Set up visualization.
    config = DEFAULT_CONFIG
    capture_config = DEFAULT_CAPTURE_CONFIG
    viz = setup_visualization(config)

    # Create digital twin instance.
    digital_twin = DigitalTwin(ts, sat_array, viz, config=config, capture_config=capture_config)

    # Set up a timer to update the digital twin at roughly 60 FPS.
    timer1 = app.Timer(interval=1 / 60.0, connect=digital_twin.update, start=True)
    app.run()

if __name__ == '__main__':
    logger.level = logging.DEBUG
    main()
