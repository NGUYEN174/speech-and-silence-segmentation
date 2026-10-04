"""
Evaluation utilities for speech/silence segmentation.

Provides:
    - Boundary matching between detected and ground-truth boundaries
    - MAE (Mean Absolute Error) calculation
    - RMSE (Root Mean Squared Error) calculation

Unit: milliseconds

Boundary matching:
    Detected and ground-truth boundary counts may differ.
    We use chronological ordered matching with validation.
"""

import numpy as np


def match_boundaries(detected_boundaries_s, gt_boundaries_s):
    """Match detected boundaries to ground-truth boundaries chronologically.
    
    Strategy:
        Uses nearest-neighbor matching in chronological order.
        Each ground-truth boundary is matched to the closest detected
        boundary that has not already been matched.
    
    This handles cases where the number of detected and ground-truth
    boundaries differ.
    
    Args:
        detected_boundaries_s (list of float): Detected boundaries in seconds.
        gt_boundaries_s (list of float): Ground-truth boundaries in seconds.
    
    Returns:
        dict with keys:
            'matched_pairs' (list of tuple): (gt_boundary, det_boundary) pairs
                in seconds
            'errors_ms' (list of float): Absolute errors in milliseconds
                for each matched pair
            'unmatched_gt' (list of float): Ground-truth boundaries without
                a match (in seconds)
            'unmatched_det' (list of float): Detected boundaries without
                a match (in seconds)
            'num_gt' (int): Number of ground-truth boundaries
            'num_det' (int): Number of detected boundaries
    """
    det = sorted(detected_boundaries_s)
    gt = sorted(gt_boundaries_s)
    
    matched_pairs = []
    errors_ms = []
    used_det = set()
    
    # For each ground-truth boundary, find the nearest unmatched detected boundary
    for gt_b in gt:
        best_idx = None
        best_dist = float('inf')
        
        for i, det_b in enumerate(det):
            if i in used_det:
                continue
            dist = abs(det_b - gt_b)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx is not None:
            used_det.add(best_idx)
            matched_pairs.append((gt_b, det[best_idx]))
            errors_ms.append(abs(det[best_idx] - gt_b) * 1000.0)
    
    # Find unmatched boundaries
    unmatched_gt = [gt_b for i, gt_b in enumerate(gt)
                    if not any(gt_b == pair[0] for pair in matched_pairs)]
    unmatched_det = [det[i] for i in range(len(det)) if i not in used_det]
    
    return {
        'matched_pairs': matched_pairs,
        'errors_ms': errors_ms,
        'unmatched_gt': unmatched_gt,
        'unmatched_det': unmatched_det,
        'num_gt': len(gt),
        'num_det': len(det),
    }


def calculate_mae(errors_ms):
    """Calculate Mean Absolute Error.
    
    Formula:
        MAE = mean(|predicted - ground_truth|)
    
    The errors_ms input should already contain absolute errors.
    
    Args:
        errors_ms (list of float): Absolute boundary errors in milliseconds.
    
    Returns:
        float: MAE in milliseconds. Returns 0.0 if no errors.
    """
    if len(errors_ms) == 0:
        return 0.0
    return float(np.mean(errors_ms))


def calculate_rmse(errors_ms):
    """Calculate Root Mean Squared Error.
    
    Formula:
        RMSE = sqrt(mean(error^2))
    
    Args:
        errors_ms (list of float): Absolute boundary errors in milliseconds.
    
    Returns:
        float: RMSE in milliseconds. Returns 0.0 if no errors.
    """
    if len(errors_ms) == 0:
        return 0.0
    errors = np.array(errors_ms)
    return float(np.sqrt(np.mean(errors ** 2)))


def evaluate_boundaries(detected_boundaries_s, gt_boundaries_s):
    """Full evaluation of detected vs ground-truth boundaries.
    
    Matches boundaries, computes MAE and RMSE, and reports statistics.
    
    Args:
        detected_boundaries_s (list of float): Detected boundaries in seconds.
        gt_boundaries_s (list of float): Ground-truth boundaries in seconds.
    
    Returns:
        dict with keys:
            'mae_ms' (float): Mean Absolute Error in milliseconds
            'rmse_ms' (float): Root Mean Squared Error in milliseconds
            'errors_ms' (list of float): Individual boundary errors in ms
            'matched_pairs' (list of tuple): Matched (gt, det) pairs
            'num_gt' (int): Number of ground-truth boundaries
            'num_det' (int): Number of detected boundaries
            'num_matched' (int): Number of matched pairs
            'num_missing' (int): Ground-truth boundaries without match
            'num_extra' (int): Detected boundaries without match
    """
    matching = match_boundaries(detected_boundaries_s, gt_boundaries_s)
    
    mae = calculate_mae(matching['errors_ms'])
    rmse = calculate_rmse(matching['errors_ms'])
    
    return {
        'mae_ms': mae,
        'rmse_ms': rmse,
        'errors_ms': matching['errors_ms'],
        'matched_pairs': matching['matched_pairs'],
        'num_gt': matching['num_gt'],
        'num_det': matching['num_det'],
        'num_matched': len(matching['matched_pairs']),
        'num_missing': len(matching['unmatched_gt']),
        'num_extra': len(matching['unmatched_det']),
    }
