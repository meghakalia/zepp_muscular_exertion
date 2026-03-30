"""
Main entrypoint for Wall Ball training evaluation.

Provides data loading, user/session management, evaluation orchestration,
web-demo formatting, and batch export.
"""

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .insight import calculate_three_dimension_assessment, generate_insights_from_csv
from .exertion import calculate_exertion, compute_cardiac_exertion, compute_combined_exertion
from .utils import (
    evaluate_cardiovascular_load,
    evaluate_control_stability,
    evaluate_output_sustainability,
    evaluate_output_sustainability_time_based,
    evaluate_pacing_strategy,
    evaluate_recovery_capacity,
    evaluate_trends,
    process_sets_by_rep_count,
)


# -----------------------------------------------------------------------------
# User / session and data loading
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_user_excel_df():
    """
    Load and cache the user demographics Excel file.
    Uses lru_cache to avoid re-reading the file on every call.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    excel_path = os.path.join(data_dir, '202512_HT.xlsx')

    if not os.path.exists(excel_path):
        return None

    try:
        excel_df = pd.read_excel(excel_path, engine='openpyxl')

        # Pre-process: Create userid column
        try:
            excel_df['collect_date_parsed'] = pd.to_datetime(excel_df['collect_date'], errors='coerce')
            excel_df['collect_date_yyyymmdd'] = excel_df['collect_date_parsed'].dt.strftime('%Y%m%d')
        except Exception:
            excel_df['collect_date_yyyymmdd'] = excel_df['collect_date'].astype(str).str.replace('-', '').str.replace('/', '').str[:8]

        excel_df['userid'] = (
            excel_df['collect_date_yyyymmdd'].astype(str) + '_' +
            excel_df['masked_id'].astype(str) + '_' +
            excel_df['scene_id'].astype(str)
        )
        return excel_df
    except ImportError:
        raise ImportError("openpyxl library is required to read Excel files. Install it with: pip install openpyxl")
    except Exception as e:
        print(f"Warning: Could not load user Excel file: {e}")
        return None


def load_user_info_from_excel(session_folder_name=None, masked_id=None, scene_id=None, collect_date=None):
    """
    Load user information from Excel file (202512_HT.xlsx).
    
    Matching strategy:
    - Creates a 'userid' column in Excel by concatenating: YYYYMMDD_masked_id_scene_id
    - collect_date is converted from Excel format (e.g., '2025-12-24') to YYYYMMDD format ('20251224')
    - Directly matches session_folder_name with the userid column
    
    Session folder name format: YYYYMMDD_masked_id_scene_id
    Example: '20251224_SSCDDQTQHOPK_HT_4kg'
    - collect_date: 20251224 (YYYYMMDD format)
    - masked_id: SSCDDQTQHOPK
    - scene_id: HT_4kg
    
    Excel columns:
    - collect_date: e.g., '2025-12-24' (will be converted to '20251224')
    - masked_id: e.g., 'SSCDDQTQHOPK'
    - scene_id: e.g., 'HT_4kg'
    
    Data extraction rules:
    - weight: Read from Excel 'weight' column
    - age: Read from Excel 'age' column
    - medicine_ball_weight_kg: Extract from scene_id by splitting by '_' and taking the last part
      (e.g., "HT_4kg" -> split -> ["HT", "4kg"] -> last part "4kg" -> extract number -> 4.0)
    - rhr: Default value 55 (not in Excel)
    
    Parameters:
    -----------
    session_folder_name : str, optional
        Session folder name, e.g., '20251224_SSCDDQTQHOPK_HT_4kg'
        Format: YYYYMMDD_masked_id_scene_id
        If provided, will be used directly to match userid column in Excel
    masked_id : str, optional
        Masked user ID, e.g., 'SSCDDQTQHOPK'
        Only used if session_folder_name is not provided
    scene_id : str, optional
        Scene ID, e.g., 'HT_4kg'
        Only used if session_folder_name is not provided
    collect_date : str, optional
        Collection date, e.g., '2025-12-24' or '20251224'
        Only used if session_folder_name is not provided
    
    Returns:
    --------
    dict: User information dictionary containing:
        - user_id: Session folder name (YYYYMMDD_masked_id_scene_id)
        - masked_id: Masked user ID
        - scene_id: Scene ID
        - age: Age in years (from Excel)
        - body_weight_kg: Body weight in kg (from Excel 'weight' column)
        - height: Height in cm (from Excel)
        - gender: Gender (from Excel)
        - medicine_ball_weight_kg: Medicine ball weight in kg (extracted from scene_id last part)
        - rhr: Resting heart rate in bpm (default 55)
        - All other fields from Excel
    
    Raises:
    --------
    FileNotFoundError: If Excel file doesn't exist
    ValueError: If user not found in Excel file
    """
    # Load cached Excel DataFrame
    excel_df = _load_user_excel_df()
    if excel_df is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        excel_path = os.path.join(data_dir, '202512_HT.xlsx')
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Determine the userid to search for
    search_userid = None
    
    if session_folder_name:
        # Use session_folder_name directly as userid
        search_userid = session_folder_name
    elif masked_id and scene_id and collect_date:
        # Build userid from individual parameters
        # Convert collect_date to YYYYMMDD format
        collect_date_str = None
        try:
            if isinstance(collect_date, str):
                if '-' in collect_date:
                    collect_date_str = datetime.strptime(collect_date, '%Y-%m-%d').strftime('%Y%m%d')
                elif '/' in collect_date:
                    collect_date_str = datetime.strptime(collect_date, '%Y/%m/%d').strftime('%Y%m%d')
                elif len(collect_date) == 8 and collect_date.isdigit():
                    collect_date_str = collect_date
                else:
                    collect_date_str = pd.to_datetime(collect_date).strftime('%Y%m%d')
            else:
                collect_date_str = pd.to_datetime(collect_date).strftime('%Y%m%d')
        except Exception as e:
            raise ValueError(f"Could not parse collect_date: {e}")
        
        if collect_date_str:
            search_userid = f"{collect_date_str}_{masked_id}_{scene_id}"
    
    if not search_userid:
        raise ValueError("Must provide either session_folder_name or (masked_id, scene_id, collect_date)")
    
    # Find matching row by userid
    matching_rows = excel_df[excel_df['userid'] == search_userid]
    
    if len(matching_rows) == 0:
        # Show available userids for debugging
        available_userids = excel_df['userid'].head(10).tolist()
        raise ValueError(f"User not found in Excel file. Searched userid: {search_userid}. "
                       f"Available userids (first 10): {available_userids}")
    
    if len(matching_rows) > 1:
        print(f"Warning: Multiple matches found, using first one. Total matches: {len(matching_rows)}")
    
    # Extract user information
    user_row = matching_rows.iloc[0]
    user_info = user_row.to_dict()
    
    # Extract medicine ball weight from scene_id
    # scene_id format: "HT_4kg" -> split by '_' -> take last part "4kg" -> extract number -> 4.0
    medicine_ball_weight_kg = 4.0  # Default
    if 'scene_id' in user_info and pd.notna(user_info['scene_id']):
        scene_id_str = str(user_info['scene_id'])
        try:
            # Split by '_' and take the last part
            scene_parts = scene_id_str.split('_')
            if len(scene_parts) > 0:
                last_part = scene_parts[-1]  # e.g., "4kg"
                # Extract number from last part (remove "kg" and convert to float)
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', last_part)
                if match:
                    medicine_ball_weight_kg = float(match.group(1))
        except Exception as e:
            print(f"Warning: Could not extract medicine ball weight from scene_id '{scene_id_str}': {e}")
    
    # Use the matched userid as session_folder_name
    if not session_folder_name:
        session_folder_name = user_info.get('userid', search_userid)
    
    # Build result dictionary
    result = {
        'user_id': session_folder_name,
        'masked_id': user_info.get('masked_id', ''),
        'scene_id': user_info.get('scene_id', ''),
        'age': float(user_info.get('age', 30)) if pd.notna(user_info.get('age')) else 30,
        'body_weight_kg': float(user_info.get('weight', 75.0)) if pd.notna(user_info.get('weight')) else 75.0,
        'height': float(user_info.get('height', 170.0)) if pd.notna(user_info.get('height')) else 170.0,
        'gender': user_info.get('gender', 'unknown'),
        'medicine_ball_weight_kg': medicine_ball_weight_kg,
        'rhr': 55.0,  # Default RHR, not in Excel
        'bmi': float(user_info.get('bmi', 0)) if pd.notna(user_info.get('bmi')) else 0,
    }
    
    # Add all other fields from Excel
    for key, value in user_info.items():
        if key not in result:
            result[key] = value
    
    return result


def load_user_info(user_id):
    """
    Load user information from Excel file (202512_HT.xlsx).

    Wrapper around load_user_info_from_excel().
    
    Parameters:
    -----------
    user_id : str
        Session folder name, e.g., '20251224_SSCDDQTQHOPK_HT_4kg'
        Format: collect_date_masked_id_scene_id
    
    Returns:
    --------
    dict: User information dictionary containing:
        - user_id: Session folder name
        - age: Age in years (from Excel)
        - rhr: Resting heart rate (bpm, default 55)
        - body_weight_kg: Body weight (kg, from Excel)
        - medicine_ball_weight_kg: Medicine ball weight (kg, extracted from scene_id)
        - All other fields from Excel
    
    Raises:
    --------
    FileNotFoundError: If Excel file doesn't exist
    ValueError: If user_id is not found in the Excel file
    """
    if not user_id:
        raise ValueError("user_id is required")
    
    # Load from Excel file
    return load_user_info_from_excel(session_folder_name=user_id)


def create_session_dict(user_id=None, age=30, rhr=55, body_weight_kg=75.0, medicine_ball_weight_kg=9.0, **kwargs):
    """
    Create a session dictionary containing basic session information.
    
    Parameters:
    -----------
    user_id : str, optional
        Session folder name, e.g., '20251224_SSCDDQTQHOPK_HT_4kg'
        Format: collect_date_masked_id_scene_id
        If provided and load_from_excel=True, will load from Excel file
    age : float, optional
        Age in years (default: 30)
    rhr : float, optional
        Resting heart rate in bpm (default: 55)
    body_weight_kg : float, optional
        Body weight in kilograms (default: 75.0)
    medicine_ball_weight_kg : float, optional
        Medicine ball weight in kilograms (default: 9.0)
    load_from_excel : bool, optional
        If True and user_id is provided, load user info from Excel (202512_HT.xlsx) (default: False)
    **kwargs : dict
        Additional session parameters
    
    Returns:
    --------
    dict: Session dictionary containing:
        - user_id: Session folder name
        - age: Age in years
        - rhr: Resting heart rate (bpm)
        - body_weight_kg: Body weight (kg)
        - medicine_ball_weight_kg: Medicine ball weight (kg)
        - bw: Alias for body_weight_kg (for convenience)
        - mb: Alias for medicine_ball_weight_kg (for convenience)
        - Any additional parameters from kwargs
    """
    # Check if we should load from Excel
    load_from_excel = kwargs.pop('load_from_excel', False)
    
    # If load_from_excel is True and user_id is provided, load from Excel
    if load_from_excel and user_id:
        try:
            user_info = load_user_info(user_id)  # Load from Excel
            # Use loaded values, but allow kwargs to override
            age = user_info.get('age', age)
            rhr = user_info.get('rhr', rhr)
            body_weight_kg = user_info.get('body_weight_kg', body_weight_kg)
            medicine_ball_weight_kg = user_info.get('medicine_ball_weight_kg', medicine_ball_weight_kg)
            
            # Also update user_id if it was constructed from Excel data
            if 'user_id' in user_info:
                user_id = user_info['user_id']
        except (FileNotFoundError, ValueError) as e:
            # If loading fails, use provided defaults
            print(f"Warning: Could not load user info from Excel: {e}")
            print("Using provided default values instead.")
    
    session_dict = {
        'user_id': user_id,
        'age': age,
        'rhr': rhr,
        'body_weight_kg': body_weight_kg,
        'medicine_ball_weight_kg': medicine_ball_weight_kg,
        'bw': body_weight_kg,  # Alias for convenience
        'mb': medicine_ball_weight_kg,  # Alias for convenience
    }
    
    # Add any additional parameters
    session_dict.update(kwargs)
    
    return session_dict


def load_data(user_id=None):
    """
    Load real data files (sets.csv, reps.csv, hr.csv) and convert to standard format
    
    Real data format:
    - sets.csv: start_time(ms), stop_time(ms), motion, total_reps
    - reps.csv: start_time, stop_time, motion, duration(ms), squat_distance(m), completed
    - hr.csv: timestamp, heart_rate(bpm)
    
    Parameters:
    -----------
    user_id : str, optional
        Session folder name, e.g., '20251224_SSCDDQTQHOPK_HT_4kg'
        If None, will look for data in data/real_data directory
    
    Returns:
    --------
    tuple: (motions_df, measures_df)
        Normalized DataFrames compatible with evaluation functions:
        motions_df: start_time, stop_time, motion, duration, waist_deviation, set_index, is_completed
        measures_df: timestamp, hr
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    real_data_dir = os.path.join(data_dir, 'real_data')
    
    # Determine file paths based on user_id
    if user_id:
        # Try multiple possible paths
        possible_paths = [
            os.path.join(real_data_dir, user_id),  # data/real_data/user_id
            os.path.join(data_dir, user_id),  # data/user_id (direct)
        ]
        
        user_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                user_dir = path
                break
        
        if user_dir is None:
            raise FileNotFoundError(f"Data directory not found for user_id: {user_id}. "
                                  f"Tried paths: {possible_paths}")
    else:
        # If no user_id, try to find any real_data session (for backward compatibility)
        if os.path.exists(real_data_dir):
            # Get first available session
            sessions = [d for d in os.listdir(real_data_dir) 
                      if os.path.isdir(os.path.join(real_data_dir, d))]
            if sessions:
                user_dir = os.path.join(real_data_dir, sessions[0])
            else:
                raise FileNotFoundError(f"No sessions found in {real_data_dir}")
        else:
            raise FileNotFoundError(f"Real data directory not found: {real_data_dir}")
    
    # Real data file paths
    sets_path = os.path.join(user_dir, 'sets.csv')
    reps_path = os.path.join(user_dir, 'reps.csv')
    hr_path = os.path.join(user_dir, 'hr.csv')
    
    # Check if files exist
    if not os.path.exists(sets_path):
        raise FileNotFoundError(f"Sets file not found: {sets_path}")
    if not os.path.exists(reps_path):
        raise FileNotFoundError(f"Reps file not found: {reps_path}")
    if not os.path.exists(hr_path):
        raise FileNotFoundError(f"HR file not found: {hr_path}")
    
    # Load and process real data
    return load_real_data(sets_path, reps_path, hr_path)


def load_real_data(sets_path, reps_path, hr_path):
    """
    Load and process real data format (sets.csv, reps.csv, hr.csv)
    
    Converts real data format to standard format compatible with evaluation functions:
    - Each rep in reps.csv becomes a wall_ball motion entry
    - Sets are used to assign set_index to reps
    - Rest periods are inferred between sets
    - waist_deviation is calculated from squat_distance variation
    
    Parameters:
    -----------
    sets_path : str
        Path to sets.csv file
    reps_path : str
        Path to reps.csv file
    hr_path : str
        Path to hr.csv file
    
    Returns:
    --------
    tuple: (motions_df, measures_df)
        Normalized DataFrames compatible with evaluation functions
    """
    # Read real data files
    sets_df = pd.read_csv(sets_path)
    reps_df = pd.read_csv(reps_path)
    hr_df = pd.read_csv(hr_path)
    
    # Normalize column names (remove units in parentheses and strip whitespace)
    sets_df.columns = sets_df.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()
    reps_df.columns = reps_df.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()
    hr_df.columns = hr_df.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()
    
    # Convert time columns to int (they're in ms)
    sets_df['start_time'] = sets_df['start_time'].astype(int)
    sets_df['stop_time'] = sets_df['stop_time'].astype(int)
    reps_df['start_time'] = reps_df['start_time'].astype(int)
    reps_df['stop_time'] = reps_df['stop_time'].astype(int)

    # Ensure sets are ordered by time for downstream rest calculations
    sets_df = sets_df.sort_values('start_time').reset_index(drop=True)
    reps_df = reps_df.sort_values('start_time').reset_index(drop=True)

    # Compute rest between adjacent sets (gap between current stop and next start)
    sets_df['rest_to_next_ms'] = sets_df['start_time'].shift(-1) - sets_df['stop_time']
    sets_df['rest_to_next_s'] = sets_df['rest_to_next_ms'] / 1000.0
    
    # Convert duration from ms to seconds for reps
    if 'duration' in reps_df.columns:
        reps_df['duration'] = reps_df['duration'] / 1000.0  # Convert ms to seconds

    # Normalize control stability source columns if present
    control_mean_col = 'core_muscle_deviation_mean'
    control_std_col = 'core_muscle_deviation_std'
    if control_mean_col in reps_df.columns:
        reps_df[control_mean_col] = pd.to_numeric(reps_df[control_mean_col], errors='coerce')
    if control_std_col in reps_df.columns:
        reps_df[control_std_col] = pd.to_numeric(reps_df[control_std_col], errors='coerce')

    # Pre-compute per-rep control CV% using core muscle deviation (std / mean)
    if {control_mean_col, control_std_col}.issubset(reps_df.columns):
        mean_values = reps_df[control_mean_col]
        std_values = reps_df[control_std_col]
        reps_df['control_cv_pct'] = std_values.divide(mean_values).replace([pd.NA, pd.NaT], pd.NA)
        reps_df['control_cv_pct'] = reps_df['control_cv_pct'].replace([float('inf'), -float('inf')], pd.NA)
        reps_df['control_cv_pct'] = reps_df['control_cv_pct'].fillna(0) * 100
    else:
        reps_df['control_cv_pct'] = None
    
    # Assign set_index to each rep based on sets.csv
    # Each rep belongs to the set that contains it
    reps_df['set_index'] = None
    for set_idx, set_row in sets_df.iterrows():
        set_start = set_row['start_time']
        set_stop = set_row['stop_time']
        
        # Find all reps that fall within this set's time range
        mask = (reps_df['start_time'] >= set_start) & (reps_df['stop_time'] <= set_stop)
        reps_df.loc[mask, 'set_index'] = set_idx
    
    # Calculate waist_deviation for each rep based on squat_distance variation within its set
    # For Control Stability, we need per-rep waist_deviation
    # We'll use the deviation from mean squat_distance as a proxy
    reps_df['waist_deviation'] = 0.0
    
    for set_idx in sets_df.index:
        set_reps = reps_df[reps_df['set_index'] == set_idx]
        if len(set_reps) > 0 and 'squat_distance' in set_reps.columns:
            # Calculate mean squat_distance for this set
            avg_squat_dist = set_reps['squat_distance'].mean()
            
            if avg_squat_dist > 0:
                # Use individual deviation from mean as proxy for waist_deviation
                # Convert to cm: deviation in meters * 100, then scale appropriately
                for rep_idx in set_reps.index:
                    rep_squat_dist = set_reps.loc[rep_idx, 'squat_distance']
                    deviation_from_mean = abs(rep_squat_dist - avg_squat_dist)
                    # Convert to approximate waist_deviation in cm
                    # Scale factor: 0.1 means 1cm deviation per 0.1m squat_distance difference
                    reps_df.loc[rep_idx, 'waist_deviation'] = (deviation_from_mean / avg_squat_dist) * 100 * 0.1
    
    # Convert each squat rep to a wall_ball motion entry
    motions_list = []
    
    # Add wall_ball entries for each rep
    for _, rep_row in reps_df.iterrows():
        motions_list.append({
            'start_time': rep_row['start_time'],
            'stop_time': rep_row['stop_time'],
            'motion': 'wall_ball',  # Convert squat to wall_ball
            'duration': rep_row['duration'],
            'waist_deviation': rep_row['waist_deviation'],
            'core_muscle_deviation_mean': rep_row.get('core_muscle_deviation_mean'),
            'core_muscle_deviation_std': rep_row.get('core_muscle_deviation_std'),
            'control_cv_pct': rep_row.get('control_cv_pct'),
            'set_index': rep_row['set_index'],
            'is_completed': 1 if rep_row.get('completed', True) else 0,
        })
    
    # Add rest periods between sets
    for idx in range(len(sets_df) - 1):
        current_set_stop = sets_df.iloc[idx]['stop_time']
        next_set_start = sets_df.iloc[idx + 1]['start_time']
        
        if next_set_start > current_set_stop:
            rest_duration = (next_set_start - current_set_stop) / 1000.0  # Convert ms to seconds
            motions_list.append({
                'start_time': current_set_stop,
                'stop_time': next_set_start,
                'motion': 'rest',
                'duration': rest_duration,
                'waist_deviation': 0.0,
                'core_muscle_deviation_mean': None,
                'core_muscle_deviation_std': None,
                'control_cv_pct': None,
                'set_index': None,  # Rest periods don't belong to a set
                'is_completed': 1,
            })
    
    # Create motions_df
    motions_df = pd.DataFrame(motions_list)
    motions_df = motions_df.sort_values('start_time').reset_index(drop=True)
    
    # Process hr.csv to create measures_df
    # hr.csv columns: timestamp, heart_rate(bpm) -> normalized to: timestamp, hr
    hr_df['timestamp'] = hr_df['timestamp'].astype(int)
    
    # Rename heart_rate column to hr
    if 'heart_rate' in hr_df.columns:
        hr_df = hr_df.rename(columns={'heart_rate': 'hr'})
    elif 'hr' not in hr_df.columns:
        raise ValueError("hr.csv must contain either 'heart_rate' or 'hr' column")
    
    # Create measures_df with standard columns
    measures_df = pd.DataFrame({
        'timestamp': hr_df['timestamp'],
        'hr': hr_df['hr'].astype(float),
    })
    
    # Add optional columns if available
    if 'breath_rate' in hr_df.columns:
        measures_df['breath_rate'] = hr_df['breath_rate']
    if 'motion_frequency' in hr_df.columns:
        measures_df['motion_frequency'] = hr_df['motion_frequency']
    
    measures_df = measures_df.sort_values('timestamp').reset_index(drop=True)
    
    return motions_df, measures_df


def get_history_file_path(user_id=None):
    """
    Get history sessions file path
    
    For real data, history sessions are stored per session or in a central location.
    
    Parameters:
    -----------
    user_id : str, optional
        Session folder name, e.g., '20251224_SSCDDQTQHOPK_HT_4kg'
        If None, returns None (no history for real data without user_id)
    
    Returns:
    --------
    str: History sessions file path, returns None if file doesn't exist
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    real_data_dir = os.path.join(data_dir, 'real_data')
    
    if user_id:
        # Try multiple possible paths
        possible_paths = [
            os.path.join(real_data_dir, user_id, 'history_sessions.csv'),  # Per-session history
            os.path.join(real_data_dir, 'history_sessions.csv'),  # Central history file
            os.path.join(data_dir, user_id, 'history_sessions.csv'),  # Direct path
        ]
        
        for history_path in possible_paths:
            if os.path.exists(history_path):
                return history_path
    
    # If no user_id or not found, try central history file
    central_history = os.path.join(real_data_dir, 'history_sessions.csv')
    if os.path.exists(central_history):
        return central_history
    
    return None


# -----------------------------------------------------------------------------
# Evaluation, formatting, and batch export
# -----------------------------------------------------------------------------


def build_result_bundle(evaluation_results):
    """
    Build a return structure containing three parts:
    1) five_dim_eval: the 5-dimension evaluation for the current session
    2) five_dim_trend: trend comparison for those 5 dimensions
    3) insight: placeholder for future insight content (None for now)
    """
    five_dim_keys = [
        'cardiovascular_load',
        'recovery_capacity',
        'output_sustainability',
        'control_stability',
        'pacing_strategy',
    ]

    five_dim_eval = {k: evaluation_results.get(k) for k in five_dim_keys}
    five_dim_trend = evaluation_results.get('trends')

    return {
        'five_dim_eval': five_dim_eval,
        'five_dim_trend': five_dim_trend,
        'insight': None,
    }


def calculate_composite_score(dimension_analysis):
    """
    Calculate Composite Performance Score based on 5 dimensions.
    
    Parameters:
    -----------
    dimension_analysis : dict
        Dictionary containing result, score, interpretation for each dimension
    
    Returns:
    --------
    dict: Composite score result containing:
        - composite_score: float (0-5.0)
        - assessment: str (e.g., "Excellent session")
        - data_quality_flags: int (0-5)
        - data_quality_assessment: str
    """
    dimension_keys = [
        'cardiovascular_load',
        'recovery_capacity',
        'output_sustainability',
        'control_stability',
        'pacing_strategy',
    ]
    
    scores = []
    data_quality_flags = 0
    all_flags = []  # Collect all flags from all dimensions
    
    # Calculate scores for each dimension
    for dim_key in dimension_keys:
        dim_data = dimension_analysis.get(dim_key, {})
        category = dim_data.get('result', 'N/A')
        score = dim_data.get('score')
        flags = dim_data.get('flags', [])
        
        # Collect flags from this dimension
        if isinstance(flags, list):
            all_flags.extend(flags)
        
        # Check if dimension has valid data
        if category == 'N/A' or score is None:
            data_quality_flags += 1
        else:
            # Use score directly (1-5 scale)
            if isinstance(score, (int, float)) and 1 <= score <= 5:
                scores.append(float(score))
    
    # Calculate composite score
    if len(scores) == 0:
        composite_score = 0.0
    else:
        composite_score = sum(scores) / len(scores)
    
    # Determine assessment based on score range
    # Score Range: 4.5-5.0 = Excellent session, 3.5-4.4 = Strong performance, 
    # 2.5-3.4 = Solid effort, 1.5-2.4 = Room for improvement, <1.5 = Significant limiters
    if composite_score >= 4.5:
        assessment = "Excellent session"
    elif composite_score >= 3.5:  # 3.5-4.4
        assessment = "Strong performance"
    elif composite_score >= 2.5:  # 2.5-3.4
        assessment = "Solid effort"
    elif composite_score >= 1.5:  # 1.5-2.4
        assessment = "Room for improvement"
    else:  # <1.5
        assessment = "Significant limiters"
    
    # Determine data quality assessment
    if data_quality_flags == 0:
        data_quality_assessment = "Clean data"
    elif data_quality_flags <= 2:
        data_quality_assessment = "Some limitations, scores still useful"
    elif data_quality_flags <= 4:
        data_quality_assessment = "Multiple concerns, use for trends only"
    else:  # 5 flags
        data_quality_assessment = "All metrics affected, limited reliability"
    
    # Count total flags (N/A dimensions + dimension-specific flags)
    total_flags = data_quality_flags + len(all_flags)
    
    return {
        'composite_score': round(composite_score, 2),
        'assessment': assessment,
        'data_quality_flags': data_quality_flags,  # Number of N/A dimensions
        'total_flags': total_flags,  # Total flags (N/A + dimension-specific flags)
        'flags': all_flags,  # List of all flags
        'data_quality_assessment': data_quality_assessment,
    }




def format_web_demo_response(evaluation_results):
    """
    Format evaluation results for web demo API response.
    
    Returns five values:
    1) dimension_analysis (dict): Contains result, score, interpretation for each dimension
    2) trend (dict): Contains baseline and label for each dimension
    3) insight (list): Insight list based on CSV rules
    4) composite_score_result (dict): Composite performance score
    5) three_dim_assessment (dict): Three-dimension assessment (Cardiac Stress, Local Muscular Endurance, Movement Control)
    
    Parameters:
    -----------
    evaluation_results : dict
        Dictionary containing evaluation results with 'trends' key
    
    Returns:
    --------
    tuple: (dimension_analysis, trend, insight, composite_score_result, three_dim_assessment)
    """
    import numpy as np
    
    # Helper function to get score value for each dimension
    def get_score_value(dim_key, dim_result):
        """Extract the primary metric score for each dimension"""
        if dim_result is None:
            return None
        
        score_mapping = {
            'cardiovascular_load': 'high_intensity_percentage',
            'recovery_capacity': 'avg_hr_drop',
            'output_sustainability': 'performance_decline',
            'control_stability': 'cv_percentage',
            'pacing_strategy': 'decline_percentage',
        }
        
        score_key = score_mapping.get(dim_key)
        if score_key and score_key in dim_result:
            value = dim_result[score_key]
            # Check if value is valid (not None and not nan)
            if value is None:
                return None
            if isinstance(value, float) and np.isnan(value):
                return None
            return value
        return None
    
    # 1. Build dimension_analysis dict
    dimension_analysis = {}
    
    dimension_keys = [
        'cardiovascular_load',
        'recovery_capacity',
        'output_sustainability',
        'control_stability',
        'pacing_strategy',
    ]
    
    for dim_key in dimension_keys:
        dim_result = evaluation_results.get(dim_key)
        if dim_result:
            # Use score field (1-5) directly, fallback to get_score_value for backward compatibility
            score_value = dim_result.get('score')
            if score_value is None:
                score_value = get_score_value(dim_key, dim_result)
            
            dimension_analysis[dim_key] = {
                'result': dim_result.get('category', 'N/A'),
                'score': score_value,  # This is now the 1-5 score, not the raw metric value
                'interpretation': dim_result.get('interpretation', 'N/A'),
                'flags': dim_result.get('flags', []),  # Include flags for this dimension
            }
        else:
            dimension_analysis[dim_key] = {
                'result': 'N/A',
                'score': None,
                'interpretation': 'Insufficient data',
                'flags': [],  # Empty flags for N/A dimensions
            }
    
    # 2. Build trend dict
    trend = {}
    trend_results = evaluation_results.get('trends', {})
    
    if trend_results and trend_results.get('status') == 'success':
        trends_data = trend_results.get('trends', {})
        for dim_key in dimension_keys:
            if dim_key in trends_data:
                trend_data = trends_data[dim_key]
                trend[dim_key] = {
                    'baseline': trend_data.get('baseline'),
                    'label': trend_data.get('trend', 'N/A'),
                }
            else:
                trend[dim_key] = {
                    'baseline': None,
                    'label': 'N/A',
                }
    else:
        # No trend data available
        for dim_key in dimension_keys:
            trend[dim_key] = {
                'baseline': None,
                'label': 'N/A',
            }
    
    # 3. Calculate Composite Performance Score
    composite_score_result = calculate_composite_score(dimension_analysis)
    
    # 4. Calculate Three-Dimension Assessment
    three_dim_assessment = calculate_three_dimension_assessment(evaluation_results)
    
    # 5. Generate Insight list based on CSV rules
    insight = generate_insights_from_csv(evaluation_results)
    
    return dimension_analysis, trend, insight, composite_score_result, three_dim_assessment


def compute_evaluation_bundle(user_id="20251224_SSCDDQTQHOPK_HT_4kg", age=30, rhr=55, session_dict=None):
    """
    Pure computation entrypoint.
    Returns the bundle of:
      - five_dim_eval
      - five_dim_trend
      - insight (None placeholder)
    No printing inside this function.
    
    Parameters:
    -----------
    user_id : str, optional
        Real data session folder name (default: "20251224_SSCDDQTQHOPK_HT_4kg")
    age : float, optional
        Age in years (default: 30)
    rhr : float, optional
        Resting heart rate in bpm (default: 55)
    session_dict : dict, optional
        Session dictionary containing user information. If provided, will override
        individual parameters (user_id, age, rhr).
    """
    # Use session_dict if provided, otherwise use individual parameters
    if session_dict is not None:
        user_id = session_dict.get('user_id', user_id)
        age = session_dict.get('age', age)
        rhr = session_dict.get('rhr', rhr)
    
    motions_df, measures_df = load_data(user_id=user_id)
    set_processing_info = process_sets_by_rep_count(motions_df)

    cv_result = evaluate_cardiovascular_load(
        measures_df,
        hr_max_method='age_formula',
        age=age,
        rhr=rhr,
        motions_df=motions_df,
    )
    recovery_result = evaluate_recovery_capacity(
        motions_df,
        measures_df,
        recovery_window_s=60,
        set_processing_info=set_processing_info,
    )
    sustainability_result = evaluate_output_sustainability(
        motions_df,
        set_processing_info=set_processing_info,
    )
    control_result = evaluate_control_stability(
        motions_df,
        use_waist_deviation=True,
        set_processing_info=set_processing_info,
    )
    pacing_result = evaluate_pacing_strategy(
        motions_df,
        set_processing_info=set_processing_info,
    )

    evaluation_results = {
        'cardiovascular_load': cv_result,
        'recovery_capacity': recovery_result,
        'output_sustainability': sustainability_result,
        'control_stability': control_result,
        'pacing_strategy': pacing_result,
    }

    history_file_path = get_history_file_path(user_id=user_id)
    trend_results = evaluate_trends(evaluation_results, history_file_path=history_file_path)
    evaluation_results['trends'] = trend_results

    result_bundle = build_result_bundle(evaluation_results)
    evaluation_results['result_bundle'] = result_bundle
    return result_bundle, evaluation_results


def process_single_user_for_web_demo(user_id):
    """
    Single-user entrypoint for web demo. Evaluates one user and returns formatted response.

    Parameters
    ----------
    user_id : str
        User/session ID, format: YYYYMMDD_masked_id_scene_id
        e.g. '20251224_SSCDDQTQHOPK_HT_4kg'

    Returns
    -------
    dict
        Web-demo response with: user_id, dimension_analysis, trend, insight,
        composite_score_result, three_dim_assessment, exertion, session_info.

    Raises
    ------
    FileNotFoundError
        If user data files are missing.
    ValueError
        If user_id is invalid or user info cannot be loaded from Excel.
    """
    try:
        session_dict = create_session_dict(user_id=user_id, load_from_excel=True)
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(f"Failed to load user info (user_id: {user_id}): {e}")

    try:
        result_bundle, evaluation_results = compute_evaluation_bundle(session_dict=session_dict)
    except Exception as e:
        raise ValueError(f"Evaluation failed (user_id: {user_id}): {e}")

    dimension_analysis, trend, insight, composite_score_result, three_dim_assessment = (
        format_web_demo_response(evaluation_results)
    )

    motions_df = None
    measures_df = None
    exertion_result = None

    try:
        motions_df, measures_df = load_data(user_id=user_id)
        exertion_result = calculate_exertion(
            'wall_ball',
            motions_df=motions_df,
            body_weight_kg=session_dict['body_weight_kg'],
            equipment_weight_kg=session_dict['medicine_ball_weight_kg'],
        )
        # Add cardiac exertion from per-second HR data
        if exertion_result is not None and measures_df is not None and len(measures_df) > 0:
            age = session_dict.get('age', 30)
            rhr = session_dict.get('rhr', 60)
            gender = session_dict.get('gender', 'Male')
            gender_key = 'female' if str(gender).lower() in ('female', 'f', '1') else 'male'
            hr_max = 207 - 0.7 * age
            hrs = measures_df['hr'].tolist() if 'hr' in measures_df.columns else []
            if hrs:
                cardiac_result = compute_cardiac_exertion(hrs, hr_max, rhr, gender=gender_key)
                cardiac_val = cardiac_result['cardiac_exertion']
                muscular_val = exertion_result.get('muscular_exertion', 0)
                combined = compute_combined_exertion(muscular_val, cardiac_val)
                exertion_result['cardiac_exertion'] = cardiac_val
                exertion_result['combined_exertion'] = combined['combined_exertion']
    except Exception as e:
        print(f"Warning: Exertion calculation failed: {e}")
        exertion_result = None

    # Convert DataFrames to list of dicts for JSON serialization
    motions_list = motions_df.to_dict('records') if motions_df is not None else []
    measures_list = measures_df.to_dict('records') if measures_df is not None else []

    response = {
        'user_id': user_id,
        'dimension_analysis': dimension_analysis,
        'trend': trend,
        'insight': insight,
        'composite_score_result': composite_score_result,
        'three_dim_assessment': three_dim_assessment,
        'exertion': exertion_result,
        'session_info': {
            'age': session_dict.get('age'),
            'rhr': session_dict.get('rhr'),
            'body_weight_kg': session_dict.get('body_weight_kg'),
            'medicine_ball_weight_kg': session_dict.get('medicine_ball_weight_kg'),
        },
        # Raw data for HR chart visualization
        'motions': motions_list,
        'measures': measures_list,
    }

    return response


def _excel_safe_value(v):
    """
    Convert nested/list/dict values to a string so Excel export won't fail and remains readable.
    """
    # Handle numpy scalar types if numpy is available
    try:
        import numpy as np  # optional dependency in this repo
        if isinstance(v, np.generic):
            v = v.item()
    except Exception:
        pass
    
    if isinstance(v, (dict, list, tuple, set)):
        try:
            return json.dumps(v, ensure_ascii=False, default=str)
        except Exception:
            return str(v)
    return v


def _excel_safe_dict(d):
    if not isinstance(d, dict):
        return {}
    return {k: _excel_safe_value(v) for k, v in d.items()}


def process_all_sessions_and_export(
    real_data_dir="data/",
    output_xlsx="evaluation_results_summary.xlsx",
    manual_default_params=None,
):
    """
    Batch-process all session folders under real_data_dir, aggregate dimension scores,
    and export to xlsx (one sheet per dimension + summary). Output is written to data/.
    """
    if manual_default_params is None:
        manual_default_params = dict(age=30, rhr=55, body_weight_kg=75.0, medicine_ball_weight_kg=4.0)
    script_dir = Path(os.path.dirname(__file__))
    real_data_dir = script_dir / real_data_dir
    session_dirs = sorted(f.name for f in real_data_dir.iterdir() if f.is_dir())

    summary_rows = []
    cv_detail_rows = []
    recovery_detail_rows = []
    sustainability_detail_rows = []
    control_detail_rows = []
    pacing_detail_rows = []
    exertion_detail_rows = []
    three_dim_detail_rows = []

    for user_id in session_dirs:
        try:
            session_dict = create_session_dict(
                user_id=user_id,
                load_from_excel=True,
                **manual_default_params,
            )
            result_bundle, evaluation_results = compute_evaluation_bundle(session_dict=session_dict)
            dimension_analysis, trend, insight, composite_score, three_dim_assessment = (
                format_web_demo_response(evaluation_results)
            )
            motions_df, measures_df = load_data(user_id=user_id)
            exertion_result = calculate_exertion(
                'wall_ball',
                motions_df=motions_df,
                body_weight_kg=session_dict['body_weight_kg'],
                equipment_weight_kg=session_dict['medicine_ball_weight_kg'],
            )
            # Add cardiac exertion from per-second HR data
            if exertion_result is not None and measures_df is not None and len(measures_df) > 0:
                age = session_dict.get('age', 30)
                rhr = session_dict.get('rhr', 60)
                gender = session_dict.get('gender', 'Male')
                gender_key = 'female' if str(gender).lower() in ('female', 'f', '1') else 'male'
                hr_max = 207 - 0.7 * age
                hrs = measures_df['hr'].tolist() if 'hr' in measures_df.columns else []
                if hrs:
                    cardiac_result = compute_cardiac_exertion(hrs, hr_max, rhr, gender=gender_key)
                    cardiac_val = cardiac_result['cardiac_exertion']
                    muscular_val = exertion_result.get('muscular_exertion', 0)
                    combined = compute_combined_exertion(muscular_val, cardiac_val)
                    exertion_result['cardiac_exertion'] = cardiac_val
                    exertion_result['combined_exertion'] = combined['combined_exertion']
        except Exception as e:
            summary_rows.append({"user_id": user_id, "error": str(e)})
            continue

        row_summary = {"user_id": user_id}
        row_summary.update(
            {k: v for k, v in composite_score.items()} if isinstance(composite_score, dict) else {}
        )
        row_summary["trend"] = str(trend)
        row_summary["insight"] = str(insight)
        
        # Add 3-dimension assessment columns to summary
        if three_dim_assessment:
            # Cardiac Stress
            cardiac_stress = three_dim_assessment.get('cardiac_stress', {})
            row_summary["cardiac_stress_score"] = cardiac_stress.get('score')
            row_summary["cardiac_stress_classification"] = cardiac_stress.get('classification')
            row_summary["cv_load_score"] = cardiac_stress.get('components', {}).get('cardiovascular_load_score')
            row_summary["recovery_capacity_score"] = cardiac_stress.get('components', {}).get('recovery_capacity_score')
            
            # Local Muscular Endurance
            local_muscular = three_dim_assessment.get('local_muscular_endurance', {})
            row_summary["local_muscular_endurance_score"] = local_muscular.get('score')
            row_summary["local_muscular_endurance_classification"] = local_muscular.get('classification')
            
            # Movement Control
            movement_control = three_dim_assessment.get('movement_control', {})
            row_summary["movement_control_score"] = movement_control.get('score')
            row_summary["movement_control_classification"] = movement_control.get('classification')
            
            # Training Recommendation
            training_rec = three_dim_assessment.get('training_recommendation', {})
            row_summary["training_recommendation_classification"] = training_rec.get('classification')
            row_summary["training_recommendation_insight"] = training_rec.get('insight')
        
        summary_rows.append(row_summary)

        cv_result = evaluation_results.get("cardiovascular_load", {})
        cv_result_flat = {"user_id": user_id, **_excel_safe_dict(cv_result)}
        cv_detail_rows.append(cv_result_flat)

        recovery_result = evaluation_results.get("recovery_capacity", {})
        recovery_result_flat = {"user_id": user_id, **_excel_safe_dict(recovery_result)}
        recovery_detail_rows.append(recovery_result_flat)

        sustainability_result = evaluation_results.get("output_sustainability", {})
        sustainability_result_flat = {"user_id": user_id, **_excel_safe_dict(sustainability_result)}
        sustainability_detail_rows.append(sustainability_result_flat)

        control_result = evaluation_results.get("control_stability", {})
        control_result_flat = {"user_id": user_id, **_excel_safe_dict(control_result)}
        control_detail_rows.append(control_result_flat)

        pacing_result = evaluation_results.get("pacing_strategy", {})
        pacing_result_flat = {"user_id": user_id, **_excel_safe_dict(pacing_result)}
        pacing_detail_rows.append(pacing_result_flat)

        exertion_row = {"user_id": user_id}
        exertion_row.update(_excel_safe_dict(exertion_result) if isinstance(exertion_result, dict) else {})
        exertion_detail_rows.append(exertion_row)
        
        # Add 3-dimension assessment detail row
        three_dim_row = {"user_id": user_id}
        if three_dim_assessment:
            three_dim_row.update(_excel_safe_dict(three_dim_assessment))
        three_dim_detail_rows.append(three_dim_row)

    summary_df = pd.DataFrame(summary_rows)
    cv_detail_df = pd.DataFrame(cv_detail_rows)
    recovery_detail_df = pd.DataFrame(recovery_detail_rows)
    sustainability_detail_df = pd.DataFrame(sustainability_detail_rows)
    control_detail_df = pd.DataFrame(control_detail_rows)
    pacing_detail_df = pd.DataFrame(pacing_detail_rows)
    exertion_df = pd.DataFrame(exertion_detail_rows)
    three_dim_df = pd.DataFrame(three_dim_detail_rows)

    data_dir = Path(os.path.dirname(__file__)) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / output_xlsx

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        cv_detail_df.to_excel(writer, sheet_name="Cardiovascular Load", index=False)
        recovery_detail_df.to_excel(writer, sheet_name="Recovery Capacity", index=False)
        sustainability_detail_df.to_excel(writer, sheet_name="Output Sustainability", index=False)
        control_detail_df.to_excel(writer, sheet_name="Control Stability", index=False)
        pacing_detail_df.to_excel(writer, sheet_name="Pacing Strategy", index=False)
        exertion_df.to_excel(writer, sheet_name="Exertion", index=False)
        three_dim_df.to_excel(writer, sheet_name="Three Dimension Assessment", index=False)
    print(f"[OK] Batch data analysis results exported: {out_path.resolve()}")


# -----------------------------------------------------------------------------
# Web App Adapter Functions
# -----------------------------------------------------------------------------

# Sample session IDs for the web demo
SAMPLE_SESSIONS = {
    "strong_performance": "20251230_WTTADQTQLRFVOBWA_HT_6kg",  # Score ~3.6
    "solid_effort": "20251225_SSCDDQTQHOPK_HT_6kg",            # Score ~2.6
    "room_for_improvement": "20251230_FZEIJCLQMGTN_HT_4kg",    # Score ~2.0
}


def load_sample_session(session_key):
    """
    Load a sample session by key name.

    Parameters
    ----------
    session_key : str
        One of: 'strong_performance', 'solid_effort', 'room_for_improvement'

    Returns
    -------
    dict
        Full response from process_single_user_for_web_demo()
    """
    if session_key not in SAMPLE_SESSIONS:
        raise ValueError(f"Unknown session key: {session_key}. Valid keys: {list(SAMPLE_SESSIONS.keys())}")

    user_id = SAMPLE_SESSIONS[session_key]
    return process_single_user_for_web_demo(user_id)


def evaluate_from_dataframes_web(hr_df, reps_df, sets_df, user_info, history=None):
    """
    Web app adapter function for evaluating uploaded CSV data.

    Accepts DataFrames directly (instead of file paths) and user profile info,
    then runs the full evaluation pipeline.

    Parameters
    ----------
    hr_df : pd.DataFrame
        Heart rate data with columns: timestamp, heart_rate (or hr)
    reps_df : pd.DataFrame
        Rep data with columns: start_time, stop_time, motion, duration, etc.
    sets_df : pd.DataFrame
        Set data with columns: start_time, stop_time, motion, total_reps
    user_info : dict
        User profile with keys: age, rhr, body_weight_kg, medicine_ball_weight_kg
        Optional: height, gender
    history : list, optional
        Historical session data for trend calculation

    Returns
    -------
    dict
        Web-demo response with: dimension_analysis, trend, insight,
        composite_score_result, three_dim_assessment, exertion, evaluation_results
    """
    import tempfile
    import os as _os

    # Create temp files from DataFrames
    with tempfile.TemporaryDirectory() as tmpdir:
        hr_path = _os.path.join(tmpdir, 'hr.csv')
        reps_path = _os.path.join(tmpdir, 'reps.csv')
        sets_path = _os.path.join(tmpdir, 'sets.csv')

        hr_df.to_csv(hr_path, index=False)
        reps_df.to_csv(reps_path, index=False)
        sets_df.to_csv(sets_path, index=False)

        # Load and convert to standard format
        motions_df, measures_df = load_real_data(sets_path, reps_path, hr_path)

    # Extract user parameters
    age = user_info.get('age', 30)
    rhr = user_info.get('rhr', 55)
    body_weight_kg = user_info.get('body_weight_kg', 70.0)
    medicine_ball_weight_kg = user_info.get('medicine_ball_weight_kg', 6.0)

    # Process sets for rep count analysis
    set_processing_info = process_sets_by_rep_count(motions_df)

    # Run 5-dimension evaluation
    cv_result = evaluate_cardiovascular_load(
        measures_df=measures_df,
        age=age,
        rhr=rhr,
        motions_df=motions_df,
    )

    recovery_result = evaluate_recovery_capacity(
        motions_df=motions_df,
        measures_df=measures_df,
        set_processing_info=set_processing_info,
    )

    sustainability_result = evaluate_output_sustainability(
        motions_df=motions_df,
        set_processing_info=set_processing_info,
    )

    control_result = evaluate_control_stability(
        motions_df=motions_df,
        set_processing_info=set_processing_info,
    )

    pacing_result = evaluate_pacing_strategy(
        motions_df=motions_df,
        set_processing_info=set_processing_info,
    )

    # Bundle evaluation results
    evaluation_results = {
        'cardiovascular_load': cv_result,
        'recovery_capacity': recovery_result,
        'output_sustainability': sustainability_result,
        'control_stability': control_result,
        'pacing_strategy': pacing_result,
        'session_type': 'First Session' if not history else 'Subsequent',
    }

    # Calculate trends if history is available
    if history and len(history) >= 4:
        # Convert history list to DataFrame format expected by evaluate_trends
        history_records = []
        for session in history:
            metrics = session.get('metrics', {})
            history_records.append({
                'session_date': session.get('date'),
                'cardiovascular_load_pct': metrics.get('Cardiovascular Load', {}).get('value'),
                'recovery_capacity_bpm': metrics.get('Recovery Capacity', {}).get('value'),
                'output_sustainability_pct': metrics.get('Performance Sustainability', {}).get('value'),
                'control_stability_cv': metrics.get('Control Stability', {}).get('value'),
                'pacing_strategy_pct': metrics.get('Pacing Strategy', {}).get('value'),
            })
        history_df = pd.DataFrame(history_records)

        # Call evaluate_trends with the DataFrame
        trend_results = evaluate_trends(
            current_results=evaluation_results,
            history_df=history_df,
        )
        evaluation_results['trends'] = trend_results

    # Format response
    dimension_analysis, trend, insight, composite_score_result, three_dim_assessment = (
        format_web_demo_response(evaluation_results)
    )

    # Calculate exertion
    try:
        exertion_result = calculate_exertion(
            'wall_ball',
            motions_df=motions_df,
            body_weight_kg=body_weight_kg,
            equipment_weight_kg=medicine_ball_weight_kg,
        )
    except Exception as e:
        print(f"Warning: Exertion calculation failed: {e}")
        exertion_result = None

    # Compute cardiac exertion from per-second HR data
    if exertion_result is not None and measures_df is not None and len(measures_df) > 0:
        gender = user_info.get('gender', 'Male')
        gender_key = 'female' if str(gender).lower() in ('female', 'f', '1') else 'male'
        hr_max = 207 - 0.7 * age
        hrs = measures_df['hr'].tolist() if 'hr' in measures_df.columns else []

        if hrs:
            cardiac_result = compute_cardiac_exertion(hrs, hr_max, rhr, gender=gender_key)
            cardiac_val = cardiac_result['cardiac_exertion']
            muscular_val = exertion_result.get('muscular_exertion', exertion_result.get('exertion', 0))
            combined = compute_combined_exertion(muscular_val, cardiac_val)
            exertion_result['cardiac_exertion'] = cardiac_val
            exertion_result['combined_exertion'] = combined['combined_exertion']

    # Convert DataFrames to list of dicts for JSON serialization (for HR chart)
    motions_list = motions_df.to_dict('records') if motions_df is not None else []
    measures_list = measures_df.to_dict('records') if measures_df is not None else []

    return {
        'dimension_analysis': dimension_analysis,
        'trend': trend,
        'insight': insight,
        'composite_score_result': composite_score_result,
        'three_dim_assessment': three_dim_assessment,
        'exertion': exertion_result,
        'evaluation_results': evaluation_results,
        'session_info': user_info,
        # Raw data for HR chart visualization
        'motions': motions_list,
        'measures': measures_list,
    }


def evaluate_time_based_web(time_series_df, hr_df, sets_df, user_info,
                            exercise_type='farmers_carry'):
    """
    Evaluate a time-based exercise session (Farmer's Carry, Sled Push, Sled Pull).

    Computes available 5-dimension scores (no Control Stability, no radar map),
    3D assessment, insight, and exertion (muscular + cardiac).

    Parameters
    ----------
    time_series_df : pd.DataFrame
        Per-second time series with columns: timestamp(ms), step_frequency(Hz)
    hr_df : pd.DataFrame
        Heart rate data with columns: timestamp, heart_rate (or hr)
    sets_df : pd.DataFrame
        Set data with columns: start_time(ms), stop_time(ms), motion, total_reps
    user_info : dict
        User profile: age, rhr, body_weight_kg, medicine_ball_weight_kg (equipment_weight_kg),
        gender (optional)
    exercise_type : str
        One of 'farmers_carry', 'sled_push', 'sled_pull'

    Returns
    -------
    dict
        Response with: dimension_analysis, three_dim_assessment, exertion, insight,
        evaluation_results, session_info
    """

    # Normalize column names
    def _norm_cols(df):
        df = df.copy()
        df.columns = df.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()
        return df

    time_series_df = _norm_cols(time_series_df)
    sets_df = _norm_cols(sets_df)
    if hr_df is not None and len(hr_df) > 0:
        hr_df = _norm_cols(hr_df)

    age = user_info.get('age', 30)
    rhr = user_info.get('rhr', 55)
    body_weight_kg = user_info.get('body_weight_kg', 70.0)
    equipment_weight_kg = user_info.get('medicine_ball_weight_kg',
                                        user_info.get('equipment_weight_kg', 0.0))
    gender = user_info.get('gender', 'Male')
    gender_key = 'female' if str(gender).lower() in ('female', 'f', '1') else 'male'

    # Set boundaries
    set_boundaries = list(zip(
        sets_df['start_time'].astype(int).tolist(),
        sets_df['stop_time'].astype(int).tolist(),
    ))

    # Step frequency array
    cadences = []
    if 'step_frequency' in time_series_df.columns:
        cadences = time_series_df['step_frequency'].tolist()

    # ── 5-dimension evaluation (partial) ──
    cv_result = None
    recovery_result = None

    # Build measures_df from hr_df if available
    measures_df = None
    if hr_df is not None and len(hr_df) > 0:
        measures_df = hr_df.copy()
        if 'heart_rate' in measures_df.columns and 'hr' not in measures_df.columns:
            measures_df['hr'] = measures_df['heart_rate']

        # Build a synthetic motions_df from sets for CV load evaluation
        motions_rows = []
        for i, (s_start, s_end) in enumerate(set_boundaries):
            motions_rows.append({
                'start_time': s_start,
                'stop_time': s_end,
                'motion': exercise_type,
                'set_index': i,
                'duration': (s_end - s_start) / 1000.0,
            })
            # Add rest between sets
            if i < len(set_boundaries) - 1:
                next_start = set_boundaries[i + 1][0]
                if next_start > s_end:
                    motions_rows.append({
                        'start_time': s_end,
                        'stop_time': next_start,
                        'motion': 'rest',
                        'set_index': i,
                        'duration': (next_start - s_end) / 1000.0,
                    })
        motions_df_synthetic = pd.DataFrame(motions_rows)

        cv_result = evaluate_cardiovascular_load(
            measures_df=measures_df,
            age=age,
            rhr=rhr,
            motions_df=motions_df_synthetic,
        )
        recovery_result = evaluate_recovery_capacity(
            motions_df=motions_df_synthetic,
            measures_df=measures_df,
        )

    # Output Sustainability from step frequency
    sustainability_result = evaluate_output_sustainability_time_based(
        cadences, set_boundaries=set_boundaries,
    )

    # Control Stability & Pacing Strategy: N/A for time-based exercises
    na_result = {
        'score': None, 'category': 'N/A',
        'interpretation': 'Not available for time-based exercises',
        'flags': [],
    }

    evaluation_results = {
        'cardiovascular_load': cv_result,
        'recovery_capacity': recovery_result,
        'output_sustainability': sustainability_result,
        'control_stability': na_result,
        'pacing_strategy': na_result,
        'session_type': 'First Session',
    }

    # 3D assessment + insight
    three_dim_assessment = calculate_three_dimension_assessment(evaluation_results)
    insight = generate_insights_from_csv(evaluation_results)

    # Exertion
    exertion_result = None
    try:
        exertion_result = calculate_exertion(
            exercise_type,
            time_series_df=time_series_df,
            body_weight_kg=body_weight_kg,
            equipment_weight_kg=equipment_weight_kg,
            set_boundaries=set_boundaries,
        )
    except Exception as e:
        print(f"Warning: Exertion calculation failed: {e}")

    # Cardiac exertion
    if exertion_result is not None and measures_df is not None and len(measures_df) > 0:
        hr_max = 207 - 0.7 * age
        hrs = measures_df['hr'].tolist() if 'hr' in measures_df.columns else []
        if hrs:
            cardiac_result = compute_cardiac_exertion(hrs, hr_max, rhr, gender=gender_key)
            cardiac_val = cardiac_result['cardiac_exertion']
            muscular_val = exertion_result.get('muscular_exertion', exertion_result.get('exertion', 0))
            combined = compute_combined_exertion(muscular_val, cardiac_val)
            exertion_result['cardiac_exertion'] = cardiac_val
            exertion_result['combined_exertion'] = combined['combined_exertion']

    # Build dimension_analysis dict (same structure as rep-based, minus radar)
    dimension_analysis = {}
    for dim_key in ['cardiovascular_load', 'recovery_capacity', 'output_sustainability',
                    'control_stability', 'pacing_strategy']:
        dim_result = evaluation_results.get(dim_key)
        if dim_result and dim_result.get('score') is not None:
            dimension_analysis[dim_key] = {
                'result': dim_result.get('category', 'N/A'),
                'score': dim_result.get('score'),
                'interpretation': dim_result.get('interpretation', 'N/A'),
                'flags': dim_result.get('flags', []),
            }
        else:
            dimension_analysis[dim_key] = {
                'result': 'N/A', 'score': None,
                'interpretation': 'Insufficient data', 'flags': [],
            }

    return {
        'dimension_analysis': dimension_analysis,
        'three_dim_assessment': three_dim_assessment,
        'exertion': exertion_result,
        'insight': insight,
        'evaluation_results': evaluation_results,
        'session_info': user_info,
    }


if __name__ == "__main__":
    # Test sample sessions
    for key in SAMPLE_SESSIONS:
        print(f"\n=== {key} ===")
        response = load_sample_session(key)
        print(f"Composite score: {response['composite_score_result']}")