"""
Exercise evaluation utility functions
"""

def is_exercise_motion(motion_type):
    """
    Determine if motion type is an exercise motion (excluding rest)
    
    Parameters:
    -----------
    motion_type : str
        Motion type string
    
    Returns:
    --------
    bool: True if exercise motion, False if rest
    """
    rest_keywords = ['rest', 'resting', 'break', 'pause']
    motion_lower = str(motion_type).lower()
    return not any(keyword in motion_lower for keyword in rest_keywords)


def filter_exercise_motions(motions_df):
    """
    Filter exercise motion data (excluding rest)
    
    Parameters:
    -----------
    motions_df : pd.DataFrame
        DataFrame containing motion column
    
    Returns:
    --------
    pd.DataFrame: DataFrame containing only exercise motions
    """
    if 'motion' not in motions_df.columns:
        return motions_df
    return motions_df[motions_df['motion'].apply(is_exercise_motion)].copy()


def process_sets_by_rep_count(motions_df):
    """
    Process sets based on rep count and determine which metrics can be calculated.
    
    Rules:
    - Sets with 10+ reps: Full analysis (all metrics including Control Stability)
    - Sets with 5-9 reps: Partial analysis (Output Sustainability, Pacing Strategy, Recovery Capacity only)
    - Sets with <5 reps: Excluded entirely
    
    Parameters:
    -----------
    motions_df : pd.DataFrame
        Contains motion data with optional 'set_index' column for grouping
    
    Returns:
    --------
    dict: Contains:
        - 'full_analysis_sets': DataFrame with sets having 10+ reps
        - 'partial_analysis_sets': DataFrame with sets having 5-9 reps
        - 'excluded_sets': DataFrame with sets having <5 reps
        - 'set_rep_counts': dict mapping set_index to rep count
        - 'set_analysis_flags': dict mapping set_index to analysis type
    """
    import pandas as pd
    import numpy as np
    
    # Filter exercise motions only
    exercise_motions = filter_exercise_motions(motions_df)
    exercise_motions = exercise_motions.sort_values('start_time').copy()
    
    # Determine how to group into sets
    if 'set_index' in exercise_motions.columns:
        # Group by set_index
        set_groups = exercise_motions.groupby('set_index')
    else:
        # No set_index provided — infer sets by contiguous exercise blocks separated by rest
        # 1) Sort full motions (including rest) by time
        sorted_motions = motions_df.sort_values('start_time').copy()
        current_set_idx = 0
        inferred_indices = []
        for _, row in sorted_motions.iterrows():
            if is_exercise_motion(row['motion']):
                inferred_indices.append(current_set_idx)
            else:
                inferred_indices.append(None)
                # Increment set index when encountering rest, so next exercise block is a new set
                current_set_idx += 1
        sorted_motions['set_index_inferred'] = inferred_indices
        
        # Map inferred set_index back to exercise motions
        exercise_motions = sorted_motions[sorted_motions['motion'].apply(is_exercise_motion)].copy()
        exercise_motions['set_index'] = exercise_motions['set_index_inferred']
        exercise_motions = exercise_motions.drop(columns=['set_index_inferred'])
        
        set_groups = exercise_motions.groupby('set_index')
    
    # Calculate rep count for each set
    set_rep_counts = {}
    set_dataframes = {}
    
    for set_idx, group_df in set_groups:
        rep_count = len(group_df)
        set_rep_counts[set_idx] = rep_count
        set_dataframes[set_idx] = group_df
    
    # Categorize sets
    full_analysis_sets = []  # 10+ reps
    partial_analysis_sets = []  # 5-9 reps
    excluded_sets = []  # <5 reps
    
    set_analysis_flags = {}
    
    for set_idx, rep_count in set_rep_counts.items():
        if rep_count >= 10:
            full_analysis_sets.append(set_idx)
            set_analysis_flags[set_idx] = 'full'
        elif rep_count >= 5:
            partial_analysis_sets.append(set_idx)
            set_analysis_flags[set_idx] = 'partial'
        else:
            excluded_sets.append(set_idx)
            set_analysis_flags[set_idx] = 'excluded'
    
    # Create DataFrames for each category
    full_df = pd.concat([set_dataframes[idx] for idx in full_analysis_sets]) if full_analysis_sets else pd.DataFrame()
    partial_df = pd.concat([set_dataframes[idx] for idx in partial_analysis_sets]) if partial_analysis_sets else pd.DataFrame()
    excluded_df = pd.concat([set_dataframes[idx] for idx in excluded_sets]) if excluded_sets else pd.DataFrame()
    
    result = {
        'full_analysis_sets': full_df,
        'partial_analysis_sets': partial_df,
        'excluded_sets': excluded_df,
        'set_rep_counts': set_rep_counts,
        'set_analysis_flags': set_analysis_flags,
        'all_exercise_motions': exercise_motions  # Keep original for metrics that don't need filtering
    }
    
    return result


def filter_exercise_measures(measures_df, motions_df):
    """
    Filter measures data based on motions data, keeping only exercise period data
    
    Parameters:
    -----------
    measures_df : pd.DataFrame
        Time series data containing timestamp column
    motions_df : pd.DataFrame
        Motion data containing start_time and stop_time, need to filter exercise periods
    
    Returns:
    --------
    pd.DataFrame: Time series data containing only exercise periods
    """
    # Filter exercise period motions
    exercise_motions = filter_exercise_motions(motions_df)
    
    if len(exercise_motions) == 0:
        return measures_df
    
    # Get all exercise period time ranges
    exercise_start = exercise_motions['start_time'].min()
    exercise_end = exercise_motions['stop_time'].max()
    
    # Filter measures data
    filtered_measures = measures_df[
        (measures_df['timestamp'] >= exercise_start) & 
        (measures_df['timestamp'] <= exercise_end)
    ].copy()
    
    return filtered_measures


def calculate_cardiorespiratory_limit(hr_data=None, method='age_formula', age=None, percentile=95):
    """
    Estimate maximum heart rate (HRmax)
    
    Parameters:
    -----------
    hr_data : pd.Series, optional
        Heart rate data (for percentile or max method)
    method : str
        Calculation method: 'age_formula' (default, uses formula 207 - 0.7 * age), 
                 'percentile', 'max', 'old_formula'
    age : float, optional
        Age (required for age_formula method)
    percentile : float
        Percentile to use if using percentile method (default 95)
    
    Returns:
    --------
    float: Estimated maximum heart rate
    """
    import pandas as pd
    
    if method == 'age_formula':
        if age is None:
            raise ValueError("age parameter is required when using 'age_formula' method")
        # Formula: HRmax = 207 - 0.7 * age
        return 207.0 - 0.7 * age
    elif method == 'max':
        if hr_data is None:
            raise ValueError("hr_data parameter is required when using 'max' method")
        return float(hr_data.max())
    elif method == 'percentile':
        if hr_data is None:
            raise ValueError("hr_data parameter is required when using 'percentile' method")
        return float(hr_data.quantile(percentile / 100))
    elif method == 'old_formula':
        if age is None:
            raise ValueError("age parameter is required when using 'old_formula' method")
        # Old formula: 220 - age
        return 220.0 - age
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: 'age_formula', 'percentile', 'max', 'old_formula'")


def evaluate_cardiovascular_load(measures_df, hr_max=None, hr_max_method='age_formula', age=None, rhr=55, motions_df=None):
    """
    Evaluate Cardiovascular Load
    
    PRIMARY METRIC: Time in High-Intensity Zone (85%+ HRmax or using Heart Rate Reserve method)
    
    Parameters:
    -----------
    measures_df : pd.DataFrame
        Time series data containing timestamp and hr columns
    hr_max : float, optional
        Maximum heart rate. If None, will be estimated from data
    hr_max_method : str
        Estimation method if hr_max is None. Default is 'age_formula' (uses formula 207 - 0.7 * age)
    age : float, optional
        Age (required when hr_max_method='age_formula')
    rhr : float, optional
        Resting Heart Rate, default 55 bpm. Used for Heart Rate Reserve (HRR) calculation
    motions_df : pd.DataFrame, optional
        Motion data containing motion column, used to filter exercise periods
    
    Returns:
    --------
    dict: Dictionary containing evaluation results
        - hr_max: Maximum heart rate used
        - rhr: Resting heart rate
        - hrr: Heart Rate Reserve (HRmax - RHR)
        - hr_threshold: 85% high-intensity threshold (using HRR method: RHR + 0.85 * HRR)
        - total_time_s: Total training time (seconds)
        - high_intensity_time_s: High-intensity time (seconds)
        - high_intensity_percentage: High-intensity time percentage
        - category: Classification ('Optimal', 'Good', 'Needs Improvement')
        - interpretation: Interpretation text
    """
    import pandas as pd
    
    # If motions_df is provided, only calculate data for exercise periods
    if motions_df is not None:
        measures_df = filter_exercise_measures(measures_df, motions_df)
    
    # Ensure data is sorted by time
    measures_df = measures_df.sort_values('timestamp').copy()
    
    # Calculate or use provided maximum heart rate
    if hr_max is None:
        if hr_max_method == 'age_formula':
            hr_max = calculate_cardiorespiratory_limit(method=hr_max_method, age=age)
        elif hr_max_method in ['percentile', 'max']:
            hr_max = calculate_cardiorespiratory_limit(measures_df['hr'], method=hr_max_method)
        else:
            hr_max = calculate_cardiorespiratory_limit(measures_df['hr'], method=hr_max_method, age=age)
    
    # Calculate Heart Rate Reserve
    hrr = hr_max - rhr
    
    # Calculate 85% / 80% high-intensity thresholds using HRR method
    # HR_threshold = RHR + p * HRR = RHR + p * (HRmax - RHR)
    hr_threshold_85 = rhr + 0.85 * hrr
    hr_threshold_80 = rhr + 0.80 * hrr
    
    # Calculate total training time (sampling per second, so total time is number of records)
    total_time_s = len(measures_df)
    
    # ------------------------------------------------------------------
    # Global (all sets combined) high-intensity time ratio (85% / 80%), used as baseline & fallback
    # ------------------------------------------------------------------
    high_intensity_time_s_85 = (measures_df['hr'] >= hr_threshold_85).sum()
    high_intensity_time_s_80 = (measures_df['hr'] >= hr_threshold_80).sum()
    
    high_intensity_percentage_85 = (
        (high_intensity_time_s_85 / total_time_s) * 100 if total_time_s > 0 else 0
    )
    high_intensity_percentage_80 = (
        (high_intensity_time_s_80 / total_time_s) * 100 if total_time_s > 0 else 0
    )
    
    # ------------------------------------------------------------------
    # Per-set high-intensity time ratio:
    # 1) Compute high_intensity_percentage within each set
    # 2) Average across all sets to get the final high_intensity_percentage
    # ------------------------------------------------------------------
    high_intensity_pct_by_set_85 = []
    high_intensity_pct_by_set_80 = []
    set_indices = []
    if motions_df is not None:
        try:
            set_processing_info = process_sets_by_rep_count(motions_df)
            exercise_motions = set_processing_info.get('all_exercise_motions')
            if exercise_motions is not None and 'set_index' in exercise_motions.columns:
                exercise_motions = exercise_motions.sort_values('start_time').copy()
                for set_idx, set_group in exercise_motions.groupby('set_index'):
                    set_start = set_group['start_time'].min()
                    set_stop = set_group['stop_time'].max()
                    # Get HR samples within this set's time window
                    set_measures = measures_df[
                        (measures_df['timestamp'] >= set_start) &
                        (measures_df['timestamp'] <= set_stop)
                    ]
                    if len(set_measures) == 0:
                        continue
                    set_total_s = len(set_measures)
                    # 85% threshold
                    set_high_s_85 = (set_measures['hr'] >= hr_threshold_85).sum()
                    set_pct_85 = (set_high_s_85 / set_total_s) * 100 if set_total_s > 0 else 0
                    # 80% threshold
                    set_high_s_80 = (set_measures['hr'] >= hr_threshold_80).sum()
                    set_pct_80 = (set_high_s_80 / set_total_s) * 100 if set_total_s > 0 else 0
                    
                    high_intensity_pct_by_set_85.append(set_pct_85)
                    high_intensity_pct_by_set_80.append(set_pct_80)
                    set_indices.append(int(set_idx) if set_idx is not None else None)
            
            # Only override global percentage with per-set average if at least one valid set exists
            if len(high_intensity_pct_by_set_85) > 0:
                high_intensity_percentage_85 = sum(high_intensity_pct_by_set_85) / len(high_intensity_pct_by_set_85)
                high_intensity_time_s_85 = (high_intensity_percentage_85 / 100.0) * total_time_s
            if len(high_intensity_pct_by_set_80) > 0:
                high_intensity_percentage_80 = sum(high_intensity_pct_by_set_80) / len(high_intensity_pct_by_set_80)
                high_intensity_time_s_80 = (high_intensity_percentage_80 / 100.0) * total_time_s
        except Exception:
            # Fall back to global calculation on any error; do not interrupt evaluation
            high_intensity_pct_by_set_85 = []
            high_intensity_pct_by_set_80 = []
            set_indices = []
    
    # Check for flags
    flags = []
    
    # Flag: If starting HR >70% HRmax → Flag "Elevated baseline from previous work"
    if len(measures_df) > 0:
        starting_hr = measures_df.iloc[0]['hr']
        hr_70_percent = rhr + 0.70 * hrr
        if starting_hr > hr_70_percent:
            flags.append("Elevated baseline from previous work")
    
    # Flag: If other exercises within 2 minutes before/after Wall Balls → Flag "Mixed exercise - HR may be elevated"
    # Note: This would require checking motions_df for other exercise types, but for now we'll skip this
    # as we're only analyzing wall_ball exercises
    
    # Classification using 1-5 scoring system
    # Use 85% high-intensity ratio as primary score; also compute 80% threshold score for comparison
    # Score 5: 10-20% - Optimal stimulus
    # Score 4: 5-10% or 20-25% - Good
    # Score 3: 25-30% - Adequate but high
    # Score 2: 30-40% or <5% - Too much OR insufficient
    # Score 1: >40% - Excessive
    # Keep legacy variable name; defaults to 85% result
    high_intensity_percentage = high_intensity_percentage_85
    
    def _score_from_pct(pct: float):
        if 10 <= pct <= 20:
            return 5, "Optimal", "Optimal stimulus"
        elif (5 <= pct < 10) or (20 < pct <= 25):
            return 4, "Good", "Good"
        elif 25 < pct <= 30:
            return 3, "Adequate", "Adequate but high"
        elif (30 < pct <= 40) or (pct < 5):
            return 2, "Needs Improvement", "Too much OR insufficient"
        else:  # >40%
            return 1, "Needs Improvement", "Excessive"
    
    # 85% primary score
    score, category, interpretation = _score_from_pct(high_intensity_percentage_85)
    # 80% comparison score
    score_80, category_80, interpretation_80 = _score_from_pct(high_intensity_percentage_80)
    
    # Legacy fields kept for backward compat; new 85p_ / 80p_ prefixed fields added for comparison
    result = {
        'hr_max': hr_max,
        'rhr': rhr,
        'hrr': hrr,
        # Legacy fields: default to 85% threshold
        'hr_threshold': hr_threshold_85,
        'total_time_s': total_time_s,
        'high_intensity_time_s': high_intensity_time_s_85,
        'high_intensity_percentage': high_intensity_percentage_85,
        'high_intensity_percentage_by_set': high_intensity_pct_by_set_85,
        'set_indices': set_indices,
        # New fields: full 85% and 80% variable sets
        '85p_hr_threshold': hr_threshold_85,
        '85p_high_intensity_time_s': high_intensity_time_s_85,
        '85p_high_intensity_percentage': high_intensity_percentage_85,
        '85p_high_intensity_percentage_by_set': high_intensity_pct_by_set_85,
        '80p_hr_threshold': hr_threshold_80,
        '80p_high_intensity_time_s': high_intensity_time_s_80,
        '80p_high_intensity_percentage': high_intensity_percentage_80,
        '80p_high_intensity_percentage_by_set': high_intensity_pct_by_set_80,
        '80p_score': score_80,
        '80p_category': category_80,
        '80p_interpretation': interpretation_80,
        'score': score,
        'category': category,
        'interpretation': interpretation,
        'flags': flags
    }
    
    return result


def evaluate_recovery_capacity(motions_df, measures_df, recovery_window_s=60, set_processing_info=None):
    """
    Evaluate Recovery Capacity
    
    PRIMARY METRIC: HR Drop Rate (ADAPTIVE)
    - What it measures: Heart rate recovery ability
    - Calculation (ADAPTIVE):
      * For inter-set rest ≥30s:
        HR Drop Rate = (Peak HR at end of set - HR at end of rest period) / Rest duration in seconds
        Average across all valid rest periods
      * For no rest (continuous work) or all rest <30s:
        Post-Session HR Drop = Peak HR at end - HR at 60s post-completion
        Convert to rate: Post-Session Drop / 60 seconds
    
    Note: Can be calculated for sets with 5+ reps (doesn't require 10+ like Control Stability)
    
    Parameters:
    -----------
    motions_df : pd.DataFrame
        Contains motion data with start_time and stop_time
    measures_df : pd.DataFrame
        Contains time series data with timestamp and hr columns
    recovery_window_s : int
        Recovery window time in seconds (default 60)
    set_processing_info : dict, optional
        Output from process_sets_by_rep_count() containing set categorization info
    
    Returns:
    --------
    dict: Contains evaluation results
    """
    import pandas as pd
    import numpy as np
    
    # Process sets by rep count if not provided
    if set_processing_info is None:
        set_processing_info = process_sets_by_rep_count(motions_df)
    
    # Use both full and partial analysis sets (5+ reps)
    # Exclude sets with <5 reps
    full_sets = set_processing_info['full_analysis_sets']
    partial_sets = set_processing_info['partial_analysis_sets']
    
    # Combine sets with 5+ reps
    if len(full_sets) > 0 and len(partial_sets) > 0:
        exercise_motions = pd.concat([full_sets, partial_sets]).sort_values('start_time').copy()
    elif len(full_sets) > 0:
        exercise_motions = full_sets.sort_values('start_time').copy()
    elif len(partial_sets) > 0:
        exercise_motions = partial_sets.sort_values('start_time').copy()
    else:
        # No sets with 5+ reps
        return {
            'avg_hr_drop': 0,
            'avg_hr_drop_rate': 0,
            'hr_drops': [],
            'hr_drop_rates': [],
            'recovery_details': [],
            'score': None,
            'category': "N/A",
            'interpretation': "No sets with 5+ reps available for analysis",
            'recovery_window_s': recovery_window_s,
            'flags': []
        }
    
    measures_df = measures_df.sort_values('timestamp').copy()
    
    # Group exercise motions by set_index to identify complete sets
    # When sets are properly separated by rest, we should calculate recovery after each complete set
    if 'set_index' not in exercise_motions.columns:
        # If no set_index, infer sets from rest periods
        sorted_motions = motions_df.sort_values('start_time').copy()
        current_set_idx = 0
        inferred_indices = []
        for _, row in sorted_motions.iterrows():
            if is_exercise_motion(row['motion']):
                inferred_indices.append(current_set_idx)
            else:
                inferred_indices.append(None)
                current_set_idx += 1
        sorted_motions['set_index_inferred'] = inferred_indices
        exercise_motions = sorted_motions[sorted_motions['motion'].apply(is_exercise_motion)].copy()
        exercise_motions['set_index'] = exercise_motions['set_index_inferred']
        exercise_motions = exercise_motions.drop(columns=['set_index_inferred'])
    
    # Group by set_index to get complete sets
    set_groups = exercise_motions.groupby('set_index')
    
    # Calculate HR Drop Rate (bpm/s) after each complete set ends
    hr_drop_rates = []  # New: HR Drop Rate (bpm/s)
    hr_drops = []  # Keep for backward compatibility
    recovery_details = []
    
    # Get all sets sorted by set_index
    set_indices = sorted(set_groups.groups.keys())
    
    for set_idx in set_indices:
        set_motions = set_groups.get_group(set_idx).sort_values('start_time')
        
        # Step 1: Find the last rep's end time (this is when the set ends)
        set_end_time_ms = set_motions['stop_time'].max()
        
        # Step 2: Find Peak HR at set end moment (the exact second when set ends)
        # Find heart rate at the exact timestamp when set ends
        peak_hr_data = measures_df[measures_df['timestamp'] == set_end_time_ms]
        if len(peak_hr_data) == 0:
            # If no exact match, use the nearest data point at or before set end time
            peak_hr_data = measures_df[measures_df['timestamp'] <= set_end_time_ms]
            if len(peak_hr_data) == 0:
                continue
            # Use the last available HR reading before or at set end
            peak_hr = peak_hr_data.iloc[-1]['hr']
        else:
            peak_hr = peak_hr_data.iloc[0]['hr']
        
        # Step 3: Find the rest period after this set and the next set's start time
        # Find rest periods that start after this set ends
        rest_periods = motions_df[
            (motions_df['motion'].apply(lambda x: not is_exercise_motion(x))) &
            (motions_df['start_time'] >= set_end_time_ms)
        ].sort_values('start_time')
        
        if len(rest_periods) == 0:
            # No rest period found (continuous work), skip this set
            # Will use post-session recovery if no valid inter-set rest periods found
            continue
        
        # Get the rest period after this set
        rest_period = rest_periods.iloc[0]
        rest_start_time_ms = rest_period['start_time']
        rest_end_time_ms = rest_period['stop_time']
        
        # Step 4: Find End HR at rest end moment (next set's first rep start_time)
        # Find the next set (if exists)
        next_set_idx = None
        for next_idx in set_indices:
            if next_idx > set_idx:
                next_set_idx = next_idx
                break
        
        if next_set_idx is None:
            # This is the last set, use rest period end time
            end_hr_timestamp_ms = rest_end_time_ms
        else:
            # Use next set's first rep start_time as rest end moment
            next_set_motions = set_groups.get_group(next_set_idx).sort_values('start_time')
            end_hr_timestamp_ms = next_set_motions.iloc[0]['start_time']
        
        # Find heart rate at the exact timestamp when rest ends (next set starts)
        end_hr_data = measures_df[measures_df['timestamp'] == end_hr_timestamp_ms]
        if len(end_hr_data) == 0:
            # If no exact match, use the nearest data point at or before rest end time
            end_hr_data = measures_df[measures_df['timestamp'] <= end_hr_timestamp_ms]
            if len(end_hr_data) == 0:
                continue
            # Use the last available HR reading before or at rest end
            end_hr = end_hr_data.iloc[-1]['hr']
        else:
            end_hr = end_hr_data.iloc[0]['hr']
        
        # Step 5: Calculate Rest Duration (from set end to next set start, or rest end)
        rest_duration_s = (end_hr_timestamp_ms - set_end_time_ms) / 1000
        
        # Step 6: Calculate HR Drop Rate for this rest period
        # HR Drop Rate = (Peak HR at end of set - HR at end of rest period) / Rest duration in seconds
        # Only calculate if rest duration >= 30 seconds (ADAPTIVE: For inter-set rest ≥30s)
        if rest_duration_s >= 30:
            hr_drop = peak_hr - end_hr
            hr_drop_rate = hr_drop / rest_duration_s  # bpm/s
            
            hr_drop_rates.append(hr_drop_rate)
            hr_drops.append(hr_drop)  # Keep for backward compatibility
            
            recovery_details.append({
                'set': int(set_idx) + 1,  # Set number (1-indexed)
                'peak_hr': peak_hr,
                'recovery_hr': end_hr,
                'hr_drop': hr_drop,
                'hr_drop_rate': hr_drop_rate,
                'recovery_time_s': rest_duration_s
            })
    
    # ADAPTIVE: If no valid recovery periods (all <30s or no rest), use post-session recovery
    # For no rest (continuous work): Post-Session HR Drop = Peak HR at end - HR at 60s post-completion
    # Convert to rate: Post-Session Drop / 60 seconds
    if len(hr_drop_rates) == 0:
        # Calculate post-session recovery: Peak HR at end - HR at 60s post-completion
        if len(set_indices) > 0:
            # Get the last set's end time
            last_set_idx = set_indices[-1]
            last_set_motions = set_groups.get_group(last_set_idx).sort_values('start_time')
            last_set_end = last_set_motions['stop_time'].max()
            
            # Find Peak HR at end of set (same logic as inter-set recovery)
            peak_hr_data = measures_df[measures_df['timestamp'] == last_set_end]
            if len(peak_hr_data) == 0:
                # If no exact match, use the nearest data point at or before set end time
                peak_hr_data = measures_df[measures_df['timestamp'] <= last_set_end]
                if len(peak_hr_data) == 0:
                    # No HR data available at set end, skip post-session recovery calculation
                    peak_hr_end = None
                else:
                    # Use the last available HR reading before or at set end
                    peak_hr_end = peak_hr_data.iloc[-1]['hr']
            else:
                peak_hr_end = peak_hr_data.iloc[0]['hr']
            
            # Calculate post-session recovery if we have peak HR data
            if peak_hr_end is not None:
                post_session_time = last_set_end + 60000  # 60 seconds after
                post_session_hr_data = measures_df[
                    (measures_df['timestamp'] > last_set_end) & 
                    (measures_df['timestamp'] <= post_session_time)
                ]
                
                if len(post_session_hr_data) > 0:
                    # Use HR at 60s post (or closest available)
                    post_session_hr = post_session_hr_data.iloc[-1]['hr']
                    hr_drop = peak_hr_end - post_session_hr
                    hr_drop_rate = hr_drop / 60.0  # Convert to rate per second: Post-Session Drop / 60 seconds
                    hr_drop_rates.append(hr_drop_rate)
                    hr_drops.append(hr_drop)  # Keep for backward compatibility
                    
                    recovery_details.append({
                        'set': 'Post-Session',
                        'peak_hr': peak_hr_end,
                        'recovery_hr': post_session_hr,
                        'hr_drop': hr_drop,
                        'hr_drop_rate': hr_drop_rate,
                        'recovery_time_s': 60.0
                    })
    
    if len(hr_drop_rates) == 0:
        avg_hr_drop_rate = 0
    else:
        avg_hr_drop_rate = np.mean(hr_drop_rates)
    
    # Classification using 1-5 scoring system based on HR Drop Rate (bpm/s)
    # Score 5: >0.60 - Elite recovery
    # Score 4: 0.45-0.60 - Strong recovery
    # Score 3: 0.30-0.45 - Adequate recovery
    # Score 2: 0.20-0.30 - Slow recovery
    # Score 1: <0.20 - Very slow recovery
    if avg_hr_drop_rate > 0.60:
        score = 5
        category = "Optimal"
        interpretation = "Elite recovery"
    elif 0.45 <= avg_hr_drop_rate <= 0.60:
        score = 4
        category = "Good"
        interpretation = "Strong recovery"
    elif 0.30 <= avg_hr_drop_rate < 0.45:
        score = 3
        category = "Adequate"
        interpretation = "Adequate recovery"
    elif 0.20 <= avg_hr_drop_rate < 0.30:
        score = 2
        category = "Needs Improvement"
        interpretation = "Slow recovery"
    else:  # <0.20
        score = 1
        category = "Needs Improvement"
        interpretation = "Very slow recovery"
    
    # Check for flags
    flags = []
    
    # Flag: If all rest <30s → Flag "Short rest - post-session recovery used"
    if len(hr_drop_rates) == 0 and len(recovery_details) > 0:
        if recovery_details[0].get('set') == 'Post-Session':
            flags.append("Short rest - post-session recovery used")
    
    # Flag: Check if any rest periods were <30s
    short_rest_count = sum(1 for detail in recovery_details if detail.get('recovery_time_s', 0) < 30)
    if short_rest_count > 0 and len(recovery_details) > short_rest_count:
        flags.append(f"{short_rest_count} rest period(s) <30s - limited recovery data")
    
    # ---- Raw values per set (for exporting / debugging) ----
    set_labels = []
    peak_hr_by_set = []
    recovery_hr_by_set = []
    hr_drop_by_set = []
    hr_drop_rate_by_set = []
    recovery_time_s_by_set = []
    
    for d in recovery_details:
        set_labels.append(d.get('set'))
        peak_hr_by_set.append(d.get('peak_hr'))
        recovery_hr_by_set.append(d.get('recovery_hr'))
        hr_drop_by_set.append(d.get('hr_drop'))
        hr_drop_rate_by_set.append(d.get('hr_drop_rate'))
        recovery_time_s_by_set.append(d.get('recovery_time_s'))
    
    result = {
        'avg_hr_drop': avg_hr_drop_rate * recovery_window_s if avg_hr_drop_rate > 0 else 0,  # Keep for backward compatibility
        'avg_hr_drop_rate': avg_hr_drop_rate,  # New: HR Drop Rate (bpm/s)
        'hr_drops': [rate * recovery_window_s for rate in hr_drop_rates] if len(hr_drop_rates) > 0 else [],  # Keep for backward compatibility
        'hr_drop_rates': hr_drop_rates,  # New: HR Drop Rates (bpm/s)
        'recovery_details': recovery_details,
        # Raw per-set values (aligned with recovery_details)
        'set_labels': set_labels,
        'peak_hr_by_set': peak_hr_by_set,
        'recovery_hr_by_set': recovery_hr_by_set,
        'hr_drop_by_set': hr_drop_by_set,
        'hr_drop_rate_by_set': hr_drop_rate_by_set,
        'recovery_time_s_by_set': recovery_time_s_by_set,
        'score': score,
        'category': category,
        'interpretation': interpretation,
        'recovery_window_s': recovery_window_s,
        'flags': flags
    }
    
    return result


def evaluate_output_sustainability(motions_df, set_processing_info=None):
    """
    Evaluate Output Sustainability
    
    Evaluates whether work output can be continuously maintained during the training process 
    and whether pacing strategy is appropriate.
    
    PRIMARY METRIC: Performance Decline Across Session (ADAPTIVE)
    - What it measures: Performance decline across Wall Ball work
    - Calculation (ADAPTIVE):
      * If discrete sets exist:
        Output Sustainability = ((First Set reps/min - Last Set reps/min) / First Set reps/min) × 100
      * If continuous work:
        Auto-segment into 10-rep virtual sets
        Compare first vs last virtual set
    
    Note: Can be calculated for sets with 5+ reps (doesn't require 10+ like Control Stability)
    
    Parameters:
    -----------
    motions_df : pd.DataFrame
        Contains motion data with duration field
    set_processing_info : dict, optional
        Output from process_sets_by_rep_count() containing set categorization info
    
    Returns:
    --------
    dict: Contains evaluation results
    """
    import pandas as pd
    import numpy as np
    
    # Process sets by rep count if not provided
    if set_processing_info is None:
        set_processing_info = process_sets_by_rep_count(motions_df)
    
    # For Output Sustainability, we compare first set vs final set
    # If all motions are in one set (common when each record is a rep), 
    # we should treat each rep as a separate "set" for this metric
    # OR use all exercise motions directly if they represent individual sets
    
    # Get all exercise motions (not filtered by rep count for this metric)
    # Output Sustainability can be calculated with any number of sets/reps
    all_exercise_motions = set_processing_info.get('all_exercise_motions', None)
    if all_exercise_motions is None or len(all_exercise_motions) == 0:
        # Fallback: use full and partial sets
        full_sets = set_processing_info['full_analysis_sets']
        partial_sets = set_processing_info['partial_analysis_sets']
        
        if len(full_sets) > 0 and len(partial_sets) > 0:
            exercise_motions = pd.concat([full_sets, partial_sets]).sort_values('start_time').copy()
        elif len(full_sets) > 0:
            exercise_motions = full_sets.sort_values('start_time').copy()
        elif len(partial_sets) > 0:
            exercise_motions = partial_sets.sort_values('start_time').copy()
        else:
            exercise_motions = filter_exercise_motions(motions_df).sort_values('start_time').copy()
    else:
        exercise_motions = all_exercise_motions.sort_values('start_time').copy()
    
    if len(exercise_motions) == 0:
        return {
            "error": "Insufficient data",
            'first_set_output': 0,
            'final_set_output': 0,
            'performance_decline': 0,
            'reps_per_min': [],
            'set_indices': [],
            'grouping_mode': "N/A",
            'reps_per_min_by_set_mean': {},
            'reps_per_min_by_set_raw': {},
            'score': None,
            'category': "N/A",
            'interpretation': "No exercise motions available for analysis",
            'flags': []
        }
    
    exercise_motions['reps_per_min'] = 60.0 / exercise_motions['duration']
    
    # ADAPTIVE: For Output Sustainability, we need to compare first set vs final set
    # Strategy: 
    # 1. If discrete sets exist (set_index with >=2 sets): group by set_index
    # 2. If continuous work (no set_index or single set): auto-segment into 10-rep virtual sets
    grouping_mode = "virtual_sets"
    set_indices = []
    reps_per_min_by_set_raw = {}
    reps_per_min_by_set_mean = {}
    
    if 'set_index' in exercise_motions.columns:
        unique_sets = exercise_motions['set_index'].nunique()
        if unique_sets >= 2:
            # Discrete sets exist: use actual sets
            grouping_mode = "discrete_sets"
            grouped = exercise_motions.groupby('set_index')['reps_per_min']
            set_performance = grouped.mean().sort_index()
            set_indices = [int(x) for x in set_performance.index.tolist()]
            reps_per_min_by_set_mean = {int(k): float(v) for k, v in set_performance.to_dict().items()}
            reps_per_min_by_set_raw = {
                int(k): [float(x) for x in grouped.get_group(k).dropna().tolist()]
                for k in set_performance.index.tolist()
            }
        else:
            # Continuous work: auto-segment into 10-rep virtual sets
            grouping_mode = "virtual_sets"
            n_reps = len(exercise_motions)
            virtual_set_size = 10
            virtual_sets = []
            
            for i in range(0, n_reps, virtual_set_size):
                virtual_set_reps = exercise_motions.iloc[i:i+virtual_set_size]
                if len(virtual_set_reps) > 0:
                    virtual_set_avg = virtual_set_reps['reps_per_min'].mean()
                    virtual_sets.append(virtual_set_avg)
                    virtual_set_idx = i // virtual_set_size
                    set_indices.append(virtual_set_idx)
                    reps_per_min_by_set_mean[virtual_set_idx] = float(virtual_set_avg)
                    reps_per_min_by_set_raw[virtual_set_idx] = virtual_set_reps['reps_per_min'].dropna().tolist()
            
            if len(virtual_sets) > 0:
                set_performance = pd.Series(virtual_sets)
            else:
                set_performance = pd.Series([exercise_motions['reps_per_min'].mean()])
    else:
        # No set_index: continuous work, auto-segment into 10-rep virtual sets
        grouping_mode = "virtual_sets"
        n_reps = len(exercise_motions)
        virtual_set_size = 10
        virtual_sets = []
        
        for i in range(0, n_reps, virtual_set_size):
            virtual_set_reps = exercise_motions.iloc[i:i+virtual_set_size]
            if len(virtual_set_reps) > 0:
                virtual_set_avg = virtual_set_reps['reps_per_min'].mean()
                virtual_sets.append(virtual_set_avg)
                virtual_set_idx = i // virtual_set_size
                set_indices.append(virtual_set_idx)
                reps_per_min_by_set_mean[virtual_set_idx] = float(virtual_set_avg)
                reps_per_min_by_set_raw[virtual_set_idx] = virtual_set_reps['reps_per_min'].dropna().tolist()
        
        if len(virtual_sets) > 0:
            set_performance = pd.Series(virtual_sets)
        else:
            set_performance = pd.Series([exercise_motions['reps_per_min'].mean()])
    
    num_sets = len(set_performance)
    if num_sets < 2:
        return {
            "error": "Need at least 2 sets to calculate performance decline",
            'first_set_output': set_performance.iloc[0] if num_sets > 0 else 0,
            'final_set_output': set_performance.iloc[-1] if num_sets > 0 else 0,
            'performance_decline': 0,
            'reps_per_min': set_performance.tolist() if num_sets > 0 else [],
            'set_indices': set_indices,
            'grouping_mode': grouping_mode,
            'reps_per_min_by_set_mean': reps_per_min_by_set_mean,
            'reps_per_min_by_set_raw': reps_per_min_by_set_raw,
            'score': None,
            'category': "N/A",
            'interpretation': "Need at least 2 sets to calculate performance decline",
            'flags': []
        }
    
    # PRIMARY METRIC: Calculate performance decline from first set to final set
    # Output Sustainability = ((First Set reps/min - Last Set reps/min) / First Set reps/min) × 100
    first_set_val = set_performance.iloc[0]
    final_set_val = set_performance.iloc[-1]
    
    if first_set_val > 0:
        decline_pct = ((first_set_val - final_set_val) / first_set_val) * 100
    else:
        decline_pct = 0
    
    # Classification using 1-5 scoring system
    # Score 5: <5% decline - Optimal sustainability
    # Score 4: 5-12% decline - Good sustainability
    # Score 3: 12-20% decline - Adequate sustainability
    # Score 2: 20-30% decline - Limited sustainability
    # Score 1: >30% decline - Very limited sustainability
    if decline_pct < 5:
        score = 5
        category = "Optimal"
        interpretation = "Optimal sustainability"
    elif 5 <= decline_pct < 12:
        score = 4
        category = "Good"
        interpretation = "Good sustainability"
    elif 12 <= decline_pct < 20:
        score = 3
        category = "Adequate"
        interpretation = "Adequate sustainability"
    elif 20 <= decline_pct <= 30:
        score = 2
        category = "Needs Improvement"
        interpretation = "Limited sustainability"
    else:  # >30%
        score = 1
        category = "Needs Improvement"
        interpretation = "Very limited sustainability"
    
    # Add evidence-based context for excessive decline
    if decline_pct > 30:
        interpretation += ". Likely glycogen depletion or inadequate aerobic base."
    
    # Check for flags
    flags = []
    
    # Flag: If total reps <30 → Flag "Limited volume"
    total_reps = len(exercise_motions)
    if total_reps < 30:
        flags.append("Limited volume")
    
    # Flag: If first set >2x average → Flag "Uneven pacing"
    if num_sets >= 2:
        avg_output = set_performance.mean()
        if first_set_val > 2 * avg_output:
            flags.append("Uneven pacing")
    
    # Flag: If other exercises between Wall Balls → Flag "Circuit structure - cumulative fatigue"
    # Note: This would require checking for other exercise types in motions_df
    # For now, we'll skip this as we're only analyzing wall_ball exercises
    
    result = {
        'first_set_output': first_set_val,
        'final_set_output': final_set_val,
        'performance_decline': decline_pct,
        'reps_per_min': set_performance.tolist(),
        # Raw per-set values
        'set_indices': set_indices,
        'grouping_mode': grouping_mode,
        'reps_per_min_by_set_mean': reps_per_min_by_set_mean,
        'reps_per_min_by_set_raw': reps_per_min_by_set_raw,
        'score': score,
        'category': category,
        'interpretation': interpretation,
        'flags': flags
    }
    
    return result


def evaluate_control_stability(motions_df, use_waist_deviation=True, set_processing_info=None):
    """
    Evaluate Control Stability
    
    PRIMARY METRIC: Movement Consistency (CV%)
    - What it measures: Movement pattern consistency
    - Calculation:
      Control Stability = Coefficient of Variation (CV%) of waist movement patterns
      CV% = (Standard Deviation / Mean) × 100
    - Auto-Segmentation:
      * Continuous work → 10-rep chunks
      * Sets <10 reps → combine to reach minimum
      * Calculate CV% across valid movement data
    
    Note: Requires 10+ reps per segment for reliable calculation of movement variability metric.
    
    Parameters:
    -----------
    motions_df : pd.DataFrame
        Contains motion data
    use_waist_deviation : bool
        Whether to use waist_deviation as proxy metric (default True)
    set_processing_info : dict, optional
        Output from process_sets_by_rep_count() containing set categorization info
    
    Returns:
    --------
    dict: Contains evaluation results with flags for excluded sets
    """
    import pandas as pd
    import numpy as np
    
    # Process sets by rep count if not provided
    if set_processing_info is None:
        set_processing_info = process_sets_by_rep_count(motions_df)
    
    # Auto-Segmentation: Create valid segments with >=10 reps
    # Strategy:
    # 1. Use sets with 10+ reps directly
    # 2. For sets <10 reps: combine multiple sets to reach minimum 10 reps
    # 3. For continuous work (no set_index or single set): segment into 10-rep chunks
    
    # Get all exercise motions
    all_exercise_motions = set_processing_info.get('all_exercise_motions', None)
    if all_exercise_motions is None or len(all_exercise_motions) == 0:
        # Fallback: combine all available sets
        full_sets = set_processing_info['full_analysis_sets']
        partial_sets = set_processing_info['partial_analysis_sets']
        excluded_sets = set_processing_info['excluded_sets']
        
        if len(full_sets) > 0 or len(partial_sets) > 0 or len(excluded_sets) > 0:
            all_parts = []
            if len(full_sets) > 0:
                all_parts.append(full_sets)
            if len(partial_sets) > 0:
                all_parts.append(partial_sets)
            if len(excluded_sets) > 0:
                all_parts.append(excluded_sets)
            all_exercise_motions = pd.concat(all_parts).sort_values('start_time').copy() if all_parts else pd.DataFrame()
        else:
            all_exercise_motions = filter_exercise_motions(motions_df).sort_values('start_time').copy()
    else:
        all_exercise_motions = all_exercise_motions.sort_values('start_time').copy()
    
    if len(all_exercise_motions) == 0:
        return {
            'cv_percentage': np.nan,
            'mean_deviation': np.nan,
            'std_deviation': np.nan,
            'movement_metric': [],
            'movement_metric_by_set_raw': {},
            'score': None,
            'category': "N/A",
            'interpretation': "No exercise motions available for Control Stability analysis",
            'note': 'Control Stability requires movement data',
            'excluded_sets': [],
            'excluded_reason': 'No data',
            'can_calculate': False,
            'flags': []
        }
    
    # Auto-Segmentation: Create segments with >=10 reps
    valid_segments = []
    segment_metadata = []
    
    if 'set_index' in all_exercise_motions.columns:
        # Group by set_index
        set_groups = all_exercise_motions.groupby('set_index')
        current_segment = []
        current_segment_sets = []
        
        for set_idx, set_data in set_groups:
            set_data = set_data.sort_values('start_time')
            n_reps_in_set = len(set_data)
            
            if n_reps_in_set >= 10:
                # Set has 10+ reps: use as standalone segment
                if len(current_segment) > 0:
                    # Save previous accumulated segment if it has >=10 reps
                    if len(current_segment) >= 10:
                        valid_segments.append(pd.concat(current_segment))
                        segment_metadata.append({
                            'type': 'combined',
                            'sets': current_segment_sets.copy()
                        })
                    current_segment = []
                    current_segment_sets = []
                
                # Add this set as a standalone segment
                valid_segments.append(set_data)
                segment_metadata.append({
                    'type': 'standalone',
                    'sets': [set_idx]
                })
            else:
                # Set has <10 reps: accumulate with others
                current_segment.append(set_data)
                current_segment_sets.append(set_idx)
                
                # If accumulated segment reaches >=10 reps, save it
                if len(current_segment) >= 10:
                    total_reps = sum(len(s) for s in current_segment)
                    if total_reps >= 10:
                        valid_segments.append(pd.concat(current_segment))
                        segment_metadata.append({
                            'type': 'combined',
                            'sets': current_segment_sets.copy()
                        })
                        current_segment = []
                        current_segment_sets = []
        
        # Handle remaining accumulated segment
        if len(current_segment) >= 10:
            total_reps = sum(len(s) for s in current_segment)
            if total_reps >= 10:
                valid_segments.append(pd.concat(current_segment))
                segment_metadata.append({
                    'type': 'combined',
                    'sets': current_segment_sets
                })
    else:
        # No set_index: continuous work, segment into 10-rep chunks
        n_reps = len(all_exercise_motions)
        chunk_size = 10
        
        for i in range(0, n_reps, chunk_size):
            chunk = all_exercise_motions.iloc[i:i+chunk_size]
            if len(chunk) >= 10:
                valid_segments.append(chunk)
                segment_metadata.append({
                    'type': 'continuous_chunk',
                    'start_idx': i,
                    'end_idx': min(i+chunk_size, n_reps)
                })
    
    # Combine all valid segments for CV% calculation
    if len(valid_segments) == 0:
        excluded_set_indices = list(set_processing_info.get('set_analysis_flags', {}).keys())
        excluded_set_indices = [idx for idx in excluded_set_indices 
                               if set_processing_info.get('set_analysis_flags', {}).get(idx) in ['excluded', 'partial']]
        
        return {
            'cv_percentage': np.nan,
            'mean_deviation': np.nan,
            'std_deviation': np.nan,
            'movement_metric': [],
            'movement_metric_by_set_raw': {},
            'score': None,
            'category': "N/A",
            'interpretation': "No valid segments with 10+ reps available for Control Stability analysis",
            'note': 'Control Stability requires segments with 10+ reps (auto-segmentation applied)',
            'excluded_sets': excluded_set_indices,
            'excluded_reason': 'Insufficient reps after auto-segmentation',
            'can_calculate': False,
            'flags': []
        }
    
    # Combine all valid segments
    full_analysis_motions = pd.concat(valid_segments).sort_values('start_time').copy()
    
    # Prefer dedicated core muscle deviation metrics if available
    using_core_metrics = False
    cv_percentage = 0
    mean_value = 0
    std_value = 0
    movement_metric = []
    movement_metric_by_set_raw = {}
    data_source = 'waist_deviation'
    
    core_cols = {'core_muscle_deviation_mean', 'core_muscle_deviation_std'}
    if core_cols.issubset(full_analysis_motions.columns):
        # Keep set_index (if present) so we can export raw values per set
        core_cols_list = ['core_muscle_deviation_mean', 'core_muscle_deviation_std']
        cols = core_cols_list
        if 'set_index' in full_analysis_motions.columns:
            cols = ['set_index'] + core_cols_list
        core_df = full_analysis_motions[cols].copy()
        for c in core_cols_list:
            core_df[c] = pd.to_numeric(core_df[c], errors='coerce')
        if 'set_index' in core_df.columns:
            core_df['set_index'] = pd.to_numeric(core_df['set_index'], errors='coerce')
        # Remove non-positive mean values to avoid divide-by-zero
        core_df = core_df[core_df['core_muscle_deviation_mean'] > 0]
        
        if len(core_df) > 0:
            # Calculate CV% from movement metric values
            # CV% = (Standard Deviation / Mean) × 100
            movement_metric = core_df['core_muscle_deviation_mean'].tolist()
            movement_series = pd.Series(movement_metric)
            mean_value = movement_series.mean() * 100  # convert to cm for readability
            std_value = movement_series.std() * 100
            cv_percentage = 0 if mean_value == 0 else (std_value / mean_value) * 100
            using_core_metrics = True
            data_source = 'core_muscle_deviation'
            
            if 'set_index' in core_df.columns:
                raw_map = (
                    core_df.dropna(subset=['set_index'])
                          .groupby('set_index')['core_muscle_deviation_mean']
                          .apply(lambda s: s.dropna().tolist())
                          .to_dict()
                )
                movement_metric_by_set_raw = {int(k): v for k, v in raw_map.items()}
    
    if not using_core_metrics:
        # Fallback to waist_deviation proxy
        # Calculate CV% = (Standard Deviation / Mean) × 100
        movement_series = pd.to_numeric(full_analysis_motions['waist_deviation'], errors='coerce')
        mean_value = movement_series.mean()
        std_value = movement_series.std()
        movement_metric = movement_series.tolist()
        cv_percentage = 0 if mean_value == 0 else (std_value / mean_value) * 100
        
        if 'set_index' in full_analysis_motions.columns:
            tmp = full_analysis_motions[['set_index']].copy()
            tmp['set_index'] = pd.to_numeric(tmp['set_index'], errors='coerce')
            tmp['movement_metric'] = movement_series
            raw_map = (
                tmp.dropna(subset=['set_index'])
                   .groupby('set_index')['movement_metric']
                   .apply(lambda s: s.dropna().tolist())
                   .to_dict()
            )
            movement_metric_by_set_raw = {int(k): v for k, v in raw_map.items()}
    
    # If set_index is missing, fall back to sequential grouping (each record as one group)
    if not movement_metric_by_set_raw:
        movement_metric_by_set_raw = {
            int(i): ([float(v)] if v is not None and not pd.isna(v) else [])
            for i, v in enumerate(movement_metric)
        }
    
    # Classification using 1-5 scoring system
    # Score 5: <10% CV - Optimal stability
    # Score 4: 10-15% CV - Good stability
    # Score 3: 15-22% CV - Adequate stability
    # Score 2: 22-30% CV - Limited stability
    # Score 1: >30% CV - Very limited stability
    if cv_percentage < 10:
        score = 5
        category = "Optimal"
        interpretation = "Optimal stability"
    elif 10 <= cv_percentage < 15:
        score = 4
        category = "Good"
        interpretation = "Good stability"
    elif 15 <= cv_percentage < 22:
        score = 3
        category = "Adequate"
        interpretation = "Adequate stability"
    elif 22 <= cv_percentage <= 30:
        score = 2
        category = "Needs Improvement"
        interpretation = "Limited stability"
    else:  # >30%
        score = 1
        category = "Needs Improvement"
        interpretation = "Very limited stability"
    
    # Identify which sets were excluded
    excluded_set_indices = []
    for set_idx, flag in set_processing_info['set_analysis_flags'].items():
        if flag != 'full':
            excluded_set_indices.append(set_idx)
    
    # Check for flags
    flags = []
    
    # Flag: If <20 total reps → Flag "Small sample size"
    # Count total reps across all analyzed sets
    total_reps = len(full_analysis_motions)
    if total_reps < 20:
        flags.append("Small sample size")
    
    # Flag: If Wall Balls done after heavy work → Flag "Pre-fatigued"
    # Note: This would require checking for other exercises before wall_ball in motions_df
    # For now, we'll skip this as we're only analyzing wall_ball exercises
    
    result = {
        'cv_percentage': cv_percentage,
        'mean_deviation': mean_value,
        'std_deviation': std_value,
        'movement_metric': movement_metric,
        'movement_metric_by_set_raw': movement_metric_by_set_raw,
        'score': score,
        'category': category,
        'interpretation': interpretation,
        'note': 'Using core muscle deviation metrics for control stability' if using_core_metrics else 'Using waist_deviation as proxy for movement consistency (gyroscope data not available)',
        'data_source': data_source,
        'excluded_sets': excluded_set_indices,
        'excluded_reason': 'Sets with <10 reps excluded from Control Stability analysis',
        'can_calculate': True,
        'sets_analyzed': full_analysis_motions['set_index'].nunique() if 'set_index' in full_analysis_motions.columns else len(full_analysis_motions),
        'sets_excluded': len(excluded_set_indices),
        'flags': flags
    }
    
    return result


def evaluate_pacing_strategy(motions_df, set_processing_info=None):
    """
    Evaluate Pacing Strategy
    
    Determines whether initial intensity is appropriate and whether there is early overexertion 
    leading to premature fatigue.
    
    PRIMARY METRIC: First vs Final Portion Performance Decline (ADAPTIVE)
    - What it measures: Performance decline from early to late reps
    - Calculation (ADAPTIVE):
      * For >=20 reps: First Third = First 33% of total Wall Ball reps, Final Third = Final 33% of total Wall Ball reps
      * For <20 reps: Use first 25% vs final 25%
    - Formula: ((First Third Rate - Final Third Rate) / First Third Rate) × 100
    
    Note: Can be calculated for sets with 5+ reps (doesn't require 10+ like Control Stability)
    
    Parameters:
    -----------
    motions_df : pd.DataFrame
        Contains motion data
    set_processing_info : dict, optional
        Output from process_sets_by_rep_count() containing set categorization info
    
    Returns:
    --------
    dict: Contains evaluation results
    """
    import pandas as pd
    import numpy as np
    
    # For Pacing Strategy, we analyze exercise motions directly
    # Each motion record represents a rep (e.g., one wall ball throw)
    # Filter exercise motions only (exclude rest periods)
    # Calculation is based on total Wall Ball reps, not sets
    exercise_motions = filter_exercise_motions(motions_df)
    exercise_motions = exercise_motions.sort_values('start_time').copy()
    
    if len(exercise_motions) == 0:
        return {
            'first_third_avg': 0,
            'final_third_avg': 0,
            'decline_percentage': 0,
            'first_third_sets': 0,
            'final_third_sets': 0,
            'reps_per_min_all': [],
            'reps_per_min_first_third_raw': [],
            'reps_per_min_final_third_raw': [],
            'reps_per_min_by_set_mean': {},
            'reps_per_min_by_set_raw': {},
            'score': None,
            'category': "N/A",
            'interpretation': "No exercise motions available for analysis",
            'flags': []
        }
    
    # Calculate performance (reps/min) for each rep
    reps_per_min = 60.0 / exercise_motions['duration']
    reps_per_min_all = reps_per_min.tolist()
    
    # Optional: if set_index exists, also expose raw values per set_index (without changing the metric logic)
    reps_per_min_by_set_raw = {}
    reps_per_min_by_set_mean = {}
    if 'set_index' in exercise_motions.columns and exercise_motions['set_index'].nunique() >= 1:
        tmp = exercise_motions[['set_index']].copy()
        tmp['set_index'] = pd.to_numeric(tmp['set_index'], errors='coerce')
        tmp['reps_per_min'] = pd.to_numeric(reps_per_min, errors='coerce')
        grouped = tmp.dropna(subset=['set_index']).groupby('set_index')['reps_per_min']
        raw_map = grouped.apply(lambda s: s.dropna().tolist()).to_dict()
        mean_map = grouped.mean().to_dict()
        reps_per_min_by_set_raw = {int(k): v for k, v in raw_map.items()}
        reps_per_min_by_set_mean = {int(k): float(v) for k, v in mean_map.items()}
    
    # Total number of Wall Ball reps
    n_reps = len(exercise_motions)
    
    # Need at least 3 reps to calculate first vs final portion
    if n_reps < 3:
        return {
            'first_third_avg': reps_per_min.mean() if n_reps > 0 else 0,
            'final_third_avg': reps_per_min.mean() if n_reps > 0 else 0,
            'decline_percentage': 0,
            'first_third_sets': 0,
            'final_third_sets': 0,
            'reps_per_min_all': reps_per_min_all,
            'reps_per_min_first_third_raw': [],
            'reps_per_min_final_third_raw': [],
            'reps_per_min_by_set_mean': reps_per_min_by_set_mean,
            'reps_per_min_by_set_raw': reps_per_min_by_set_raw,
            'score': None,
            'category': "N/A",
            'interpretation': "Need at least 3 reps to calculate first vs final portion decline",
            'flags': []
        }
    
    # ADAPTIVE: For <20 reps, use first 25% vs final 25%; otherwise use first 33% vs final 33%
    if n_reps < 20:
        # Use first 25% vs final 25% for <20 reps
        first_portion_size = int(np.ceil(n_reps * 0.25))
        final_portion_size = int(np.ceil(n_reps * 0.25))
    else:
        # Use first 33% vs final 33% for >=20 reps
        first_portion_size = int(np.ceil(n_reps / 3))
        final_portion_size = int(np.ceil(n_reps / 3))
    
    # Calculate average output for first portion and final portion
    first_portion_slice = reps_per_min.iloc[:first_portion_size]
    final_portion_slice = reps_per_min.iloc[-final_portion_size:]
    first_third_avg = first_portion_slice.mean()
    final_third_avg = final_portion_slice.mean()
    
    # Calculate decline percentage: ((First Third Rate - Final Third Rate) / First Third Rate) × 100
    if first_third_avg > 0:
        decline_percentage = ((first_third_avg - final_third_avg) / first_third_avg) * 100
    else:
        decline_percentage = 0
    
    # Classification using 1-5 scoring system
    # Score 5: <10% decline - Optimal pacing strategy
    # Score 4: 10-18% decline - Good pacing strategy
    # Score 3: 18-25% decline - Adequate pacing strategy
    # Score 2: 25-35% decline - Poor pacing strategy
    # Score 1: >35% decline - Very poor pacing strategy
    if decline_percentage < 10:
        score = 5
        category = "Optimal"
        interpretation = "Optimal pacing strategy"
    elif 10 <= decline_percentage < 18:
        score = 4
        category = "Good"
        interpretation = "Good pacing strategy"
    elif 18 <= decline_percentage < 25:
        score = 3
        category = "Adequate"
        interpretation = "Adequate pacing strategy"
    elif 25 <= decline_percentage <= 35:
        score = 2
        category = "Needs Improvement"
        interpretation = "Poor pacing strategy"
    else:  # >35%
        score = 1
        category = "Needs Improvement"
        interpretation = "Very poor pacing strategy"
    
    # Check for flags
    flags = []
    
    # Flag: If total reps <20 → Flag "Limited volume"
    if n_reps < 20:
        flags.append("Limited volume")
    
    # Flag: If circuit structure → Flag "Circuit pacing"
    # Note: This would require checking for other exercises between wall_ball sets
    # For now, we'll skip this as we're only analyzing wall_ball exercises
    
    result = {
        'first_third_avg': first_third_avg,
        'final_third_avg': final_third_avg,
        'decline_percentage': decline_percentage,
        'first_third_sets': first_portion_size,  # Number of reps in first portion
        'final_third_sets': final_portion_size,  # Number of reps in final portion
        # Raw values used to compute the two group averages
        'reps_per_min_all': reps_per_min_all,
        'reps_per_min_first_third_raw': first_portion_slice.tolist(),
        'reps_per_min_final_third_raw': final_portion_slice.tolist(),
        # Raw values grouped by set_index (if available)
        'reps_per_min_by_set_mean': reps_per_min_by_set_mean,
        'reps_per_min_by_set_raw': reps_per_min_by_set_raw,
        'score': score,
        'category': category,
        'interpretation': interpretation,
        'flags': flags
    }
    
    return result


def load_history_sessions(history_file_path=None):
    """
    Load historical session data from CSV file
    
    Parameters:
    -----------
    history_file_path : str, optional
        Path to history sessions CSV file. If None, uses default path.
    
    Returns:
    --------
    pd.DataFrame: Historical session data with columns:
        - session_date: Date of session
        - cardiovascular_load_pct: High-intensity zone percentage
        - recovery_capacity_bpm: Average 60-second HR drop
        - output_sustainability_pct: Performance decline percentage
        - control_stability_cv: Coefficient of variation percentage
        - pacing_strategy_pct: First vs final third decline percentage
    """
    import pandas as pd
    import os
    
    if history_file_path is None:
        # Default path: data/history_sessions.csv relative to utils.py
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        history_file_path = os.path.join(utils_dir, 'data', 'history_sessions.csv')
    
    if not os.path.exists(history_file_path):
        return pd.DataFrame()
    
    history_df = pd.read_csv(history_file_path)
    history_df['session_date'] = pd.to_datetime(history_df['session_date'])
    history_df = history_df.sort_values('session_date')
    
    return history_df


def calculate_trend(current_value, baseline_value, metric_name, improvement_direction='higher'):
    """
    Calculate trend classification for a metric
    
    Parameters:
    -----------
    current_value : float
        Current session value
    baseline_value : float
        Rolling 4-week average baseline value
    metric_name : str
        Name of the metric (for error messages)
    improvement_direction : str
        'higher' if higher values are better, 'lower' if lower values are better
    
    Returns:
    --------
    str: Trend classification - "Maintaining", "Improving", or "Decreasing"
    """
    if baseline_value == 0:
        return "N/A"
    
    # Calculate percentage change
    if improvement_direction == 'higher':
        # For metrics where higher is better (e.g., Recovery Capacity HR drop)
        pct_change = ((current_value - baseline_value) / baseline_value) * 100
    else:
        # For metrics where lower is better (e.g., Output Sustainability decline, Control Stability CV)
        pct_change = ((baseline_value - current_value) / baseline_value) * 100
    
    # Classify trend
    if abs(pct_change) <= 5:
        return "Maintaining"
    elif pct_change > 5:
        return "Improving"
    else:  # pct_change < -5
        return "Decreasing"


def evaluate_trends(current_results, history_file_path=None, weeks=4, history_df=None):
    """
    Evaluate trends by comparing current session to rolling 4-week average

    Parameters:
    -----------
    current_results : dict
        Dictionary containing current session evaluation results:
        - cardiovascular_load: dict with 'high_intensity_percentage'
        - recovery_capacity: dict with 'avg_hr_drop'
        - output_sustainability: dict with 'performance_decline'
        - control_stability: dict with 'cv_percentage'
        - pacing_strategy: dict with 'decline_percentage'
    history_file_path : str, optional
        Path to history sessions CSV file
    weeks : int
        Number of weeks to use for rolling average (default: 4)
    history_df : pd.DataFrame, optional
        Pre-loaded history DataFrame. If provided, takes precedence over history_file_path.
        Expected columns: cardiovascular_load_pct, recovery_capacity_bpm,
        output_sustainability_pct, control_stability_cv, pacing_strategy_pct

    Returns:
    --------
    dict: Trend evaluation results for each dimension
    """
    import pandas as pd
    import numpy as np

    # Helper function to check if value is valid (not None and not nan)
    def is_valid_value(val):
        if val is None:
            return False
        if isinstance(val, float) and np.isnan(val):
            return False
        return True

    # Use provided history_df if available, otherwise load from file
    if history_df is None:
        history_df = load_history_sessions(history_file_path)
    
    if len(history_df) == 0:
        return {
            'status': 'insufficient_data',
            'message': 'No historical data available. Need at least 4 sessions to calculate trends.',
            'trends': {}
        }
    
    # Get last N weeks of data (assuming weekly sessions)
    recent_history = history_df.tail(weeks * 7)  # Approximate: 7 sessions per week
    
    if len(recent_history) < 4:
        return {
            'status': 'insufficient_data',
            'message': f'Insufficient historical data. Have {len(history_df)} sessions, need at least 4 sessions.',
            'trends': {}
        }
    
    # Calculate rolling averages
    baseline_metrics = {
        'cardiovascular_load_pct': recent_history['cardiovascular_load_pct'].mean(),
        'recovery_capacity_bpm': recent_history['recovery_capacity_bpm'].mean(),
        'output_sustainability_pct': recent_history['output_sustainability_pct'].mean(),
        'control_stability_cv': recent_history['control_stability_cv'].mean(),
        'pacing_strategy_pct': recent_history['pacing_strategy_pct'].mean()
    }
    
    trends = {}
    
    # 1. Cardiovascular Load
    # Check if we have the metric value (not just category)
    if 'cardiovascular_load' in current_results:
        current_cv = current_results['cardiovascular_load'].get('high_intensity_percentage', None)
        # Check if value is valid (not None and not nan)
        if is_valid_value(current_cv):
            baseline_cv = baseline_metrics['cardiovascular_load_pct']
            # For CV Load, optimal is 10-25%, so we want to maintain in that range
            # If current is in optimal range and baseline is also, it's maintaining
            # If current moves closer to optimal, it's improving
            if baseline_cv > 0:
                # Calculate how close to optimal range (10-25%)
                optimal_center = 17.5  # Middle of optimal range
                current_distance = abs(current_cv - optimal_center)
                baseline_distance = abs(baseline_cv - optimal_center)
                
                if abs(current_distance - baseline_distance) <= 2.5:  # Within 5% tolerance
                    trend = "Maintaining"
                elif current_distance < baseline_distance:
                    trend = "Improving"
                else:
                    trend = "Decreasing"
            else:
                trend = "N/A"
            
            trends['cardiovascular_load'] = {
                'current': current_cv,
                'baseline': baseline_cv,
                'trend': trend,
                'category': current_results['cardiovascular_load'].get('category', 'N/A')
            }
    
    # 2. Recovery Capacity (higher is better)
    # Use avg_hr_drop_rate (bpm/s) if available, otherwise fallback to avg_hr_drop (bpm)
    if 'recovery_capacity' in current_results:
        current_rc = current_results['recovery_capacity'].get('avg_hr_drop_rate', None)
        if current_rc is None or not is_valid_value(current_rc):
            # Fallback to avg_hr_drop for backward compatibility
            current_rc = current_results['recovery_capacity'].get('avg_hr_drop', None)
        
        # Check if value is valid (not None and not nan)
        if is_valid_value(current_rc):
            baseline_rc = baseline_metrics['recovery_capacity_bpm']
            # If using rate, convert baseline to rate (assuming 60s recovery window)
            # Otherwise compare directly
            if current_results['recovery_capacity'].get('avg_hr_drop_rate') is not None:
                # Convert baseline from bpm to bpm/s (assuming 60s recovery)
                baseline_rc_rate = baseline_rc / 60.0 if baseline_rc > 0 else 0
                trend = calculate_trend(current_rc, baseline_rc_rate, 'Recovery Capacity', improvement_direction='higher')
                trends['recovery_capacity'] = {
                    'current': current_rc,
                    'baseline': baseline_rc_rate,
                    'trend': trend,
                    'category': current_results['recovery_capacity'].get('category', 'N/A')
                }
            else:
                # Using absolute HR drop (backward compatibility)
                trend = calculate_trend(current_rc, baseline_rc, 'Recovery Capacity', improvement_direction='higher')
                trends['recovery_capacity'] = {
                    'current': current_rc,
                    'baseline': baseline_rc,
                    'trend': trend,
                    'category': current_results['recovery_capacity'].get('category', 'N/A')
                }
    
    # 3. Output Sustainability (lower decline is better)
    if 'output_sustainability' in current_results:
        current_os = current_results['output_sustainability'].get('performance_decline', None)
        # Check if value is valid (not None and not nan)
        if is_valid_value(current_os):
            baseline_os = baseline_metrics['output_sustainability_pct']
            trend = calculate_trend(current_os, baseline_os, 'Output Sustainability', improvement_direction='lower')
            
            trends['output_sustainability'] = {
                'current': current_os,
                'baseline': baseline_os,
                'trend': trend,
                'category': current_results['output_sustainability'].get('category', 'N/A')
            }
    
    # 4. Control Stability (lower CV is better)
    if 'control_stability' in current_results:
        current_cs = current_results['control_stability'].get('cv_percentage', None)
        baseline_cs = baseline_metrics['control_stability_cv']
        
        # Check if value is valid (not None and not nan)
        if is_valid_value(current_cs):
            trend = calculate_trend(current_cs, baseline_cs, 'Control Stability', improvement_direction='lower')
        else:
            trend = "N/A"
        
        trends['control_stability'] = {
            'current': current_cs if is_valid_value(current_cs) else None,
            'baseline': baseline_cs,
            'trend': trend,
            'category': current_results['control_stability'].get('category', 'N/A')
        }
    
    # 5. Pacing Strategy (optimal is 10-20%, so we want to maintain in that range)
    if 'pacing_strategy' in current_results:
        current_ps = current_results['pacing_strategy'].get('decline_percentage', None)
        # Check if value is valid (not None and not nan)
        if is_valid_value(current_ps):
            baseline_ps = baseline_metrics['pacing_strategy_pct']
            # Similar to CV Load, optimal is 10-20%
            if baseline_ps > 0:
                optimal_center = 15.0  # Middle of optimal range
                current_distance = abs(current_ps - optimal_center)
                baseline_distance = abs(baseline_ps - optimal_center)
                
                if abs(current_distance - baseline_distance) <= 0.75:  # Within 5% tolerance
                    trend = "Maintaining"
                elif current_distance < baseline_distance:
                    trend = "Improving"
                else:
                    trend = "Decreasing"
            else:
                trend = "N/A"
            
            trends['pacing_strategy'] = {
                'current': current_ps,
                'baseline': baseline_ps,
                'trend': trend,
                'category': current_results['pacing_strategy'].get('category', 'N/A')
            }
    
    return {
        'status': 'success',
        'baseline_period_weeks': weeks,
        'baseline_sessions_count': len(recent_history),
        'trends': trends
    }


# ════════════════════════════════════════════════════════════════════
# Time-based evaluation functions (Farmer's Carry, Sled Push, Sled Pull)
# ════════════════════════════════════════════════════════════════════

def evaluate_output_sustainability_time_based(cadences, set_boundaries=None):
    """
    Evaluate Output Sustainability for time-based exercises using step frequency.

    Compares first 1/3 vs last 1/3 of per-second step_frequency values.
    Decline = (first_avg - last_avg) / first_avg * 100

    Scoring (same as rep-based):
      5: <5%   Optimal
      4: 5-12%  Good
      3: 12-20% Adequate
      2: 20-30% Needs Improvement
      1: >30%  Very limited

    Parameters
    ----------
    cadences : list[float]
        Per-second step frequency (Hz). Invalid values (<=0) are filtered out.
    set_boundaries : list[tuple], optional
        List of (start_ms, stop_ms) per set. Currently unused but kept for
        future per-set analysis.

    Returns
    -------
    dict : Same structure as evaluate_output_sustainability().
    """
    import numpy as np

    valid = [c for c in cadences if c is not None and c > 0]

    if len(valid) < 6:
        return {
            'first_set_output': 0,
            'final_set_output': 0,
            'performance_decline': 0,
            'score': None,
            'category': "N/A",
            'interpretation': "Insufficient step frequency data for analysis",
            'flags': [],
            'valid_samples': len(valid),
        }

    n = len(valid)
    third = max(n // 3, 1)
    first_avg = float(np.mean(valid[:third]))
    last_avg = float(np.mean(valid[-third:]))

    if first_avg > 0:
        decline_pct = ((first_avg - last_avg) / first_avg) * 100
    else:
        decline_pct = 0.0

    if decline_pct < 5:
        score, category, interpretation = 5, "Optimal", "Optimal sustainability"
    elif decline_pct < 12:
        score, category, interpretation = 4, "Good", "Good sustainability"
    elif decline_pct < 20:
        score, category, interpretation = 3, "Adequate", "Adequate sustainability"
    elif decline_pct <= 30:
        score, category, interpretation = 2, "Needs Improvement", "Limited sustainability"
    else:
        score, category, interpretation = 1, "Needs Improvement", "Very limited sustainability"
        interpretation += ". Likely glycogen depletion or inadequate aerobic base."

    flags = []
    if n < 30:
        flags.append("Limited volume")

    return {
        'first_set_output': first_avg,
        'final_set_output': last_avg,
        'performance_decline': decline_pct,
        'score': score,
        'category': category,
        'interpretation': interpretation,
        'flags': flags,
        'valid_samples': n,
    }


