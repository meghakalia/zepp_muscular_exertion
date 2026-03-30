"""
HYROX Exercise Exertion Calculations (V1)

Unified framework for all 8 HYROX exercises:
    muscular_exertion = movement_coef × (T / 60) × f_load × f_density × f_stability

Two exercise families:
  - Rep-based: Wall Ball, SkiErg, Rowing, Sandbag Lunges, Burpee Broad Jumps
  - Time-based: Farmer's Carry, Sled Push, Sled Pull

Two input paths (both extract scalars, then call the same compute_exertion):
  - DataFrame path: calculate_exertion_df() — accepts pandas DataFrames (app/demo usage)
  - Firmware path:  calculate_exertion_firmware() — accepts flat scalars/arrays (C struct style)

No sigmoid output transform — values are raw exertion units.
"""

import math

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Constants & Configuration
# ─────────────────────────────────────────────

ALPHA = 0.7
G_MIN = 0.8          # micro rest threshold (seconds)

# Density factor parameters
D_BASELINE = 1.0
D_AMPLITUDE = 0.3
D_REST_SENSITIVITY = 1.0

# Stability factor parameters
S_BASELINE = 1.0
S_AMPLITUDE = 0.2
S_SENSITIVITY = 0.08

# Firmware station_type enum → exercise_type string
STATION_TYPE_MAP = {
    'HYROX_SPORT_TYPE_MEDICINE': 'wall_ball',       # 9
    'HYROX_SPORT_TYPE_SKING': 'skierg',             # 2
    'HYROX_SPORT_TYPE_ROW': 'rowing',               # 6
    'HYROX_SPORT_TYPE_LUNGE': 'sandbag_lunges',     # 8
    'HYROX_SPORT_TYPE_JUMP': 'burpee_broad_jumps',  # 5
    'HYROX_SPORT_TYPE_FAMER': 'farmers_carry',      # 7
    'HYROX_SPORT_TYPE_SLEDGE': 'sled_push',         # 3
    'HYROX_SPORT_TYPE_SLED': 'sled_pull',           # 4
}
# Also support integer enum values
STATION_TYPE_INT_MAP = {
    9: 'wall_ball',
    2: 'skierg',
    6: 'rowing',
    8: 'sandbag_lunges',
    5: 'burpee_broad_jumps',
    7: 'farmers_carry',
    3: 'sled_push',
    4: 'sled_pull',
}

EXERCISE_CONFIG = {
    'wall_ball': {
        'family': 'rep',
        't_ref': 1.55,
        'movement_coef': 1.5,
        'has_core_deviation': True,
    },
    'skierg': {
        'family': 'rep',
        't_ref': 1.5,
        'movement_coef': 1.0,
        'has_core_deviation': True,
    },
    'rowing': {
        'family': 'rep',
        't_ref': 2.2,
        'movement_coef': 1.0,
        'has_core_deviation': False,
    },
    'sandbag_lunges': {
        'family': 'rep',
        't_ref': 1.5,
        'movement_coef': 2.1,
        'has_core_deviation': True,
    },
    'burpee_broad_jumps': {
        'family': 'rep',
        't_ref': 4.0,
        'movement_coef': 1.95,
        'has_core_deviation': False,
    },
    'farmers_carry': {
        'family': 'time',
        'f_ref': 2.0,
        'movement_coef': 1.8,
        'has_core_deviation': False,
    },
    'sled_push': {
        'family': 'time',
        'f_ref': 1.6,
        'movement_coef': 1.95,
        'has_core_deviation': False,
    },
    'sled_pull': {
        'family': 'time',
        'f_ref': 1.6,
        'movement_coef': 1.95,
        'has_core_deviation': False,
    },
}


# ─────────────────────────────────────────────
# Factor Calculations
# ─────────────────────────────────────────────

def _calculate_density_factor(active_time, total_rest):
    """
    V1 density factor.
    f_density = D_BASELINE + D_AMPLITUDE * active_time / (active_time + D_REST_SENSITIVITY * total_rest)
    """
    denominator = active_time + D_REST_SENSITIVITY * total_rest
    if denominator <= 0:
        return D_BASELINE
    return D_BASELINE + D_AMPLITUDE * (active_time / denominator)


def _calculate_stability_factor(stability_cv):
    """
    V1 stability factor (sigmoid curve).
    f_stability = S_BASELINE + S_AMPLITUDE * (1 - exp(-S_SENSITIVITY * stability_cv))
    """
    if stability_cv is None or stability_cv <= 0:
        return S_BASELINE
    return S_BASELINE + S_AMPLITUDE * (1.0 - np.exp(-S_SENSITIVITY * stability_cv))


def _compute_rep_based_T(rep_count, active_time, t_ref, alpha=ALPHA):
    """
    Effective work quantity for rep-based exercises.
    work_unit = α·rep_count + (1-α)·(active_time/t_ref)
    T = t_ref · work_unit
    Returns (work_unit, T)
    """
    work_unit = alpha * rep_count + (1.0 - alpha) * (active_time / t_ref)
    T = t_ref * work_unit
    return work_unit, T


def _compute_time_based_T(active_time, avg_step_frequency, f_ref,
                           cadence_min=0.7, cadence_max=1.3):
    """
    Effective work quantity for time-based exercises.
    cadence_factor = clamp(avg_step_frequency / f_ref, cadence_min, cadence_max)
    T = active_time · cadence_factor
    Returns (cadence_factor, T)
    """
    if f_ref is None or f_ref <= 0 or avg_step_frequency is None:
        return 1.0, active_time
    cadence_factor = min(max(avg_step_frequency / f_ref, cadence_min), cadence_max)
    T = active_time * cadence_factor
    return cadence_factor, T


# ─────────────────────────────────────────────
# Stability CV Helpers
# ─────────────────────────────────────────────

def _compute_stability_cv_from_dataframe(exercise_motions, config):
    """
    Compute stability CV% from per-rep core_muscle_deviation_mean in a DataFrame.
    Returns float CV% or None if not applicable/available.
    """
    if not config.get('has_core_deviation', False):
        return None

    col = 'core_muscle_deviation_mean'
    if col not in exercise_motions.columns:
        return None

    values = pd.to_numeric(exercise_motions[col], errors='coerce').dropna()
    if len(values) < 2 or values.mean() <= 0:
        return None

    return (values.std() / values.mean()) * 100.0


def _compute_stability_cv_from_array(core_stability_values, has_core_deviation):
    """
    Compute stability CV% from a flat array of per-rep core_stability floats.
    Values of -1 are treated as invalid and filtered out.
    Returns float CV% or None.
    """
    if not has_core_deviation:
        return None
    valid = [v for v in core_stability_values if v is not None and v >= 0]
    if len(valid) < 2:
        return None
    arr = np.array(valid, dtype=float)
    mean_val = arr.mean()
    if mean_val <= 0:
        return None
    return float((arr.std() / mean_val) * 100.0)


# ─────────────────────────────────────────────
# Cardiac Exertion (Banister TRIMP)
# ─────────────────────────────────────────────

TRIMP_PARAMS = {
    'male':   {'a': 0.64, 'b': 1.92},
    'female': {'a': 0.86, 'b': 1.67},
}


def compute_cardiac_exertion(hrs, hr_max, hr_rest, gender='male'):
    """
    Compute cardiac exertion using Banister's exponential TRIMP.

    Banister TRIMP: each second contributes (1/60) * hr_r * a * exp(b * hr_r).
    - HR validity: 30 < hr < 200
    - Median filter (window=3) for noise smoothing

    Parameters
    ----------
    hrs : list or array
        Per-second heart rate values.
    hr_max : float
        Maximum heart rate
    hr_rest : float
        Resting heart rate
    gender : str
        'male' or 'female' (default 'male')

    Returns
    -------
    dict with cardiac_exertion (float) and valid_samples (int)
    """
    params = TRIMP_PARAMS.get(gender.lower(), TRIMP_PARAMS['male'])
    a, b = params['a'], params['b']

    hr_range = hr_max - hr_rest
    if hr_range <= 0:
        return {'cardiac_exertion': 0.0, 'valid_samples': 0}

    # Running median filter (window=3), matching Strain implementation
    filtered = []
    for i in range(len(hrs)):
        start = max(0, i - 2)
        window = sorted(hrs[start:i + 1])
        filtered.append(window[len(window) // 2])

    total_trimp = 0.0
    valid = 0

    for hr in filtered:
        if hr is None:
            continue
        if not (30 < hr < 200):
            continue
        hr_r = max((hr - hr_rest) / hr_range, 0.0)
        total_trimp += (1.0 / 60.0) * hr_r * a * math.exp(b * hr_r)
        valid += 1

    return {
        'cardiac_exertion': round(total_trimp, 4),
        'valid_samples': valid,
    }


def compute_combined_exertion(muscular_exertion, cardiac_exertion):
    """
    Combine muscular and cardiac exertion.

    Returns dict with combined_exertion, muscular_exertion, cardiac_exertion.
    """
    m = muscular_exertion if muscular_exertion else 0.0
    c = cardiac_exertion if cardiac_exertion else 0.0
    return {
        'combined_exertion': round(m + c, 4),
        'muscular_exertion': round(m, 4),
        'cardiac_exertion': round(c, 4),
    }


# ─────────────────────────────────────────────
# Rest Calculation Helpers
# ─────────────────────────────────────────────

def _calculate_macro_rest(set_boundaries):
    """
    Compute macro rest from set boundary timestamps (ms).
    Returns total macro rest in seconds.
    """
    R_macro = 0.0
    for i in range(len(set_boundaries) - 1):
        gap = set_boundaries[i + 1][0] - set_boundaries[i][1]
        if gap > 0:
            R_macro += gap / 1000.0
    return R_macro


def _calculate_micro_rest(rep_timestamps, g_min=G_MIN):
    """
    Compute micro rest between consecutive reps within a set.
    Returns total micro rest in seconds.
    """
    R_micro = 0.0
    for i in range(len(rep_timestamps) - 1):
        gap_ms = rep_timestamps[i + 1][0] - rep_timestamps[i][1]
        gap_s = max(0.0, gap_ms / 1000.0)
        if gap_s >= g_min:
            R_micro += gap_s
    return R_micro


# ─────────────────────────────────────────────
# Core Compute (single function for both families)
# ─────────────────────────────────────────────

def compute_exertion(exercise_type, rep_count, active_time_s, total_rest_s,
                     stability_cv=None, avg_step_frequency=None,
                     body_weight_kg=0.0, equipment_weight_kg=0.0):
    """
    Unified exertion calculation for any HYROX exercise.

    Both the DataFrame path and the firmware path extract scalars from their
    respective inputs, then call this function.

    Parameters
    ----------
    exercise_type : str
    rep_count : int or None     Total reps (rep-based only)
    active_time_s : float       Sum of rep durations (rep) or moving time (time-based)
    total_rest_s : float        Total rest in seconds
    stability_cv : float        CV% of core deviation, or None
    avg_step_frequency : float  Mean step frequency Hz (time-based only), or None
    body_weight_kg : float
    equipment_weight_kg : float

    Returns
    -------
    dict with exertion results
    """
    config = EXERCISE_CONFIG[exercise_type]
    family = config['family']
    movement_coef = config['movement_coef']

    # ── Compute T ──
    if family == 'rep':
        t_ref = config['t_ref']
        work_unit, T = _compute_rep_based_T(rep_count, active_time_s, t_ref)
        cadence_factor = 1.0
    else:
        t_ref = None
        f_ref = config.get('f_ref')
        cadence_factor, T = _compute_time_based_T(active_time_s, avg_step_frequency, f_ref)
        work_unit = None

    # ── Compute factors ──
    f_load = 1.0
    f_density = _calculate_density_factor(active_time_s, total_rest_s)
    f_stability = _calculate_stability_factor(stability_cv) if family == 'rep' else 1.0

    # ── Exertion ──
    # T is converted from seconds to minutes to align with cardiac exertion (TRIMP) time unit
    exertion = movement_coef * (T / 60.0) * f_load * f_density * f_stability

    # ── Build result ──
    result = {
        # V1 fields
        'muscular_exertion': round(exertion, 4),
        'movement_coef': movement_coef,
        'f_load': round(f_load, 4),
        'f_density': round(f_density, 4),
        'f_stability': round(f_stability, 4),
        'stability_cv': round(stability_cv, 2) if stability_cv is not None else None,
        'cadence_factor': round(cadence_factor, 4),
        # Backward-compatible aliases
        'exertion': round(exertion, 4),
        'exertion_raw': round(exertion, 4),
        'density_factor': round(f_density, 4),
        'quality_factor': round(f_stability, 4),
        # Shared fields
        'exercise_type': exercise_type,
        'total_reps': rep_count,
        'total_work_time': round(active_time_s, 2),
        'total_rest_time': round(total_rest_s, 2),
        'effective_work_time': round(T, 2),
        'mixed_work_quantity': round(work_unit, 2) if work_unit is not None else None,
        'parameters': {
            'exercise_type': exercise_type,
            'movement_coef': movement_coef,
            'body_weight_kg': body_weight_kg,
            'equipment_weight_kg': equipment_weight_kg,
            'version': 'v1',
            **(({'t_ref': t_ref, 'alpha': ALPHA} if family == 'rep'
                else {'f_ref': config.get('f_ref')})),
        },
    }
    return result


def _empty_result(exercise_type, body_weight_kg, equipment_weight_kg):
    """Return zeroed result for an exercise with no data."""
    config = EXERCISE_CONFIG.get(exercise_type, {})
    family = config.get('family', 'rep')
    result = compute_exertion(
        exercise_type,
        rep_count=0 if family == 'rep' else None,
        active_time_s=0.0,
        total_rest_s=0.0,
        body_weight_kg=body_weight_kg,
        equipment_weight_kg=equipment_weight_kg,
    )
    result.update(
        macro_rest=0.0, micro_rest=0.0, sets=0, set_details=[],
        error='No exercise motions found' if family == 'rep' else 'No time series data found',
    )
    return result


# ─────────────────────────────────────────────
# DataFrame Extractors
# ─────────────────────────────────────────────

def _extract_rep_based(exercise_type, motions_df, body_weight_kg,
                       equipment_weight_kg=0.0, extra_data=None, **_kwargs):
    """
    Extract scalars from a rep-based DataFrame, compute exertion,
    and add DataFrame-specific fields (set_details, macro/micro rest).
    """
    from .utils import is_exercise_motion, filter_exercise_motions

    config = EXERCISE_CONFIG[exercise_type]

    # Filter to exercise motions only
    exercise_motions = filter_exercise_motions(motions_df)
    exercise_motions = exercise_motions.sort_values('start_time').copy()

    if len(exercise_motions) == 0:
        return _empty_result(exercise_type, body_weight_kg, equipment_weight_kg)

    # ── Determine sets ──
    if 'set_index' in exercise_motions.columns:
        set_groups = exercise_motions.groupby('set_index')
        set_indices = sorted(set_groups.groups.keys())
    else:
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
        exercise_motions = sorted_motions[
            sorted_motions['motion'].apply(is_exercise_motion)
        ].copy()
        exercise_motions['set_index'] = exercise_motions['set_index_inferred']
        exercise_motions = exercise_motions.drop(columns=['set_index_inferred'])
        set_groups = exercise_motions.groupby('set_index')
        set_indices = sorted(set_groups.groups.keys())

    S = len(set_indices)

    # ── Extract scalars ──
    N = len(exercise_motions)
    W = exercise_motions['duration'].sum()

    R_macro = 0.0
    R_micro = 0.0
    set_details = []

    for idx, set_idx in enumerate(set_indices):
        set_motions = set_groups.get_group(set_idx).sort_values('start_time')
        r_s = len(set_motions)
        rep_durations = set_motions['duration'].tolist()

        rep_ts = list(zip(
            set_motions['start_time'].tolist(),
            set_motions['stop_time'].tolist()
        ))

        set_micro_rest = _calculate_micro_rest(rep_ts)
        R_micro += set_micro_rest

        rest_s = 0.0
        if idx < len(set_indices) - 1:
            set_end_time = set_motions['stop_time'].max()
            next_set_idx = set_indices[idx + 1]
            next_set_motions = set_groups.get_group(next_set_idx).sort_values('start_time')
            next_set_start_time = next_set_motions['start_time'].min()

            rest_periods = motions_df[
                (motions_df['motion'].apply(lambda x: not is_exercise_motion(x))) &
                (motions_df['start_time'] >= set_end_time) &
                (motions_df['stop_time'] <= next_set_start_time)
            ]
            if len(rest_periods) > 0:
                rest_s = rest_periods['duration'].sum()
            else:
                rest_s = max(0.0, (next_set_start_time - set_end_time) / 1000.0)

        R_macro += rest_s

        set_details.append({
            'set_index': int(set_idx),
            'reps': r_s,
            'rep_durations': rep_durations,
            'set_work_time': sum(rep_durations),
            'rest_time': rest_s,
            'micro_rest': round(set_micro_rest, 2),
            'avg_rep_duration': np.mean(rep_durations) if rep_durations else 0.0,
        })

    R = R_macro + R_micro
    stability_cv = _compute_stability_cv_from_dataframe(exercise_motions, config)

    # ── Compute ──
    result = compute_exertion(
        exercise_type, N, W, R, stability_cv,
        body_weight_kg=body_weight_kg, equipment_weight_kg=equipment_weight_kg,
    )

    # ── DataFrame-specific fields ──
    result['macro_rest'] = round(R_macro, 2)
    result['micro_rest'] = round(R_micro, 2)
    result['sets'] = S
    result['set_details'] = set_details
    return result


def _extract_time_based(exercise_type, time_series_df, body_weight_kg,
                        equipment_weight_kg=0.0, set_boundaries=None):
    """
    Extract scalars from a time-based DataFrame, compute exertion,
    and add DataFrame-specific fields.
    """
    config = EXERCISE_CONFIG[exercise_type]

    if len(time_series_df) == 0:
        return _empty_result(exercise_type, body_weight_kg, equipment_weight_kg)

    # ── Extract scalars ──
    moving_mask = time_series_df['status'] != 'pause'
    active_time = float(moving_mask.sum())

    R = 0.0
    S = 1
    if set_boundaries and len(set_boundaries) > 1:
        R = _calculate_macro_rest(set_boundaries)
        S = len(set_boundaries)

    avg_step_frequency = None
    f_ref = config.get('f_ref')
    if f_ref is not None and 'step_frequency' in time_series_df.columns:
        moving_data = time_series_df[moving_mask]
        if len(moving_data) > 0:
            valid_freq = moving_data['step_frequency'][moving_data['step_frequency'] > 0]
            if len(valid_freq) > 0:
                avg_step_frequency = valid_freq.mean()

    # ── Compute ──
    result = compute_exertion(
        exercise_type, None, active_time, R,
        avg_step_frequency=avg_step_frequency,
        body_weight_kg=body_weight_kg, equipment_weight_kg=equipment_weight_kg,
    )

    # ── DataFrame-specific fields ──
    result['macro_rest'] = round(R, 2)
    result['micro_rest'] = 0.0
    result['sets'] = S
    result['set_details'] = []
    return result


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def calculate_exertion_df(exercise_type, **kwargs):
    """
    Calculate exertion from DataFrame inputs (app/demo usage).

    Parameters
    ----------
    exercise_type : str
        One of: wall_ball, skierg, rowing, sandbag_lunges, burpee_broad_jumps,
                farmers_carry, sled_push, sled_pull

    For rep-based exercises:
        motions_df : pd.DataFrame (required)
        body_weight_kg : float (required)
        equipment_weight_kg : float (default 0.0)
        extra_data : dict (optional)

    For time-based exercises:
        time_series_df : pd.DataFrame (required)
        body_weight_kg : float (required)
        equipment_weight_kg : float (default 0.0)
        set_boundaries : list of (start_ms, end_ms) (optional)

    Returns
    -------
    dict with exertion results including muscular_exertion, f_density, f_stability, etc.
    """
    if exercise_type not in EXERCISE_CONFIG:
        raise ValueError(
            f"Unknown exercise type '{exercise_type}'. "
            f"Valid types: {list(EXERCISE_CONFIG.keys())}"
        )

    family = EXERCISE_CONFIG[exercise_type]['family']

    if family == 'rep':
        return _extract_rep_based(exercise_type, **kwargs)
    else:
        return _extract_time_based(exercise_type, **kwargs)


# Backward-compatible alias
calculate_exertion = calculate_exertion_df


def _resolve_station_type(station_type):
    """Resolve station_type (str enum name, int, or exercise_type string) to exercise_type."""
    if isinstance(station_type, int):
        if station_type in STATION_TYPE_INT_MAP:
            return STATION_TYPE_INT_MAP[station_type]
        raise ValueError(f"Unknown station_type int: {station_type}")
    if station_type in EXERCISE_CONFIG:
        return station_type
    if station_type in STATION_TYPE_MAP:
        return STATION_TYPE_MAP[station_type]
    raise ValueError(
        f"Unknown station_type '{station_type}'. "
        f"Valid: {list(EXERCISE_CONFIG.keys())} or {list(STATION_TYPE_MAP.keys())} or int enum"
    )


def calculate_exertion_firmware(station):
    """
    Calculate exertion from firmware-style station input (C struct).

    Extracts scalars from the station dict, then calls compute_exertion().

    Parameters
    ----------
    station : dict with keys:
        station_type : str or int       Exercise type (enum name, int, or exercise_type string)
        body_weight_kg : float          User body weight
        equipment_weight_kg : float     Equipment weight (-1 means none)
        total_duration_ms : int         Total station duration (ms)
        total_rest_duration_ms : int    Total rest duration (ms)

        Rep-based exercises additionally:
            rep_count : int
            reps : list of dict with 'start_ms', 'end_ms', 'core_stability'

        Time-based exercises additionally:
            cadences : list of float    Per-second step frequency (Hz), -1 = invalid

    Returns
    -------
    dict with exertion results (same format as calculate_exertion_df)
    """
    exercise_type = _resolve_station_type(station['station_type'])
    config = EXERCISE_CONFIG[exercise_type]
    family = config['family']

    body_weight_kg = station.get('body_weight_kg', 0.0)
    equipment_weight_kg = station.get('equipment_weight_kg', 0.0)
    if equipment_weight_kg < 0:
        equipment_weight_kg = 0.0

    total_duration_ms = station.get('total_duration_ms', 0)
    total_rest_duration_ms = station.get('total_rest_duration_ms', 0)
    total_rest_s = total_rest_duration_ms / 1000.0

    if family == 'rep':
        reps = station.get('reps', [])
        rep_count = station.get('rep_count', len(reps))

        active_time_s = sum(
            max(0, r['end_ms'] - r['start_ms']) / 1000.0 for r in reps
        )

        core_values = [r.get('core_stability', -1) for r in reps]
        stability_cv = _compute_stability_cv_from_array(
            core_values, config.get('has_core_deviation', False)
        )

        result = compute_exertion(
            exercise_type, rep_count, active_time_s, total_rest_s,
            stability_cv=stability_cv,
            body_weight_kg=body_weight_kg, equipment_weight_kg=equipment_weight_kg,
        )

    else:  # time-based
        cadences = station.get('cadences', [])
        active_time_s = (total_duration_ms - total_rest_duration_ms) / 1000.0

        valid_cadences = [c for c in cadences if c > 0]
        avg_step_frequency = float(np.mean(valid_cadences)) if valid_cadences else None

        result = compute_exertion(
            exercise_type, None, active_time_s, total_rest_s,
            avg_step_frequency=avg_step_frequency,
            body_weight_kg=body_weight_kg, equipment_weight_kg=equipment_weight_kg,
        )

    # ── Cardiac exertion (if HR data provided) ──
    hrs = station.get('hrs', [])
    hr_max = station.get('hr_max')
    hr_rest = station.get('hr_rest')
    gender = station.get('gender', 'male')

    if hrs and hr_max and hr_rest:
        cardiac = compute_cardiac_exertion(hrs, hr_max, hr_rest, gender)
        result['cardiac_exertion'] = cardiac['cardiac_exertion']
        combined = compute_combined_exertion(
            result['muscular_exertion'], cardiac['cardiac_exertion']
        )
        result['combined_exertion'] = combined['combined_exertion']
    else:
        result['cardiac_exertion'] = -1
        result['combined_exertion'] = -1

    return result
