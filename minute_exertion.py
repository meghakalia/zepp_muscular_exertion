"""
Minute-Level Exertion (V0)

Computes per-minute incremental and cumulative muscular, cardiac, and unified
exertion from 1-second resolution sample data.

V0 muscular formula (simplified — no load/density/stability factors):
  Rep-based:   ΔME_t = m * (α * t_ref * Δrep_t + (1 - α))
  Time-based:  ΔME_t = m * clamp(f_t / f_ref, 0.7, 1.3)

Cardiac exertion: Banister TRIMP, per-minute via existing compute_cardiac_exertion.
Unified:  ΔUE_t = ΔCE_t + ΔME_t

Usage:
    python minute_exertion.py                        # all sample sessions
    python minute_exertion.py --exercise skierg      # single exercise
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from wall_ball.exertion import EXERCISE_CONFIG, ALPHA, compute_cardiac_exertion
from wall_ball.run_exertion import run_sample_session, run_session

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

CADENCE_MIN = 0.7
CADENCE_MAX = 1.3

# Real sessions: (folder_name, exercise_type, body_weight_kg, equipment_weight_kg, age, rhr, gender)
SESSIONS = [
    ('20251225_SSCDDQTQHOPK_HT_6kg',      'wall_ball', 78.0, 6.0, 32, 58, 'male'),
    ('20251230_FZEIJCLQMGTN_HT_4kg',       'wall_ball', 78.0, 4.0, 32, 58, 'male'),
    ('20251230_WTTADQTQLRFVOBWA_HT_6kg',   'wall_ball', 78.0, 6.0, 32, 58, 'male'),
]

# Sample sessions: (exercise_type, folder, eq_kg, family, bw_kg, age, rhr, gender)
SAMPLE_SESSIONS = [
    ('skierg',             'sample_skierg',             0.0,   'rep',  78.0, 32, 58, 'male'),
    ('rowing',             'sample_rowing',             0.0,   'rep',  78.0, 32, 58, 'male'),
    ('sandbag_lunges',     'sample_sandbag_lunges',     20.0,  'rep',  78.0, 32, 58, 'male'),
    ('burpee_broad_jumps', 'sample_burpee_broad_jumps', 0.0,   'rep',  78.0, 32, 58, 'male'),
    ('farmers_carry',      'sample_farmers_carry',      24.0,  'time', 78.0, 32, 58, 'male'),
    ('sled_push',          'sample_sled_push',          100.0, 'time', 78.0, 32, 58, 'male'),
    ('sled_pull',          'sample_sled_pull',          100.0, 'time', 78.0, 32, 58, 'male'),
]


# Per-minute max ΔME used for normalisation (one value per exercise)
# NORMALIZATION_MAX_DELTA_ME = {
#     'wall_ball':          12.45,
#     'sandbag_lunges':     17.43,
#     'burpee_broad_jumps': 16.18,
#     'skierg':              8.30,
#     'rowing':              8.30,
#     'farmers_carry':       5.11,
#     'sled_push':           5.54,
#     'sled_pull':           5.54,
# }

# new values by mingshu
NORMALIZATION_MAX_DELTA_ME = {
    'wall_ball': 5.57,
    'sandbag_lunges': 7.80,
    'burpee_broad_jumps': 7.24,
    'skierg': 3.71,
    'rowing': 3.71,
    'farmers_carry': 5.11,
    'sled_push': 5.54,
    'sled_pull': 5.54,
    }
# ─────────────────────────────────────────────
# Core V0 Computations
# ─────────────────────────────────────────────

def compute_minute_muscular(movement_type, movement_mode, delta_rep=None,
                             step_frequency=None):
    """
    V0 muscular exertion increment for one minute.

    Rep-based:   ΔME = m * (α * t_ref * Δrep + (1 - α))
    Time-based:  ΔME = m * clamp(f / f_ref, CADENCE_MIN, CADENCE_MAX)

    Returns float ΔME_t.
    """
    cfg = EXERCISE_CONFIG[movement_type]
    m = cfg['movement_coef']

    if movement_mode == 'rep_based':
        t_ref = cfg['t_ref']
        reps = delta_rep if delta_rep is not None else 0

        if reps > 0:
            delta_T = (ALPHA * t_ref * reps + (1.0 - ALPHA)*60.0)/60.0
        else:
            delta_T = 0.0
    else:
        f_ref = cfg['f_ref']
        f = step_frequency if step_frequency is not None else 0.0
        if f_ref > 0 and f > 0:
            delta_T = min(max(f / f_ref, CADENCE_MIN), CADENCE_MAX)
        else:
            delta_T = 0.0 # should this be 1.0 or 0.0 instead?

    return m * delta_T


def compute_minute_cardiac(hr_list, ts_list, hr_max, hr_rest, gender='male'):
    """
    Cardiac exertion increment for one minute using Banister TRIMP.
    Passes the second-level HR readings and their ms timestamps to the
    existing compute_cardiac_exertion function.

    Returns float ΔCE_t.
    """
    if not hr_list:
        return 0.0
    result = compute_cardiac_exertion(hr_list, hr_max, hr_rest, gender,
                                      timestamps=ts_list)
    return result['cardiac_exertion']


# ─────────────────────────────────────────────
# Stateful Accumulator
# ─────────────────────────────────────────────

class MinuteExertionAccumulator:
    """Stateful per-workout accumulator for V0 minute-level exertion."""

    def __init__(self, workout_id, user_id, hr_max, hr_rest, gender='male'):
        self.workout_id = workout_id
        self.user_id = user_id
        self.hr_max = hr_max
        self.hr_rest = hr_rest
        self.gender = gender
        self._cumulative_muscular = 0.0
        self._cumulative_cardiac = 0.0
        self._cumulative_unified = 0.0
        self._cumulative_normalized_muscular = 0.0

    def process_minute(self, record):
        """
        Process one per-minute record and return augmented output dict.

        Expected record keys:
            minute_index, movement_type, movement_mode,
            hr_list, ts_list,            # second-level HR for cardiac
            delta_rep (rep_based only),
            step_frequency (time_based only)
        """

        movement_type = record['movement_type']
        movement_mode = record['movement_mode']

        # delta me and delta ce for this minute, then sum to delta ue
        delta_me = compute_minute_muscular(
            movement_type,
            movement_mode,
            delta_rep=record.get('delta_rep'),
            step_frequency=record.get('step_frequency'),
        )

        delta_ce = compute_minute_cardiac(
            record.get('hr_list', []),
            record.get('ts_list', []),
            self.hr_max,
            self.hr_rest,
            self.gender,
        )

        delta_ue = delta_me + delta_ce

        # normalised muscular exertion for this minute
        norm_factor = NORMALIZATION_MAX_DELTA_ME.get(movement_type, 1.0)
        if norm_factor:
            norm_delta_me = min(delta_me / norm_factor, 1.0)
        else:
            norm_delta_me = 0.0

        # cumulate all metrics over the workout
        self._cumulative_muscular += delta_me
        self._cumulative_cardiac += delta_ce
        self._cumulative_unified += delta_ue
        self._cumulative_normalized_muscular += norm_delta_me

        cfg = EXERCISE_CONFIG[movement_type]
        exercise_family = 'rep_based' if cfg['family'] == 'rep' else 'time_based'

        return {
            'workout_id': self.workout_id,
            'user_id': self.user_id,
            'minute_index': record['minute_index'],
            'exercise_type': movement_type,
            'exercise_family': exercise_family,
            'normalizing_factor': norm_factor,
            'movement_mode': movement_mode,
            'delta_rep': record.get('delta_rep'),
            'step_frequency': record.get('step_frequency'),
            'delta_muscular_exertion': round(delta_me, 4),
            'cumulative_muscular_exertion': round(self._cumulative_muscular, 4),
            'normalized_delta_muscular_exertion': round(norm_delta_me, 6),
            'normalized_cumulative_muscular_exertion': round(self._cumulative_normalized_muscular, 6),
            'delta_cardiac_exertion': round(delta_ce, 4),
            'cumulative_cardiac_exertion': round(self._cumulative_cardiac, 4),
            'delta_unified_exertion': round(delta_ue, 4),
            'cumulative_unified_exertion': round(self._cumulative_unified, 4),
        }


# ─────────────────────────────────────────────
# Data Loading & Session Processing
# ─────────────────────────────────────────────

def _normalize_cols(df):
    df = df.copy()
    df.columns = df.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()
    return df


def _load_hr(session_dir):
    """Load hr.csv, return (ts_array_ms, hr_array)."""
    hr_df = _normalize_cols(pd.read_csv(os.path.join(session_dir, 'hr.csv')))
    ts_col = hr_df.columns[0]
    hr_col = hr_df.columns[1]
    return hr_df[ts_col].tolist(), hr_df[hr_col].tolist()


def _group_hr_by_minute(hr_ts, hr_vals, workout_start_ms):
    """
    Group heart rate measurements by minute of workout.

    Groups heart rate readings into buckets based on which minute of the workout
    they occurred in. Filters out invalid (non-positive) heart rate values.

    Args:
        hr_ts (list): Timestamps of heart rate measurements in milliseconds.
        hr_vals (list): Corresponding heart rate values in beats per minute.
        workout_start_ms (int): Workout start time in milliseconds (epoch/reference time).

    Returns:
        dict: A dictionary where:
            - Keys are minute indices (0-based) from workout start
            - Values are tuples of (hr_list, ts_list) containing:
                - hr_list: Heart rate values for that minute
                - ts_list: Corresponding timestamps for that minute

    Example:
        >>> _group_hr_by_minute([1000, 61000], [120, 130], 0)
        {0: ([120], [1000]), 1: ([130], [61000])}
    """
    """Return dict: minute_index -> (hr_list, ts_list)."""
    groups = {}
    for ts, hr in zip(hr_ts, hr_vals):
        if hr <= 0:
            continue
        minute = int((ts - workout_start_ms) // 60000)
        groups.setdefault(minute, ([], []))
        groups[minute][0].append(hr)
        groups[minute][1].append(ts)
    return groups


def process_session(exercise_type, session_dir, hr_max, hr_rest,
                    gender='male', workout_id=None, user_id=None):
    """
    Load sample data, aggregate to minutes, run accumulator.

    Returns (per_minute_results: list[dict], hr_ts: list, hr_vals: list)
    """
    cfg = EXERCISE_CONFIG[exercise_type]
    family = cfg['family']
    movement_mode = 'rep_based' if family == 'rep' else 'time_based'

    hr_ts, hr_vals = _load_hr(session_dir)
    workout_start_ms = hr_ts[0] if hr_ts else 0

    hr_by_minute = _group_hr_by_minute(hr_ts, hr_vals, workout_start_ms)

    if family == 'rep':
        reps_df = _normalize_cols(pd.read_csv(os.path.join(session_dir, 'reps.csv')))
        reps_df['minute_index'] = ((reps_df['start_time'] - workout_start_ms) // 60000).astype(int)
        rep_by_minute = reps_df.groupby('minute_index').size().to_dict()
        all_minutes = sorted(set(rep_by_minute) | set(hr_by_minute))
    else:
        ts_df = _normalize_cols(pd.read_csv(os.path.join(session_dir, 'time_series.csv')))
        ts_df['minute_index'] = (ts_df.index // 60).astype(int)
        moving = ts_df[ts_df['status'] != 'pause']
        if 'step_frequency' in moving.columns:
            freq_by_minute = (
                moving[moving['step_frequency'] > 0]
                .groupby('minute_index')['step_frequency']
                .mean()
                .to_dict()
            )
        else:
            freq_by_minute = {}
        all_minutes = sorted(set(ts_df['minute_index']) | set(hr_by_minute))

    acc = MinuteExertionAccumulator(
        workout_id=workout_id or exercise_type,
        user_id=user_id or 'demo',
        hr_max=hr_max,
        hr_rest=hr_rest,
        gender=gender,
    )

    results = []
    for minute in all_minutes:
        hr_list, ts_list = hr_by_minute.get(minute, ([], []))
        if family == 'rep':
            record = {
                'minute_index': minute,
                'movement_type': exercise_type,
                'movement_mode': movement_mode,
                'delta_rep': rep_by_minute.get(minute, 0),
                'hr_list': hr_list,
                'ts_list': ts_list,
            }
        else:
            record = {
                'minute_index': minute,
                'movement_type': exercise_type,
                'movement_mode': movement_mode,
                'step_frequency': freq_by_minute.get(minute),
                'hr_list': hr_list,
                'ts_list': ts_list,
            }
        results.append(acc.process_minute(record))

    return results, hr_ts, hr_vals


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_minute_exertion(results, exercise_type, hr_ts, hr_vals,
                         run_exertion_result=None, filename=None, data_label=None):
    """
    3-subplot figure:

    Subplot 1 — combined exertion overview (dual y-axes):
        Left : grouped bars (ΔMuscular, ΔCardiac, ΔUnified per minute)
               + cumulative lines (Muscular, Cardiac, Unified)
        Right: Heart Rate (bpm), averaged per minute

    Subplot 2 — normalised muscular exertion:
        Normalised ΔMuscular (per minute) and Normalised Cumulative Muscular

    Subplot 3 — V0 (minute-level) vs V1 (run_exertion) total comparison:
        Vertical grouped bars for Muscular, Cardiac, and Combined/Unified totals.
        Shown only when run_exertion_result is provided.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    minutes    = [r['minute_index'] for r in results]
    delta_me   = [r['delta_muscular_exertion'] for r in results]
    delta_ce   = [r['delta_cardiac_exertion'] for r in results]
    delta_ue   = [r['delta_unified_exertion'] for r in results]
    cum_me     = [r['cumulative_muscular_exertion'] for r in results]
    cum_ce     = [r['cumulative_cardiac_exertion'] for r in results]
    cum_ue     = [r['cumulative_unified_exertion'] for r in results]
    norm_delta = [r['normalized_delta_muscular_exertion'] for r in results]
    norm_cum   = [r['normalized_cumulative_muscular_exertion'] for r in results]

    # Plot HR in its original form per second (not averaged per minute)
    hr_sec_x = []
    hr_sec_y = []
    if hr_ts:
        t0 = hr_ts[0]
        for ts, hr in zip(hr_ts, hr_vals):
            if hr > 0:
                # x-axis: minute index + fractional second within minute
                minute_index = (ts - t0) / 60000.0
                hr_sec_x.append(minute_index)
                hr_sec_y.append(hr)

    n_rows = 3 if run_exertion_result is not None else 2
    height_ratios = [3, 2, 2] if n_rows == 3 else [3, 2]
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 5 * n_rows),
                             gridspec_kw={'height_ratios': height_ratios})
    label_suffix = f'  [{data_label}]' if data_label else ''
    fig.suptitle(f'Minute-Level Exertion (V0) — {exercise_type}{label_suffix}',
                 fontsize=13, fontweight='bold')

    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if n_rows == 3 else None

    x = np.arange(len(minutes))
    width = 0.2

    # ── Subplot 1: grouped delta bars + cumulative lines ──
    ax1.bar(x - width, delta_me, width, label='ΔMuscular', color='#4472C4', alpha=0.55)
    ax1.bar(x,         delta_ce, width, label='ΔCardiac',  color='#C0504D', alpha=0.55)
    ax1.bar(x + width, delta_ue, width, label='ΔUnified',  color='#404040', alpha=0.55)

    ax1.plot(x, cum_me, marker='o', markersize=3, color='#4472C4',
             linewidth=1.8, label='Cum Muscular')
    ax1.plot(x, cum_ce, marker='^', markersize=3, color='#C0504D',
             linewidth=1.8, label='Cum Cardiac')
    ax1.plot(x, cum_ue, marker='s', markersize=3, color='#1A1A1A',
             linewidth=1.8, label='Cum Unified')

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(m) for m in minutes], fontsize=8)
    ax1.set_xlabel('Minute index', fontsize=9)
    ax1.set_ylabel('Exertion', fontsize=9)
    ax1.set_title('Per-Minute Increments & Cumulative Exertion', fontsize=9, loc='left')
    ax1.tick_params(labelsize=8)
    ax1.grid(True, axis='y', alpha=0.2)

    if minutes:
        for val, color in [(cum_me[-1], '#4472C4'), (cum_ce[-1], '#C0504D'),
                           (cum_ue[-1], '#1A1A1A')]:
            ax1.annotate(f'{val:.2f}', xy=(x[-1], val),
                         xytext=(5, 0), textcoords='offset points',
                         fontsize=7, color=color, va='center')

    # Right axis: HR (per second, original form)
    ax1_hr = ax1.twinx()
    if hr_sec_x and hr_sec_y:
        ax1_hr.plot(hr_sec_x, hr_sec_y, color='#E06060', linewidth=1.2,
                    linestyle=':', alpha=0.9, label='HR (bpm, per sec)')
    ax1_hr.set_ylabel('Heart Rate (bpm)', fontsize=9, color='#E06060')
    ax1_hr.tick_params(axis='y', labelsize=8, labelcolor='#E06060')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_hr.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2, loc='upper left')

    # ── Subplot 2: normalised muscular curves ──
    ax2.plot(x, norm_delta, linestyle='--', marker='o', markersize=3,
             color='#70A0D4', linewidth=1.5, label='Norm ΔMuscular (per min)')
    ax2.plot(x, norm_cum, linestyle='-', marker='s', markersize=3,
             color='#2255A0', linewidth=1.5, label='Norm Cumulative Muscular')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(m) for m in minutes], fontsize=8)
    ax2.set_xlabel('Minute index', fontsize=9)
    ax2.set_ylabel('Normalised muscular exertion', fontsize=9)
    ax2.set_title('Normalised Muscular Exertion', fontsize=9, loc='left')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.tick_params(labelsize=8)
    ax2.grid(True, alpha=0.2)

    if minutes:
        ax2.annotate(f'{norm_cum[-1]:.3f}', xy=(x[-1], norm_cum[-1]),
                     xytext=(5, 0), textcoords='offset points',
                     fontsize=7, color='#2255A0', va='center')

    # ── Subplot 3: V1 (run_exertion) total exertion bars ──
    if ax3 is not None and run_exertion_result is not None:
        v1_muscular = run_exertion_result['muscular']['muscular_exertion']
        v1_cardiac  = run_exertion_result['cardiac']['cardiac_exertion']
        v1_combined = run_exertion_result['combined']['combined_exertion']

        categories = ['Muscular', 'Cardiac', 'Combined']
        values     = [v1_muscular, v1_cardiac, v1_combined]
        colors     = ['#4472C4', '#C0504D', '#404040']

        cx = np.arange(len(categories))
        bars = ax3.bar(cx, values, width=0.4, color=colors, alpha=0.7)

        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax3.set_xticks(cx)
        ax3.set_xticklabels(categories, fontsize=9)
        ax3.set_ylabel('Total exertion', fontsize=9)
        ax3.set_title('V1 Total Exertion (run_exertion)', fontsize=9, loc='left')
        ax3.tick_params(labelsize=8)
        ax3.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()

    if filename is None:
        filename = f'minute_{exercise_type}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'  [plot saved → {filename}]')
    plt.close(fig)


# ─────────────────────────────────────────────
# CSV Export
# ─────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

def save_results_csv(results, session_name):
    """
    Save per-minute metrics for a session to results/{session_name}_per_minute_value.csv.

    Columns saved:
        minute_index, exercise_type, exercise_family, normalizing_factor,
        delta_muscular_exertion, cumulative_muscular_exertion,
        normalized_delta_muscular_exertion, normalized_cumulative_muscular_exertion,
        delta_cardiac_exertion, cumulative_cardiac_exertion,
        delta_unified_exertion, cumulative_unified_exertion
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    columns = [
        'minute_index', 'exercise_type', 'exercise_family', 'normalizing_factor',
        'delta_muscular_exertion', 'cumulative_muscular_exertion',
        'normalized_delta_muscular_exertion', 'normalized_cumulative_muscular_exertion',
        'delta_cardiac_exertion', 'cumulative_cardiac_exertion',
        'delta_unified_exertion', 'cumulative_unified_exertion',
    ]
    df = pd.DataFrame(results)[columns]
    path = os.path.join(RESULTS_DIR, f'{session_name}_per_minute_value.csv')
    df.to_csv(path, index=False)
    print(f'  [csv  saved → {path}]')


# ─────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────

def print_results(results, exercise_type):
    print(f'\n{"─"*80}')
    print(f'  {exercise_type}')
    print(f'{"─"*80}')
    header = (f"{'Min':>4}  {'ΔMuscular':>10}  {'CumMusc':>10}  "
              f"{'ΔCardiac':>10}  {'ΔUnified':>10}  {'CumUnified':>11}")
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for r in results:
        dr = r.get('delta_rep')
        sf = r.get('step_frequency')
        extra = f"  reps={dr}" if dr is not None else (f"  freq={sf:.2f}" if sf else '')
        print(f"  {r['minute_index']:>3}  "
              f"{r['delta_muscular_exertion']:>10.4f}  "
              f"{r['cumulative_muscular_exertion']:>10.4f}  "
              f"{r['delta_cardiac_exertion']:>10.4f}  "
              f"{r['delta_unified_exertion']:>10.4f}  "
              f"{r['cumulative_unified_exertion']:>11.4f}"
              f"{extra}")
    if results:
        last = results[-1]
        print(f"\n  Total muscular:  {last['cumulative_muscular_exertion']:.4f}")
        print(f"  Total unified:   {last['cumulative_unified_exertion']:.4f}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minute-level V0 exertion from sample or real data')
    parser.add_argument('--mode', choices=['sample', 'real'], default='sample',
                        help='Data source: "sample" (synthetic) or "real" (real user sessions). Default: sample.')
    parser.add_argument('--exercise', default=None,
                        help='Run a single exercise type (e.g. skierg, wall_ball). Default: all.')
    parser.add_argument('--age',    type=int,   default=32)
    parser.add_argument('--rhr',    type=int,   default=58)
    parser.add_argument('--gender', default='male')
    parser.add_argument('--hyperparameter_identifier', default='new_max')
    args = parser.parse_args()

    hr_max = 207 - 0.7 * args.age

    if args.mode == 'sample':
        sessions = [s for s in SAMPLE_SESSIONS if args.exercise is None or s[0] == args.exercise]
        if not sessions:
            print(f"No sample sessions found for exercise '{args.exercise}'.")
            sys.exit(1)

        for exercise_type, folder, eq_wt, family, bw, age, rhr, gender in sessions:
            session_dir = os.path.join(DATA_DIR, folder)
            if not os.path.isdir(session_dir):
                print(f'[skip] {folder} — directory not found')
                continue

            results, hr_ts, hr_vals = process_session(
                exercise_type, session_dir,
                hr_max=hr_max, hr_rest=rhr, gender=gender,
            )
            print_results(results, exercise_type)

            try:
                v1_result = run_sample_session(exercise_type, folder, eq_wt, family,
                                               bw, age, rhr, gender)
            except Exception as e:
                print(f'  [warn] run_sample_session failed for {exercise_type}: {e}')
                v1_result = None

            plot_minute_exertion(
                results, exercise_type, hr_ts, hr_vals,
                run_exertion_result=v1_result,
                filename=os.path.join(RESULTS_DIR, f'sample_{args.hyperparameter_identifier}_{exercise_type}_minute.png'),
                data_label='Sample Data',
            )
            save_results_csv(results, f'sample_{args.hyperparameter_identifier}_{exercise_type}')

    else:  # real
        sessions = [s for s in SESSIONS if args.exercise is None or s[1] == args.exercise]
        if not sessions:
            print(f"No real sessions found for exercise '{args.exercise}'.")
            sys.exit(1)

        for folder, exercise_type, bw, eq_wt, age, rhr, gender in sessions:
            session_dir = os.path.join(DATA_DIR, folder)
            if not os.path.isdir(session_dir):
                print(f'[skip] {folder} — directory not found')
                continue

            results, hr_ts, hr_vals = process_session(
                exercise_type, session_dir,
                hr_max=hr_max, hr_rest=rhr, gender=gender,
                workout_id=folder,
            )
            print_results(results, exercise_type)

            try:
                v1_result = run_session(folder, bw, eq_wt, age, rhr, gender)
            except Exception as e:
                print(f'  [warn] run_session failed for {folder}: {e}')
                v1_result = None

            os.makedirs(RESULTS_DIR, exist_ok=True)
            plot_minute_exertion(
                results, exercise_type, hr_ts, hr_vals,
                run_exertion_result=v1_result,
                filename=os.path.join(RESULTS_DIR, f'real_{args.hyperparameter_identifier}_{folder}_minute.png'),
                data_label='Real User Data',
            )
            save_results_csv(results, f'real_{args.hyperparameter_identifier}_{folder}')
