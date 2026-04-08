"""
Run muscular + cardiac exertion on real Wall Ball sessions.

Usage:
    python wall_ball/run_exertion.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from wall_ball.exertion import (
    calculate_exertion,
    compute_cardiac_exertion,
    compute_combined_exertion,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Real sessions: (folder_name, body_weight_kg, equipment_weight_kg, age, rhr, gender)
SESSIONS = [
    ('20251225_SSCDDQTQHOPK_HT_6kg', 78.0, 6.0, 32, 58, 'male'),
    ('20251230_FZEIJCLQMGTN_HT_4kg',  78.0, 4.0, 32, 58, 'male'),
    ('20251230_WTTADQTQLRFVOBWA_HT_6kg', 78.0, 6.0, 32, 58, 'male'),
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


def normalize_columns(df):
    """Strip units like (ms), (m), (bpm) from column names."""
    df = df.copy()
    df.columns = df.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()
    return df


def load_and_prepare(session_dir):
    """Load reps, sets, hr CSVs and prepare for exertion calculation."""
    reps_df = normalize_columns(pd.read_csv(os.path.join(session_dir, 'reps.csv')))
    sets_df = normalize_columns(pd.read_csv(os.path.join(session_dir, 'sets.csv')))
    hr_df = normalize_columns(pd.read_csv(os.path.join(session_dir, 'hr.csv')))

    # Convert duration from ms to seconds if needed
    if 'duration' in reps_df.columns and reps_df['duration'].median() > 100:
        reps_df['duration'] = reps_df['duration'] / 1000.0
    reps_df['motion'] = 'exercise'

    # Assign set_index based on set boundaries
    set_boundaries = list(zip(sets_df['start_time'], sets_df['stop_time']))
    set_indices = []
    for _, rep in reps_df.iterrows():
        assigned = 0
        for s_idx, (s_start, s_end) in enumerate(set_boundaries):
            if s_start <= rep['start_time'] <= s_end:
                assigned = s_idx
                break
        set_indices.append(assigned)
    reps_df['set_index'] = set_indices

    return reps_df, sets_df, hr_df


def run_session(name, bw, eq_wt, age, rhr, gender):
    """Run muscular + cardiac exertion for one session."""
    session_dir = os.path.join(DATA_DIR, name)
    reps_df, sets_df, hr_df = load_and_prepare(session_dir)

    # Muscular exertion (V1)
    result = calculate_exertion(
        'wall_ball',
        motions_df=reps_df,
        body_weight_kg=bw,
        equipment_weight_kg=eq_wt,
    )

    # Cardiac exertion (Banister TRIMP)
    hr_max = 207 - 0.7 * age  # Gellish formula
    hr_timestamps = hr_df.iloc[:, 0].tolist()
    hr_values = hr_df.iloc[:, 1].tolist()
    valid_pairs = [(t, h) for t, h in zip(hr_timestamps, hr_values) if h > 0]
    ts_list, hrs = (list(zip(*valid_pairs)) if valid_pairs else ([], []))
    ts_list, hrs = list(ts_list), list(hrs)
    cardiac = compute_cardiac_exertion(hrs, hr_max, rhr, gender=gender, timestamps=ts_list)
    combined = compute_combined_exertion(result['muscular_exertion'], cardiac['cardiac_exertion'])

    return {
        'name': name,
        'muscular': result,
        'cardiac': cardiac,
        'combined': combined,
        'hr_max': hr_max,
        'valid_hrs': len(hrs),
    }


def print_result(r):
    """Print exertion results for one session."""
    m = r['muscular']
    c = r['cardiac']
    comb = r['combined']

    print(f"  Muscular Exertion:  {m['muscular_exertion']:.4f}")
    print(f"    movement_coef:    {m['movement_coef']}")
    print(f"    T (eff work):     {m['effective_work_time']}s")
    print(f"    f_load:           {m['f_load']}")
    print(f"    f_density:        {m['f_density']}")
    print(f"    f_stability:      {m['f_stability']}")
    if m.get('stability_cv') is not None:
        print(f"    stability_cv:     {m['stability_cv']}%")
    print(f"    Reps/Sets:        {m['total_reps']} / {m['sets']}")
    print(f"    Work/Rest:        {m['total_work_time']}s / {m['total_rest_time']}s")

    print(f"  Cardiac Exertion:   {c['cardiac_exertion']:.4f}  (TRIMP, {c['valid_samples']}s = {c['valid_samples']/60:.1f}min)")
    print(f"    HRmax:            {r['hr_max']:.1f}")

    print(f"  Combined:           {comb['combined_exertion']:.4f}")
    if comb['combined_exertion'] > 0:
        m_pct = comb['muscular_exertion'] / comb['combined_exertion'] * 100
        c_pct = comb['cardiac_exertion'] / comb['combined_exertion'] * 100
        print(f"    Ratio:            muscular {m_pct:.1f}% / cardiac {c_pct:.1f}%")


def run_sample_session(exercise_type, name, eq_wt, family, bw, age, rhr, gender):
    """Run muscular + cardiac exertion for one sample session (any exercise type)."""
    session_dir = os.path.join(DATA_DIR, name)
    hr_df = normalize_columns(pd.read_csv(os.path.join(session_dir, 'hr.csv')))
    sets_df = normalize_columns(pd.read_csv(os.path.join(session_dir, 'sets.csv')))

    if family == 'rep':
        reps_df, _, _ = load_and_prepare(session_dir)
        result = calculate_exertion(
            exercise_type,
            motions_df=reps_df,
            body_weight_kg=bw,
            equipment_weight_kg=eq_wt,
        )
    else:
        ts_df = pd.read_csv(os.path.join(session_dir, 'time_series.csv'))
        set_boundaries = list(zip(sets_df['start_time'], sets_df['stop_time']))
        result = calculate_exertion(
            exercise_type,
            time_series_df=ts_df,
            body_weight_kg=bw,
            equipment_weight_kg=eq_wt,
            set_boundaries=set_boundaries,
        )

    hr_max = 207 - 0.7 * age
    hr_timestamps = hr_df.iloc[:, 0].tolist()
    hr_values = hr_df.iloc[:, 1].tolist()
    valid_pairs = [(t, h) for t, h in zip(hr_timestamps, hr_values) if h > 0]
    ts_list, hrs = (list(zip(*valid_pairs)) if valid_pairs else ([], []))
    ts_list, hrs = list(ts_list), list(hrs)
    cardiac = compute_cardiac_exertion(hrs, hr_max, rhr, gender=gender, timestamps=ts_list)
    combined = compute_combined_exertion(result['muscular_exertion'], cardiac['cardiac_exertion'])

    return {
        'name': exercise_type,
        'muscular': result,
        'cardiac': cardiac,
        'combined': combined,
        'hr_max': hr_max,
        'valid_hrs': len(hrs),
    }


def plot_sessions(results, filename='exertion_plot.png'):
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(7, 3.5 * n))
    fig.suptitle("Exertion Breakdown", fontsize=13, fontweight='bold')
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        m = r['muscular']['muscular_exertion']
        c = r['cardiac']['cardiac_exertion']
        combined = r['combined']['combined_exertion']

        labels = ['Muscular', 'Cardiac', 'Combined']
        values = [m, c, combined]
        colors = ['#4472C4', '#C0504D', '#1A1A1A']
        alphas = [0.4,        0.4,        1.0      ]
        edgews = [0,          0,          2        ]

        bars = ax.barh(labels, values, color=colors, height=0.5)
        for bar, a, lw, col in zip(bars, alphas, edgews, colors):
            bar.set_alpha(a)
            bar.set_linewidth(lw)
            bar.set_edgecolor('#1A1A1A' if lw > 0 else col)

        ax.set_title(r['name'], fontsize=9, loc='left')
        ax.set_xlabel('Exertion', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis='x', alpha=0.2)

        for bar, val in zip(bars, values):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n[plot saved → {filename}]")
    # plt.show()


FOLDER_TO_EXERCISE = {
    'wallball':      'wall_ball',
    'burpee_broad_jumps': 'burpee_broad_jumps',
    'walking_lunges': 'sandbag_lunges',
    'sandbag_lunges': 'sandbag_lunges',
    'skierg':         'skierg',
    'rowing':         'rowing',
}


def run_user_date_sessions(user_id, date, bw=78.0, eq_wt=6.0, age=32, rhr=58, gender='male'):
    """Run exertion for all exercise folders under data/{user_id}/{date}/."""
    import json
    session_dir = os.path.join(DATA_DIR, str(user_id), str(date))
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(f"Session folder not found: {session_dir}")

    user_info_path = os.path.join(session_dir, 'user_info.json')
    if os.path.exists(user_info_path):
        with open(user_info_path) as f:
            info = json.load(f)
        bw      = info.get('body_weight_kg', bw)
        age     = info.get('age', age)
        rhr     = info.get('rhr', rhr)
        gender  = info.get('gender', gender)
        print(f"[user_info.json] age={age}, rhr={rhr}, bw={bw}kg, gender={gender}")
    else:
        print(f"  [warning] user_info.json missing in {session_dir} — using defaults")

    results = []
    for folder_name in sorted(os.listdir(session_dir)):
        folder_path = os.path.join(session_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        required = ['reps.csv', 'sets.csv', 'hr.csv']
        if not all(os.path.exists(os.path.join(folder_path, f)) for f in required):
            continue

        exercise_type = FOLDER_TO_EXERCISE.get(folder_name)
        if exercise_type is None:
            print(f"  [skip] unknown exercise folder: {folder_name}")
            continue

        reps_df, sets_df, hr_df = load_and_prepare(folder_path)

        # Extract mean RPE if available
        rpe = None
        if 'rpe' in sets_df.columns:
            valid = sets_df['rpe'].dropna()
            if len(valid) > 0:
                rpe = valid.mean()

        result = calculate_exertion(
            exercise_type,
            motions_df=reps_df,
            body_weight_kg=bw,
            equipment_weight_kg=eq_wt,
        )

        hr_max = 207 - 0.7 * age
        hr_timestamps = hr_df.iloc[:, 0].tolist()
        hr_values = hr_df.iloc[:, 1].tolist()
        valid_pairs = [(t, h) for t, h in zip(hr_timestamps, hr_values) if h > 0]
        ts_list, hrs = (list(zip(*valid_pairs)) if valid_pairs else ([], []))
        ts_list, hrs = list(ts_list), list(hrs)
        cardiac = compute_cardiac_exertion(hrs, hr_max, rhr, gender=gender, timestamps=ts_list)
        combined = compute_combined_exertion(result['muscular_exertion'], cardiac['cardiac_exertion'])

        results.append({
            'name': folder_name,
            'muscular': result,
            'cardiac': cardiac,
            'combined': combined,
            'hr_max': hr_max,
            'valid_hrs': len(hrs),
            'rpe': rpe,
        })

    return results


def plot_user_date_sessions(results, filename='user_date_plot.png'):
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(7, 3.5 * n))
    fig.suptitle("Exertion Breakdown", fontsize=13, fontweight='bold')
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        m = r['muscular']['muscular_exertion']
        c = r['cardiac']['cardiac_exertion']
        combined = r['combined']['combined_exertion']
        rpe = r.get('rpe')

        if rpe is not None:
            labels = ['RPE (avg)', 'Muscular', 'Cardiac', 'Combined']
            values = [rpe, m, c, combined]
            colors = ['#70AD47', '#4472C4', '#C0504D', '#1A1A1A']
            alphas = [0.6,       0.4,       0.4,       1.0      ]
            edgews = [0,         0,         0,         2        ]
        else:
            labels = ['Muscular', 'Cardiac', 'Combined']
            values = [m, c, combined]
            colors = ['#4472C4', '#C0504D', '#1A1A1A']
            alphas = [0.4,       0.4,       1.0      ]
            edgews = [0,         0,         2        ]

        bars = ax.barh(labels, values, color=colors, height=0.5)
        for bar, a, lw, col in zip(bars, alphas, edgews, colors):
            bar.set_alpha(a)
            bar.set_linewidth(lw)
            bar.set_edgecolor('#1A1A1A' if lw > 0 else col)

        ax.set_title(r['name'], fontsize=9, loc='left')
        ax.set_xlabel('Exertion', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis='x', alpha=0.2)

        for bar, val in zip(bars, values):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n[plot saved → {filename}]")
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='Plot exertion breakdown for real sessions')
    parser.add_argument('--plot_sampled_data', action='store_true', help='Run and plot exertion for all sample_* sessions')
    parser.add_argument('--plot_user_date', action='store_false', help='Plot exertion for all exercises under data/{user_id}/{date}/')
    parser.add_argument('--user_id', default='3095587934', help='User ID folder name (default: 3095278624)')
    parser.add_argument('--date', default='2026-03-25', help='Session date folder name (default: 2026-04-01)')
    parser.add_argument('--bw', type=float, default=78.0, help='Body weight in kg (default: 78.0)')
    parser.add_argument('--eq_wt', type=float, default=6.0, help='Equipment weight in kg (default: 6.0)')
    parser.add_argument('--age', type=int, default=32, help='Age in years (default: 32)')
    parser.add_argument('--rhr', type=int, default=58, help='Resting heart rate (default: 58)')
    parser.add_argument('--gender', default='male', help='Gender: male or female (default: male)')
    args = parser.parse_args()

    print("=" * 60)
    print("Wall Ball Exertion — Real Sessions")
    print("=" * 60)

    all_results = []
    for name, bw, eq_wt, age, rhr, gender in SESSIONS:
        print(f"\n--- {name} ---")
        r = run_session(name, bw, eq_wt, age, rhr, gender)
        print_result(r)
        all_results.append(r)

    print()

    if args.plot:
        plot_sessions(all_results)

    if args.plot_sampled_data:
        print("\n" + "=" * 60)
        print("Sample Sessions Exertion")
        print("=" * 60)
        sample_results = []
        for ex_type, folder, eq_wt, family, bw, age, rhr, gender in SAMPLE_SESSIONS:
            print(f"\n--- {ex_type} ({folder}) ---")
            r = run_sample_session(ex_type, folder, eq_wt, family, bw, age, rhr, gender)
            print_result(r)
            sample_results.append(r)
        print()
        plot_sessions(sample_results, filename='sample_data.png')

    if args.plot_user_date:
        print("\n" + "=" * 60)
        print(f"User {args.user_id} — {args.date}")
        print("=" * 60)
        ud_results = run_user_date_sessions(
            args.user_id, args.date,
            bw=args.bw, eq_wt=args.eq_wt, age=args.age, rhr=args.rhr, gender=args.gender,
        )
        for r in ud_results:
            print(f"\n--- {r['name']} ---")
            print_result(r)
            if r['rpe'] is not None:
                print(f"  RPE (avg):          {r['rpe']:.1f}")
        print()
        plot_user_date_sessions(ud_results, filename=f"{args.user_id}_{args.date}_plot.png")
