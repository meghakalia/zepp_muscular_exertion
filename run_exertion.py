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
    hrs = [h for h in hr_df.iloc[:, 1].tolist() if h > 0]
    cardiac = compute_cardiac_exertion(hrs, hr_max, rhr, gender=gender)
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


def plot_sessions(results):
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(7, 3.5 * n))
    fig.suptitle("Wall Ball Exertion", fontsize=13, fontweight='bold')
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
    plt.savefig('exertion_plot.png', dpi=150, bbox_inches='tight')
    print("\n[plot saved → exertion_plot.png]")
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='Plot exertion breakdown')
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
