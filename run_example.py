"""
Standalone example: run exertion + full session evaluation on sample data.

Usage:
    cd wall_ball
    python run_example.py
"""
import os
import sys

# Allow running directly from inside wall_ball/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from wall_ball.exertion import calculate_exertion
from wall_ball.main import evaluate_from_dataframes_web, evaluate_time_based_web, process_all_sessions_and_export


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def normalize_columns(df):
    """Strip units like (ms), (m) from column names."""
    df = df.copy()
    df.columns = df.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()
    return df


def run_rep_based(exercise_type, session_dir, body_weight_kg, equipment_weight_kg):
    """Run exertion for a rep-based exercise."""
    reps_df = normalize_columns(pd.read_csv(os.path.join(session_dir, 'reps.csv')))
    sets_df = normalize_columns(pd.read_csv(os.path.join(session_dir, 'sets.csv')))

    if 'duration' in reps_df.columns and reps_df['duration'].median() > 100:
        reps_df['duration'] = reps_df['duration'] / 1000.0
    reps_df['motion'] = 'exercise'

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

    extra_data = None
    if exercise_type == 'sandbag_lunges' and 'squat_distance' in reps_df.columns:
        extra_data = {'squat_depth': reps_df['squat_distance'].tolist()}

    return calculate_exertion(
        exercise_type,
        motions_df=reps_df,
        body_weight_kg=body_weight_kg,
        equipment_weight_kg=equipment_weight_kg,
        extra_data=extra_data,
    )


def run_time_based(exercise_type, session_dir, body_weight_kg, equipment_weight_kg):
    """Run exertion for a time-based exercise."""
    ts_df = pd.read_csv(os.path.join(session_dir, 'time_series.csv'))
    sets_df = normalize_columns(pd.read_csv(os.path.join(session_dir, 'sets.csv')))
    set_boundaries = list(zip(sets_df['start_time'].tolist(), sets_df['stop_time'].tolist()))

    return calculate_exertion(
        exercise_type,
        time_series_df=ts_df,
        body_weight_kg=body_weight_kg,
        equipment_weight_kg=equipment_weight_kg,
        set_boundaries=set_boundaries,
    )


def run_full_evaluation(session_dir, user_info):
    """Run full pipeline: exertion + 5 dimensions + 3D assessment + insight."""
    hr_df = pd.read_csv(os.path.join(session_dir, 'hr.csv'))
    reps_df = pd.read_csv(os.path.join(session_dir, 'reps.csv'))
    sets_df = pd.read_csv(os.path.join(session_dir, 'sets.csv'))

    return evaluate_from_dataframes_web(hr_df, reps_df, sets_df, user_info)


def print_exertion(result):
    """Print exertion result."""
    print(f"  Exertion:        {result['muscular_exertion']:.4f}")
    print(f"  movement_coef:   {result['movement_coef']}")
    print(f"  T (eff work):    {result['effective_work_time']}s")
    print(f"  f_load:          {result['f_load']}")
    print(f"  f_density:       {result['f_density']}")
    print(f"  f_stability:     {result['f_stability']}")
    if result.get('stability_cv') is not None:
        print(f"  stability_cv:    {result['stability_cv']}%")
    if result.get('cadence_factor') and result['cadence_factor'] != 1.0:
        print(f"  cadence_factor:  {result['cadence_factor']}")
    print(f"  Work time (W):   {result['total_work_time']}s")
    print(f"  Rest time (R):   {result['total_rest_time']}s")
    if result.get('total_reps') is not None:
        print(f"  Total reps:      {result['total_reps']}")
    if result.get('sets'):
        print(f"  Sets:            {result['sets']}")


def print_full_evaluation(response):
    """Print full evaluation: exertion + 5 dimensions + 3D + insight."""
    exertion = response.get('exertion')
    if exertion:
        muscular = exertion.get('muscular_exertion', exertion.get('exertion', 0))
        print(f"\n  Muscular Exertion: {muscular:.4f}")
        if exertion.get('cardiac_exertion') is not None:
            print(f"  Cardiac Exertion:  {exertion['cardiac_exertion']:.4f}")
        if exertion.get('combined_exertion') is not None:
            print(f"  Combined Exertion: {exertion['combined_exertion']:.4f}")

    # 5-dimension scores
    dim = response.get('dimension_analysis', {})
    print("\n  5-Dimension Scores:")
    for name, data in dim.items():
        score = data.get('score', '—')
        print(f"    {name}: {score}/5")

    # 3D assessment
    three_dim = response.get('three_dim_assessment', {})
    print("\n  3D Assessment:")
    for name, data in three_dim.items():
        classification = data.get('classification', '—')
        print(f"    {name}: {classification}")

    # Training recommendation
    rec = three_dim.get('training_recommendation', {})
    if rec.get('insight'):
        print(f"\n  Recommendation [{rec.get('classification', '—')}]:")
        print(f"    {rec['insight']}")

    # Session insight
    insight = response.get('insight')
    if insight:
        print("\n  Session Insights:")
        for item in insight:
            print(f"    [{item['metric']}] {item['message']}")


if __name__ == '__main__':
    BW = 78.0
    # process_all_sessions_and_export()
    # ════════════════════════════════════════════════
    # Part 1: Exertion only (all 8 exercises)
    # ════════════════════════════════════════════════
    print("=" * 60)
    print("PART 1: EXERTION CALCULATION (all 8 exercises)")
    print("=" * 60)

    exercises = [
        ('wall_ball', '20251225_SSCDDQTQHOPK_HT_6kg', 6.0, 'rep'),
        ('skierg', 'sample_skierg', 0.0, 'rep'),
        ('rowing', 'sample_rowing', 0.0, 'rep'),
        ('sandbag_lunges', 'sample_sandbag_lunges', 20.0, 'rep'),
        ('burpee_broad_jumps', 'sample_burpee_broad_jumps', 0.0, 'rep'),
        ('farmers_carry', 'sample_farmers_carry', 24.0, 'time'),
        ('sled_push', 'sample_sled_push', 100.0, 'time'),
        ('sled_pull', 'sample_sled_pull', 100.0, 'time'),
    ]

    for ex_type, folder, eq_weight, family in exercises:
        print(f"\n--- {ex_type} ---")
        session_dir = os.path.join(DATA_DIR, folder)
        if family == 'rep':
            result = run_rep_based(ex_type, session_dir, BW, eq_weight)
        else:
            result = run_time_based(ex_type, session_dir, BW, eq_weight)
        print_exertion(result)

    # ════════════════════════════════════════════════
    # Part 2: Full evaluation (exertion + 5 dim + 3D + insight)
    # ════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 2: FULL SESSION EVALUATION (all rep-based exercises)")
    print("=" * 60)

    # Sessions with hr.csv + reps.csv + sets.csv (rep-based only)
    full_eval_sessions = [
        ('Wall Ball (real)',       '20251225_SSCDDQTQHOPK_HT_6kg', 6.0),
        ('SkiErg (sample)',        'sample_skierg',                 0.0),
        ('Rowing (sample)',        'sample_rowing',                 0.0),
        ('Sandbag Lunges (sample)','sample_sandbag_lunges',         20.0),
        ('Burpee Broad Jumps (sample)', 'sample_burpee_broad_jumps', 0.0),
    ]

    for label, folder, eq_weight in full_eval_sessions:
        print(f"\n--- {label} ---")
        user_info = {
            'age': 32,
            'rhr': 58,
            'body_weight_kg': BW,
            'medicine_ball_weight_kg': eq_weight,
            'gender': 'Male',
        }
        response = run_full_evaluation(
            os.path.join(DATA_DIR, folder),
            user_info,
        )
        print_full_evaluation(response)

    # ════════════════════════════════════════════════
    # Part 3: Time-based full evaluation (3D + insight)
    # ════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 3: TIME-BASED FULL EVALUATION (Farmer's Carry, Sled Push, Sled Pull)")
    print("=" * 60)

    time_based_sessions = [
        ("Farmer's Carry (sample)", 'sample_farmers_carry', 24.0, 'farmers_carry'),
        ("Sled Push (sample)",      'sample_sled_push',     100.0, 'sled_push'),
        ("Sled Pull (sample)",      'sample_sled_pull',     100.0, 'sled_pull'),
    ]

    for label, folder, eq_weight, ex_type in time_based_sessions:
        print(f"\n--- {label} ---")
        session_dir = os.path.join(DATA_DIR, folder)
        ts_df = pd.read_csv(os.path.join(session_dir, 'time_series.csv'))
        sets_df = pd.read_csv(os.path.join(session_dir, 'sets.csv'))

        # Load HR if available
        hr_path = os.path.join(session_dir, 'hr.csv')
        hr_df = pd.read_csv(hr_path) if os.path.exists(hr_path) else None

        user_info = {
            'age': 32,
            'rhr': 58,
            'body_weight_kg': BW,
            'equipment_weight_kg': eq_weight,
            'gender': 'Male',
        }
        response = evaluate_time_based_web(
            ts_df, hr_df, sets_df, user_info, exercise_type=ex_type,
        )
        print_full_evaluation(response)
