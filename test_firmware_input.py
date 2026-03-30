"""
Test exertion calculation via firmware-style input for all 8 HYROX exercises.

Generates fake station data matching the C struct format:
  alg_hte_station_input_t {
      station_type, total_duration_ms, total_rest_duration_ms,
      equipment_weight_kg, reps[], rep_count, cadences[], cadence_count
  }
"""

import os
import sys

# Allow running directly from inside wall_ball/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from wall_ball.exertion import calculate_exertion_firmware, EXERCISE_CONFIG


BODY_WEIGHT = 78.0


def generate_hr_data(total_seconds, rhr=58, hr_max=185, intensity=0.7, rng=None):
    """Generate simulated per-second HR data for a station."""
    if rng is None:
        rng = np.random.default_rng(99)
    target_hr = rhr + intensity * (hr_max - rhr)
    hrs = []
    hr = rhr + 10  # start slightly above resting
    for i in range(total_seconds):
        # Ramp up toward target HR, with noise
        hr += (target_hr - hr) * 0.05 + rng.normal(0, 1.5)
        hr = max(rhr, min(hr_max, hr))
        hrs.append(round(hr))
    return hrs


def generate_rep_station(station_type, exercise_type, rep_count, rep_duration_range,
                         rest_ratio, equipment_weight_kg, has_core_deviation,
                         add_hr=False, age=32, rhr=58, gender='male'):
    """Generate a firmware-style station dict for a rep-based exercise."""
    rng = np.random.default_rng(42)
    config = EXERCISE_CONFIG[exercise_type]
    t_ref = config['t_ref']

    reps = []
    cursor_ms = 0

    for i in range(rep_count):
        dur_s = rng.uniform(*rep_duration_range)
        dur_ms = int(dur_s * 1000)
        start_ms = cursor_ms
        end_ms = cursor_ms + dur_ms

        core_stability = -1.0
        if has_core_deviation:
            # Simulate core deviation in meters (0.02 - 0.12 range)
            core_stability = float(rng.uniform(0.02, 0.12))

        reps.append({
            'start_ms': start_ms,
            'end_ms': end_ms,
            'core_stability': core_stability,
        })

        # Small gap between reps (100-500ms)
        cursor_ms = end_ms + int(rng.uniform(100, 500))

    # Total active time from reps
    active_ms = sum(r['end_ms'] - r['start_ms'] for r in reps)
    # Rest = active_time * rest_ratio
    rest_ms = int(active_ms * rest_ratio)
    total_duration_ms = active_ms + rest_ms

    station = {
        'station_type': station_type,
        'body_weight_kg': BODY_WEIGHT,
        'equipment_weight_kg': equipment_weight_kg,
        'total_duration_ms': total_duration_ms,
        'total_rest_duration_ms': rest_ms,
        'rep_count': rep_count,
        'reps': reps,
    }

    if add_hr:
        hr_max = 207 - 0.7 * age
        total_seconds = total_duration_ms // 1000
        station['hrs'] = generate_hr_data(total_seconds, rhr=rhr, hr_max=hr_max)
        station['hr_max'] = hr_max
        station['hr_rest'] = rhr
        station['gender'] = gender

    return station


def generate_time_station(station_type, exercise_type, moving_seconds, rest_seconds,
                          equipment_weight_kg, freq_range,
                          add_hr=False, age=32, rhr=58, gender='male'):
    """Generate a firmware-style station dict for a time-based exercise."""
    rng = np.random.default_rng(123)

    # Per-second cadences: valid during moving, -1 during rest
    cadences = []
    total_seconds = moving_seconds + rest_seconds

    for i in range(total_seconds):
        if i < moving_seconds:
            cadences.append(float(rng.uniform(*freq_range)))
        else:
            cadences.append(-1.0)  # rest / invalid

    total_duration_ms = total_seconds * 1000
    total_rest_duration_ms = rest_seconds * 1000

    station = {
        'station_type': station_type,
        'body_weight_kg': BODY_WEIGHT,
        'equipment_weight_kg': equipment_weight_kg,
        'total_duration_ms': total_duration_ms,
        'total_rest_duration_ms': total_rest_duration_ms,
        'cadences': cadences,
    }

    if add_hr:
        hr_max = 207 - 0.7 * age
        station['hrs'] = generate_hr_data(total_seconds, rhr=rhr, hr_max=hr_max)
        station['hr_max'] = hr_max
        station['hr_rest'] = rhr
        station['gender'] = gender

    return station


def test_all_firmware():
    """Run exertion calculation for all 8 exercises via firmware input path."""
    print("=" * 80)
    print("FIRMWARE INPUT PATH — All 8 Exercises")
    print("=" * 80)

    stations = [
        # Rep-based
        generate_rep_station(
            station_type=9,  # HYROX_SPORT_TYPE_MEDICINE
            exercise_type='wall_ball',
            rep_count=75, rep_duration_range=(1.3, 1.8),
            rest_ratio=1.0, equipment_weight_kg=9.0, has_core_deviation=True,
            add_hr=True,
        ),
        generate_rep_station(
            station_type=2,  # HYROX_SPORT_TYPE_SKING
            exercise_type='skierg',
            rep_count=100, rep_duration_range=(1.2, 1.8),
            rest_ratio=0.3, equipment_weight_kg=0.0, has_core_deviation=True,
            add_hr=True,
        ),
        generate_rep_station(
            station_type=6,  # HYROX_SPORT_TYPE_ROW
            exercise_type='rowing',
            rep_count=80, rep_duration_range=(1.8, 2.6),
            rest_ratio=0.3, equipment_weight_kg=0.0, has_core_deviation=False,
            add_hr=True,
        ),
        generate_rep_station(
            station_type=8,  # HYROX_SPORT_TYPE_LUNGE
            exercise_type='sandbag_lunges',
            rep_count=60, rep_duration_range=(1.2, 1.8),
            rest_ratio=0.8, equipment_weight_kg=20.0, has_core_deviation=True,
            add_hr=True,
        ),
        generate_rep_station(
            station_type=5,  # HYROX_SPORT_TYPE_JUMP
            exercise_type='burpee_broad_jumps',
            rep_count=30, rep_duration_range=(3.2, 4.8),
            rest_ratio=0.5, equipment_weight_kg=0.0, has_core_deviation=False,
            add_hr=True,
        ),
        # Time-based
        generate_time_station(
            station_type=7,  # HYROX_SPORT_TYPE_FAMER
            exercise_type='farmers_carry',
            moving_seconds=180, rest_seconds=20,
            equipment_weight_kg=24.0, freq_range=(1.8, 2.2),
            add_hr=True,
        ),
        generate_time_station(
            station_type=3,  # HYROX_SPORT_TYPE_SLEDGE
            exercise_type='sled_push',
            moving_seconds=150, rest_seconds=15,
            equipment_weight_kg=100.0, freq_range=(1.4, 1.8),
            add_hr=True,
        ),
        generate_time_station(
            station_type=4,  # HYROX_SPORT_TYPE_SLED
            exercise_type='sled_pull',
            moving_seconds=160, rest_seconds=25,
            equipment_weight_kg=80.0, freq_range=(1.4, 1.8),
            add_hr=True,
        ),
    ]

    # ── Summary Table ──
    print()
    header = (f"{'Exercise':<22} {'Type':<5} {'Coef':>6} {'T':>8} "
              f"{'fDens':>8} {'fStab':>8} {'Cadence':>8} "
              f"{'Muscular':>10} {'Cardiac':>10} {'Combined':>10}")
    print(header)
    print("-" * len(header))

    results = {}
    all_ok = True

    for station in stations:
        r = calculate_exertion_firmware(station)
        name = r['exercise_type']
        results[name] = r

        family = EXERCISE_CONFIG[name]['family']
        muscular = r['muscular_exertion']
        cardiac = r.get('cardiac_exertion', -1)
        combined = r.get('combined_exertion', -1)
        status = "OK" if muscular > 0 else "FAIL"
        if muscular <= 0:
            all_ok = False

        cardiac_str = f"{cardiac:>10.4f}" if cardiac >= 0 else f"{'N/A':>10}"
        combined_str = f"{combined:>10.4f}" if combined >= 0 else f"{'N/A':>10}"

        print(f"{name:<22} {family:<5} {r['movement_coef']:>6.2f} "
              f"{r['effective_work_time']:>8.2f} {r['f_density']:>8.4f} "
              f"{r['f_stability']:>8.4f} {r['cadence_factor']:>8.4f} "
              f"{muscular:>10.4f} {cardiac_str} {combined_str}  {status}")

    # ── Detail ──
    for name, r in results.items():
        print(f"\n--- {name} ---")
        print(f"  muscular_exertion:  {r['muscular_exertion']}")
        cardiac = r.get('cardiac_exertion', -1)
        combined = r.get('combined_exertion', -1)
        if cardiac >= 0:
            print(f"  cardiac_exertion:   {cardiac}")
            print(f"  combined_exertion:  {combined}")
        else:
            print(f"  cardiac_exertion:   N/A (no HR data)")
        print(f"  movement_coef:      {r['movement_coef']}")
        print(f"  f_load:             {r['f_load']}")
        print(f"  f_density:          {r['f_density']}")
        print(f"  f_stability:        {r['f_stability']}")
        if r.get('stability_cv') is not None:
            print(f"  stability_cv:       {r['stability_cv']}%")
        print(f"  Total Work Time:    {r['total_work_time']}s")
        print(f"  Total Rest Time:    {r['total_rest_time']}s")
        print(f"  Effective Work (T): {r['effective_work_time']}s")
        if r.get('cadence_factor') != 1.0:
            print(f"  cadence_factor:     {r['cadence_factor']}")
        if r.get('total_reps') is not None:
            print(f"  Total reps:         {r['total_reps']}")

    print()
    if all_ok:
        print("ALL 8 EXERCISES PASSED — non-zero exertion via firmware input.")
    else:
        print("SOME EXERCISES FAILED — check results above.")

    return results


if __name__ == '__main__':
    test_all_firmware()
