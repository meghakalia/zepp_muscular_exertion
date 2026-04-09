# Wall Ball — HYROX Exercise Exertion Module

Unified exertion calculation for all 8 HYROX exercises (V1).

## Setup

```bash
pip install -r requirements.txt
```

## Run Example

```bash
cd wall_ball
python run_example.py
```

This runs exertion calculation on all 8 exercises using the sample data in `data/`.

## Usage in Code

### DataFrame path (app/demo usage)

```python
from wall_ball.exertion import calculate_exertion

# Rep-based exercise (Wall Ball, SkiErg, Rowing, Sandbag Lunges, Burpee Broad Jumps)
result = calculate_exertion(
    'wall_ball',
    motions_df=reps_df,          # DataFrame with start_time, stop_time, duration, motion, set_index
    body_weight_kg=78.0,
    equipment_weight_kg=9.0,
)

# Time-based exercise (Farmer's Carry, Sled Push, Sled Pull)
result = calculate_exertion(
    'farmers_carry',
    time_series_df=ts_df,        # DataFrame with timestamp, status, step_frequency
    body_weight_kg=78.0,
    equipment_weight_kg=24.0,
    set_boundaries=[(start_ms, end_ms), ...],
)
```

### Firmware path (C struct style input)

```python
from wall_ball.exertion import calculate_exertion_firmware

# Rep-based — matches alg_hte_station_input_t
result = calculate_exertion_firmware({
    'station_type': 9,               # HYROX_SPORT_TYPE_MEDICINE (wall_ball)
    'body_weight_kg': 78.0,
    'equipment_weight_kg': 9.0,
    'total_duration_ms': 178850,
    'total_rest_duration_ms': 60000,
    'rep_count': 75,
    'reps': [
        {'start_ms': 0, 'end_ms': 1600, 'core_stability': 0.05},
        {'start_ms': 1800, 'end_ms': 3400, 'core_stability': 0.07},
        # ...
    ],
})

# Time-based — matches alg_hte_station_input_t
result = calculate_exertion_firmware({
    'station_type': 7,               # HYROX_SPORT_TYPE_FAMER (farmers_carry)
    'body_weight_kg': 78.0,
    'equipment_weight_kg': 24.0,
    'total_duration_ms': 200000,
    'total_rest_duration_ms': 20000,
    'cadences': [1.8, 1.9, 2.0, -1, 1.8, ...],  # Hz per second, -1 = invalid
})
```

### Result

```python
result['muscular_exertion']  # raw exertion value (unbounded)
result['movement_coef']      # per-exercise coefficient
result['f_load']             # 1.0 (disabled in V1)
result['f_density']          # density factor [1.0, 1.3]
result['f_stability']        # stability factor [1.0, 1.2]
result['stability_cv']       # CV% of core deviation, or None
result['cadence_factor']     # cadence factor (time-based only)
```

## Formula (V1)

```
muscular_exertion = movement_coef · T · f_load · f_density · f_stability
```

- **movement_coef**: per-exercise coefficient (0.3–0.7)
- **T**: effective work quantity (rep-based: mixed work unit; time-based: active_time × cadence_factor)
- **f_load**: 1.0 (disabled in V1; future: calibrated load ratio)
- **f_density**: `1.0 + 0.3 · active_time / (active_time + total_rest)`
- **f_stability**: `1.0 + 0.2 · (1 - exp(-0.08 · CV))` — only for Wall Ball, SkiErg, Sandbag Lunges

## Module Structure

- `exertion.py` — Core exertion formula (V1) with DataFrame and firmware input paths
- `utils.py` — 5-dimension session evaluation (Cardiovascular Load, Recovery Capacity, Output Sustainability, Control Stability, Pacing Strategy)
- `insight.py` — Insight generation and 3D stress assessment
- `main.py` — Orchestration (evaluation pipeline, data loading)
- `data/` — Sample data for all 8 exercises

## Station Type Mapping

| Int | C Enum | Exercise |
|-----|--------|----------|
| 2 | HYROX_SPORT_TYPE_SKING | SkiErg |
| 3 | HYROX_SPORT_TYPE_SLEDGE | Sled Push |
| 4 | HYROX_SPORT_TYPE_SLED | Sled Pull |
| 5 | HYROX_SPORT_TYPE_JUMP | Burpee Broad Jumps |
| 6 | HYROX_SPORT_TYPE_ROW | Rowing |
| 7 | HYROX_SPORT_TYPE_FAMER | Farmer's Carry |
| 8 | HYROX_SPORT_TYPE_LUNGE | Sandbag Lunges |
| 9 | HYROX_SPORT_TYPE_MEDICINE | Wall Ball |
