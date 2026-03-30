"""
Insight generation logic for Wall Ball sessions.

This module reads the Excel file (Hybrid Training - 5-radar interpretations.xlsx)
and matches them against the 5-dimension evaluation + trend results to produce
a list of structured insights. Each sheet in the Excel file corresponds to one dimension.

Also includes three-dimension assessment logic for generating
training recommendations based on Cardiac Stress, Local Muscular Endurance,
and Movement Control classifications.
"""

from typing import Any, Dict, List
import os
from functools import lru_cache
import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _load_excel_sheets() -> Dict[str, pd.DataFrame]:
    """
    Load and cache all sheets from the Excel file.
    Uses lru_cache to avoid re-reading the file on every call.
    """
    xlsx_path = os.path.join(os.path.dirname(__file__), "Hybrid Training - 5-radar interpretations.xlsx")
    if not os.path.exists(xlsx_path):
        return {}

    try:
        excel_file = pd.ExcelFile(xlsx_path, engine='openpyxl')
        sheets_dict = {}
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine='openpyxl')
            # Handle NaN values in Trend column - convert to "N/A" for First Session
            if "Trend" in df.columns:
                df["Trend"] = df["Trend"].fillna("N/A")
            # Normalize key text columns to avoid hidden whitespace mismatches
            for col in ["Trend", "Session_Type"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
                    if col == "Trend":
                        df[col] = df[col].replace(["nan", "NaN", "None"], "N/A")
            sheets_dict[sheet_name] = df
        return sheets_dict
    except Exception as e:
        print(f"Warning: Could not load Excel file: {e}")
        return {}


def _fill_placeholders(template: str, values: List[str]) -> str:
    """
    Replace [X] placeholders in the feedback template sequentially
    with provided string values.
    """
    text = str(template)
    for v in values:
        if "[X]" in text:
            text = text.replace("[X]", v, 1)
    return text


def generate_insights_from_csv(evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate a list of Insights based on the Excel file rules.
    Each sheet in the Excel file corresponds to one dimension.

    Input:
      - evaluation_results: full evaluation dict including:
          - 5 dimension results under keys:
              * 'cardiovascular_load'
              * 'recovery_capacity'
              * 'output_sustainability'
              * 'control_stability'
              * 'pacing_strategy'
          - trend results under key 'trends' (output of evaluate_trends)

    Output:
      - List[dict] where each dict represents one Insight:
          * story_id: Story_ID from Excel
          * metric: Metric name (e.g., "Cardiovascular Load")
          * category: Category ("Optimal", "Good", "Needs Improvement")
          * trend: Trend label ("Improving" / "Maintaining" / "Decreasing" / "N/A")
          * session_type: "First Session" or "Subsequent"
          * value: primary metric value for this session
          * baseline: baseline metric value from history (if available)
          * delta_from_baseline: absolute difference between value and baseline (if available)
          * message: Feedback_Message with [X] placeholders replaced with real numbers
    """
    # Dimension configuration: how each internal key maps to Excel sheet name and which value to use
    dimension_config = {
        "cardiovascular_load": {
            "sheet_name": "CV Load",
            "metric_name": "Cardiovascular Load",
            "value_key": "high_intensity_percentage",
        },
        "recovery_capacity": {
            "sheet_name": "Recovery Capacity",
            "metric_name": "Recovery Capacity",
            "value_key": "avg_hr_drop_rate",
        },
        "output_sustainability": {
            "sheet_name": "Output Sustainability",
            "metric_name": "Output Sustainability",
            "value_key": "performance_decline",
        },
        "control_stability": {
            "sheet_name": "Control Stability",
            "metric_name": "Control Stability",
            "value_key": "cv_percentage",
        },
        "pacing_strategy": {
            "sheet_name": "Pacing Strategy",
            "metric_name": "Pacing Strategy",
            "value_key": "decline_percentage",
        },
    }

    # Load Excel file (cached)
    sheets_dict = _load_excel_sheets()
    if not sheets_dict:
        return []

    # Trend results may be 'insufficient_data'
    trend_results = evaluation_results.get("trends", {}) or {}
    trend_status = trend_results.get("status")
    trends_data = trend_results.get("trends", {}) if trend_status == "success" else {}

    insights: List[Dict[str, Any]] = []

    for dim_key, cfg in dimension_config.items():
        dim_result = evaluation_results.get(dim_key)
        if not dim_result:
            continue

        # Get score from evaluation result
        score = dim_result.get("score")
        if score is None or (isinstance(score, float) and np.isnan(score)):
            continue
        
        # Round score to nearest integer for matching (scores are 1-5)
        score_int = int(round(float(score)))
        if score_int < 1 or score_int > 5:
            continue

        # Primary metric value for this dimension
        value = dim_result.get(cfg["value_key"])
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        metric_name = cfg["metric_name"]
        sheet_name = cfg["sheet_name"]

        # Determine session type and trend label
        if dim_key in trends_data:
            session_type = "Subsequent"
            trend_label = trends_data[dim_key].get("trend", "N/A")
            baseline = trends_data[dim_key].get("baseline")
        else:
            session_type = "First Session"
            trend_label = "N/A"
            baseline = None

        # Get the sheet for this dimension
        if sheet_name not in sheets_dict:
            continue
        
        rules_df = sheets_dict[sheet_name]

        # Filter candidates in Excel sheet for this dimension/setup
        # Match by Score, Session_Type, and Trend
        candidates = rules_df[
            (rules_df["Score"] == score_int)
            & (rules_df["Session_Type"] == session_type)
            & (rules_df["Trend"] == trend_label)
        ]

        if candidates.empty:
            continue

        # Select the first matching row
        selected_row = candidates.iloc[0]

        template = selected_row["Feedback_Message"]
        placeholder_count = str(template).count("[X]")

        # First [X] is always the current value (1 decimal)
        placeholder_values = [f"{float(value):.1f}"]

        # Second [X] (if present) is |current - baseline|, if baseline exists
        delta = None
        if baseline is not None and isinstance(baseline, (int, float)) and not (
            isinstance(baseline, float) and np.isnan(baseline)
        ):
            delta = abs(float(value) - float(baseline))
        if placeholder_count >= 2 and delta is not None:
            placeholder_values.append(f"{delta:.1f}")

        message = _fill_placeholders(template, placeholder_values)

        # Get category from evaluation result for backward compatibility
        category = dim_result.get("category", "N/A")

        insight_struct: Dict[str, Any] = {
            "story_id": int(selected_row["Story_ID"]),
            "metric": metric_name,
            "category": category,
            "trend": trend_label,
            "session_type": session_type,
            "value": float(value),
            "baseline": float(baseline) if baseline is not None else None,
            "delta_from_baseline": float(delta) if delta is not None else None,
            "message": message,
        }
        insights.append(insight_struct)

    return insights


def generate_three_dimension_insight(cardiac_stress_class, local_muscular_endurance_class, movement_control_class):
    """
    Generate a single Insight based on the combination of three dimension classifications.
    
    Matches the provided table with 8 combinations:
    - Cardiac Stress, Local Muscular Endurance, Movement Control
    
    Parameters:
    -----------
    cardiac_stress_class : str
        Classification for Cardiac Stress ("Need Improve", "Good", "Optimal", or "N/A")
    local_muscular_endurance_class : str
        Classification for Local Muscular Endurance ("Need Improve", "Good", "Optimal", or "N/A")
    movement_control_class : str
        Classification for Movement Control ("Need Improve", "Good", "Optimal", or "N/A")
    
    Returns:
    --------
    tuple: (recommendation_classification, insight_message)
        - recommendation_classification: "Need Improve" or "Good/Optimal"
        - insight_message: Detailed training recommendation text
    """
    # Helper function to normalize classification (Good and Optimal are treated as Good/Optimal)
    def normalize_classification(classification):
        if classification in ("Optimal", "Good"):
            return "Good/Optimal"
        return classification

    cs_available = cardiac_stress_class not in ("N/A", None)
    lme_available = local_muscular_endurance_class not in ("N/A", None)
    mc_available = movement_control_class not in ("N/A", None)

    cs_norm = normalize_classification(cardiac_stress_class) if cs_available else None
    lme_norm = normalize_classification(local_muscular_endurance_class) if lme_available else None
    mc_norm = normalize_classification(movement_control_class) if mc_available else None

    # ── Route based on which dimensions are genuinely available ──

    if cs_available and lme_available and mc_available:
        # All 3 dimensions available — original 8 branches
        return _insight_all_three(cs_norm, lme_norm, mc_norm)

    if cs_available and lme_available and not mc_available:
        # CS + LME available, MC = N/A (e.g., time-based exercises without core deviation)
        return _insight_cs_lme(cs_norm, lme_norm)

    if cs_available and not lme_available:
        # CS only (no step frequency / no reps data)
        return _insight_cs_only(cs_norm)

    if lme_available and not cs_available:
        # LME only (no HR data)
        return _insight_lme_only(lme_norm)

    # Nothing available
    return ("Need Improve",
            "Insufficient data to generate a detailed recommendation. "
            "Try adding heart rate monitoring and ensure enough exercise volume for analysis.")


def _insight_all_three(cs_norm, lme_norm, mc_norm):
    """Original 8-branch insight when all 3 dimensions are available."""

    if cs_norm == "Need Improve" and lme_norm == "Need Improve" and mc_norm == "Need Improve":
        recommendation_class = "Need Improve"
        insight = "Start with lower volume (3-5 sets of 5-10 reps) with full rest (90-120s). Practice technique drills at slow tempo before adding volume. Add aerobic base building with 20-30 min easy runs 3x/week using Zepp Coach for Runners. Keep Wall Ball frequency at 2x/week with focus on quality over quantity. Goal is to master movement pattern while building work capacity."

    elif cs_norm == "Need Improve" and lme_norm == "Need Improve" and mc_norm == "Good/Optimal":
        recommendation_class = "Need Improve"
        insight = "Maintain good form while increasing volume to 4-6 sets of 12-15 reps. Reduce rest periods gradually (start 60s, work toward 45s). Add mixed-modal conditioning like Wall Balls combined with runs or rows in circuits. Supplement with interval training such as 8x400m runs at threshold pace using Zepp Coach. Goal is to build endurance without sacrificing technique."

    elif cs_norm == "Need Improve" and lme_norm == "Good/Optimal" and mc_norm == "Need Improve":
        recommendation_class = "Need Improve"
        insight = "Your endurance is good but energy leaks are driving up cardiac stress. Practice breathing patterns (exhale on throw, inhale on catch). Use tempo work like 5x10 reps with 3-second eccentric (squat down slowly). Record sets with video to identify wasted movement. Add Zone 2 cardio with 30-40 min easy runs to improve aerobic efficiency using Zepp Coach for Runners. Goal is to reduce energy cost per rep through better mechanics."

    elif cs_norm == "Need Improve" and lme_norm == "Good/Optimal" and mc_norm == "Good/Optimal":
        recommendation_class = "Need Improve"
        insight = "Excellent muscular endurance and technique - you just need cardiovascular conditioning. Use Zepp Coach for Runners with focus on VO2max interval sessions. For Wall Balls, use high-density formats like EMOM or short rest circuits. Add cross-training with rowing, assault bike, or running intervals. Example workout: 10 min EMOM with 15 Wall Balls each minute. Goal is to improve aerobic and anaerobic systems to handle sustained work."

    elif cs_norm == "Good/Optimal" and lme_norm == "Need Improve" and mc_norm == "Need Improve":
        recommendation_class = "Need Improve"
        insight = "Good cardio but need muscular endurance and movement quality. Use higher rep schemes like 5-8 sets of 15-20 reps with moderate rest (60s). Slow down and prioritize consistency over speed. Add accessory work such as goblet squats, wall sits, and overhead holds to build strength endurance. Monitor Control Stability metric and aim for less than 15% CV. Goal is to build local endurance while refining movement patterns."

    elif cs_norm == "Good/Optimal" and lme_norm == "Need Improve" and mc_norm == "Good/Optimal":
        recommendation_class = "Need Improve"
        insight = "Only muscular endurance needs work - perfect opportunity for volume accumulation. Use progressive overload by adding 5-10 reps per week to total volume. Try time-based sets like 30s, 45s, or 60s continuous work with equal rest. Practice pacing to hold consistent reps per minute across all sets with no drop-off. Example: 5 rounds of 30 Wall Balls at steady pace, rest 90s. Goal is to increase Output Sustainability and Pacing scores to 4 or higher."

    elif cs_norm == "Good/Optimal" and lme_norm == "Good/Optimal" and mc_norm == "Need Improve":
        recommendation_class = "Need Improve"
        insight = "Fitness is excellent - just need technical consistency. Use quality drills like 10x5 Wall Balls with 60s rest, focusing on making every rep identical. Try tempo variations such as 3-1-1 tempo (3s down, 1s pause, 1s up). Record every session with video to compare movement patterns. Add separate skill work sessions when not fatigued. Monitor Control Stability and target less than 10% CV. Goal is to reduce movement variability for efficiency under fatigue."

    elif cs_norm == "Good/Optimal" and lme_norm == "Good/Optimal" and mc_norm == "Good/Optimal":
        recommendation_class = "Good/Optimal"
        insight = "Well-rounded performance - maintain current level or advance to harder variations. For maintenance, complete 2x/week Wall Ball sessions at current volume. Progression options include using a heavier medicine ball (increase by 2-4 lbs), higher target (increase wall height 6-12 inches), faster pace (reduce rest, increase density), or Hyrox-specific competition prep workouts. Use Zepp Coach for Runners to optimize running performance for complete Hyrox readiness. Goal is to continue progressive overload or specialize for competition."

    else:
        recommendation_class = "Need Improve"
        insight = "Focus on building aerobic base, increasing muscular endurance training, and strengthening technical movement patterns. Prioritize recovery and control training intensity to allow for consistent improvement across all areas."

    return recommendation_class, insight


def _insight_cs_lme(cs_norm, lme_norm):
    """Insight when Cardiac Stress + LME are available but Movement Control is N/A.

    Derived from the all-three branches: extract the CS-specific and LME-specific
    advice from the original 8 combinations.
    """
    # CS NI + LME NI: combines advice from branch (NI, NI, NI) — aerobic base + lower volume
    if cs_norm == "Need Improve" and lme_norm == "Need Improve":
        return ("Need Improve",
                "Start with lower volume (3-5 sets) with full rest (90-120s). "
                "Add aerobic base building with 20-30 min easy runs 3x/week using Zepp Coach for Runners. "
                "Keep frequency at 2x/week with focus on quality over quantity. "
                "Goal is to build work capacity across both energy systems.")

    # CS NI + LME G/O: from branch (NI, G/O, G/O) — cardio conditioning
    if cs_norm == "Need Improve" and lme_norm == "Good/Optimal":
        return ("Need Improve",
                "Excellent muscular endurance - you just need cardiovascular conditioning. "
                "Use Zepp Coach for Runners with focus on VO2max interval sessions. "
                "Use high-density formats like EMOM or short rest circuits. "
                "Add cross-training with rowing, assault bike, or running intervals. "
                "Goal is to improve aerobic and anaerobic systems to handle sustained work.")

    # CS G/O + LME NI: from branch (G/O, NI, G/O) — volume accumulation
    if cs_norm == "Good/Optimal" and lme_norm == "Need Improve":
        return ("Need Improve",
                "Only muscular endurance needs work - perfect opportunity for volume accumulation. "
                "Use progressive overload by adding 5-10 reps per week to total volume. "
                "Try time-based sets like 30s, 45s, or 60s continuous work with equal rest. "
                "Practice pacing to hold consistent output across all sets with no drop-off. "
                "Goal is to increase Output Sustainability score to 4 or higher.")

    # CS G/O + LME G/O: from branch (G/O, G/O, G/O) — maintain or advance
    if cs_norm == "Good/Optimal" and lme_norm == "Good/Optimal":
        return ("Good/Optimal",
                "Well-rounded performance - maintain current level or advance to harder variations. "
                "For maintenance, complete 2-3 sessions per week at current intensity. "
                "Progression options include increasing load, reducing rest periods, or faster pace. "
                "Use Zepp Coach for Runners to optimize running performance for complete Hyrox readiness. "
                "Goal is to continue progressive overload or specialize for competition.")

    return ("Need Improve",
            "Focus on building aerobic base and increasing muscular endurance training. "
            "Prioritize recovery and control training intensity to allow for consistent improvement.")


def _insight_cs_only(cs_norm):
    """Insight when only Cardiac Stress is available (no LME or MC).

    Derived from the CS-specific advice in the all-three branches.
    """
    # CS NI: from branches (NI, *, *) — aerobic base building
    if cs_norm == "Need Improve":
        return ("Need Improve",
                "Add aerobic base building with 20-30 min easy runs 3x/week using Zepp Coach for Runners. "
                "Reduce rest periods gradually as fitness improves. "
                "Supplement with interval training such as 8x400m runs at threshold pace. "
                "Goal is to improve aerobic efficiency and lower cardiac stress during exercise.")
    # CS G/O: from branches (G/O, *, *) — maintain and progress
    return ("Good/Optimal",
            "Good cardiovascular response during the session. "
            "Continue current training intensity and consider adding more volume or reducing rest periods. "
            "Use Zepp Coach for Runners to optimize running performance for complete Hyrox readiness.")


def _insight_lme_only(lme_norm):
    """Insight when only Local Muscular Endurance is available (no HR data).

    Derived from the LME-specific advice in the all-three branches.
    """
    # LME NI: from branches (*, NI, *) — progressive overload
    if lme_norm == "Need Improve":
        return ("Need Improve",
                "Use progressive overload by adding 5-10 reps per week to total volume. "
                "Try time-based sets like 30s, 45s, or 60s continuous work with equal rest. "
                "Practice pacing to hold consistent output across all sets with no drop-off. "
                "Goal is to increase Output Sustainability score to 4 or higher.")
    # LME G/O: from branches (*, G/O, *) — maintain or advance
    return ("Good/Optimal",
            "Strong muscular endurance - output was well-maintained throughout the session. "
            "Progression options include increasing load, reducing rest periods, or faster pace. "
            "Goal is to continue progressive overload or specialize for competition.")


def calculate_three_dimension_assessment(evaluation_results):
    """
    Calculate three-dimension assessment based on 5-Radar Score Mapping.
    
    Three Dimensions:
    1. Cardiac Stress --> Average of (Cardiovascular Load + Recovery Capacity) scores
    2. Local Muscular Endurance --> Output Sustainability score
    3. Movement Control --> Control Stability score
    
    Classification:
    - Score <= 3 --> Need Improve
    - Score (3, 4] --> Good
    - Score (4, 5] --> Optimal
    
    Parameters:
    -----------
    evaluation_results : dict
        Dictionary containing evaluation results for all 5 dimensions
    
    Returns:
    --------
    dict: Three-dimension assessment containing:
        - cardiac_stress: dict with score, classification, components
        - local_muscular_endurance: dict with score, classification
        - movement_control: dict with score, classification
        - training_recommendation: dict with classification and insight (single combined recommendation)
    """
    # Helper function to get score from dimension result
    def get_dimension_score(dim_key):
        dim_result = evaluation_results.get(dim_key)
        if dim_result is None:
            return None
        score = dim_result.get('score')
        if score is None:
            return None
        if isinstance(score, (int, float)) and 1 <= score <= 5:
            return float(score)
        return None
    
    # Helper function to classify score
    def classify_score(score):
        if score is None or np.isnan(score):
            return "N/A"
        if score <= 3:
            return "Need Improve"
        elif score <= 4:
            return "Good"
        else:  # score > 4 and <= 5
            return "Optimal"
    
    # 1. Cardiac Stress: Average of Cardiovascular Load + Recovery Capacity
    cv_score = get_dimension_score('cardiovascular_load')
    recovery_score = get_dimension_score('recovery_capacity')
    
    if cv_score is not None and recovery_score is not None:
        cardiac_stress_score = (cv_score + recovery_score) / 2.0
    elif cv_score is not None:
        cardiac_stress_score = cv_score
    elif recovery_score is not None:
        cardiac_stress_score = recovery_score
    else:
        cardiac_stress_score = None
    
    cardiac_stress_classification = classify_score(cardiac_stress_score)
    
    # 2. Local Muscular Endurance: Output Sustainability
    local_muscular_endurance_score = get_dimension_score('output_sustainability')
    local_muscular_endurance_classification = classify_score(local_muscular_endurance_score)
    
    # 3. Movement Control: Control Stability
    movement_control_score = get_dimension_score('control_stability')
    movement_control_classification = classify_score(movement_control_score)
    
    # Generate combined Insight
    recommendation_classification, combined_insight = generate_three_dimension_insight(
        cardiac_stress_classification,
        local_muscular_endurance_classification,
        movement_control_classification
    )
    
    return {
        'cardiac_stress': {
            'score': round(cardiac_stress_score, 2) if cardiac_stress_score is not None else None,
            'classification': cardiac_stress_classification,
            'components': {
                'cardiovascular_load_score': round(cv_score, 2) if cv_score is not None else None,
                'recovery_capacity_score': round(recovery_score, 2) if recovery_score is not None else None,
            }
        },
        'local_muscular_endurance': {
            'score': round(local_muscular_endurance_score, 2) if local_muscular_endurance_score is not None else None,
            'classification': local_muscular_endurance_classification,
        },
        'movement_control': {
            'score': round(movement_control_score, 2) if movement_control_score is not None else None,
            'classification': movement_control_classification,
        },
        'training_recommendation': {
            'classification': recommendation_classification,
            'insight': combined_insight,
        },
    }

