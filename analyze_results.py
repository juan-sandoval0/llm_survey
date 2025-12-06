"""
LLM vs Human Comparison Analysis
=================================

This script compares the LLM survey results to the human baseline data
and computes various metrics for evaluating cultural simulation accuracy.

Usage:
    python analyze_results.py --llm-data llm_survey_results.csv --human-data human_responses.csv
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Human baseline proportions (from our earlier analysis)
# Format: {dilemma_id: {country: {choice_label: proportion}}}
HUMAN_BASELINE = {
    "D1_family": {
        "United States": {"Family/Honor": 0.125, "Individual/Medical": 0.875},
        "Mexico": {"Family/Honor": 0.0, "Individual/Medical": 1.0},
        "India": {"Family/Honor": 0.462, "Individual/Medical": 0.538}
    },
    "D2_halal": {
        "United States": {"Report/Transparency": 0.875, "Stay quiet/Economic": 0.125},
        "Mexico": {"Report/Transparency": 0.444, "Stay quiet/Economic": 0.556},
        "India": {"Report/Transparency": 0.923, "Stay quiet/Economic": 0.077}
    },
    "D3_loyalty": {
        "United States": {"Protect friend": 0.375, "Report friend": 0.625},
        "Mexico": {"Protect friend": 0.444, "Report friend": 0.556},
        "India": {"Protect friend": 0.615, "Report friend": 0.385}
    },
    "D4_hierarchy": {
        "United States": {"Speak up": 1.0, "Defer to authority": 0.0},
        "Mexico": {"Speak up": 0.889, "Defer to authority": 0.111},
        "India": {"Speak up": 0.923, "Defer to authority": 0.077}
    },
    "D5_sacred": {
        "United States": {"Excavate/Science": 0.875, "Respect indigenous": 0.125},
        "Mexico": {"Excavate/Science": 0.889, "Respect indigenous": 0.111},
        "India": {"Excavate/Science": 0.923, "Respect indigenous": 0.077}
    }
}

# Human sample sizes
HUMAN_N = {
    "United States": 8,
    "Mexico": 9,
    "India": 13
}


def load_llm_data(filepath: str) -> pd.DataFrame:
    """Load LLM survey results."""
    df = pd.read_csv(filepath)
    # Filter to successfully parsed responses only
    df = df[df['parse_success'] == True]
    return df


def compute_llm_proportions(df: pd.DataFrame) -> Dict:
    """Compute choice proportions from LLM data."""
    proportions = {}
    
    for dilemma_id in df['dilemma_id'].unique():
        proportions[dilemma_id] = {}
        dilemma_df = df[df['dilemma_id'] == dilemma_id]
        
        for country in df['country'].unique():
            country_df = dilemma_df[dilemma_df['country'] == country]
            if len(country_df) == 0:
                continue
            
            choice_counts = country_df['choice_label'].value_counts(normalize=True)
            proportions[dilemma_id][country] = choice_counts.to_dict()
    
    return proportions


def jensen_shannon_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    Lower = more similar (0 = identical).
    """
    # Get all unique keys
    all_keys = set(p.keys()) | set(q.keys())
    
    # Create aligned arrays with smoothing for zero values
    epsilon = 1e-10
    p_arr = np.array([p.get(k, epsilon) for k in sorted(all_keys)])
    q_arr = np.array([q.get(k, epsilon) for k in sorted(all_keys)])
    
    # Normalize
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    
    return jensenshannon(p_arr, q_arr)


def mean_absolute_error(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute mean absolute error between proportions."""
    all_keys = set(p.keys()) | set(q.keys())
    errors = [abs(p.get(k, 0) - q.get(k, 0)) for k in all_keys]
    return np.mean(errors)


def chi_square_test(llm_df: pd.DataFrame, human_props: Dict[str, float], 
                    dilemma_id: str, country: str) -> Tuple[float, float]:
    """
    Perform chi-square goodness of fit test.
    Tests whether LLM distribution matches human distribution.
    """
    subset = llm_df[(llm_df['dilemma_id'] == dilemma_id) & (llm_df['country'] == country)]
    if len(subset) < 5:
        return np.nan, np.nan
    
    observed = subset['choice_label'].value_counts()
    
    # Expected counts based on human proportions
    expected = {k: v * len(subset) for k, v in human_props.items()}
    
    # Align
    all_keys = set(observed.index) | set(expected.keys())
    obs_arr = np.array([observed.get(k, 0) for k in sorted(all_keys)])
    exp_arr = np.array([expected.get(k, 0.001) for k in sorted(all_keys)])  # Avoid zero
    
    try:
        chi2, p = stats.chisquare(obs_arr, exp_arr)
        return chi2, p
    except:
        return np.nan, np.nan


def analyze_stereotyping(llm_props: Dict, human_baseline: Dict) -> Dict:
    """
    Detect stereotyping patterns:
    - Over-differentiation: LLM exaggerates cultural differences
    - Under-differentiation: LLM doesn't capture real differences
    - Directional errors: LLM predicts opposite direction
    """
    results = {}
    
    for dilemma_id in human_baseline.keys():
        results[dilemma_id] = {}
        
        human_d = human_baseline[dilemma_id]
        llm_d = llm_props.get(dilemma_id, {})
        
        # Get the "key" choice for analysis (first option)
        key_choice = list(list(human_d.values())[0].keys())[0]
        
        for country in human_d.keys():
            if country not in llm_d:
                continue
            
            human_prop = human_d[country].get(key_choice, 0)
            llm_prop = llm_d[country].get(key_choice, 0)
            
            # Compare to US as baseline (for detecting stereotyping)
            if country != "United States" and "United States" in human_d:
                human_diff_from_us = human_prop - human_d["United States"].get(key_choice, 0)
                llm_diff_from_us = llm_prop - llm_d.get("United States", {}).get(key_choice, 0)
                
                # Stereotyping = LLM exaggerates the difference
                if abs(llm_diff_from_us) > abs(human_diff_from_us) * 1.5:
                    pattern = "OVER-DIFFERENTIATION"
                elif abs(llm_diff_from_us) < abs(human_diff_from_us) * 0.5:
                    pattern = "UNDER-DIFFERENTIATION"
                elif human_diff_from_us * llm_diff_from_us < 0:  # Different signs
                    pattern = "DIRECTIONAL ERROR"
                else:
                    pattern = "OK"
                
                results[dilemma_id][country] = {
                    "human_prop": human_prop,
                    "llm_prop": llm_prop,
                    "human_diff_from_us": human_diff_from_us,
                    "llm_diff_from_us": llm_diff_from_us,
                    "pattern": pattern
                }
    
    return results


def analyze_religiosity_effect(llm_df: pd.DataFrame) -> Dict:
    """
    Test whether LLM captures the religiosity effect we found in human data.
    (In India, religiosity strongly predicted D1 choice)
    """
    india_d1 = llm_df[(llm_df['country'] == 'India') & (llm_df['dilemma_id'] == 'D1_family')]

    if len(india_d1) == 0:
        return {"error": "No India D1 data"}

    results = {}
    for religiosity in india_d1['religiosity'].unique():
        if pd.isna(religiosity):
            continue
        subset = india_d1[india_d1['religiosity'] == religiosity]
        family_prop = (subset['choice_label'] == 'Family/Honor').mean()
        results[religiosity] = {
            "n": len(subset),
            "family_proportion": family_prop
        }

    return results


def analyze_baseline_vs_personas(llm_df: pd.DataFrame) -> Dict:
    """
    Analyze the baseline (no-persona) responses to understand the model's default behavior.
    """
    baseline_df = llm_df[llm_df['persona_id'] == 'BASELINE']
    persona_df = llm_df[llm_df['persona_id'] != 'BASELINE']

    if len(baseline_df) == 0:
        return {"error": "No baseline data found"}

    results = {
        "baseline_proportions": {},
        "persona_vs_baseline": {},
        "persona_effect_magnitude": {}
    }

    # Compute baseline proportions for each dilemma
    for dilemma_id in baseline_df['dilemma_id'].unique():
        dilemma_baseline = baseline_df[baseline_df['dilemma_id'] == dilemma_id]
        choice_props = dilemma_baseline['choice_label'].value_counts(normalize=True).to_dict()
        results["baseline_proportions"][dilemma_id] = {
            "n": len(dilemma_baseline),
            "proportions": choice_props
        }

    # Compare each country's personas to baseline
    for dilemma_id in HUMAN_BASELINE.keys():
        results["persona_vs_baseline"][dilemma_id] = {}

        baseline_subset = baseline_df[baseline_df['dilemma_id'] == dilemma_id]
        if len(baseline_subset) == 0:
            continue

        baseline_props = baseline_subset['choice_label'].value_counts(normalize=True).to_dict()
        key_choice = list(HUMAN_BASELINE[dilemma_id][list(HUMAN_BASELINE[dilemma_id].keys())[0]].keys())[0]
        baseline_val = baseline_props.get(key_choice, 0)

        for country in ["United States", "Mexico", "India"]:
            country_subset = persona_df[(persona_df['dilemma_id'] == dilemma_id) &
                                       (persona_df['country'] == country)]
            if len(country_subset) == 0:
                continue

            country_props = country_subset['choice_label'].value_counts(normalize=True).to_dict()
            country_val = country_props.get(key_choice, 0)
            human_val = HUMAN_BASELINE[dilemma_id][country].get(key_choice, 0)

            # Calculate shifts
            persona_shift = country_val - baseline_val
            human_shift = human_val - baseline_val

            results["persona_vs_baseline"][dilemma_id][country] = {
                "baseline_prop": baseline_val,
                "persona_prop": country_val,
                "human_prop": human_val,
                "persona_shift_from_baseline": persona_shift,
                "human_shift_from_baseline": human_shift,
                "persona_matches_human_direction": (persona_shift * human_shift) > 0 or abs(human_shift) < 0.05,
                "persona_effect_appropriate": abs(persona_shift) > 0.05 and (persona_shift * human_shift) > 0
            }

    # Summary: Does persona prompting add signal or just elicit bias?
    all_comparisons = []
    for dilemma_data in results["persona_vs_baseline"].values():
        for country_data in dilemma_data.values():
            if isinstance(country_data, dict):
                all_comparisons.append(country_data)

    if all_comparisons:
        pct_correct_direction = sum(1 for c in all_comparisons if c["persona_matches_human_direction"]) / len(all_comparisons)
        pct_appropriate_shift = sum(1 for c in all_comparisons if c["persona_effect_appropriate"]) / len(all_comparisons)

        results["summary"] = {
            "n_comparisons": len(all_comparisons),
            "pct_personas_shift_correct_direction": pct_correct_direction,
            "pct_personas_provide_appropriate_shift": pct_appropriate_shift,
            "interpretation": (
                "Personas effectively shift responses toward human patterns" if pct_appropriate_shift > 0.6 else
                "Personas provide weak or inconsistent shifts" if pct_appropriate_shift > 0.3 else
                "Personas do not meaningfully shift responses from baseline"
            )
        }

    return results


def full_analysis(llm_data_path: str, output_path: str = "analysis_results.json"):
    """Run complete analysis comparing LLM to human baseline."""
    
    print("=" * 70)
    print("LLM vs HUMAN BASELINE COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading LLM data from: {llm_data_path}")
    llm_df = load_llm_data(llm_data_path)
    print(f"Loaded {len(llm_df)} valid LLM responses")
    
    # Compute LLM proportions
    llm_props = compute_llm_proportions(llm_df)
    
    # ============== MAIN COMPARISON TABLE ==============
    print("\n" + "=" * 70)
    print("1. CHOICE PROPORTIONS: LLM vs HUMAN")
    print("=" * 70)
    
    all_results = []
    
    for dilemma_id, dilemma_data in HUMAN_BASELINE.items():
        print(f"\n--- {dilemma_id} ---")
        
        for country in ["United States", "Mexico", "India"]:
            human_p = dilemma_data.get(country, {})
            llm_p = llm_props.get(dilemma_id, {}).get(country, {})
            
            if not llm_p:
                print(f"  {country}: No LLM data")
                continue
            
            # Compute metrics
            js_div = jensen_shannon_divergence(human_p, llm_p)
            mae = mean_absolute_error(human_p, llm_p)
            chi2, p_val = chi_square_test(llm_df, human_p, dilemma_id, country)
            
            # Get key proportions for display
            key_choice = list(human_p.keys())[0]
            human_val = human_p.get(key_choice, 0)
            llm_val = llm_p.get(key_choice, 0)
            
            result = {
                "dilemma": dilemma_id,
                "country": country,
                "key_choice": key_choice,
                "human_prop": human_val,
                "llm_prop": llm_val,
                "difference": llm_val - human_val,
                "js_divergence": js_div,
                "mae": mae,
                "chi2": chi2,
                "chi2_p": p_val
            }
            all_results.append(result)
            
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {country}:")
            print(f"    {key_choice}: Human={human_val:.1%} vs LLM={llm_val:.1%} (Δ={llm_val-human_val:+.1%})")
            print(f"    JS divergence: {js_div:.4f}")
            print(f"    χ² p-value: {p_val:.4f} {sig}")
    
    # ============== STEREOTYPING ANALYSIS ==============
    print("\n" + "=" * 70)
    print("2. STEREOTYPING DETECTION")
    print("=" * 70)
    
    stereo_results = analyze_stereotyping(llm_props, HUMAN_BASELINE)
    
    pattern_counts = {"OVER-DIFFERENTIATION": 0, "UNDER-DIFFERENTIATION": 0, 
                      "DIRECTIONAL ERROR": 0, "OK": 0}
    
    for dilemma_id, countries in stereo_results.items():
        print(f"\n--- {dilemma_id} ---")
        for country, data in countries.items():
            pattern = data["pattern"]
            pattern_counts[pattern] += 1
            
            if pattern != "OK":
                print(f"  {country}: {pattern}")
                print(f"    Human diff from US: {data['human_diff_from_us']:+.1%}")
                print(f"    LLM diff from US: {data['llm_diff_from_us']:+.1%}")
    
    print("\n--- Pattern Summary ---")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count}")
    
    # ============== RELIGIOSITY EFFECT ==============
    print("\n" + "=" * 70)
    print("3. RELIGIOSITY EFFECT (India D1)")
    print("=" * 70)
    print("Human finding: 'Very active' = 100% family, 'Not religious' = 0% family")
    
    relig_results = analyze_religiosity_effect(llm_df)
    print("\nLLM results:")
    for religiosity, data in sorted(relig_results.items(), 
                                     key=lambda x: x[1].get('family_proportion', 0), 
                                     reverse=True):
        if isinstance(data, dict) and 'family_proportion' in data:
            print(f"  {religiosity}: {data['family_proportion']:.1%} chose family (n={data['n']})")
    
    # Check if gradient exists
    if len(relig_results) >= 2:
        props = [d.get('family_proportion', 0) for d in relig_results.values() if isinstance(d, dict)]
        if max(props) - min(props) > 0.2:
            print("\n  ✓ LLM shows religiosity gradient")
        else:
            print("\n  ✗ LLM does NOT capture religiosity effect")
    
    # ============== BASELINE ANALYSIS ==============
    print("\n" + "=" * 70)
    print("4. BASELINE VS PERSONA ANALYSIS")
    print("=" * 70)
    print("Comparing model's default behavior (no persona) to persona-prompted responses")

    baseline_results = analyze_baseline_vs_personas(llm_df)

    if "error" not in baseline_results:
        print("\nBaseline proportions (model's default behavior):")
        for dilemma_id, data in baseline_results["baseline_proportions"].items():
            print(f"\n  {dilemma_id} (n={data['n']}):")
            for choice, prop in sorted(data["proportions"].items(), key=lambda x: x[1], reverse=True):
                print(f"    {choice}: {prop:.1%}")

        print("\n" + "-" * 70)
        print("Persona Effects (shift from baseline):")
        for dilemma_id, countries in baseline_results["persona_vs_baseline"].items():
            print(f"\n  {dilemma_id}:")
            for country, data in countries.items():
                if isinstance(data, dict):
                    shift_arrow = "→" if abs(data["persona_shift_from_baseline"]) < 0.05 else "↑" if data["persona_shift_from_baseline"] > 0 else "↓"
                    direction_mark = "✓" if data["persona_matches_human_direction"] else "✗"
                    print(f"    {country}: Baseline {data['baseline_prop']:.1%} {shift_arrow} Persona {data['persona_prop']:.1%} (Human: {data['human_prop']:.1%}) {direction_mark}")

        if "summary" in baseline_results:
            print("\n" + "-" * 70)
            print("Summary:")
            print(f"  Personas shift in correct direction: {baseline_results['summary']['pct_personas_shift_correct_direction']:.1%}")
            print(f"  Personas provide appropriate magnitude shift: {baseline_results['summary']['pct_personas_provide_appropriate_shift']:.1%}")
            print(f"  Interpretation: {baseline_results['summary']['interpretation']}")
    else:
        print(f"\n  {baseline_results['error']}")

    # ============== CONFIDENCE ANALYSIS ==============
    print("\n" + "=" * 70)
    print("5. CONFIDENCE ANALYSIS")
    print("=" * 70)

    print("\nMean confidence by country:")
    for country in ["United States", "Mexico", "India", "Baseline"]:
        country_df = llm_df[llm_df['country'] == country]
        if len(country_df) > 0:
            mean_conf = country_df['confidence'].mean()
            print(f"  {country}: {mean_conf:.2f}")
    
    # ============== SAVE RESULTS ==============
    full_results = {
        "comparison_table": all_results,
        "stereotyping_analysis": {k: {ck: cv for ck, cv in v.items()}
                                   for k, v in stereo_results.items()},
        "pattern_counts": pattern_counts,
        "religiosity_effect": relig_results,
        "baseline_analysis": baseline_results,
        "llm_proportions": llm_props
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nFull results saved to: {output_path}")
    
    # ============== SUMMARY METRICS ==============
    print("\n" + "=" * 70)
    print("6. OVERALL ACCURACY METRICS")
    print("=" * 70)
    
    js_values = [r['js_divergence'] for r in all_results if not np.isnan(r['js_divergence'])]
    mae_values = [r['mae'] for r in all_results if not np.isnan(r['mae'])]
    
    print(f"\nJensen-Shannon Divergence (lower = better):")
    print(f"  Mean: {np.mean(js_values):.4f}")
    print(f"  Median: {np.median(js_values):.4f}")
    print(f"  Range: [{min(js_values):.4f}, {max(js_values):.4f}]")
    
    print(f"\nMean Absolute Error:")
    print(f"  Mean: {np.mean(mae_values):.4f}")
    print(f"  Median: {np.median(mae_values):.4f}")
    
    # By country
    print("\nBy Country (mean JS divergence):")
    for country in ["United States", "Mexico", "India"]:
        country_js = [r['js_divergence'] for r in all_results 
                     if r['country'] == country and not np.isnan(r['js_divergence'])]
        if country_js:
            print(f"  {country}: {np.mean(country_js):.4f}")
    
    return full_results


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM survey results vs human baseline")
    parser.add_argument("--llm-data", type=str, required=True,
                       help="Path to LLM survey results CSV")
    parser.add_argument("--output", type=str, default="analysis_results.json",
                       help="Output path for analysis results")
    
    args = parser.parse_args()
    full_analysis(args.llm_data, args.output)


if __name__ == "__main__":
    main()
