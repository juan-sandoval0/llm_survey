"""
Monitor survey progress and automatically run analyses when complete
"""
import time
import subprocess
import os
import sys

def check_survey_complete():
    """Check if full_results.csv exists and has 776 rows (775 data + 1 header)"""
    if not os.path.exists('full_results.csv'):
        return False

    # Count lines
    with open('full_results.csv', 'r') as f:
        lines = sum(1 for _ in f)

    return lines == 776  # 775 responses + 1 header

def run_analyses():
    """Run all analyses and visualizations"""
    print("\n" + "="*70)
    print("SURVEY COMPLETE! Running all analyses...")
    print("="*70 + "\n")

    steps = [
        ("Basic analysis", "python analyze_results.py --llm-data full_results.csv --output full_analysis.json"),
        ("MVP analysis", "python analyze_full_mvp.py --llm-data full_results.csv --output full_mvp_analysis.json"),
        ("Main visualizations", "python create_visualizations.py"),
        ("Additional visualizations", "python create_additional_visualizations.py"),
        ("Confidence visualization", "python create_confidence_visualization.py"),
    ]

    for step_name, command in steps:
        print(f"\n{step_name}...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ {step_name} complete")
        else:
            print(f"  ✗ {step_name} failed:")
            print(result.stderr)

    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • full_results.csv - Raw survey data (775 responses)")
    print("  • full_analysis.json - Basic statistical analysis")
    print("  • full_mvp_analysis.json - MVP framework evaluation")
    print("  • visualizations/ - 14 publication-ready figures")
    print("\nReview your results in the visualizations/ folder!")

def main():
    print("Monitoring survey progress...")
    print("Will automatically run analyses when survey completes.")
    print("Press Ctrl+C to stop monitoring.\n")

    check_interval = 60  # Check every 60 seconds

    try:
        while True:
            if check_survey_complete():
                run_analyses()
                break

            # Show current progress if file exists
            if os.path.exists('full_results.csv'):
                with open('full_results.csv', 'r') as f:
                    lines = sum(1 for _ in f) - 1  # Subtract header
                print(f"Progress: {lines}/775 responses ({lines/775*100:.1f}%)")
            else:
                print("Waiting for survey to start...")

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
