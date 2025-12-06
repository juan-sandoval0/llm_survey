# Cross-Cultural Moral Reasoning LLM Study

This toolkit tests whether LLMs can accurately simulate culturally-specific moral reasoning by comparing LLM responses to human baseline data collected from participants in the United States, Mexico, and India.

## Quick Start

### 1. Setup

```bash
# Clone or download this directory, then:
cd llm_cultural_survey

# Install dependencies
pip install anthropic pandas numpy scipy tqdm

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 2. Run the Survey

**Dry run (see sample prompts, no API calls):**
```bash
python run_survey.py --dry-run
```

**Full run (default: 5 samples per persona-dilemma):**
```bash
python run_survey.py --output results.csv
```

**Customized run:**
```bash
# Fewer samples (faster, cheaper)
python run_survey.py --samples 3 --output results_quick.csv

# Different model
python run_survey.py --model claude-3-opus-20240229 --output results_opus.csv

# Higher temperature (more variance)
python run_survey.py --temperature 1.2 --output results_high_temp.csv
```

### 3. Analyze Results

```bash
python analyze_results.py --llm-data results.csv --output analysis.json
```

---

## File Structure

```
llm_cultural_survey/
├── config.py           # Personas, dilemmas, prompt templates
├── run_survey.py       # Main survey runner (makes API calls)
├── analyze_results.py  # Comparison analysis (LLM vs human)
└── README.md           # This file
```

### Troubleshooting

**"ANTHROPIC_API_KEY not set"**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

**Rate limiting**
The script automatically waits 60s on rate limits. If persistent, reduce concurrent load or wait.

**Parse errors**
Some responses may not follow the exact format. These are logged and excluded from analysis. Expect ~95%+ success rate.

---

## Disclosure for CS120 class

The survey analysis code (analyze_results and run_survey) was done with the help of Claude Code.
