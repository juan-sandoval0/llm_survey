"""
LLM Cultural Survey Runner
===========================

This script runs the moral dilemmas through the Anthropic API with varied personas
and collects responses for comparison with human baseline data.

Usage:
    python run_survey.py [--samples N] [--output FILE] [--dry-run]

Requirements:
    pip install anthropic pandas tqdm

Set your API key:
    export ANTHROPIC_API_KEY="your-key-here"
"""

import os
import re
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import csv

try:
    import anthropic
except ImportError:
    print("Please install anthropic: pip install anthropic")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

from config import PERSONAS, DILEMMAS, API_CONFIG, SAMPLES_PER_PERSONA, build_prompt, Persona


@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    persona_id: str
    country: str
    age: Optional[int]
    religion: Optional[str]
    religiosity: Optional[str]
    dilemma_id: str
    dilemma_name: str
    sample_num: int
    choice: str  # "A" or "B"
    choice_label: str  # Human-readable label
    confidence: int
    reasoning: str
    raw_response: str
    timestamp: str
    model: str
    parse_success: bool
    error_message: Optional[str] = None


def parse_llm_response(raw_text: str, dilemma_key: str) -> Tuple[str, int, str, bool, str]:
    """
    Parse the structured response from the LLM.
    
    Returns: (choice, confidence, reasoning, success, error_message)
    """
    try:
        # Extract CHOICE
        choice_match = re.search(r'CHOICE:\s*([AB])', raw_text, re.IGNORECASE)
        if not choice_match:
            # Try alternative patterns
            choice_match = re.search(r'\b([AB])\)', raw_text) or re.search(r'choose\s+([AB])', raw_text, re.IGNORECASE)
        
        choice = choice_match.group(1).upper() if choice_match else None
        
        # Extract CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', raw_text, re.IGNORECASE)
        if not conf_match:
            conf_match = re.search(r'confidence[:\s]+(\d+)', raw_text, re.IGNORECASE)
        
        confidence = int(conf_match.group(1)) if conf_match else None
        if confidence and (confidence < 1 or confidence > 10):
            confidence = max(1, min(10, confidence))  # Clamp to valid range
        
        # Extract REASONING
        reason_match = re.search(r'REASONING:\s*(.+?)(?:\n\n|$)', raw_text, re.IGNORECASE | re.DOTALL)
        if not reason_match:
            # Take everything after the confidence as reasoning
            reason_match = re.search(r'CONFIDENCE:\s*\d+\s*\n(.+)', raw_text, re.DOTALL)
        
        reasoning = reason_match.group(1).strip() if reason_match else ""
        
        # Validate
        if choice is None:
            return None, None, "", False, "Could not parse choice (A/B)"
        if confidence is None:
            return choice, 5, reasoning, True, "Could not parse confidence, defaulting to 5"
        
        return choice, confidence, reasoning, True, ""
        
    except Exception as e:
        return None, None, "", False, f"Parse error: {str(e)}"


def call_anthropic_api(prompt: str, config: Dict) -> Tuple[str, Optional[str]]:
    """
    Make a single API call to Anthropic.
    
    Returns: (response_text, error_message)
    """
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    
    try:
        message = client.messages.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            temperature=config.get("temperature", 1.0),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text, None
        
    except anthropic.RateLimitError as e:
        return "", f"Rate limit hit: {e}. Waiting..."
    except anthropic.APIError as e:
        return "", f"API error: {e}"
    except Exception as e:
        return "", f"Unexpected error: {e}"


def run_single_trial(
    persona: Persona,
    dilemma_key: str,
    sample_num: int,
    config: Dict,
    retry_count: int = 3
) -> LLMResponse:
    """Run a single trial (one persona, one dilemma, one sample)."""
    
    prompt = build_prompt(persona, dilemma_key)
    dilemma = DILEMMAS[dilemma_key]
    
    for attempt in range(retry_count):
        raw_response, api_error = call_anthropic_api(prompt, config)
        
        if api_error:
            if "Rate limit" in api_error:
                print(f"\n  Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            else:
                # Return error response
                return LLMResponse(
                    persona_id=persona.id,
                    country=persona.country,
                    age=persona.age,
                    religion=persona.religion,
                    religiosity=persona.religiosity,
                    dilemma_id=dilemma_key,
                    dilemma_name=dilemma["name"],
                    sample_num=sample_num,
                    choice="",
                    choice_label="",
                    confidence=0,
                    reasoning="",
                    raw_response="",
                    timestamp=datetime.now().isoformat(),
                    model=config["model"],
                    parse_success=False,
                    error_message=api_error
                )
        
        # Parse the response
        choice, confidence, reasoning, success, parse_error = parse_llm_response(raw_response, dilemma_key)
        
        choice_label = ""
        if choice:
            choice_label = dilemma["option_mapping"].get(choice, choice)
        
        return LLMResponse(
            persona_id=persona.id,
            country=persona.country,
            age=persona.age,
            religion=persona.religion,
            religiosity=persona.religiosity,
            dilemma_id=dilemma_key,
            dilemma_name=dilemma["name"],
            sample_num=sample_num,
            choice=choice or "",
            choice_label=choice_label,
            confidence=confidence or 0,
            reasoning=reasoning,
            raw_response=raw_response,
            timestamp=datetime.now().isoformat(),
            model=config["model"],
            parse_success=success,
            error_message=parse_error if not success else None
        )
    
    # All retries failed
    return LLMResponse(
        persona_id=persona.id,
        country=persona.country,
        age=persona.age,
        religion=persona.religion,
        religiosity=persona.religiosity,
        dilemma_id=dilemma_key,
        dilemma_name=dilemma["name"],
        sample_num=sample_num,
        choice="",
        choice_label="",
        confidence=0,
        reasoning="",
        raw_response="",
        timestamp=datetime.now().isoformat(),
        model=config["model"],
        parse_success=False,
        error_message="Max retries exceeded"
    )


def run_full_survey(
    personas: List[Persona],
    dilemmas: Dict,
    samples_per_persona: int,
    config: Dict,
    output_file: str,
    dry_run: bool = False
) -> List[LLMResponse]:
    """Run the complete survey across all personas and dilemmas."""
    
    results = []
    total_trials = len(personas) * len(dilemmas) * samples_per_persona
    
    print(f"\n{'='*60}")
    print("CROSS-CULTURAL MORAL REASONING LLM SURVEY")
    print(f"{'='*60}")
    print(f"Personas: {len(personas)}")
    print(f"Dilemmas: {len(dilemmas)}")
    print(f"Samples per persona-dilemma: {samples_per_persona}")
    print(f"Total API calls: {total_trials}")
    print(f"Model: {config['model']}")
    print(f"Temperature: {config.get('temperature', 1.0)}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    if dry_run:
        print("DRY RUN - showing sample prompts only\n")
        for dilemma_key in list(dilemmas.keys())[:2]:
            sample_persona = personas[0]
            print(f"--- {dilemma_key} with {sample_persona.id} ---")
            print(build_prompt(sample_persona, dilemma_key))
            print("\n")
        return []
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("Run: export ANTHROPIC_API_KEY='your-key-here'")
        return []
    
    # Create progress bar
    pbar = tqdm(total=total_trials, desc="Running survey")
    
    # Run trials
    for persona in personas:
        for dilemma_key in dilemmas.keys():
            for sample_num in range(1, samples_per_persona + 1):
                response = run_single_trial(persona, dilemma_key, sample_num, config)
                results.append(response)
                
                # Update progress
                status = "✓" if response.parse_success else "✗"
                pbar.set_postfix({
                    "persona": persona.id,
                    "dilemma": dilemma_key,
                    "status": status
                })
                pbar.update(1)
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
    
    pbar.close()
    
    # Save results
    save_results(results, output_file)
    
    # Print summary
    print_summary(results)
    
    return results


def save_results(results: List[LLMResponse], output_file: str):
    """Save results to CSV file."""
    if not results:
        return
    
    # Convert to dicts
    rows = [asdict(r) for r in results]
    
    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nResults saved to: {output_file}")
    
    # Also save as JSON for easier analysis
    json_file = output_file.replace('.csv', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print(f"JSON backup saved to: {json_file}")


def print_summary(results: List[LLMResponse]):
    """Print a summary of the survey results."""
    if not results:
        return
    
    print(f"\n{'='*60}")
    print("SURVEY SUMMARY")
    print(f"{'='*60}")
    
    total = len(results)
    successful = sum(1 for r in results if r.parse_success)
    print(f"Total responses: {total}")
    print(f"Successfully parsed: {successful} ({successful/total*100:.1f}%)")
    
    # Quick breakdown by country
    print(f"\n--- Responses by Country ---")
    countries = {}
    for r in results:
        if r.country not in countries:
            countries[r.country] = {"total": 0, "choices": {}}
        countries[r.country]["total"] += 1
        if r.choice:
            countries[r.country]["choices"][r.choice] = countries[r.country]["choices"].get(r.choice, 0) + 1
    
    for country, data in countries.items():
        print(f"\n{country}: {data['total']} responses")
        for choice, count in data["choices"].items():
            pct = count / data["total"] * 100
            print(f"  Choice {choice}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run cross-cultural moral reasoning LLM survey")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_PERSONA,
                       help=f"Samples per persona-dilemma (default: {SAMPLES_PER_PERSONA})")
    parser.add_argument("--output", type=str, default="llm_survey_results.csv",
                       help="Output CSV file (default: llm_survey_results.csv)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show sample prompts without making API calls")
    parser.add_argument("--model", type=str, default=API_CONFIG["model"],
                       help=f"Model to use (default: {API_CONFIG['model']})")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for sampling (default: 1.0)")
    
    args = parser.parse_args()
    
    # Update config
    config = API_CONFIG.copy()
    config["model"] = args.model
    config["temperature"] = args.temperature
    
    # Run survey
    results = run_full_survey(
        personas=PERSONAS,
        dilemmas=DILEMMAS,
        samples_per_persona=args.samples,
        config=config,
        output_file=args.output,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
