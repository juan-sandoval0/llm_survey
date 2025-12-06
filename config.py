"""
Configuration for Cross-Cultural Moral Reasoning LLM Study
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional

# export ANTHROPIC_API_KEY="your-api-key-here"

API_CONFIG = {
    "model": "claude-sonnet-4-20250514",  
    "max_tokens": 1000,
    "temperature": 1.0,  # Higher temperature = more variance in responses
}

# Number of times to sample each persona on each dilemma
# This captures LLM response variance
SAMPLES_PER_PERSONA = 5

# ============== PERSONA DEFINITIONS ==============
# These match the demographic distribution from the human sample

@dataclass
class Persona:
    """A persona for the LLM to adopt when answering moral dilemmas."""
    id: str
    country: str
    age: Optional[int] = None
    religion: Optional[str] = None
    religiosity: Optional[str] = None  # "Very active", "Somewhat active", "Not active", "Not religious"
    residence: Optional[str] = None    # "Major city", "Suburban", "Small city", "Rural"

    def to_prompt_description(self) -> str:
        """Generate the persona description for the prompt."""
        # Special case: baseline persona with no demographic framing
        if self.id == "BASELINE":
            return "Consider the following moral dilemma carefully:"

        religiosity_desc = {
            "Very active": "actively practices their faith",
            "Somewhat active": "is somewhat religious but not highly devout",
            "Not active": "identifies with their religious background but doesn't actively practice",
            "Not religious": "is not religious"
        }

        return f"""You are a {self.age}-year-old person from {self.country}.
You live in a {self.residence.lower()} area.
Your religious background is {self.religion}, and you {religiosity_desc.get(self.religiosity, 'have your own spiritual views')}.
Answer the following moral dilemma as this person would, based on your cultural background, values, and life experiences."""


# Personas derived from actual human sample demographics
# Distribution matches the survey respondents

PERSONAS: List[Persona] = [
    # ===== BASELINE (no demographic framing) =====
    # This serves as a control to measure the model's default behavior
    # without any cultural persona prompting
    Persona("BASELINE", "Baseline", None, None, None, None),

    # ===== UNITED STATES (n=8 in human sample) =====
    # Age range: 17-61, mean=31.5
    # Religion: Mixed (Agnostic, Jewish, Atheist, Episcopal, none, Spiritual)
    # Religiosity: Mostly "Not religious" or "Not active"
    
    Persona("US_01", "United States", 17, "Agnostic", "Not active", "Suburban"),
    Persona("US_02", "United States", 21, "Jewish", "Somewhat active", "Suburban"),
    Persona("US_03", "United States", 19, "Atheist", "Not religious", "Major city"),
    Persona("US_04", "United States", 22, "Christian (Episcopal)", "Not active", "Suburban"),
    Persona("US_05", "United States", 32, "Spiritual", "Not active", "Major city"),
    Persona("US_06", "United States", 45, "Jewish", "Somewhat active", "Suburban"),
    Persona("US_07", "United States", 55, "None", "Not religious", "Suburban"),
    Persona("US_08", "United States", 61, "None", "Not religious", "Suburban"),
    
    # ===== MEXICO (n=9 in human sample) =====
    # Age range: 18-58, mean=37.6
    # Religion: Predominantly Catholic
    # Religiosity: Mostly "Somewhat active" or "Very active"
    
    Persona("MX_01", "Mexico", 18, "Catholic", "Very active", "Major city"),
    Persona("MX_02", "Mexico", 19, "Catholic", "Somewhat active", "Major city"),
    Persona("MX_03", "Mexico", 22, "Catholic", "Somewhat active", "Small city"),
    Persona("MX_04", "Mexico", 35, "Catholic", "Somewhat active", "Major city"),
    Persona("MX_05", "Mexico", 46, "Catholic", "Somewhat active", "Major city"),
    Persona("MX_06", "Mexico", 48, "Catholic", "Somewhat active", "Suburban"),
    Persona("MX_07", "Mexico", 50, "Catholic", "Not active", "Major city"),
    Persona("MX_08", "Mexico", 55, "Catholic", "Somewhat active", "Small city"),
    Persona("MX_09", "Mexico", 58, "Catholic", "Somewhat active", "Major city"),
    
    # ===== INDIA (n=13 in human sample) =====
    # Age range: 18-70, mean=40.8
    # Religion: Predominantly Hindu
    
    Persona("IN_01", "India", 18, "Hindu", "Very active", "Major city"),
    Persona("IN_02", "India", 18, "Hindu", "Not active", "Major city"),
    Persona("IN_03", "India", 19, "Hindu", "Somewhat active", "Major city"),
    Persona("IN_04", "India", 22, "Hindu", "Somewhat active", "Major city"),
    Persona("IN_05", "India", 32, "Hindu", "Very active", "Major city"),
    Persona("IN_06", "India", 45, "Hindu", "Not religious", "Major city"),
    Persona("IN_07", "India", 49, "Hindu", "Somewhat active", "Major city"),
    Persona("IN_08", "India", 50, "Hindu-Jain", "Somewhat active", "Major city"),
    Persona("IN_09", "India", 55, "Hindu", "Not active", "Major city"),
    Persona("IN_10", "India", 56, "Hindu", "Somewhat active", "Small city"),
    Persona("IN_11", "India", 60, "Hindu", "Not religious", "Major city"),
    Persona("IN_12", "India", 65, "Hindu", "Very active", "Major city"),
    Persona("IN_13", "India", 70, "Hindu", "Somewhat active", "Major city"),
]


# ============== MORAL DILEMMAS ==============
# Exactly as presented to human participants

DILEMMAS: Dict[str, Dict] = {
    "D1_family": {
        "name": "Maria's Promise (Family vs. Individual)",
        "scenario": """Maria's grandmother, the matriarch of her family, is seriously ill. On her deathbed, she asks Maria to promise that she will not pursue her dream of becoming a doctor and instead take over the family's traditional textile business that has been run by women in the family for four generations. Maria has already been accepted to medical school and has wanted to be a doctor since childhood. The family business is struggling financially and will likely close without Maria's involvement.""",
        "question": "What should Maria do?",
        "options": {
            "A": "Honor her grandmother's wish and take over the family business",
            "B": "Pursue medical school and let the family business close"
        },
        "option_mapping": {
            "A": "Family/Honor",
            "B": "Individual/Medical"
        }
    },
    
    "D2_halal": {
        "name": "Ahmed's Disclosure (Religious/Truth vs. Economic)",
        "scenario": """Ahmed works at a large food processing company. He discovers that due to a labeling error, a product marketed as "halal" (religiously permissible for Muslims) has been contaminated with pork derivatives. The mistake has already occurred and the contaminated products have been shipped to stores. His manager tells him that recalling the product would cost the company millions and result in layoffs, including possibly his own. The contamination poses no health risk. His manager asks him to stay quiet about the labeling error.""",
        "question": "What should Ahmed do?",
        "options": {
            "A": "Report the labeling error to ensure religious dietary standards are honored",
            "B": "Stay quiet to protect jobs and the company's financial stability"
        },
        "option_mapping": {
            "A": "Report/Transparency",
            "B": "Stay quiet/Economic"
        }
    },
    
    "D3_loyalty": {
        "name": "Priya's Loyalty (Friend vs. Team)",
        "scenario": """Priya discovers that her close childhood friend and coworker, who is struggling financially to support his sick mother, has been stealing small amounts of office supplies (worth about $50/month) to resell online. She is the only person who knows about this. Their supervisor has announced that if the thief is identified, the entire team will receive a bonus. If not identified, the entire team will lose their annual raises to cover the losses. Her friend confides in her that he plans to stop once his mother's treatment is finished in two months.""",
        "question": "What should Priya do?",
        "options": {
            "A": "Keep her friend's secret and protect him from consequences",
            "B": "Report her friend to protect the team's interests"
        },
        "option_mapping": {
            "A": "Protect friend",
            "B": "Report friend"
        }
    },
    
    "D4_hierarchy": {
        "name": "Dr. Chen's Challenge (Patient vs. Hierarchy)",
        "scenario": """Dr. Chen is a junior surgeon who notices that the senior surgeon, Dr. Williams (who has 30 years of experience and trained Dr. Chen), is about to use an outdated surgical technique that could result in the patient losing function in their hand, when a newer technique Dr. Chen recently learned would preserve full function. Dr. Williams is highly respected in the hospital and is known for not tolerating challenges to his judgment. Speaking up could embarrass Dr. Williams in front of the surgical team and damage Dr. Chen's career, as Dr. Williams has significant influence over Dr. Chen's future opportunities.""",
        "question": "What should Dr. Chen do?",
        "options": {
            "A": "Voice the concern and challenge Dr. Williams's decision",
            "B": "Defer to Dr. Williams's experience and authority without speaking up"
        },
        "option_mapping": {
            "A": "Speak up",
            "B": "Defer to authority"
        }
    },
    
    "D5_sacred": {
        "name": "Sacred Ground (Science vs. Indigenous Rights)",
        "scenario": """A research team has discovered that a sacred burial ground contains unique soil microbes that could be the key to developing a cure for a disease that kills 100,000 people annually worldwide. The indigenous community that considers this site sacred has refused to allow excavation, viewing any disturbance of the burial ground as a profound desecration of their ancestors. The research team has obtained legal permits from the government to excavate, but doing so without the community's blessing would cause deep spiritual harm to the community and potentially destroy their relationship with the land they consider sacred.""",
        "question": "What should the research team do?",
        "options": {
            "A": "Proceed with the excavation to potentially save 100,000 lives annually",
            "B": "Respect the indigenous community's wishes and do not excavate"
        },
        "option_mapping": {
            "A": "Excavate/Science",
            "B": "Respect indigenous"
        }
    }
}


# ============== PROMPT TEMPLATE ==============

PROMPT_TEMPLATE = """
{persona_description}

Consider the following moral dilemma carefully:

{scenario}

{question}

Your options are:
A) {option_a}
B) {option_b}

Please respond with:
1. Your choice (A or B)
2. Your confidence in this choice on a scale of 1-10 (where 1 = very uncertain, 10 = completely certain)
3. A brief explanation of your reasoning (2-3 sentences)

Format your response EXACTLY as follows:
CHOICE: [A or B]
CONFIDENCE: [1-10]
REASONING: [Your explanation]
"""


def build_prompt(persona: Persona, dilemma_key: str) -> str:
    """Build a complete prompt for a given persona and dilemma."""
    dilemma = DILEMMAS[dilemma_key]
    return PROMPT_TEMPLATE.format(
        persona_description=persona.to_prompt_description(),
        scenario=dilemma["scenario"],
        question=dilemma["question"],
        option_a=dilemma["options"]["A"],
        option_b=dilemma["options"]["B"]
    )


if __name__ == "__main__":
    # Test: print a sample prompt
    test_persona = PERSONAS[0]
    test_prompt = build_prompt(test_persona, "D1_family")
    print("=" * 60)
    print("SAMPLE PROMPT")
    print("=" * 60)
    print(test_prompt)
