#!/usr/bin/env python3
"""
CAT Dimension Scorer

Scores meditation instruction texts on Jose's 10 CAT dimensions using
regex-based pattern matching and keyword analysis.

This is a first-pass heuristic scorer. The plan is to validate these scores
against Jose's manual labels, then optionally move to LLM-based scoring
for higher accuracy.

Dimensions:
  D1. Attentional Constraint (AC) [0-100]
  D2. Somatic Engagement (SE) [0-100]
  D3. Startup Modality (categorical)
  D4. Object Nature (categorical, multi-valued)
  D5. Object Density (OD) [0-100]
  D6. Temporal Dynamics (TD) [0-100]
  D7. Affective Cultivation (AffC) [0-100]
  D8. Interoceptive Demand (ID) [0-100]
  D9. Metacognitive Load (ML) [0-100]
  D10. Relational Orientation (RO) [0-100]
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Pattern dictionaries ──────────────────────────────────────────────

# D1: Attentional Constraint
AC_OPEN = [
    r"let\s+(your\s+)?mind\s+rest",
    r"choiceless\s+awareness",
    r"whatever\s+arises",
    r"notice\s+whatever",
    r"open\s+awareness",
    r"let\s+go\s+of\s+(any\s+)?control",
    r"no\s+need\s+to\s+(focus|concentrate|direct)",
    r"simply\s+be\s+(present|aware|here)",
    r"rest\s+in\s+(awareness|presence|being)",
    r"allow\s+(everything|anything|all)",
    r"without\s+(trying|effort|directing)",
    r"nowhere\s+to\s+go",
    r"nothing\s+to\s+do",
    r"let\s+thoughts?\s+(come\s+and\s+go|pass|float|drift)",
    r"spacious\s+awareness",
]

AC_SOFT_FIELD = [
    r"notice\s+what(ever)?\s+(is\s+)?happening",
    r"observe\s+(what|whatever)",
    r"aware\s+of\s+whatever",
    r"gently\s+(notice|observe|attend)",
    r"open\s+monitoring",
    r"noting\s+practice",
    r"label\s+(each\s+)?(thought|sensation|experience)",
    r"name\s+what\s+(you\s+)?(notice|observe|feel)",
]

AC_GUIDED_FIELD = [
    r"move\s+(your\s+)?attention\s+(to|through|from)",
    r"shift\s+(your\s+)?(attention|focus|awareness)\s+to",
    r"bring\s+(your\s+)?(attention|awareness|focus)\s+to",
    r"direct\s+(your\s+)?(attention|awareness)",
    r"body\s+scan",
    r"scan\s+(through\s+)?(your\s+)?body",
    r"now\s+(move|turn|shift|bring)\s+(to|your)",
]

AC_FIRM_ANCHOR = [
    r"return\s+to\s+(the\s+)?breath",
    r"come\s+back\s+to\s+(the\s+)?(breath|anchor|object)",
    r"keep\s+(your\s+)?(attention|focus|awareness)\s+on",
    r"maintain\s+(your\s+)?(attention|focus)",
    r"stay\s+with\s+(the\s+)?(breath|sensation)",
    r"anchor\s+(your\s+)?(attention|awareness)",
    r"whenever\s+(you\s+)?notice\s+(you.ve\s+)?(wandered|drifted)",
    r"gently\s+bring\s+(it|your\s+attention)\s+back",
]

AC_LOCKED = [
    r"hold\s+(your\s+)?(attention|focus|gaze)\s+(on|at|steady)",
    r"fix(ate)?\s+(your\s+)?(attention|focus|gaze)",
    r"without\s+wavering",
    r"unwavering\s+(focus|attention|concentration)",
    r"single[- ]?point(ed)?\s+(focus|concentration|attention)",
    r"lock\s+(your\s+)?(attention|focus)",
    r"do\s+not\s+let\s+(your\s+)?(mind|attention)\s+(wander|move|drift)",
    r"concentrate\s+(intensely|deeply|fully)\s+on",
    r"trataka",
    r"kasina",
]

# D2: Somatic Engagement
SE_COGNITIVE = [
    r"contemplate\s+(the\s+)?(meaning|nature|concept|idea)",
    r"reflect\s+on\s+(the\s+)?(meaning|nature|concept|question)",
    r"think\s+about",
    r"consider\s+(the\s+)?(question|idea|meaning)",
    r"koan",
    r"analytical\s+meditation",
    r"intellectual(ly)?",
    r"ponder",
]

SE_LIGHT_SOMATIC = [
    r"notice\s+any\s+sensations?\s+(that\s+)?arise",
    r"aware\s+of\s+(any\s+)?(sensations?|feelings?)\s+in\s+(your\s+)?body",
    r"if\s+you\s+notice\s+(any\s+)?(tension|tightness|sensation)",
]

SE_BODY_COEQUAL = [
    r"observe\s+(the\s+)?breath\s+and",
    r"breath\s+and\s+(the\s+)?(thoughts?|feelings?|sensations?)",
    r"body\s+and\s+mind",
    r"sensations?\s+and\s+thoughts?",
    r"notice\s+both",
]

SE_BODY_DOMINANT = [
    r"feel\s+(the\s+)?sensations?\s+in\s+(each|every|your)",
    r"body\s+scan",
    r"scan\s+(through\s+)?(your\s+)?body",
    r"feel\s+(the\s+)?(weight|heaviness|pressure|warmth|tingling)",
    r"notice\s+(the\s+)?sensations?\s+in\s+(your\s+)?(feet|legs|hands|arms|torso|head|face|chest|belly|abdomen|back|shoulders|neck)",
    r"relax\s+(your\s+)?(muscles?|body|face|jaw|shoulders?|neck|arms?|hands?|legs?|feet)",
]

SE_FULLY_SOMATIC = [
    r"feel\s+(the\s+)?weight\s+of\s+(your\s+)?body",
    r"nothing\s+(else|but\s+(the\s+)?body|but\s+sensation)",
    r"only\s+(the\s+)?(body|sensation|feeling)",
    r"entire(ly)?\s+(in\s+)?(your\s+)?(body|somatic)",
    r"pure(ly)?\s+(body|somatic|physical)\s+(awareness|sensation|experience)",
]

# D3: Startup Modality
STARTUP_BREATH = [
    r"(take|begin\s+with)\s+(a\s+few|three|several|some)\s+deep\s+breaths?",
    r"start\s+(by|with)\s+(taking\s+)?deep\s+breaths?",
    r"(let.s|let\s+us)\s+begin\s+(with|by\s+taking)\s+(a\s+few\s+)?deep\s+breaths?",
    r"inhale\s+deeply",
    r"pranayama",
    r"counted\s+breaths?",
]

STARTUP_BODY_SCAN = [
    r"(start|begin)\s+(by|with)\s+(a\s+)?body\s+scan",
    r"(start|begin)\s+(by\s+)?(scanning|noticing)\s+(your\s+)?body",
    r"start\s+(at|from|with)\s+(the\s+)?(top\s+of\s+your\s+head|your\s+feet|your\s+toes)",
]

STARTUP_MOVEMENT = [
    r"(start|begin)\s+(by|with)\s+(gentle\s+)?(stretching|moving|walking)",
    r"(start|begin)\s+(by|with)\s+(a\s+)?(gentle\s+)?movement",
    r"walking\s+meditation",
    r"mudra",
    r"(begin|start)\s+(by|with)\s+(some\s+)?gentle\s+(stretches|movements)",
]

STARTUP_RELAXATION = [
    r"(start|begin)\s+(by|with)\s+(progressive\s+)?(muscle\s+)?relaxation",
    r"tense\s+and\s+(then\s+)?relax",
    r"progressive\s+muscle\s+relaxation",
    r"let\s+(your\s+)?body\s+(become\s+)?(heavy|relaxed|soft|loose)",
    r"relax\s+(each|every)\s+(part|muscle|area)",
    r"allow\s+(yourself\s+to\s+)?(relax|soften|let\s+go)",
]

# D4: Object Nature
OBJ_SOMATIC = [
    r"breath\b", r"breathing", r"inhale", r"exhale",
    r"body\s+sensation", r"bodily\s+sensation",
    r"sensation\s+in\s+(your\s+)?(body|hands?|feet|legs?|arms?)",
    r"nostrils?", r"abdomen", r"belly", r"chest",
    r"rise\s+and\s+fall", r"physical\s+sensation",
]

OBJ_VERBAL = [
    r"mantra", r"repeat\s+(the\s+)?(word|phrase|mantra)",
    r"affirmation", r"prayer",
    r"say\s+(to\s+yourself|silently|internally|quietly)",
    r"recit(e|ing)",
]

OBJ_VISUAL = [
    r"visuali[sz](e|ation)", r"imagine\s+(a|the|your)",
    r"picture\s+(a|the|your|in\s+your\s+mind)",
    r"see\s+(in\s+your\s+mind|with\s+your\s+mind.s\s+eye)",
    r"mental\s+image",
    r"kasina", r"mandala", r"candle\s+flame",
]

OBJ_AUDITORY = [
    r"listen\s+to\s+(the\s+)?(sound|sounds|silence|bell|music)",
    r"sound\s+(of|around)", r"hear\s+(the\s+)?sound",
    r"ambient\s+(sound|noise)",
    r"notice\s+(the\s+)?sounds?\s+(around|in\s+the\s+room)",
]

OBJ_COGNITIVE = [
    r"thought\s+as\s+(an?\s+)?object",
    r"observe\s+(your\s+)?thoughts?",
    r"watch\s+(your\s+)?thoughts?",
    r"koan", r"contemplate\s+(the\s+)?(question|meaning)",
    r"reflect\s+on",
]

OBJ_AFFECTIVE = [
    r"(feeling|emotion)\s+(of|as)\s+(an?\s+)?object",
    r"cultivate\s+(the\s+)?(feeling|emotion|sense)\s+of",
    r"generate\s+(a\s+)?(feeling|sense|emotion)\s+of",
    r"metta", r"loving[- ]?kindness",
    r"compassion\s+for",
    r"may\s+(you|I|all\s+beings|they)\s+be\s+(happy|peaceful|safe|free|well|healthy)",
]

OBJ_NONE = [
    r"choiceless\s+awareness",
    r"no\s+(particular\s+)?object",
    r"open\s+awareness",
    r"just\s+(sit|be|rest)",
    r"no\s+need\s+to\s+focus\s+on\s+anything",
]

# D7: Affective Cultivation
AFFC_NEUTRAL = [
    r"simply\s+observe",
    r"non[- ]?judg(ment|ing|mental)",
    r"without\s+judg(ment|ing)",
    r"neutral(ly)?",
    r"objectively\s+observe",
    r"note\s+it\s+and\s+(let\s+it\s+)?(go|pass)",
]

AFFC_GENTLE_WARMTH = [
    r"(bring|add)\s+a\s+(quality|sense|touch)\s+of\s+(kindness|warmth|gentleness|softness|care)",
    r"gentle\s+(kindness|warmth|care|compassion)",
    r"kind(ly|ness)\s+(to|toward)\s+(yourself|your)",
    r"self[- ]?compassion",
    r"be\s+gentle\s+with\s+yourself",
    r"treat\s+yourself\s+with\s+kindness",
]

AFFC_ACTIVE_CULTIVATION = [
    r"generate\s+(a\s+)?(feeling|sense|emotion)\s+of\s+(compassion|love|joy|peace|gratitude)",
    r"cultivate\s+(a\s+)?(feeling|sense|emotion)\s+of",
    r"breathe\s+(in|out)\s+(love|compassion|peace|joy|gratitude)",
    r"fill\s+(yourself|your\s+(heart|body|being))\s+with\s+(love|compassion|peace|joy|gratitude|warmth)",
]

AFFC_EMOTION_PRIMARY = [
    r"metta", r"loving[- ]?kindness",
    r"tonglen",
    r"feel\s+(love|compassion|joy|gratitude)\s+(radiating|flowing|spreading|filling)",
    r"(love|compassion)\s+from\s+(your\s+)?heart",
    r"send\s+(love|compassion|healing|kindness)\s+to",
    r"may\s+(you|I|they|all\s+beings)\s+be",
]

AFFC_INTENSE = [
    r"devotion",
    r"bhakti",
    r"let\s+(love|devotion|gratitude)\s+fill\s+every\s+(cell|part|fiber)",
    r"surrender\s+to\s+(love|the\s+divine|god|grace)",
    r"overwhelming\s+(love|gratitude|joy|devotion)",
]

# D8: Interoceptive Demand
ID_NONE = [
    r"mantra\s+repetition",
    r"repeat\s+(the\s+)?(word|phrase|mantra)",
    r"visuali[sz](e|ation)",
]

ID_PERIPHERAL = [
    r"notice\s+if\s+(you.re|you\s+are)\s+(comfortable|tense|relaxed)",
    r"check\s+in\s+with\s+(your\s+)?body",
    r"how\s+does\s+(your\s+)?body\s+feel",
]

ID_MODERATE = [
    r"rise\s+and\s+fall\s+of\s+(the\s+|your\s+)?abdomen",
    r"movement\s+of\s+(the\s+|your\s+)?(breath|chest|belly)",
    r"notice\s+(the\s+)?(rhythm|pace|quality)\s+of\s+(your\s+)?breath",
    r"feel\s+(the\s+)?air\s+(entering|leaving|at\s+(the\s+)?(nostrils|tip\s+of))",
]

ID_HIGH = [
    r"subtle\s+(tingling|vibration|sensation|pulsing|energy)",
    r"pulse\s+(in|at)\s+(your\s+)?(wrist|fingertips|temple)",
    r"feel\s+(the\s+)?tingling",
    r"detect\s+(the\s+)?(boundary|edge|transition)",
    r"fine[- ]?grained\s+(attention|awareness|sensitivity)",
    r"micro[- ]?sensations?",
]

ID_MAXIMAL = [
    r"detect\s+(the\s+)?boundary\s+between",
    r"sweeping\s+(attention|awareness)\s+(through|across)",
    r"cell[- ]?by[- ]?cell",
    r"atomic\s+sensation",
    r"dissolved?\s+(into\s+)?(vibration|sensation)",
]

# D9: Metacognitive Load
ML_NONE = [
    r"just\s+repeat\s+(the\s+)?mantra",
    r"no\s+need\s+to\s+(monitor|watch|track)",
    r"don.t\s+worry\s+about\s+(how|whether)",
]

ML_LIGHT = [
    r"when(ever)?\s+you\s+notice\s+(you.ve\s+)?(wandered|drifted|been\s+distracted)",
    r"if\s+you\s+find\s+(your\s+)?mind\s+(has\s+)?(wandered|drifted)",
    r"gently\s+(bring|return|come\s+back)",
]

ML_ACTIVE = [
    r"notice\s+whether\s+(your\s+)?(attention|mind|focus)\s+is\s+(stable|steady|wavering|wandering|distracted)",
    r"quality\s+of\s+(your\s+)?(attention|awareness|focus|concentration)",
    r"how\s+(focused|distracted|stable)\s+(are\s+you|is\s+your)",
    r"aware\s+of\s+(the\s+)?state\s+of\s+(your\s+)?(mind|attention)",
]

ML_MONITORING = [
    r"aware\s+of\s+(the\s+)?quality\s+of\s+(your\s+)?(awareness|attention|consciousness)\s+itself",
    r"watch\s+(the\s+)?watcher",
    r"observe\s+(the\s+)?observer",
    r"awareness\s+(of|watching|observing)\s+(itself|awareness)",
    r"who\s+is\s+(watching|observing|aware)",
    r"metacogniti",
]

# D10: Relational Orientation
RO_SOLITARY = [
    # Default: no mentions of others
]

RO_ABSTRACT = [
    r"may\s+all\s+beings",
    r"all\s+(living\s+)?(beings|creatures|sentient)",
    r"the\s+world",
    r"everyone\s+everywhere",
    r"all\s+of\s+humanity",
    r"universal\s+(love|compassion|peace|kindness)",
]

RO_SPECIFIC = [
    r"bring\s+to\s+mind\s+(a|someone|a\s+loved\s+one|a\s+friend|a\s+person|a\s+neutral)",
    r"think\s+of\s+(a\s+)?(loved\s+one|friend|family|person|someone)",
    r"(loved\s+one|friend|neutral\s+person|difficult\s+person|enemy|stranger)",
    r"picture\s+(a\s+)?(specific\s+)?person",
    r"someone\s+you\s+(love|care\s+about|know|find\s+difficult)",
]

RO_ACTIVE_RELATIONAL = [
    r"take\s+on\s+(their|his|her)\s+suffering",
    r"tonglen",
    r"breathe\s+in\s+(their|his|her)\s+(pain|suffering|difficulty)",
    r"breathe\s+out\s+(love|compassion|healing|peace|relief)\s+(to|toward|for)\s+(them|him|her)",
    r"send\s+(love|compassion|healing|peace)\s+to\s+(this\s+person|them|him|her)",
    r"imagine\s+(their|his|her)\s+(suffering|pain|difficulty)",
]

RO_INTERPERSONAL = [
    r"dyadic\s+meditation",
    r"look\s+into\s+(each\s+other|your\s+partner)",
    r"with\s+(your\s+)?partner",
    r"face\s+(each\s+other|your\s+partner)",
    r"relational\s+mindfulness",
]


def _count_matches(text: str, patterns: list[str]) -> int:
    """Count total pattern matches in text (case-insensitive)."""
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, text, re.IGNORECASE))
    return total


def _has_match(text: str, patterns: list[str]) -> bool:
    """Check if any pattern matches in text."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _score_by_levels(text: str, levels: list[tuple[list[str], int]],
                     default: int = 0) -> int:
    """Score by checking from highest to lowest level, weighted by match count.
    Higher levels get extra weight (1.5x) because they are more specific/diagnostic."""
    total_weight = 0
    weighted_score = 0

    for patterns, score in levels:
        count = _count_matches(text, patterns)
        if count > 0:
            # Higher scores get a 1.5x weight boost -- they are more
            # specific patterns and should dominate when present
            boost = 1.0 + (score / 200.0)  # 1.0 at score=0, 1.5 at score=100
            w = count * boost
            total_weight += w
            weighted_score += w * score

    if total_weight == 0:
        return default

    return int(round(weighted_score / total_weight))


@dataclass
class CATScore:
    """CAT dimension scores for a single meditation text."""
    title: str = ""
    source: str = ""

    # Continuous dimensions [0-100]
    d1_attentional_constraint: int = 50
    d2_somatic_engagement: int = 50
    d5_object_density: int = 20
    d6_temporal_dynamics: int = 25
    d7_affective_cultivation: int = 0
    d8_interoceptive_demand: int = 25
    d9_metacognitive_load: int = 25
    d10_relational_orientation: int = 0

    # Categorical dimensions
    d3_startup_modality: str = "none"
    d4_object_nature: list = field(default_factory=lambda: ["somatic"])

    # Metadata
    word_count: int = 0
    confidence: str = "low"  # low, medium, high

    def to_dict(self):
        return asdict(self)


def score_d1_attentional_constraint(text: str) -> int:
    """D1: How much does the instruction narrow the attentional aperture?"""
    levels = [
        (AC_LOCKED, 100),
        (AC_FIRM_ANCHOR, 75),
        (AC_GUIDED_FIELD, 50),
        (AC_SOFT_FIELD, 25),
        (AC_OPEN, 0),
    ]
    return _score_by_levels(text, levels, default=50)


def score_d2_somatic_engagement(text: str) -> int:
    """D2: How much does the practice recruit bodily awareness?"""
    levels = [
        (SE_FULLY_SOMATIC, 100),
        (SE_BODY_DOMINANT, 75),
        (SE_BODY_COEQUAL, 50),
        (SE_LIGHT_SOMATIC, 25),
        (SE_COGNITIVE, 0),
    ]
    return _score_by_levels(text, levels, default=50)


def score_d3_startup_modality(text: str) -> str:
    """D3: What does the practice use to initiate the meditative state?
    Only checks the first ~20% of the text (the startup phase)."""
    # Focus on the first portion of the text
    words = text.split()
    startup_text = " ".join(words[:max(100, len(words) // 5)])

    checks = [
        (STARTUP_BREATH, "breath_regulation"),
        (STARTUP_BODY_SCAN, "body_scan"),
        (STARTUP_RELAXATION, "relaxation_induction"),
        (STARTUP_MOVEMENT, "physical_movement"),
    ]

    best = ("none", 0)
    for patterns, label in checks:
        count = _count_matches(startup_text, patterns)
        if count > best[1]:
            best = (label, count)

    return best[0]


def score_d4_object_nature(text: str) -> list:
    """D4: The primary object(s) of attention."""
    checks = [
        (OBJ_SOMATIC, "somatic"),
        (OBJ_VERBAL, "verbal"),
        (OBJ_VISUAL, "visual"),
        (OBJ_AUDITORY, "auditory"),
        (OBJ_COGNITIVE, "cognitive"),
        (OBJ_AFFECTIVE, "affective"),
        (OBJ_NONE, "none/open"),
    ]

    results = []
    for patterns, label in checks:
        count = _count_matches(text, patterns)
        if count >= 2:  # Require at least 2 matches to count
            results.append((label, count))

    if not results:
        # If no strong matches, check for any match
        for patterns, label in checks:
            if _has_match(text, patterns):
                results.append((label, 1))

    if not results:
        return ["somatic"]  # Default

    # Sort by frequency, return labels
    results.sort(key=lambda x: -x[1])
    return [r[0] for r in results]


def score_d5_object_density(text: str) -> int:
    """D5: How many distinct objects of attention does the practice employ?"""
    # Count distinct body parts mentioned (proxy for object density)
    body_parts = [
        r"\b(head|scalp|forehead|temples?|crown)\b",
        r"\b(eyes?|eyelids?)\b",
        r"\b(face|cheeks?|jaw|chin|mouth|lips?|tongue)\b",
        r"\b(neck|throat)\b",
        r"\b(shoulders?)\b",
        r"\b(arms?|upper\s+arms?|forearms?|elbows?|wrists?)\b",
        r"\b(hands?|fingers?|fingertips?|palms?)\b",
        r"\b(chest|heart\s+center|heart\s+area|ribcage)\b",
        r"\b(belly|abdomen|stomach|navel|solar\s+plexus)\b",
        r"\b(back|upper\s+back|lower\s+back|spine)\b",
        r"\b(hips?|pelvis)\b",
        r"\b(legs?|thighs?|knees?|calves?|shins?)\b",
        r"\b(feet|foot|toes?|soles?|ankles?)\b",
    ]

    distinct_objects = sum(1 for p in body_parts if re.search(p, text, re.IGNORECASE))

    # Also check for non-somatic objects
    other_objects = [
        r"\bbreath\b", r"\bsound", r"\bmantra\b", r"\bvisuali",
        r"\bthought", r"\bemotion", r"\bfeeling",
    ]
    distinct_objects += sum(1 for p in other_objects if re.search(p, text, re.IGNORECASE))

    # Check for open awareness (high density)
    if _has_match(text, OBJ_NONE):
        return 85

    # Map count to 0-100
    if distinct_objects == 0:
        return 0
    elif distinct_objects == 1:
        return 20
    elif distinct_objects <= 3:
        return 35
    elif distinct_objects <= 5:
        return 50
    elif distinct_objects <= 8:
        return 65
    elif distinct_objects <= 12:
        return 75
    else:
        return 90


def score_d6_temporal_dynamics(text: str) -> int:
    """D6: How much does the attentional task change over the course?"""
    # Split text into quarters and check how different they are
    words = text.split()
    n = len(words)
    if n < 50:
        return 25  # Too short to judge

    quarters = [
        " ".join(words[:n//4]),
        " ".join(words[n//4:n//2]),
        " ".join(words[n//2:3*n//4]),
        " ".join(words[3*n//4:]),
    ]

    # Count transition markers
    transition_patterns = [
        r"\bnow\b", r"\bnext\b", r"\band\s+now\b",
        r"\bshift\b", r"\bmove\s+(to|your)\b",
        r"\bturn\s+(to|your)\b", r"\bbring\s+(your\s+)?attention\s+to\b",
        r"\blet\s+go\s+of\b", r"\brelease\b",
        r"\bwhen\s+you.re\s+ready\b", r"\bslowly\b",
        r"\bgently\s+(shift|move|turn|bring)\b",
        r"\bnew\s+(phase|stage|part|section)\b",
    ]

    transition_count = _count_matches(text, transition_patterns)

    # Normalize by text length (transitions per 100 words)
    transition_density = (transition_count / max(n, 1)) * 100

    if transition_density < 0.5:
        return 10
    elif transition_density < 1.5:
        return 25
    elif transition_density < 3.0:
        return 50
    elif transition_density < 5.0:
        return 75
    else:
        return 90


def score_d7_affective_cultivation(text: str) -> int:
    """D7: How much does the practice actively cultivate emotional states?"""
    levels = [
        (AFFC_INTENSE, 100),
        (AFFC_EMOTION_PRIMARY, 75),
        (AFFC_ACTIVE_CULTIVATION, 50),
        (AFFC_GENTLE_WARMTH, 25),
        (AFFC_NEUTRAL, 0),
    ]
    return _score_by_levels(text, levels, default=10)


def score_d8_interoceptive_demand(text: str) -> int:
    """D8: How much does the practice require sensitivity to internal body signals?"""
    levels = [
        (ID_MAXIMAL, 100),
        (ID_HIGH, 75),
        (ID_MODERATE, 50),
        (ID_PERIPHERAL, 25),
        (ID_NONE, 0),
    ]
    return _score_by_levels(text, levels, default=25)


def score_d9_metacognitive_load(text: str) -> int:
    """D9: How much does the practice ask to monitor one's own mental process?"""
    levels = [
        (ML_MONITORING, 100),
        (ML_ACTIVE, 75),
        (ML_LIGHT, 25),
        (ML_NONE, 0),
    ]
    return _score_by_levels(text, levels, default=15)


def score_d10_relational_orientation(text: str) -> int:
    """D10: How much does the practice involve others?"""
    levels = [
        (RO_INTERPERSONAL, 100),
        (RO_ACTIVE_RELATIONAL, 75),
        (RO_SPECIFIC, 50),
        (RO_ABSTRACT, 25),
    ]
    score = _score_by_levels(text, levels, default=0)
    return score


def score_meditation(text: str, title: str = "", source: str = "") -> CATScore:
    """Score a meditation text on all 10 CAT dimensions."""
    score = CATScore(
        title=title,
        source=source,
        word_count=len(text.split()),
        d1_attentional_constraint=score_d1_attentional_constraint(text),
        d2_somatic_engagement=score_d2_somatic_engagement(text),
        d3_startup_modality=score_d3_startup_modality(text),
        d4_object_nature=score_d4_object_nature(text),
        d5_object_density=score_d5_object_density(text),
        d6_temporal_dynamics=score_d6_temporal_dynamics(text),
        d7_affective_cultivation=score_d7_affective_cultivation(text),
        d8_interoceptive_demand=score_d8_interoceptive_demand(text),
        d9_metacognitive_load=score_d9_metacognitive_load(text),
        d10_relational_orientation=score_d10_relational_orientation(text),
    )

    # Confidence based on text length
    if score.word_count < 50:
        score.confidence = "low"
    elif score.word_count < 200:
        score.confidence = "medium"
    else:
        score.confidence = "high"

    return score


def score_from_file(filepath: str) -> Optional[CATScore]:
    """Score a meditation from a text file (expects our standard format)."""
    with open(filepath, "r") as f:
        content = f.read()

    # Parse metadata header if present
    title = ""
    source = ""
    text = content

    if "---" in content:
        parts = content.split("---", 1)
        header = parts[0]
        text = parts[1] if len(parts) > 1 else content

        for line in header.split("\n"):
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Source:"):
                source = line.replace("Source:", "").strip()

    return score_meditation(text, title=title, source=source)


def score_from_json(json_path: str, output_path: str = None):
    """Score all meditations in a JSON file (from scraper output)."""
    with open(json_path, "r") as f:
        meditations = json.load(f)

    scores = []
    for med in meditations:
        score = score_meditation(
            med["text"],
            title=med.get("title", ""),
            source=med.get("source", "")
        )
        scores.append(score.to_dict())

    if output_path is None:
        output_path = json_path.replace(".json", "_scored.json")

    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print(f"Scored {len(scores)} meditations -> {output_path}")
    return scores


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        if input_path.endswith(".json"):
            score_from_json(input_path, output_path)
        else:
            score = score_from_file(input_path)
            if score:
                print(json.dumps(score.to_dict(), indent=2))
    else:
        # Demo with sample texts
        samples = [
            ("Classic Vipassana", "Sit comfortably and close your eyes. Take a few deep breaths. Now bring your attention to the breath at the nostrils. Feel the air entering and leaving. Whenever you notice the mind has wandered, gently bring your attention back to the breath. Simply observe the natural rhythm of breathing without trying to change it. If thoughts arise, note them and let them pass. Return to the breath."),
            ("Body Scan", "Begin by taking three deep breaths. Now bring your attention to the top of your head. Notice any sensations in your scalp. Feel the weight of your head. Move your attention to your forehead, your temples, your eyes and eyelids. Relax your face, your cheeks, your jaw. Let the muscles of your neck soften. Notice sensations in your shoulders, your upper arms, your elbows, your forearms, your wrists, your hands, your fingertips."),
            ("Loving Kindness", "Close your eyes and settle into a comfortable position. Bring to mind someone you love deeply. Picture this person clearly. Now silently repeat: May you be happy. May you be healthy. May you be safe. May you be at peace. Feel love radiating from your heart center toward this person. Now think of a neutral person -- someone you see regularly but don't know well. Send them the same wishes: May you be happy. May you be healthy."),
        ]

        for title, text in samples:
            score = score_meditation(text, title=title, source="demo")
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")
            print(f"  D1 Attentional Constraint: {score.d1_attentional_constraint}")
            print(f"  D2 Somatic Engagement:     {score.d2_somatic_engagement}")
            print(f"  D3 Startup Modality:       {score.d3_startup_modality}")
            print(f"  D4 Object Nature:          {', '.join(score.d4_object_nature)}")
            print(f"  D5 Object Density:         {score.d5_object_density}")
            print(f"  D6 Temporal Dynamics:       {score.d6_temporal_dynamics}")
            print(f"  D7 Affective Cultivation:  {score.d7_affective_cultivation}")
            print(f"  D8 Interoceptive Demand:   {score.d8_interoceptive_demand}")
            print(f"  D9 Metacognitive Load:     {score.d9_metacognitive_load}")
            print(f"  D10 Relational Orientation: {score.d10_relational_orientation}")
            print(f"  Words: {score.word_count}")
