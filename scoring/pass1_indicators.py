#!/usr/bin/env python3
"""Pass 1: Keyword and pattern detection for CAT dimensions.

Scans meditation transcripts for indicator phrases and computes preliminary
scores for each continuous dimension (D1, D2, D5-D10). Categorical dimensions
(D3, D4) get keyword-based suggestions that Pass 2 (LLM) will finalize.

Usage:
    from scoring.pass1_indicators import score_transcript
    result = score_transcript(transcript_text, transcript_with_timestamps)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IndicatorResult:
    """Result of indicator detection for a single dimension."""
    dimension: str
    low_count: int = 0
    mid_count: int = 0
    high_count: int = 0
    key_phrases: list[str] = field(default_factory=list)
    estimated_score: float | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Indicator phrase sets
# ---------------------------------------------------------------------------

# D1: Attentional Constraint (AC)
D1_LOW = [
    r"let\s+go\s+of",
    r"no\s+need\s+to\s+focus",
    r"whatever\s+arises",
    r"choiceless",
    r"rest\s+in\s+awareness",
    r"spacious\s+awareness",
    r"no\s+particular\s+object",
    r"open\s+awareness",
    r"allow\s+everything",
    r"nothing\s+to\s+do",
    r"no\s+effort",
    r"let\s+the\s+mind\s+rest",
    r"simply\s+be",
    r"just\s+being",
    r"effortless",
    r"no\s+agenda",
]

D1_MID = [
    r"gently\s+bring\s+(?:your\s+)?attention",
    r"move\s+your\s+awareness",
    r"scan\s+(?:through|your|the)",
    r"shift\s+(?:your\s+)?(?:attention|awareness|focus)",
    r"notice\b.*?\bthen\s+notice",
    r"when\s+you(?:'re|\s+are)\s+ready",
    r"softly\s+(?:bring|return|redirect)",
    r"allow\s+your\s+attention\s+to\s+(?:move|drift|shift)",
]

D1_HIGH = [
    r"focus\s+(?:on|your|all)",
    r"concentrate",
    r"hold\s+your\s+attention",
    r"fix\s+your\s+(?:gaze|attention|mind)",
    r"don(?:'t|not)\s+let\s+go",
    r"return\s+immediately",
    r"stay\s+with",
    r"single[- ]?pointed",
    r"one[- ]?pointed",
    r"count\s+(?:your\s+)?breaths?",
    r"count(?:ing)?\s+(?:from|to|back)",
    r"keep\s+(?:your\s+)?(?:attention|focus|mind)\s+(?:on|at|fixed)",
    r"(?:always|immediately)\s+(?:come|bring|return)\s+back",
    r"unwavering",
    r"steady\s+(?:your\s+)?(?:gaze|attention|focus)",
]

# D2: Somatic Engagement (SE)
D2_LOW = [
    r"think\s+about",
    r"contemplate",
    r"reflect\s+on",
    r"consider\s+(?:the|what|how|why)",
    r"imagine\s+(?:a|that|the|yourself)",
    r"visualize",
    r"what\s+is\s+the\s+meaning",
    r"ponder",
    r"analyze",
    r"intellectual",
]

D2_MID = [
    r"notice\s+the\s+breath",
    r"feel\s+the\s+breath",
    r"aware\s+of\s+(?:your\s+)?(?:body|breath|breathing)",
    r"breath\s+(?:in|out|flowing)",
    r"breathing\s+(?:in|out|naturally)",
]

D2_HIGH = [
    r"\bfeel\b",
    r"\bsensation[s]?\b",
    r"\btingling\b",
    r"\bwarmth\b",
    r"\bheaviness\b",
    r"\bmuscle[s]?\b",
    r"\bchest\b",
    r"\bbelly\b",
    r"\bhand[s]?\b",
    r"\bfeet\b",
    r"\bfoot\b",
    r"\bspine\b",
    r"\bshoulders?\b",
    r"\bjaw\b",
    r"\bforehead\b",
    r"\btoes?\b",
    r"\bfinger(?:s|tips)?\b",
    r"\bhips?\b",
    r"\blegs?\b",
    r"\barms?\b",
    r"\bneck\b",
    r"\bstomach\b",
    r"\babdomen\b",
    r"\bphysical\b",
    r"\bbody\s+part",
    r"\bweight\s+of\s+(?:your\s+)?body\b",
]

# D5: Object Density (OD) -- we count distinct attention objects
ATTENTION_OBJECTS = [
    r"\bbreath\b",
    r"\bbreathing\b",
    r"\bbody\b",
    r"\bbody\s+scan\b",
    r"\bsound[s]?\b",
    r"\bthought[s]?\b",
    r"\bemotion[s]?\b",
    r"\bfeeling[s]?\b",
    r"\bmantra\b",
    r"\bphrase[s]?\b",
    r"\bimage\b",
    r"\bvisualization\b",
    r"\blight\b",
    r"\bflame\b",
    r"\bcandle\b",
    r"\bheart\s+center\b",
    r"\bthird\s+eye\b",
    r"\bnavel\b",
    r"\bnose(?:trils?)?\b",
    r"\blips?\b",
    r"\bspace\b",
    r"\bsilence\b",
    r"\bgap\s+between\b",
    r"\bpain\b",
    r"\bdiscomfort\b",
    r"\bcolor\b",
    r"\benergy\b",
    r"\bchakra\b",
    r"\beverything\b",
    r"\ball\s+(?:of\s+)?(?:experience|reality|that\s+is)\b",
]

# D6: Temporal Dynamics (TD) -- phase transition markers
PHASE_MARKERS = [
    r"\bnow\s+(?:let(?:'s)?|we(?:'ll)?|I\s+(?:want|invite)|begin|move|shift|turn|bring|open|expand|release)",
    r"\bnext\b",
    r"\band\s+now\b",
    r"\bmoving\s+(?:on|to|into)\b",
    r"\blet(?:'s)?\s+(?:shift|change|move|transition|turn)\b",
    r"\bthe\s+(?:next|second|third|final)\s+(?:step|stage|phase|part)\b",
    r"\bwhen\s+you(?:'re)?\s+ready\s*,\s*(?:let|begin|open|move|shift|expand)",
    r"\bgradually\b",
    r"\bslowly\s+(?:begin|start|expand|open|shift|bring)",
    r"\bexpand(?:ing)?\s+(?:your\s+)?(?:awareness|attention|field)\b",
    r"\bwiden(?:ing)?\b",
    r"\bnarrow(?:ing)?\b",
    r"\bdeepen(?:ing)?\b",
]

# D7: Affective Cultivation (AffC)
D7_LOW = [
    r"\bobserve\b",
    r"\bnotice\b",
    r"\bnote\b",
    r"\blabel\b",
    r"\bsimply\s+watch\b",
    r"\bjust\s+(?:notice|observe|watch|see)\b",
    r"\bwithout\s+(?:judging|judgment|reacting)\b",
    r"\bnon[- ]?(?:reactive|judgmental)\b",
]

D7_MID = [
    r"\bkindly\b",
    r"\bgently\b",
    r"\bwith\s+warmth\b",
    r"\bcompassionate(?:ly)?\b",
    r"\bself[- ]?compassion\b",
    r"\bsoft(?:ness|ly|en)\b",
    r"\btender(?:ness|ly)?\b",
    r"\bgentle\s+(?:smile|kindness|attention)\b",
]

D7_HIGH = [
    r"\blove\b",
    r"\bloving[- ]?kindness\b",
    r"\bcompassion\b",
    r"\bgratitude\b",
    r"\bjoy\b",
    r"\bdevotion\b",
    r"\bforgiveness\b",
    r"\bradiate\b",
    r"\bsend\b.*\b(?:love|kindness|compassion|light|warmth)\b",
    r"\boffer\b",
    r"\bgenerate\s+(?:the\s+)?feeling\b",
    r"\bmay\s+(?:you|I|we|they|all|he|she)\s+be\s+(?:happy|peaceful|safe|free|healthy|well)\b",
    r"\bmay\s+all\s+beings\b",
    r"\bheart\s+(?:full|filled|open|center|space)\b",
    r"\bfill(?:ed|ing)?\s+with\s+(?:love|light|warmth|compassion|gratitude|joy)\b",
    r"\bblessing[s]?\b",
]

# D8: Interoceptive Demand (ID)
D8_LOW = [
    r"\bsound[s]?\b",
    r"\bvisual\b",
    r"\bsee\b",
    r"\blisten\b",
    r"\bhear\b",
    r"\bexternal\b",
]

D8_HIGH = [
    r"\bsubtle\b",
    r"\bpulse\b",
    r"\bheartbeat\b",
    r"\btemperature\b",
    r"\bvibration[s]?\b",
    r"\btingling\b",
    r"\bboundary\s+of\s+sensation\b",
    r"\bfine[- ]?grained\b",
    r"\bmicro[- ]?sensation[s]?\b",
    r"\bdissolving\b",
    r"\binternal\s+(?:organ|sensation|signal)\b",
    r"\bblood\s+(?:flow|pressure)\b",
    r"\bdigest(?:ion|ive)\b",
    r"\binner\s+(?:body|sensation|feeling|landscape)\b",
]

# D9: Metacognitive Load (ML)
D9_LOW = [
    r"\bjust\s+repeat\b",
    r"\bsimply\s+follow\b",
    r"\bjust\s+(?:do|keep|continue)\b",
    r"\bno\s+need\s+to\s+(?:analyze|think|understand)\b",
]

D9_HIGH = [
    r"\bnotice\s+that\s+you(?:'re|\s+are)\s+noticing\b",
    r"\baware\s+of\s+(?:your\s+)?awareness\b",
    r"\bquality\s+of\s+(?:your\s+)?(?:attention|awareness|mind)\b",
    r"\b(?:is|are)\s+(?:the|your)\s+mind\s+(?:dull|sharp|agitated|calm|scattered|focused)\b",
    r"\bwatch\s+the\s+watcher\b",
    r"\bwho\s+is\s+(?:observing|watching|noticing|aware)\b",
    r"\bmind\s+(?:itself|watching\s+(?:itself|the\s+mind))\b",
    r"\bdullness\b",
    r"\bagitation\b",
    r"\bclarity\b",
    r"\bstability\s+of\s+(?:attention|mind|awareness)\b",
    r"\bnotice\s+(?:the\s+)?(?:quality|state|condition)\s+of\b",
    r"\baware(?:ness)?\s+(?:of|watching|observing)\s+(?:itself|awareness)\b",
    r"\brecognize\s+(?:when|that|if)\s+(?:the\s+)?mind\b",
    r"\bnature\s+of\s+(?:mind|awareness|consciousness)\b",
]

# D10: Relational Orientation (RO)
D10_LOW_SIGNALS: list[str] = [
    # Absence indicators -- if ONLY these appear, RO is low
    # (scored by absence of D10_HIGH)
]

D10_HIGH = [
    r"\bloved\s+one\b",
    r"\bfriend\b",
    r"\bdifficult\s+person\b",
    r"\bstranger\b",
    r"\ball\s+beings\b",
    r"\btheir\s+(?:suffering|pain|joy|happiness|well[- ]?being)\b",
    r"\bsend\s+(?:them|him|her|this\s+person)\b",
    r"\bimagine\s+(?:them|him|her|a\s+person|someone)\b",
    r"\btonglen\b",
    r"\bbreathe\s+in\s+(?:their|the)\s+(?:pain|suffering)\b",
    r"\bsomeone\s+(?:you\s+)?(?:love|care|know)\b",
    r"\bother\s+(?:people|beings|person)\b",
    r"\bthey\s+(?:are|feel|suffer|struggle)\b",
    r"\bperson\s+(?:who|that)\b",
    r"\bmay\s+(?:they|he|she|this\s+person)\b",
    r"\bfamily\b",
    r"\bpartner\b",
    r"\bchild(?:ren)?\b",
    r"\bmother\b",
    r"\bfather\b",
    r"\bparent[s]?\b",
    r"\bteacher\b",
    r"\bbenefactor\b",
    r"\bneighbor\b",
    r"\bcommunity\b",
]


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _count_matches(text: str, patterns: list[str]) -> tuple[int, list[str]]:
    """Count total matches and collect matched phrases."""
    total = 0
    found: list[str] = []
    text_lower = text.lower()
    for pat in patterns:
        matches = re.findall(pat, text_lower, re.IGNORECASE)
        if matches:
            total += len(matches)
            # Keep up to 3 examples per pattern
            for m in matches[:3]:
                phrase = m if isinstance(m, str) else m[0] if m else pat
                if phrase not in found:
                    found.append(phrase)
    return total, found


def _score_from_levels(low: int, mid: int, high: int, text_len: int) -> float:
    """Estimate a 0-100 score from low/mid/high indicator counts.

    Uses a weighted approach normalized by transcript length.
    """
    if low + mid + high == 0:
        return 50.0  # no signal -> neutral

    # Normalize by approximate word count (rough: 5 chars per word)
    word_count = max(text_len / 5, 1)
    # Density per 1000 words
    density_factor = 1000 / word_count

    low_d = low * density_factor
    mid_d = mid * density_factor
    high_d = high * density_factor

    # Weighted average: low pulls toward 15, mid toward 50, high toward 85
    weighted_sum = low_d * 15 + mid_d * 50 + high_d * 85
    total_weight = low_d + mid_d + high_d

    if total_weight == 0:
        return 50.0

    return min(100.0, max(0.0, weighted_sum / total_weight))


def score_d1_attentional_constraint(text: str) -> IndicatorResult:
    """Score D1: Attentional Constraint."""
    low, low_phrases = _count_matches(text, D1_LOW)
    mid, mid_phrases = _count_matches(text, D1_MID)
    high, high_phrases = _count_matches(text, D1_HIGH)

    score = _score_from_levels(low, mid, high, len(text))

    return IndicatorResult(
        dimension="D1_attentional_constraint",
        low_count=low,
        mid_count=mid,
        high_count=high,
        key_phrases=low_phrases[:5] + mid_phrases[:5] + high_phrases[:5],
        estimated_score=round(score, 1),
    )


def score_d2_somatic_engagement(text: str) -> IndicatorResult:
    """Score D2: Somatic Engagement."""
    low, low_phrases = _count_matches(text, D2_LOW)
    mid, mid_phrases = _count_matches(text, D2_MID)
    high, high_phrases = _count_matches(text, D2_HIGH)

    score = _score_from_levels(low, mid, high, len(text))

    return IndicatorResult(
        dimension="D2_somatic_engagement",
        low_count=low,
        mid_count=mid,
        high_count=high,
        key_phrases=low_phrases[:5] + mid_phrases[:5] + high_phrases[:5],
        estimated_score=round(score, 1),
    )


def score_d5_object_density(text: str) -> IndicatorResult:
    """Score D5: Object Density -- count distinct attention objects."""
    text_lower = text.lower()
    found_objects: list[str] = []

    for pat in ATTENTION_OBJECTS:
        if re.search(pat, text_lower, re.IGNORECASE):
            found_objects.append(pat.strip(r"\b"))

    n = len(found_objects)
    if n <= 1:
        score = 10.0
    elif n <= 4:
        score = 20.0 + (n - 1) * 10.0  # 30-50
    elif n <= 10:
        score = 50.0 + (n - 4) * 4.2   # ~50-75
    else:
        score = min(100.0, 75.0 + (n - 10) * 2.5)

    return IndicatorResult(
        dimension="D5_object_density",
        key_phrases=found_objects[:15],
        estimated_score=round(score, 1),
        notes=f"{n} distinct attention objects detected",
    )


def score_d6_temporal_dynamics(
    text: str,
    timestamps: list[dict[str, Any]] | None = None,
) -> IndicatorResult:
    """Score D6: Temporal Dynamics -- count phase transitions."""
    transitions, phrases = _count_matches(text, PHASE_MARKERS)

    # Also check for silence gaps if timestamps are available
    silence_ratio = 0.0
    if timestamps and len(timestamps) > 1:
        total_duration = sum(s.get("duration", 0) for s in timestamps)
        if total_duration > 0:
            last_seg = timestamps[-1]
            session_length = last_seg.get("start", 0) + last_seg.get("duration", 0)
            if session_length > 0:
                silence_ratio = 1.0 - (total_duration / session_length)

    # Score based on transition count
    if transitions <= 1:
        score = 12.5
    elif transitions <= 3:
        score = 25.0 + (transitions - 1) * 12.5
    elif transitions <= 6:
        score = 50.0 + (transitions - 3) * 8.3
    else:
        score = min(100.0, 75.0 + (transitions - 6) * 5.0)

    notes = f"{transitions} phase transitions detected"
    if silence_ratio > 0:
        notes += f", silence ratio: {silence_ratio:.1%}"

    return IndicatorResult(
        dimension="D6_temporal_dynamics",
        key_phrases=phrases[:10],
        estimated_score=round(score, 1),
        notes=notes,
    )


def score_d7_affective_cultivation(text: str) -> IndicatorResult:
    """Score D7: Affective Cultivation."""
    low, low_phrases = _count_matches(text, D7_LOW)
    mid, mid_phrases = _count_matches(text, D7_MID)
    high, high_phrases = _count_matches(text, D7_HIGH)

    score = _score_from_levels(low, mid, high, len(text))

    return IndicatorResult(
        dimension="D7_affective_cultivation",
        low_count=low,
        mid_count=mid,
        high_count=high,
        key_phrases=low_phrases[:5] + mid_phrases[:5] + high_phrases[:5],
        estimated_score=round(score, 1),
    )


def score_d8_interoceptive_demand(text: str) -> IndicatorResult:
    """Score D8: Interoceptive Demand."""
    low, low_phrases = _count_matches(text, D8_LOW)
    high, high_phrases = _count_matches(text, D8_HIGH)

    # For D8, low indicators pull down, high indicators pull up
    score = _score_from_levels(low, 0, high, len(text))

    return IndicatorResult(
        dimension="D8_interoceptive_demand",
        low_count=low,
        high_count=high,
        key_phrases=low_phrases[:5] + high_phrases[:5],
        estimated_score=round(score, 1),
    )


def score_d9_metacognitive_load(text: str) -> IndicatorResult:
    """Score D9: Metacognitive Load."""
    low, low_phrases = _count_matches(text, D9_LOW)
    high, high_phrases = _count_matches(text, D9_HIGH)

    score = _score_from_levels(low, 0, high, len(text))

    return IndicatorResult(
        dimension="D9_metacognitive_load",
        low_count=low,
        high_count=high,
        key_phrases=low_phrases[:5] + high_phrases[:5],
        estimated_score=round(score, 1),
    )


def score_d10_relational_orientation(text: str) -> IndicatorResult:
    """Score D10: Relational Orientation."""
    high, high_phrases = _count_matches(text, D10_HIGH)

    # RO is primarily about presence of relational language
    # No relational language -> low score
    word_count = max(len(text) / 5, 1)
    density = high * (1000 / word_count)

    if high == 0:
        score = 5.0
    elif density < 2:
        score = 15.0 + density * 10
    elif density < 5:
        score = 35.0 + (density - 2) * 10
    elif density < 10:
        score = 65.0 + (density - 5) * 5
    else:
        score = min(100.0, 90.0 + (density - 10) * 1)

    return IndicatorResult(
        dimension="D10_relational_orientation",
        high_count=high,
        key_phrases=high_phrases[:10],
        estimated_score=round(min(100.0, score), 1),
    )


def suggest_d3_startup_modality(text: str) -> IndicatorResult:
    """Suggest D3: Startup Modality (categorical) based on early transcript.

    Looks at roughly the first 20% of the transcript for startup cues.
    """
    # Use first 20% of text
    cutoff = len(text) // 5
    early_text = text[:max(cutoff, 500)].lower()

    suggestions: dict[str, int] = {
        "none": 0,
        "breath_regulation": 0,
        "body_scan": 0,
        "physical_movement": 0,
        "relaxation_induction": 0,
    }

    breath_pats = [
        r"deep\s+breath", r"breathe\s+(?:in|out|deeply)",
        r"inhale", r"exhale", r"count\s+(?:your\s+)?breath",
        r"pranayama", r"breath\s+(?:work|exercise)",
    ]
    scan_pats = [
        r"body\s+scan", r"scan\s+(?:through|your|the)\s+body",
        r"starting\s+(?:from|with|at)\s+(?:the|your)\s+(?:feet|toes|head|crown)",
    ]
    movement_pats = [
        r"walk(?:ing)", r"stretch", r"move\s+your",
        r"gentle\s+movement", r"mudra", r"shake",
    ]
    relaxation_pats = [
        r"relax\s+(?:your|the|each)", r"progressive\s+muscle",
        r"tension.*release", r"let\s+(?:go|the\s+tension)",
        r"soften", r"melt", r"sink\s+(?:into|down)",
    ]

    for pat in breath_pats:
        if re.search(pat, early_text):
            suggestions["breath_regulation"] += 1
    for pat in scan_pats:
        if re.search(pat, early_text):
            suggestions["body_scan"] += 1
    for pat in movement_pats:
        if re.search(pat, early_text):
            suggestions["physical_movement"] += 1
    for pat in relaxation_pats:
        if re.search(pat, early_text):
            suggestions["relaxation_induction"] += 1

    # If nothing detected, suggest "none"
    best = max(suggestions, key=suggestions.get)  # type: ignore[arg-type]
    if suggestions[best] == 0:
        best = "none"

    return IndicatorResult(
        dimension="D3_startup_modality",
        notes=f"suggested: {best} (scores: {suggestions})",
        key_phrases=[best],
    )


def suggest_d4_object_nature(text: str) -> IndicatorResult:
    """Suggest D4: Object Nature (categorical, multi-valued)."""
    text_lower = text.lower()
    categories: dict[str, int] = {
        "somatic": 0,
        "verbal": 0,
        "visual": 0,
        "auditory": 0,
        "cognitive": 0,
        "affective": 0,
        "none_open": 0,
    }

    somatic_pats = [
        r"\bbreath\b", r"\bbreathing\b", r"\bsensation[s]?\b",
        r"\bbody\b", r"\bphysical\b", r"\bpain\b",
    ]
    verbal_pats = [
        r"\bmantra\b", r"\bprayer\b", r"\baffirmation\b",
        r"\brepeat\s+(?:the|this|a)\s+(?:word|phrase)\b",
        r"\bchant\b",
    ]
    visual_pats = [
        r"\bvisualize\b", r"\bimagine\s+(?:a\s+)?(?:light|color|scene|place|ball)\b",
        r"\bpicture\b", r"\bsee\s+(?:a|the|in)\b",
        r"\bmandala\b", r"\bflame\b", r"\bcandle\b",
    ]
    auditory_pats = [
        r"\bsound[s]?\b", r"\blisten\b", r"\bhear\b",
        r"\bbell\b", r"\bambient\b", r"\bmusic\b",
    ]
    cognitive_pats = [
        r"\bthought[s]?\b", r"\bkoan\b", r"\bquestion\b",
        r"\bcontemplate\b", r"\binquiry\b", r"\bwho\s+am\s+I\b",
    ]
    affective_pats = [
        r"\bloving[- ]?kindness\b", r"\bmetta\b",
        r"\bcompassion\b", r"\bgratitude\b",
        r"\bgenerate\s+(?:the\s+)?feeling\b",
    ]
    open_pats = [
        r"\bchoiceless\b", r"\bopen\s+awareness\b",
        r"\bno\s+(?:particular\s+)?object\b",
        r"\bwhatever\s+arises\b",
    ]

    for pat in somatic_pats:
        categories["somatic"] += len(re.findall(pat, text_lower))
    for pat in verbal_pats:
        categories["verbal"] += len(re.findall(pat, text_lower))
    for pat in visual_pats:
        categories["visual"] += len(re.findall(pat, text_lower))
    for pat in auditory_pats:
        categories["auditory"] += len(re.findall(pat, text_lower))
    for pat in cognitive_pats:
        categories["cognitive"] += len(re.findall(pat, text_lower))
    for pat in affective_pats:
        categories["affective"] += len(re.findall(pat, text_lower))
    for pat in open_pats:
        categories["none_open"] += len(re.findall(pat, text_lower))

    # Return categories with non-zero counts, sorted by count
    detected = sorted(
        [(k, v) for k, v in categories.items() if v > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    primary = [k for k, v in detected[:3]] if detected else ["none_open"]

    return IndicatorResult(
        dimension="D4_object_nature",
        notes=f"detected: {dict(detected)}" if detected else "no clear object detected",
        key_phrases=primary,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_transcript(
    transcript_text: str,
    transcript_with_timestamps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Pass 1 scoring on a transcript.

    Args:
        transcript_text: full transcript text
        transcript_with_timestamps: optional list of {start, duration, text} segments

    Returns:
        Dictionary with dimension keys mapping to IndicatorResult-like dicts.
    """
    results: dict[str, Any] = {}

    # Continuous dimensions
    for scorer in [
        score_d1_attentional_constraint,
        score_d2_somatic_engagement,
        score_d5_object_density,
        score_d7_affective_cultivation,
        score_d8_interoceptive_demand,
        score_d9_metacognitive_load,
        score_d10_relational_orientation,
    ]:
        r = scorer(transcript_text)
        results[r.dimension] = {
            "low_count": r.low_count,
            "mid_count": r.mid_count,
            "high_count": r.high_count,
            "key_phrases": r.key_phrases,
            "estimated_score": r.estimated_score,
            "notes": r.notes,
        }

    # D6 needs timestamps
    r6 = score_d6_temporal_dynamics(transcript_text, transcript_with_timestamps)
    results[r6.dimension] = {
        "low_count": r6.low_count,
        "mid_count": r6.mid_count,
        "high_count": r6.high_count,
        "key_phrases": r6.key_phrases,
        "estimated_score": r6.estimated_score,
        "notes": r6.notes,
    }

    # Categorical dimensions
    r3 = suggest_d3_startup_modality(transcript_text)
    results[r3.dimension] = {
        "suggested": r3.key_phrases[0] if r3.key_phrases else "none",
        "notes": r3.notes,
    }

    r4 = suggest_d4_object_nature(transcript_text)
    results[r4.dimension] = {
        "suggested": r4.key_phrases,
        "notes": r4.notes,
    }

    return results


def format_pass1_summary(results: dict[str, Any]) -> str:
    """Format Pass 1 results as a human-readable summary."""
    lines = ["=== Pass 1 Indicator Detection ===", ""]

    for dim, data in results.items():
        if "estimated_score" in data and data["estimated_score"] is not None:
            lines.append(f"{dim}: {data['estimated_score']:.0f}/100")
            if data.get("key_phrases"):
                lines.append(f"  Phrases: {', '.join(data['key_phrases'][:5])}")
            if data.get("notes"):
                lines.append(f"  Notes: {data['notes']}")
        elif "suggested" in data:
            lines.append(f"{dim}: {data['suggested']}")
            if data.get("notes"):
                lines.append(f"  Notes: {data['notes']}")
        lines.append("")

    return "\n".join(lines)
