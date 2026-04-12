#!/usr/bin/env python3
"""Tests for the CAT scoring pipeline."""

import json
import sys
import tempfile
from pathlib import Path

# Ensure imports work from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring.pass1_indicators import score_transcript, format_pass1_summary
from scoring.pipeline import run_pass1, merge_scores, read_jsonl, write_jsonl


# --- Synthetic transcripts based on Jose's archetypes ---

ANAPANASATI_TRANSCRIPT = """
Welcome to this breath meditation. Find a comfortable seated position.
Let your spine be tall, shoulders relaxed. Take three deep breaths to settle.
Breathe in deeply through your nose. And breathe out slowly.
Again, deep breath in. And slowly exhale.
One more deep breath. And release.

Now let the breath return to its natural rhythm. There is no need to control it.
Simply observe the natural breath as it flows in and out.
Bring your attention to the nostrils. Notice the subtle sensation of air
entering and leaving. Stay with this sensation.

When you notice the mind has wandered -- and it will wander -- gently return
your attention to the breath at the nostrils. No judgment. Simply come back.
Stay with the breath. Keep your focus here.

Return to the breath. Always come back to the breath.
The breath is your anchor. Stay with this single point of focus.

When thoughts arise, notice them, and return to the breath.
Stay with the breath at the nostrils. Keep your attention steady here.

And now slowly begin to widen your awareness. Take a deep breath.
Gently open your eyes when you are ready.
"""

METTA_TRANSCRIPT = """
Welcome to this loving kindness meditation. Sit comfortably and close your eyes.
Take a moment to settle into stillness.

Now bring to mind yourself. Picture yourself sitting here, just as you are.
With warmth and tenderness, begin to repeat these phrases:
May I be happy. May I be healthy. May I be safe. May I live with ease.

Generate the feeling behind the words. Feel love and compassion filling your heart.
Let warmth radiate from your heart center.
May I be happy. May I be healthy. May I be safe. May I live with ease.

Now bring to mind a loved one. Someone who makes you smile. Picture them clearly.
Send them this same love and compassion.
May you be happy. May you be healthy. May you be safe. May you live with ease.
Fill them with light and warmth.

Now think of a neutral person. Perhaps a stranger you saw today. A neighbor.
Extend this same kindness to them.
May you be happy. May you be healthy. May you be safe. May you live with ease.

Now bring to mind a difficult person. Someone who challenges you.
With courage and compassion, extend kindness even here.
May you be happy. May you be healthy. May you be safe. May you live with ease.
Offer them forgiveness and understanding.

Finally, expand to all beings everywhere.
May all beings be happy. May all beings be healthy.
May all beings be safe. May all beings live with ease.
Feel this love radiating outward in all directions, touching all beings.
Joy and compassion without limit.

Now gently return your awareness to the room. Notice the warmth in your heart.
Carry this feeling with you.
"""

BODY_SCAN_TRANSCRIPT = """
Welcome. Lie down in a comfortable position.
Take three deep breaths to settle. Breathe in. And out.

Now we will systematically move attention through the body.
Begin at the top of your head. Feel any sensations there.
Tingling, pressure, warmth, or nothing at all. Simply observe.

Move your attention to your forehead. Notice any tension.
Let it soften. Feel the skin of the forehead.

Now the eyes. The area around the eyes. Notice subtle sensations.
The jaw. Is there tightness? Feel the weight of the jaw.

Move to the neck. The throat. Feel the pulse there if you can.
The shoulders. Heavy or light? Notice the temperature.

Now the arms. Upper arms. Elbows. Forearms. Wrists.
The hands. Each finger. Feel the tingling in your fingertips.
Notice the subtle vibrations in your hands.

Now the chest. Feel the heartbeat. The rise and fall of the ribcage.
The belly. The abdomen. Feel the internal sensations.
The digestive process. The warmth of the inner body.

Move to the hips. The thighs. Notice any heaviness.
The knees. The shins. The calves.
The feet. The soles of the feet. Each toe.

Now scan back up. Observe with equanimity. Whatever sensations arise --
gross or subtle -- observe without reacting. Tingling, pressure,
heat, cold, numbness. All sensations are impermanent.
Detect the boundary between sensation and no-sensation.

Feel the vibrations throughout the entire body.
Dissolving boundaries. The inner landscape of sensation.

And now gradually bring your awareness back to the room.
Gently wiggle your fingers and toes. Open your eyes when ready.
"""


def make_record(vid: str, transcript: str) -> dict:
    """Create a minimal record for testing."""
    return {
        "video_id": vid,
        "title": f"Test: {vid}",
        "channel": "test_channel",
        "duration_seconds": 1200,
        "transcript_text": transcript,
        "transcript_with_timestamps": None,
        "search_query_source": "test",
    }


def test_pass1_anapanasati():
    """Anapanasati should score high on D1 (attentional constraint)."""
    result = score_transcript(ANAPANASATI_TRANSCRIPT)

    d1 = result["D1_attentional_constraint"]["estimated_score"]
    d2 = result["D2_somatic_engagement"]["estimated_score"]
    d7 = result["D7_affective_cultivation"]["estimated_score"]
    d10 = result["D10_relational_orientation"]["estimated_score"]

    # Anapanasati: high AC, moderate SE, low AffC, zero RO
    assert d1 > 50, f"D1 should be > 50 for anapanasati, got {d1}"
    assert d7 < 40, f"D7 should be < 40 for anapanasati, got {d7}"
    assert d10 < 20, f"D10 should be < 20 for anapanasati, got {d10}"
    print(f"  Anapanasati: D1={d1}, D2={d2}, D7={d7}, D10={d10} -- OK")


def test_pass1_metta():
    """Metta should score high on D7 (affective) and D10 (relational)."""
    result = score_transcript(METTA_TRANSCRIPT)

    d7 = result["D7_affective_cultivation"]["estimated_score"]
    d10 = result["D10_relational_orientation"]["estimated_score"]

    assert d7 > 50, f"D7 should be > 50 for metta, got {d7}"
    assert d10 > 30, f"D10 should be > 30 for metta, got {d10}"
    print(f"  Metta: D7={d7}, D10={d10} -- OK")


def test_pass1_body_scan():
    """Body scan should score high on D2 (somatic) and D8 (interoceptive)."""
    result = score_transcript(BODY_SCAN_TRANSCRIPT)

    d2 = result["D2_somatic_engagement"]["estimated_score"]
    d8 = result["D8_interoceptive_demand"]["estimated_score"]

    assert d2 > 60, f"D2 should be > 60 for body scan, got {d2}"
    assert d8 > 40, f"D8 should be > 40 for body scan, got {d8}"
    print(f"  Body scan: D2={d2}, D8={d8} -- OK")


def test_pipeline_roundtrip():
    """Test that records survive the write -> read -> score cycle."""
    records = [
        make_record("anapanasati_001", ANAPANASATI_TRANSCRIPT),
        make_record("metta_001", METTA_TRANSCRIPT),
        make_record("bodyscan_001", BODY_SCAN_TRANSCRIPT),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        # Write input
        write_jsonl(records, input_path)

        # Read back
        loaded = read_jsonl(input_path)
        assert len(loaded) == 3

        # Score each
        scored = []
        for rec in loaded:
            rec = run_pass1(rec)
            rec = merge_scores(rec)
            scored.append(rec)

        # All should have final_scores
        for rec in scored:
            assert "final_scores" in rec, f"Missing final_scores for {rec['video_id']}"
            fs = rec["final_scores"]
            assert fs.get("D1_attentional_constraint") is not None

        # Write and re-read
        write_jsonl(scored, output_path)
        reloaded = read_jsonl(output_path)
        assert len(reloaded) == 3

        print(f"  Pipeline roundtrip: {len(reloaded)} records -- OK")


def test_format_summary():
    """Test that format_pass1_summary produces readable output."""
    result = score_transcript(ANAPANASATI_TRANSCRIPT)
    summary = format_pass1_summary(result)
    assert "D1_attentional_constraint" in summary
    assert "/100" in summary
    print(f"  Format summary: {len(summary)} chars -- OK")


def test_categorical_dimensions():
    """Test D3 and D4 suggestions."""
    result = score_transcript(ANAPANASATI_TRANSCRIPT)

    d3 = result["D3_startup_modality"]
    assert d3["suggested"] == "breath_regulation", f"D3 should be breath_regulation, got {d3['suggested']}"

    d4 = result["D4_object_nature"]
    assert "somatic" in d4["suggested"], f"D4 should include somatic, got {d4['suggested']}"

    print(f"  Categoricals: D3={d3['suggested']}, D4={d4['suggested']} -- OK")


if __name__ == "__main__":
    print("Running CAT pipeline tests...\n")

    tests = [
        test_pass1_anapanasati,
        test_pass1_metta,
        test_pass1_body_scan,
        test_pipeline_roundtrip,
        test_format_summary,
        test_categorical_dimensions,
    ]

    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests.")
    sys.exit(1 if failed else 0)
