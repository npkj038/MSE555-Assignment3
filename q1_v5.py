"""
q1.py  –  Prompt Engineering: Extracting Per-Session Progress Scores

Pipeline overview
-----------------
    labeled_notes.json   ──► score ──► compute_metrics() ──► print results  (Q1a: validate prompt)
    unlabeled_notes.json ──► score ──► save                                  (Q1b: score at scale)

The LLM's job
-------------
For each client, the model receives the full sequence of session notes and
must return one progress score per consecutive note pair:

    notes 1→2 : score
    notes 2→3 : score
    ...
    notes 11→12 : score

Scores are integers 1–4, returned as a JSON list, e.g. [3, 2, 1, 2, ...].

What is already done for you
------------------------------
- Parsing and validating the LLM's JSON response
- Retrying once automatically if the response is malformed
- Looping over every client in a dataset
- Aligning true vs. predicted scores into a flat list of (true, predicted) pairs
- Building and printing the confusion matrix
- Saving all outputs to JSON

Your tasks  (search for # TODO to find each one)
--------------------------------------------------
1. build_prompt()      Write the prompt that instructs the LLM.
2. call_llm()          Wire up your chosen LLM API (OpenAI, Gemini, Anthropic, etc.).
3. compute_metrics()   Define and compute the performance metric(s) you will use
                       to evaluate and compare prompt versions.

Expected inputs:
    data/labeled_notes.json     – hand-scored by Patel; use this to test your prompt
    data/unlabeled_notes.json   – apply your validated prompt here

Expected outputs:
    output/evaluated_labeled_results.json   – scored test set with true labels (Q1a)
    output/scored_notes.json                – scored unlabeled clients (Q1b, feeds Q2)


Note: Gen-AI was used in the completion of this code
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
import os
from dotenv import load_dotenv
from openai import OpenAI

# this loads variables from your .env file into environment
load_dotenv()


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class BaseQ1Config:
    client_id_key: str = "client_id"
    notes_key: str = "notes"
    note_number_key: str = "note_number"
    note_text_key: str = "note_text"
    true_vector_key: str = "scored_progress"
    pred_vector_key: str = "estimated_trajectory_vector"

    valid_scores: tuple[int, ...] = (0, 1, 2, 3)


@dataclass
class Q1ALabeledConfig(BaseQ1Config):
    test_path: str = "data/test_notes.json"
    evaluated_output_path: str = "outputs/evaluated_labeled_results.json"


@dataclass
class Q1BUnlabeledConfig(BaseQ1Config):
    unlabeled_path: str = "data/unlabeled_notes.json"
    output_path: str = "outputs/scored_notes.json"


# ============================================================================
# DATA LOADING / SAVING
# ============================================================================

def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent folders for an output path and return it as a Path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load a top-level JSON list from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {path}.")
    return data


def save_json(data: Any, path: str) -> None:
    """Save JSON to disk and create parent folders if needed."""
    output_path = ensure_parent_dir(path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def get_vector_pair(
    record: Dict[str, Any],
    config: BaseQ1Config,
) -> tuple[str, List[int], List[int]]:
    """Pull the client id, true vector, and estimated vector from one scored record."""
    client_id = str(record[config.client_id_key])
    true_vector = record.get(config.true_vector_key, [])
    estimated_vector = record.get(config.pred_vector_key, [])
    return client_id, true_vector, estimated_vector


def build_step_comparisons(
    client_id: str,
    true_vector: List[int],
    estimated_vector: List[int],
) -> List[Dict[str, Any]]:
    """Build one row per compared step between the true and estimated vectors."""
    rows = []
    for step_idx, (true_score, estimated_score) in enumerate(
        zip(true_vector, estimated_vector),
        start=1,
    ):
        rows.append(
            {
                "client_id": client_id,
                "step_number": step_idx,
                "true_score": true_score,
                "estimated_score": estimated_score,
            }
        )
    return rows


def build_client_comparison(
    record: Dict[str, Any],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """Create the per-client comparison payload used by evaluation code."""
    client_id, true_vector, estimated_vector = get_vector_pair(record, config)
    step_rows = build_step_comparisons(client_id, true_vector, estimated_vector)
    return {
        "client_id": client_id,
        "true_vector": true_vector,
        "estimated_vector": estimated_vector,
        "n_true_scores": len(true_vector),
        "n_estimated_scores": len(estimated_vector),
        "n_compared_scores": len(step_rows),
        "step_comparisons": step_rows,
    }


def build_evaluation_comparisons(
    scored_test_data: List[Dict[str, Any]],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """Build client-level and step-level comparison tables for evaluation."""
    client_level_comparisons = []
    step_level_comparisons = []

    for record in scored_test_data:
        client_summary = build_client_comparison(record, config)
        client_level_comparisons.append(client_summary)
        step_level_comparisons.extend(client_summary["step_comparisons"])

    return {
        "n_clients": len(scored_test_data),
        "client_level_comparisons": client_level_comparisons,
        "step_level_comparisons": step_level_comparisons,
    }


def build_confusion_matrix(
    step_rows: List[Dict[str, Any]],
    valid_scores: List[int] | tuple[int, ...],
) -> Dict[str, Any]:
    """Build a confusion matrix with row totals, column totals, and a printable table."""
    matrix = {
        true_score: {estimated_score: 0 for estimated_score in valid_scores}
        for true_score in valid_scores
    }

    for row in step_rows:
        true_score = row["true_score"]
        estimated_score = row["estimated_score"]
        if true_score in matrix and estimated_score in matrix[true_score]:
            matrix[true_score][estimated_score] += 1

    row_totals = {
        true_score: sum(
            matrix[true_score][estimated_score] for estimated_score in valid_scores
        )
        for true_score in valid_scores
    }
    column_totals = {
        estimated_score: sum(
            matrix[true_score][estimated_score] for true_score in valid_scores
        )
        for estimated_score in valid_scores
    }
    grand_total = sum(row_totals.values())

    headers = ["true\\pred", *[str(score) for score in valid_scores], "Total"]
    row_label_width = max(
        len(headers[0]),
        len("Total"),
        max(len(str(score)) for score in valid_scores),
    )
    cell_width = max(
        5,
        max(
            len(str(value))
            for value in [
                *[
                    matrix[true_score][estimated_score]
                    for true_score in valid_scores
                    for estimated_score in valid_scores
                ],
                *row_totals.values(),
                *column_totals.values(),
                grand_total,
            ]
        ),
    )

    header_line = " | ".join(
        [headers[0].rjust(row_label_width)]
        + [header.rjust(cell_width) for header in headers[1:]]
    )
    separator_line = "-+-".join(
        ["-" * row_label_width] + ["-" * cell_width for _ in headers[1:]]
    )

    table_lines = [header_line, separator_line]
    for true_score in valid_scores:
        row_values = [
            str(matrix[true_score][estimated_score])
            for estimated_score in valid_scores
        ]
        row_line = " | ".join(
            [str(true_score).rjust(row_label_width)]
            + [value.rjust(cell_width) for value in row_values]
            + [str(row_totals[true_score]).rjust(cell_width)]
        )
        table_lines.append(row_line)

    total_line = " | ".join(
        ["Total".rjust(row_label_width)]
        + [
            str(column_totals[estimated_score]).rjust(cell_width)
            for estimated_score in valid_scores
        ]
        + [str(grand_total).rjust(cell_width)]
    )
    table_lines.append(separator_line)
    table_lines.append(total_line)

    return {
        "labels": list(valid_scores),
        "counts": matrix,
        "row_totals": row_totals,
        "column_totals": column_totals,
        "grand_total": grand_total,
        "table": "\n".join(table_lines),
    }


# ============================================================================
# TODO 1 of 3 — PROMPT
# ============================================================================

def build_prompt(notes_json_str: str) -> str:
    """
    Prompt that instructs the LLM to score a client's note sequence.

    Uses David Patel's rubric (0–3), SLP persona, few-shot examples,
    and a strict JSON output constraint: {"scores": [...]}
    """
    return f"""You are David Patel, a senior Speech-Language Pathologist (SLP) with 20 years of experience evaluating paediatric therapy records. You are reviewing a child's ordered session notes and scoring the progress made between each consecutive pair of sessions.

RATING SCALE
For every consecutive note pair (Session N → Session N+1), assign exactly one integer score:

  0 – Maintenance / minimal change
      The child is functioning at essentially the same level as before.
      Accuracy, cueing needs, and hierarchy level remain similar.
      Use when notes show the same goals, same cueing demands, same errors, and no clear shift.

  1 – Small but clear improvement
      Modest progress within the same general level — slightly better consistency,
      less cueing, or improved carryover — without a major increase in independence
      or hierarchy level.
      Use when there is a noticeable but incremental gain, not a step-change.

  2 – Meaningful clinical progress
      An obvious step forward that matters clinically: moving from inconsistent to
      fairly consistent performance, clearly requiring less support, or showing
      broader generalisation across activities.
      Use when the child has clearly consolidated a skill or moved up within the
      goal hierarchy in a meaningful way.

  3 – Major gain / step up in level
      A clear breakthrough: a hierarchy jump, major gain in independence, or a new
      level of spontaneous use. Use sparingly — this is a landmark session.

SCORING RULES
- Judge ONLY the change between the two notes in each pair. Do not reward high absolute performance; reward improvement relative to the previous session.
- Ground scores in observable clinical indicators: accuracy rates, cueing level (maximal / moderate / minimal / none), spontaneous use, generalisation, goal hierarchy level reached, and clinician plan decisions (e.g., introducing a harder level signals progress).
- Score 0 when goals, cueing needs, and accuracy are essentially the same — even if the child is doing well overall.
- Score 3 sparingly. Reserve it for genuine breakthroughs (hierarchy jump, major independence gain, or new spontaneous use).
- If notes are ambiguous or brief, default to the lower of the two candidate scores.

FEW-SHOT EXAMPLES

Example A — 3 notes → output has 2 scores
Session 1: "Child produced /k/ in isolation with maximal cueing (~40% accuracy). No word-level attempts."
Session 2: "Child produced /k/ in isolation with moderate cueing (~55% accuracy). Brief word-level trials with maximal support."
Session 3: "Child produced /k/ in CV syllables consistently with minimal cueing. Beginning word-level imitation with moderate support."
Correct output: {{"scores": [1, 2]}}
Reasoning: S1→S2: modest accuracy gain, same level → 1. S2→S3: moved to syllable level, less cueing → meaningful step → 2.

Example B — 3 notes → output has 2 scores
Session 1: "Practised /f/ at word level with moderate cueing, ~60% accuracy. Phrases not yet attempted."
Session 2: "Continued /f/ at word level, moderate cueing, ~65% accuracy. Began phrase level with maximal support."
Session 3: "/f/ phrases with minimal cueing, ~80% accuracy. Spontaneous use noted twice during play."
Correct output: {{"scores": [1, 3]}}
Reasoning: S1→S2: small accuracy gain plus phrase-level introduction → 1. S2→S3: high accuracy, minimal cueing, spontaneous use → major gain → 3.

Example C — 3 notes → output has 2 scores
Session 1: "/s/ in isolation inconsistently. Lateral productions frequent."
Session 2: "/s/ in isolation inconsistently. Lateral productions still frequent. No change in cueing."
Session 3: "/s/ in isolation inconsistently. Lateral productions still frequent. No change in cueing."
Correct output: {{"scores": [0, 0]}}
Reasoning: S1→S2 and S2→S3: no change in accuracy, cueing, or hierarchy → 0 both times.

END OF EXAMPLES

Now score the following client. Read every note carefully before scoring.

SESSION NOTES (JSON):
{notes_json_str}

INSTRUCTIONS
- Compare each consecutive note pair in order.
- Assign one score per pair using the rubric above.
- Return ONLY a JSON object in this exact format — no explanation, no markdown:
{{"scores": [score1, score2, ...]}}"""


# ============================================================================
# TODO 2 of 3 — LLM CALL
# ============================================================================

def call_llm(prompt: str) -> str:
    """
    Send a prompt to your chosen LLM and return the raw response text.

    Parameters
    ----------
    prompt : str
        The string returned by build_prompt().

    Returns
    -------
    str
        The model's raw text response (the pipeline will parse it).

    Instructions
    ------------
    Pick ONE of the three provider examples below, uncomment it, and add
    your API key.  Delete the other two and the raise at the bottom.

    Tips
    ----
    - Set temperature=0.0 so results are deterministic and reproducible.
    - Do not post-process the response here — return it raw.  Parsing and
      validation happen in parse_vector_from_response().
    """
    import time

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "503" in err or "502" in err:
                wait = 10 * (attempt + 1)
                print(f"\nAPI busy (attempt {attempt+1}/5), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("OpenAI API unavailable after 5 retries.")


# ============================================================================
# CLIENT-LEVEL SCORING
# ============================================================================

def parse_vector_from_response(
    response_text: str,
    expected_length: int,
    valid_scores: List[int] | tuple[int, ...] = (0, 1, 2, 3),
) -> List[int]:
    """
    Parse the model's response into one full trajectory vector.

    This function checks that:
    - the response is a JSON list
    - every item is an allowed score
    - the list length matches the number of note-to-note transitions

    Example valid response:
    [3, 2, 1]
    """
    import re

    text = response_text.strip()

    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    try:
        data = json.loads(text)

        # Accept either {"scores": [...]} (preferred) or a bare list
        if isinstance(data, dict) and "scores" in data:
            raw_list = data["scores"]
        elif isinstance(data, list):
            raw_list = data
        else:
            raise ValueError("Response is neither a list nor a dict with 'scores' key")

        cleaned = []
        for value in raw_list:
            score = int(value)
            if score not in valid_scores:
                raise ValueError(f"Invalid score value: {score}")
            cleaned.append(score)

        if len(cleaned) != expected_length:
            raise ValueError(
                f"Expected {expected_length} scores, got {len(cleaned)}"
            )
        return cleaned
    except Exception as e:
        print(f"  [PARSE ERROR] {e} | raw: {repr(response_text[:120])}")
        return []


def get_validated_vector_from_llm(
    prompt: str,
    expected_length: int,
    config: BaseQ1Config,
    client_id: str,
) -> List[int]:
    """
    Call the LLM, validate the returned vector, and retry once if needed.

    If the first response is empty or malformed, this function runs the same
    prompt one more time. If the second response is still invalid, it raises an
    error so the whole program stops instead of continuing with bad outputs.
    """
    if expected_length == 0:
        return []

    for attempt in (1, 2):
        raw_response = call_llm(prompt)
        estimated_vector = parse_vector_from_response(
            raw_response,
            expected_length=expected_length,
            valid_scores=config.valid_scores,
        )
        if estimated_vector:
            return estimated_vector

        if attempt == 1:
            print(
                f"Invalid LLM response for client {client_id}. "
                "Retrying once with the same prompt..."
            )

    raise RuntimeError(
        f"LLM returned an invalid trajectory vector twice for client {client_id}. "
        "Stopping program."
    )


def score_client_record(
    client_record: Dict[str, Any],
    config: BaseQ1Config,
) -> Dict[str, Any]:
    """
    Score one client's full note sequence.

    What this function does:
    - pulls all notes for one client
    - turns those notes into a JSON string for the prompt
    - calls the LLM once for the whole sequence
    - parses the returned vector of progress scores
    - returns one output record with the estimated vector

    If the input record already has a true scored vector, it is copied into the
    output too so the evaluation step can compare true vs estimated values.
    """
    all_notes = client_record[config.notes_key]
    client_id = str(client_record[config.client_id_key])
    notes_json_str = json.dumps(all_notes, ensure_ascii=False, indent=2)
    expected_length = max(len(all_notes) - 1, 0)

    prompt = build_prompt(notes_json_str)
    estimated_vector = get_validated_vector_from_llm(
        prompt=prompt,
        expected_length=expected_length,
        config=config,
        client_id=client_id,
    )

    scored_record = {
        config.client_id_key: client_record[config.client_id_key],
        config.notes_key: client_record[config.notes_key],
        config.pred_vector_key: estimated_vector,
    }
    if config.true_vector_key in client_record:
        scored_record[config.true_vector_key] = client_record[config.true_vector_key]
    return scored_record


def score_dataset(
    data: List[Dict[str, Any]],
    config: BaseQ1Config,
    progress_desc: str,
) -> List[Dict[str, Any]]:
    """Score every client record in a dataset and return the scored records."""
    scored = []

    for client_record in tqdm(data, desc=progress_desc):
        scored_record = score_client_record(client_record, config)
        scored.append(scored_record)

    return scored


# ============================================================================
# EVALUATION SECTION
# ============================================================================

# ============================================================================
# TODO 3 of 3 — PERFORMANCE METRICS
# ============================================================================

def compute_metrics(step_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute one or more performance metrics from the step-level comparisons.

    The assignment asks you to choose and justify an evaluation approach that
    is appropriate for this task.  Implement your chosen metric(s) here.

    Parameters
    ----------
    step_rows : List[Dict[str, Any]]
        One dict per scored note transition across all clients.
        Each dict has at minimum:
            "true_score"      – Patel's hand-assigned score (int, 1–4)
            "estimated_score" – your LLM's predicted score  (int, 1–4)
        Example:
            [
              {"client_id": "C_0011", "step_number": 1,
               "true_score": 3, "estimated_score": 2},
              {"client_id": "C_0011", "step_number": 2,
               "true_score": 2, "estimated_score": 2},
              ...
            ]

    Returns
    -------
    Dict[str, Any]
        A dict mapping metric name → value.  Whatever you return here will
        be printed by print_evaluation().  Example shape:
            {"metricA": 0.61, "metricB": 0.88}

    """
    # Step 1: extract true and predicted scores
    true_scores = [row["true_score"] for row in step_rows]
    pred_scores = [row["estimated_score"] for row in step_rows]
    n = len(true_scores)

    if n == 0:
        return {"error": "No step comparisons found"}

    # Step 2: compute metrics

    # Exact accuracy — strict match
    exact_matches = sum(t == p for t, p in zip(true_scores, pred_scores))
    exact_accuracy = exact_matches / n

    # Adjacent accuracy — within ±1 (appropriate for ordinal 0-3 scale)
    adjacent_matches = sum(abs(t - p) <= 1 for t, p in zip(true_scores, pred_scores))
    adjacent_accuracy = adjacent_matches / n

    # Mean Absolute Error — measures average magnitude of ordinal error
    mae = sum(abs(t - p) for t, p in zip(true_scores, pred_scores)) / n

    # Quadratic Weighted Kappa — gold standard for ordinal inter-rater agreement
    # Penalises large disagreements more than small ones
    num_classes = 4  # scores 0, 1, 2, 3
    # Build observed and expected frequency matrices
    O = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(true_scores, pred_scores):
        O[t][p] += 1

    true_hist = [0] * num_classes
    pred_hist = [0] * num_classes
    for t, p in zip(true_scores, pred_scores):
        true_hist[t] += 1
        pred_hist[p] += 1

    # Weight matrix: w[i][j] = (i - j)^2 / (num_classes - 1)^2
    W = [
        [(i - j) ** 2 / (num_classes - 1) ** 2 for j in range(num_classes)]
        for i in range(num_classes)
    ]

    # Expected matrix E[i][j] = true_hist[i] * pred_hist[j] / n
    E = [
        [true_hist[i] * pred_hist[j] / n for j in range(num_classes)]
        for i in range(num_classes)
    ]

    num = sum(W[i][j] * O[i][j] for i in range(num_classes) for j in range(num_classes))
    den = sum(W[i][j] * E[i][j] for i in range(num_classes) for j in range(num_classes))
    qwk = 1.0 - (num / den) if den != 0 else 0.0

    # Step 3: return metrics dict
    return {
        "n_comparisons": n,
        "exact_accuracy": round(exact_accuracy, 4),
        "adjacent_accuracy_within_1": round(adjacent_accuracy, 4),
        "mean_absolute_error": round(mae, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
    }


def evaluate_predictions(
    config: Q1ALabeledConfig,
) -> Dict[str, Any]:
    """
    Compare each client's true scored_vector with the predicted
    estimated_trajectory_vector, then compute metrics and the confusion matrix.
    """
    scored_test_data = load_json(config.evaluated_output_path)
    comparisons = build_evaluation_comparisons(scored_test_data, config)
    step_rows = comparisons["step_level_comparisons"]

    metrics = compute_metrics(step_rows)
    confusion_matrix = build_confusion_matrix(step_rows, config.valid_scores)

    return {
        **metrics,
        "confusion_matrix": confusion_matrix,
    }


def print_evaluation(results: Dict[str, Any]) -> None:
    print("\n=== Evaluation Results ===")
    for key, value in results.items():
        if key == "confusion_matrix" and isinstance(value, dict):
            print("confusion_matrix:")
            print(value.get("table", ""))
        else:
            print(f"{key}: {value}")


# ============================================================================
# PIPELINES
# ============================================================================

def run_test_pipeline(config: Q1ALabeledConfig) -> List[Dict[str, Any]]:
    """Run the Q1 pipeline on labeled test data."""
    test_data = load_json(config.test_path)

    scored_test_data = score_dataset(
        data=test_data,
        config=config,
        progress_desc="Scoring labeled clients",
    )
    save_json(scored_test_data, config.evaluated_output_path)

    results = evaluate_predictions(config)
    print_evaluation(results)

    return scored_test_data


def save_scored_notes_csv(
    scored_data: List[Dict[str, Any]],
    config: BaseQ1Config,
    csv_path: str,
) -> None:
    """
    Save scored unlabeled data as a CSV with columns: client_id, session, score.

    'session' is the 1-based index of the note-pair transition
    (session 1 = Notes 1→2, session 2 = Notes 2→3, etc.).
    """
    import csv

    output_path = ensure_parent_dir(csv_path)
    rows = []
    for record in scored_data:
        client_id = record[config.client_id_key]
        vector = record.get(config.pred_vector_key, [])
        for session_idx, score in enumerate(vector, start=1):
            rows.append({"client_id": client_id, "session": session_idx, "score": score})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "session", "score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}  ({len(rows)} rows)")


def run_unlabeled_pipeline(config: Q1BUnlabeledConfig) -> List[Dict[str, Any]]:
    """Run the Q1 pipeline on unlabeled note data and save scored outputs."""
    unlabeled_data = load_json(config.unlabeled_path)

    scored_unlabeled_data = score_dataset(
        data=unlabeled_data,
        config=config,
        progress_desc="Scoring unlabeled clients",
    )
    save_json(scored_unlabeled_data, config.output_path)

    # Q1b requirement: also save as scored_notes.csv
    csv_path = str(Path(config.output_path).parent / "scored_notes.csv")
    save_scored_notes_csv(scored_unlabeled_data, config, csv_path)

    return scored_unlabeled_data


# ============================================================================
# ENTRY POINT
# ============================================================================
#
# HOW TO WORK THROUGH THIS FILE
# ──────────────────────────────
# There are three functions marked # TODO that you must implement:
#
#   1. build_prompt()      Write the prompt that tells the LLM what to do.
#   2. call_llm()          Wire up your LLM API (uncomment one of the three
#                          provider options and add your API key).
#   3. compute_metrics()   Define the metric(s) you will use to evaluate and
#                          compare prompt versions.
#
# Recommended order:
#   Step 1 — implement build_prompt(), call_llm(), and compute_metrics()
#   Step 2 — run run_test_pipeline(LABELED_CONFIG) to score the labeled set
#             and see your metrics + confusion matrix printed to the terminal
#   Step 3 — iterate on your prompt; re-run Step 2 to compare versions
#   Step 4 — once satisfied, run run_unlabeled_pipeline(UNLABELED_CONFIG)
#             to score all 300 clients → produces scored_notes.json for Q2
#
# TIP: before running at scale, test your prompt on a single client record:
#
#   import json
#   sample = load_json("data/labeled_notes.json")[0]
#   notes_str = json.dumps(sample["notes"], indent=2)
#   print(build_prompt(notes_str))           # inspect the prompt visually
#   print(call_llm(build_prompt(notes_str))) # check the raw model response
# ============================================================================

if __name__ == "__main__":
    # Uncomment below to list available models for your API key:
    # from google import genai as _genai
    # _c = _genai.Client(api_key="PASTE_YOUR_KEY_HERE")
    # for m in _c.models.list():
    #     print(m.name)

    LABELED_CONFIG = Q1ALabeledConfig(
        test_path="data/labeled_notes.json",
        evaluated_output_path="output/evaluated_labeled_results_v5.json",
    )
    UNLABELED_CONFIG = Q1BUnlabeledConfig(
        unlabeled_path="data/unlabeled_notes.json",
        output_path="output/scored_notes_v5.json",
    )

    # Step 2: validate your prompt on the labeled test set
    run_test_pipeline(LABELED_CONFIG)

    # Step 4: score all unlabeled clients (only after prompt is validated)
    run_unlabeled_pipeline(UNLABELED_CONFIG)  
    # ← uncomment this once Q1a looks good

    