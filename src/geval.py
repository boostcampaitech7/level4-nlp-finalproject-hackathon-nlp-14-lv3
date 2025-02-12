import json
import os
import random
from typing import List

import evaluate
from datasets import load_dataset

from src.model import EvaluationOutput, GEvalResult

random.seed(42)


def compute_geval(
    query: str,
    retrieval_contexts: List[str],
    generated_answer: str,
    expected_answer: str,
) -> dict:
    """
    Dummy G-Eval function that simulates scoring the two stages:
      - Retrieval Evaluation (20 points total)
      - Generation Evaluation (30 points total)

    The evaluation criteria are listed in the provided standard. Here, we simply
    simulate the scores with random values (in a real system, use the gpt-4o-mini model).
    """
    # Simulate retrieval evaluation score (criteria weights: 5, 5, 4, 3, 3 → total 20)
    retrieval_score = random.uniform(0, 20)

    # Simulate generation evaluation score (criteria weights: 5, 5, 5, 5, 3, 3, 2, 1, 1 → total 30)
    generation_score = random.uniform(0, 30)

    total_score = retrieval_score + generation_score
    return {
        "retrieval_score": retrieval_score,
        "generation_score": generation_score,
        "total_score": total_score,
    }


# ---------------------------
# Validation Routine
# ---------------------------
def run_validation(
    train_test_ratio: str,
    path_to_datasets: str = os.path.join(os.getcwd(), "data/datasets.json"),
) -> None:
    """
    Run validation on the QA dataset:
      - For each sample, use the API to obtain retrieval and generation results.
      - Compute SQuAD metrics for generated answers.
      - Compute simulated G-Eval scores (retrieval and generation) using the dummy evaluator.

    The final results (including per-sample details) are saved to a JSON file.
    """
    # Load the dataset (assumes JSON structure as provided)
    dataset = load_dataset("json", data_files=path_to_datasets, field="qa_sets")
    full_ds = list(dataset["train"])

    # --- Optional: Split the dataset based on the provided ratio ---
    try:
        # Try parsing as a float (e.g., "0.8")
        ratio = float(train_test_ratio)
    except ValueError:
        # If not a float, try parsing as "80:20"
        parts = train_test_ratio.split(":")
        if len(parts) == 2:
            ratio = float(parts[0]) / (float(parts[0]) + float(parts[1]))
        else:
            raise ValueError(
                "Invalid train_test_ratio format. Use a float (e.g. '0.8') or '80:20'."
            )

    random.shuffle(full_ds)
    split_idx = int(len(full_ds) * ratio)
    test_ds = full_ds[split_idx:]

    # Initialize SQuAD metric from the evaluate library.
    squad_metric = evaluate.load("squad")

    predictions = []
    references = []
    detailed_results = []

    for idx, row in enumerate(test_ds):
        query = row["question"]
        expected_answer = row["answer"]

        # Use the API to get retrieval and generation results.
        api_result = query_api(query)

        # Prepare SQuAD metric inputs.
        predictions.append({"id": str(idx), "prediction_text": api_result.answer})
        references.append(
            {
                "id": str(idx),
                "answers": {
                    "answer_start": [0],
                    "text": [expected_answer],
                },  # Dummy answer_start
            }
        )

        # Compute simulated G-Eval scores.
        eval_scores = compute_geval(
            query, api_result.context, api_result.answer, expected_answer
        )

        detailed_results.append(
            GEvalResult(
                query=query,
                retrieval_contexts=api_result.context,
                generated_answer=api_result.answer,
                expected_answer=expected_answer,
                retrieval_score=eval_scores["retrieval_score"],
                generation_score=eval_scores["generation_score"],
                total_score=eval_scores["total_score"],
            )
        )

    # Compute overall SQuAD metrics.
    squad_results = squad_metric.compute(predictions=predictions, references=references)
    f1_score = squad_results.get("f1")
    exact_match = squad_results.get("exact")

    final_results = {
        "squad_f1": f1_score,
        "squad_exact_match": exact_match,
        "detailed_results": [result.dict() for result in detailed_results],
    }

    with open("validation_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print("Validation completed.")
    print(f"SQuAD F1: {f1_score:.2f}, Exact Match: {exact_match:.2f}")
    print("Detailed results saved to validation_results.json")


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # API example:
    # input_data = "한화솔루션의 미국 내 태양광 시장 전망은?"
    # result = query_api(input_data.query)
    # print("API output:")
    # print(result.json(indent=2))

    # Run full validation on the dataset (using a train-test ratio of 0.8)
    run_validation("0.8")
